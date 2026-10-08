"""Deterministic replays: configuration + seed + action log. ``verify_replay`` re-simulates and compares."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import ENV_VERSION
from .config import EnvParams, load_params, rules_hash
from .env import KilithattuEnv, make_env
from .state import Actions, EnvState


def state_hash(s: EnvState) -> str:
    h = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(s):
        h.update(np.asarray(leaf).tobytes())
    return h.hexdigest()[:16]


def stack_actions(actions: list[Actions]) -> np.ndarray:
    """-> int8 array [ticks, agents, 5 heads]."""
    return np.stack([np.stack([np.asarray(a.move), np.asarray(a.dash), np.asarray(a.turn), np.asarray(a.clap), np.asarray(a.call)], -1) for a in actions]).astype(np.int8)


def _to_actions(arr) -> Actions:
    arr = jnp.asarray(arr, jnp.int32)
    return Actions(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4])


def simulate(env: KilithattuEnv, seed: int, action_log: np.ndarray):
    """Re-run an action log. Returns (final_state, events list of dicts, per-tick points)."""
    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)
    _, s = env.reset(k0)
    step = jax.jit(env.step)
    events = []
    for t in range(len(action_log)):
        key, k = jax.random.split(key)
        _, s, _, done, ev = step(k, s, _to_actions(action_log[t]))
        for name in ("tagged", "fouled", "side_exit", "false_start", "crossed", "delivered"):
            for i in np.flatnonzero(np.asarray(getattr(ev, name))):
                events.append({"tick": t, "type": name, "crosser": int(i)})
        if bool(ev.clapped):
            events.append({"tick": t, "type": "clap"})
        if bool(done):
            break
    return s, events


def write_replay(path: str | Path, params: EnvParams, seed: int, action_log: np.ndarray, controllers: dict | None = None) -> dict:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    env = make_env(params)
    final, events = simulate(env, seed, action_log)
    header = {
        "format": 1, "env_version": ENV_VERSION, "rules_hash": rules_hash(params), "params": dataclasses.asdict(params),
        "seed": seed, "ticks": int(len(action_log)), "controllers": controllers or {}, "final_hash": state_hash(final),
        "points": float(final.points),
    }
    (path / "header.json").write_text(json.dumps(header, indent=2))
    np.savez_compressed(path / "actions.npz", actions=action_log)
    (path / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    return header


def read_replay(path: str | Path):
    path = Path(path)
    header = json.loads((path / "header.json").read_text())
    actions = np.load(path / "actions.npz")["actions"]
    return header, actions


def verify_replay(path: str | Path) -> bool:
    header, actions = read_replay(path)
    if header["env_version"].split(".")[0] != ENV_VERSION.split(".")[0]:
        return False
    params = load_params(header["params"])
    final, _ = simulate(make_env(params), header["seed"], actions)
    return state_hash(final) == header["final_hash"]
