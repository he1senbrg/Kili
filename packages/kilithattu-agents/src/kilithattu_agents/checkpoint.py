"""Checkpoints: flax msgpack parameters + metadata.json with provenance and compatibility checks."""
from __future__ import annotations

import dataclasses
import json
import subprocess
import warnings
from pathlib import Path

from flax import serialization

import kilithattu_core
from kilithattu_core import EnvParams, load_params, rules_hash

from .networks import NetConfig
from .policy import init_networks


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def save_checkpoint(path: str | Path, actor_params, env_params: EnvParams, cfg: NetConfig, side: str, run_id: str = "adhoc",
                    step: int = 0, role_heads: list[str] | None = None, extra: dict | None = None, critic_params=None) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "actor.msgpack").write_bytes(serialization.to_bytes(actor_params))
    if critic_params is not None:
        (path / "critic.msgpack").write_bytes(serialization.to_bytes(critic_params))
    meta = {
        "run_id": run_id, "step": step, "side": side,
        "role_heads": role_heads or (["crosser"] if side == "attack" else ["guard", "kili"]),
        "git_commit": _git_commit(), "env_version": kilithattu_core.ENV_VERSION, "rules_hash": rules_hash(env_params),
        "config": {"env_params": dataclasses.asdict(env_params)}, "network": dataclasses.asdict(cfg), **(extra or {}),
    }
    (path / "metadata.json").write_text(json.dumps(meta, indent=2))
    return path


def load_checkpoint(path: str | Path, env_params: EnvParams | None = None):
    """-> (actor_params, NetConfig, metadata). Fails on a major ENV_VERSION mismatch, warns on a rules-hash mismatch."""
    import jax

    path = Path(path)
    meta = json.loads((path / "metadata.json").read_text())
    if meta["env_version"].split(".")[0] != kilithattu_core.ENV_VERSION.split(".")[0]:
        raise RuntimeError(f"checkpoint env_version {meta['env_version']} incompatible with {kilithattu_core.ENV_VERSION}")
    train_params = load_params(meta["config"]["env_params"])
    if env_params is not None and rules_hash(env_params) != meta["rules_hash"]:
        warnings.warn("policy was trained on different rules (rules_hash mismatch)")
    cfg = NetConfig(**meta["network"])
    _, _, a_tmpl, c_tmpl = init_networks(jax.random.PRNGKey(0), train_params, cfg)
    actor = serialization.from_bytes(a_tmpl, (path / "actor.msgpack").read_bytes())
    return actor, cfg, meta
