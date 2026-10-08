"""Behaviour detectors on simulated trajectories (pincer, decoy, guard mirroring, fouls)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from kilithattu_agents.policy import join_actions
from kilithattu_core import EnvParams, make_env


def trajectory(p: EnvParams, att, dfn, seed: int = 0) -> dict:
    """Record one innings with python-level stepping -> dict of numpy arrays."""
    env = make_env(p)
    step = jax.jit(env.step)
    aact, dact = jax.jit(att.act), jax.jit(dfn.act)
    key = jax.random.PRNGKey(seed)
    key, k0, k1, k2 = jax.random.split(key, 4)
    obs, s = env.reset(k0)
    ca, cd = att.init_carry(k1), dfn.init_carry(k2)
    rec = {k: [] for k in ("c_pos", "g_x", "k_pos", "status", "kappa", "crossed", "tagged", "fouled")}
    for _ in range(p.max_ticks + p.clap_max_ticks):
        key, ka, kd, ke = jax.random.split(key, 4)
        aa, ca, _ = aact(ka, ca, obs)
        ad, cd, _ = dact(kd, cd, obs)
        obs, s, _, done, ev = step(ke, s, join_actions(aa, ad))
        rec["c_pos"].append(np.asarray(s.c_pos)); rec["g_x"].append(np.asarray(s.g_x)); rec["k_pos"].append(np.asarray(s.k_pos))
        rec["status"].append(np.asarray(s.c_status)); rec["kappa"].append(np.asarray(s.c_kappa))
        rec["crossed"].append(np.asarray(ev.crossed)); rec["tagged"].append(np.asarray(ev.tagged)); rec["fouled"].append(np.asarray(ev.fouled))
        if bool(done):
            break
    return {k: np.stack(v) for k, v in rec.items()}


def detect_pincers(p: EnvParams, tr: dict, window_s: float = 1.0, min_sep: float = 2.5) -> int:
    """Two crossers pass the same line within ``window_s`` while at least ``min_sep`` metres apart."""
    w = int(window_s / p.dt)
    cross, pos, kap = tr["crossed"], tr["c_pos"], tr["kappa"]
    n = 0
    for t in range(len(cross)):
        idx = np.flatnonzero(cross[t])
        for i in idx:
            for j in range(p.n_crossers):
                if j <= i:
                    continue
                lo, hi = max(0, t - w), min(len(cross), t + w)
                if cross[lo:hi, j].any() and abs(pos[t, i, 0] - pos[t, j, 0]) >= min_sep:
                    n += 1
                    break
    return n


def guard_mirroring(p: EnvParams, tr: dict, near: float = 2.0) -> float:
    """Mean lateral tracking error (m) of guards to the nearest crosser when one is within ``near`` m of the line."""
    errs = []
    for t in range(len(tr["g_x"])):
        for j in range(p.n_guards):
            y = (j + 1) * p.row_d
            alive = tr["status"][t] != 2
            d = np.abs(tr["c_pos"][t, :, 1] - y)
            d[~alive] = 99
            k = int(np.argmin(d))
            if d[k] < near:
                errs.append(abs(tr["g_x"][t, j] - tr["c_pos"][t, k, 0]))
    return float(np.mean(errs)) if errs else float("nan")


def detect_decoys(p: EnvParams, tr: dict, lateral: float = 2.5, window_s: float = 1.0, follow_s: float = 2.0) -> int:
    """A crosser sprints sideways near a line, and another crosser then passes that line within ``follow_s``."""
    w, f = int(window_s / p.dt), int(follow_s / p.dt)
    pos, cross = tr["c_pos"], tr["crossed"]
    n = 0
    for i in range(p.n_crossers):
        for t in range(w, len(pos) - 1, w):
            near_line = np.abs((pos[t, i, 1] / p.row_d) - np.round(pos[t, i, 1] / p.row_d)) * p.row_d < 2.0 and 0 < pos[t, i, 1] < p.field_l
            if near_line and abs(pos[t, i, 0] - pos[t - w, i, 0]) >= lateral:
                if any(cross[t : t + f, j].any() for j in range(p.n_crossers) if j != i):
                    n += 1
    return n


def behaviour_report(p: EnvParams, att, dfn, seeds=(0, 1, 2)) -> dict:
    out = {"pincers": 0, "decoys": 0, "mirroring_error": [], "fouls": 0, "tags": 0}
    for s in seeds:
        tr = trajectory(p, att, dfn, s)
        out["pincers"] += detect_pincers(p, tr); out["decoys"] += detect_decoys(p, tr)
        out["mirroring_error"].append(guard_mirroring(p, tr)); out["fouls"] += int(tr["fouled"].sum()); out["tags"] += int(tr["tagged"].sum())
    out["mirroring_error"] = float(np.nanmean(out["mirroring_error"]))
    return {k: (v / len(seeds) if k in ("pincers", "decoys", "fouls", "tags") else v) for k, v in out.items()}
