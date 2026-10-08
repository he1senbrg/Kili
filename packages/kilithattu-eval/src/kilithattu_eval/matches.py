"""Match play and summary metrics."""
from __future__ import annotations

import numpy as np

from kilithattu_core import EnvParams
from kilithattu_train.play import play_innings


def summarize(p: EnvParams, r: dict, win_points: float = 18.0) -> dict:
    n = len(r["points"])
    pts = r["points"]
    return {
        "innings": n,
        "points_mean": float(pts.mean()), "points_std": float(pts.std()),
        "points_norm": float(pts.mean() / (p.n_crossers * p.trip_points)),
        "trips_mean": float(r["trips"].mean()),
        "win_rate": float((pts >= win_points).mean()),
        "all_out_rate": float(r["all_out"].mean()),
        "tags_per_innings": float(r["tags"].mean()), "fouls_per_innings": float(r["fouls"].mean()),
        "live_seconds": float(r["ticks"].mean() * p.dt),
    }


def play_match(p: EnvParams, att, dfn, n: int = 64, seed: int = 0, win_points: float = 18.0) -> dict:
    return summarize(p, play_innings(p, att, dfn, n, seed), win_points)


def play_pair(p: EnvParams, A, B, n: int = 64, seed: int = 0) -> float:
    """Score of agent A against agent B in [-1, 1]. Each agent is a dict {"attack": Policy, "defence": Policy};
    A's attack faces B's defence, then B's attack faces A's defence. Positive means A is stronger."""
    ra = play_innings(p, A["attack"], B["defence"], n, seed)
    rb = play_innings(p, B["attack"], A["defence"], n, seed + 1)
    scale = p.n_crossers * p.trip_points
    return float((ra["points"].mean() - rb["points"].mean()) / scale)
