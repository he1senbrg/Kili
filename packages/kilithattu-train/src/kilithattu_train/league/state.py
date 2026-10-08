"""Resume support: read the persisted league (snapshots + payoff) back into a pool."""
from __future__ import annotations

import json
from pathlib import Path

from kilithattu_agents import load_checkpoint

from .pool import LeaguePool, Snapshot


def load_pool(run_dir: str | Path) -> LeaguePool:
    d = Path(run_dir) / "league"
    meta = json.loads((d / "league.json").read_text())
    pool = LeaguePool()
    for s in meta["snapshots"]:
        actor, cfg, m = load_checkpoint(d / s["id"])
        critic = None
        pool.add(Snapshot(s["id"], s["side"], s["kind"], s["step"], {"actor": actor, "critic": critic}))
    pool.payoff = {(a, b): v for a, b, v in meta["payoff"]}
    pool.points = {(a, b): v for a, b, v in meta["points"]}
    return pool
