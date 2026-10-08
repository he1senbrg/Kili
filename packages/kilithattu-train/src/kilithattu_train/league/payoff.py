"""Payoff-table helpers: win-rate matrix, opponent hardness and a Nash support estimate."""
from __future__ import annotations

import numpy as np

from .pool import LeaguePool


def hardness(pool: LeaguePool, side: str) -> dict[str, float]:
    """Mean win rate of the *other* side against each snapshot of ``side`` (higher = easier to beat)."""
    att, dfn, m = pool.matrix()
    out: dict[str, float] = {}
    if side == "defence":
        for j, d in enumerate(dfn):
            col = m[:, j]
            out[d] = float(np.nanmean(col)) if np.isfinite(col).any() else 0.5
    else:
        for i, a in enumerate(att):
            row = 1.0 - m[i, :]
            out[a] = float(np.nanmean(row)) if np.isfinite(row).any() else 0.5
    return out
