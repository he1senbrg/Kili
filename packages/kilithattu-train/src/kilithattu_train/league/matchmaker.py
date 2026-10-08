"""Prioritised fictitious self-play (PFSP) opponent sampling."""
from __future__ import annotations

import numpy as np


def pfsp_weights(p_win_learner: np.ndarray, power: float = 2.0) -> np.ndarray:
    """Weight opponents the learner beats rarely: f(p) = (1 - p)^power, normalised (uniform if all are beaten)."""
    w = (1.0 - np.clip(p_win_learner, 0.0, 1.0)) ** power + 1e-3
    return w / w.sum()


def sample_opponent(rng: np.random.Generator, ids: list[str], p_win_learner: np.ndarray, power: float = 2.0) -> str:
    return ids[int(rng.choice(len(ids), p=pfsp_weights(np.asarray(p_win_learner, np.float64), power)))]
