"""Elo and Nash averaging for a population of agents from a pairwise score matrix."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def elo_from_matrix(score: np.ndarray, iters: int = 400, k: float = 16.0) -> np.ndarray:
    """score[i, j] in [-1, 1] (positive: i beats j). Returns Elo ratings centred on 1000."""
    n = score.shape[0]
    r = np.full(n, 1000.0)
    prob = (np.clip(score, -1, 1) + 1) / 2  # observed win expectation
    for _ in range(iters):
        exp = 1 / (1 + 10 ** ((r[None, :] - r[:, None]) / 400))
        g = ((prob - exp) * ~np.eye(n, dtype=bool)).sum(1)
        r += k * g / max(n - 1, 1)
    return r - r.mean() + 1000.0


def nash_average(score: np.ndarray):
    """Maximum-entropy-free Nash of the antisymmetric zero-sum game ``score`` via LP.

    Returns (mixture over agents, game value, nash-averaged payoff of each agent against the mixture).
    """
    n = score.shape[0]
    A = np.asarray(score, float)
    # maximise v s.t. sum_i x_i A[i, j] >= v for all j, x in simplex
    c = np.zeros(n + 1); c[-1] = -1
    A_ub = np.hstack([-A.T, np.ones((n, 1))])
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(n), A_eq=[np.r_[np.ones(n), 0]], b_eq=[1], bounds=[(0, 1)] * n + [(None, None)])
    x = np.clip(res.x[:n], 0, None)
    x = x / x.sum()
    return x, float(res.x[-1]), A @ x
