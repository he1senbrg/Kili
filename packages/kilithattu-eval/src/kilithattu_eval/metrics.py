"""Population and outcome metrics from PLAN.md section 10.1."""
from __future__ import annotations

import numpy as np


def outcome_entropy(points: np.ndarray, n_bins: int = 6) -> float:
    """Shannon entropy (bits) of the binned innings-points distribution: higher = more varied outcomes."""
    hist, _ = np.histogram(points, bins=n_bins)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def competitiveness(points: np.ndarray, trips: np.ndarray, trip_points: int = 18) -> dict:
    return {"p_zero": float((points == 0).mean()), "p_three_trips": float((trips >= 3).mean()), "entropy": outcome_entropy(points),
            "ppi": float(points.mean())}


def nontransitivity(score: np.ndarray, tol: float = 0.02) -> float:
    """Fraction of agent triples forming a rock-paper-scissors cycle in the pairwise score matrix."""
    n = score.shape[0]
    beats = score > tol
    cyc = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                total += 1
                if (beats[i, j] and beats[j, k] and beats[k, i]) or (beats[j, i] and beats[k, j] and beats[i, k]):
                    cyc += 1
    return cyc / total if total else 0.0


def normalised_retention(held_out_ppi: float, in_dist_ppi: float) -> float:
    """E3: held-out points per innings relative to in-distribution (1.0 = no loss)."""
    return float(held_out_ppi / in_dist_ppi) if in_dist_ppi > 0 else float("nan")
