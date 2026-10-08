"""Analysis helpers for the human-AI teaming study (SUS scores, paired comparisons)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def sus_score(responses: list[int]) -> float:
    """System Usability Scale: ten 1-5 answers, odd items positive, even items negative -> 0..100."""
    assert len(responses) == 10
    odd = sum(r - 1 for r in responses[0::2])
    even = sum(5 - r for r in responses[1::2])
    return (odd + even) * 2.5


def paired_comparison(a: list[float], b: list[float]) -> dict:
    """Wilcoxon signed-rank test and a paired effect size (Cohen's dz) for within-subject conditions."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    res = stats.wilcoxon(a, b) if np.any(d != 0) else None
    return {"mean_diff": float(d.mean()), "dz": float(d.mean() / (d.std(ddof=1) + 1e-9)), "p": float(res.pvalue) if res else 1.0, "n": len(d)}


def team_score_by_condition(rows: list[dict]) -> dict:
    """rows: {"participant", "condition", "points"} -> mean points per condition."""
    out: dict[str, list[float]] = {}
    for r in rows:
        out.setdefault(r["condition"], []).append(r["points"])
    return {k: float(np.mean(v)) for k, v in out.items()}
