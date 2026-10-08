"""Matplotlib figures: learning curves, payoff heatmaps, rating bars."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def read_metrics(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def learning_curve(metrics_path: str | Path, out: str | Path, keys=("ep/points", "eval/points")) -> Path:
    rows = read_metrics(metrics_path)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for k in keys:
        xs = [r["step"] for r in rows if k in r]
        ys = [r[k] for r in rows if k in r]
        if xs:
            ax.plot(xs, ys, label=k)
    ax.set_xlabel("update"); ax.set_ylabel("points / innings"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return Path(out)


def heatmap(matrix: np.ndarray, names_x: list[str], names_y: list[str], out: str | Path, title: str = "") -> Path:
    fig, ax = plt.subplots(figsize=(1 + 0.6 * len(names_x), 1 + 0.5 * len(names_y)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-1 if np.nanmin(matrix) < 0 else 0, vmax=1)
    ax.set_xticks(range(len(names_x)), names_x, rotation=70, ha="right", fontsize=7)
    ax.set_yticks(range(len(names_y)), names_y, fontsize=7)
    ax.set_title(title); fig.colorbar(im); fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return Path(out)


def rating_bars(names: list[str], elo: np.ndarray, out: str | Path) -> Path:
    order = np.argsort(elo)
    fig, ax = plt.subplots(figsize=(5, 0.4 * len(names) + 1))
    ax.barh([names[i] for i in order], elo[order]); ax.set_xlabel("Elo"); fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return Path(out)
