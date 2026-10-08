"""Round-robin tournaments over a population; writes payoff matrix, Elo and Nash average."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kilithattu_core import EnvParams

from .matches import play_pair
from .ratings import elo_from_matrix, nash_average


def round_robin(p: EnvParams, agents: dict[str, dict], n: int = 64, seed: int = 0) -> dict:
    names = list(agents)
    m = len(names)
    score = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            s = play_pair(p, agents[names[i]], agents[names[j]], n, seed + 17 * (i * m + j))
            score[i, j], score[j, i] = s, -s
    elo = elo_from_matrix(score)
    mix, value, vs_nash = nash_average(score) if m > 1 else (np.ones(1), 0.0, np.zeros(1))
    return {"names": names, "score": score, "elo": elo, "nash_mix": mix, "nash_value": value, "vs_nash": vs_nash}


def save_tournament(res: dict, out: str | Path) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "score_matrix.csv", res["score"], delimiter=",", header=",".join(res["names"]))
    (out / "ratings.json").write_text(json.dumps({"names": res["names"], "elo": res["elo"].tolist(), "nash_mix": res["nash_mix"].tolist(), "nash_value": res["nash_value"], "vs_nash": res["vs_nash"].tolist()}, indent=2))
