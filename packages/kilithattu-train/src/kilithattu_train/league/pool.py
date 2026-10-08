"""League pool: frozen snapshots per side, a payoff table, persistence."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kilithattu_agents import save_checkpoint
from kilithattu_core import EnvParams
from kilithattu_agents.networks import NetConfig


@dataclass
class Snapshot:
    id: str
    side: str  # attack | defence
    kind: str  # main | main_exploiter | league_exploiter | past
    step: int
    params: Any  # {"actor", "critic"}


@dataclass
class LeaguePool:
    snaps: dict[str, Snapshot] = field(default_factory=dict)
    payoff: dict[tuple[str, str], float] = field(default_factory=dict)  # (attack id, defence id) -> EMA of attack win rate
    points: dict[tuple[str, str], float] = field(default_factory=dict)  # EMA of mean attack points
    ema: float = 0.5

    def add(self, snap: Snapshot) -> None:
        self.snaps[snap.id] = snap

    def side(self, side: str) -> list[Snapshot]:
        return [s for s in self.snaps.values() if s.side == side]

    def record(self, att_id: str, def_id: str, win_rate: float, points: float) -> None:
        k = (att_id, def_id)
        self.payoff[k] = win_rate if k not in self.payoff else (1 - self.ema) * self.payoff[k] + self.ema * win_rate
        self.points[k] = points if k not in self.points else (1 - self.ema) * self.points[k] + self.ema * points

    def attack_win(self, att_id: str, def_id: str, default: float = 0.5) -> float:
        return self.payoff.get((att_id, def_id), default)

    def matrix(self):
        att = [s.id for s in self.side("attack")]
        dfn = [s.id for s in self.side("defence")]
        m = np.full((len(att), len(dfn)), np.nan)
        for i, a in enumerate(att):
            for j, d in enumerate(dfn):
                if (a, d) in self.payoff:
                    m[i, j] = self.payoff[(a, d)]
        return att, dfn, m

    def save(self, run_dir: Path, env_params: EnvParams, net: NetConfig, run_id: str) -> None:
        d = run_dir / "league"
        d.mkdir(parents=True, exist_ok=True)
        for s in self.snaps.values():
            if not (d / s.id).exists():
                save_checkpoint(d / s.id, s.params["actor"], env_params, net, s.side, run_id, s.step, extra={"kind": s.kind}, critic_params=s.params["critic"])
        meta = {"snapshots": [{"id": s.id, "side": s.side, "kind": s.kind, "step": s.step} for s in self.snaps.values()],
                "payoff": [[a, d_, v] for (a, d_), v in self.payoff.items()], "points": [[a, d_, v] for (a, d_), v in self.points.items()]}
        (d / "league.json").write_text(json.dumps(meta, indent=2))
