"""Environment parameters, rule toggles (Classic / Kilithattu+), YAML presets and the rules hash."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EnvParams:
    """All rule parameters. Frozen and hashable, so it can be closed over by jitted functions.

    Defaults are the Classic rule set of PLAN.md section 4.6 with balance v1 speeds (RULES.md section 9). Toggles P1-P4 and P6 are implemented in the
    environment; P5 (procedural fields) is realised by building one environment per sampled field size
    (see ``sample_field``); P7 (human-fair reflexes) lives in ``kilithattu_agents.human_fair``.
    """

    # field geometry (metres)
    n_lanes: int = 2
    n_rows: int = 5
    n_crossers: int = 4
    lane_w: float = 3.0
    row_d: float = 3.0
    zone: float = 2.0
    # time
    dt: float = 0.05
    innings_s: float = 120.0
    clap_auto_s: float = 2.0
    # tagging and boundaries
    r_tag: float = 0.5
    side_tol: float = 0.1
    # classic speeds (m/s)
    v_cross: float = 5.0
    v_guard: float = 3.0
    v_kili: float = 3.6
    # --- rule toggles ---
    p1_momentum: bool = False  # acceleration limit, sprint / dash with stamina
    p2_facing: bool = False  # facing, 180 degree tag cone, turning time
    p3_vision: bool = False  # limited vision (fog of war) in observations
    p4_clap: bool = False  # strategic clap window, false start = out
    p6_calls: bool = False  # discrete team calls
    # P1
    walk_cross: float = 3.5
    sprint_cross: float = 6.0
    walk_guard: float = 3.5
    dash_guard: float = 5.0
    walk_kili: float = 4.0
    dash_kili: float = 5.5
    a_max: float = 20.0
    stam_drain: float = 0.35
    stam_regen: float = 0.20
    stam_min: float = 0.10
    # P2
    t_turn: float = 0.30
    # P3 (radius in metres)
    vis_cross: float = 9.0
    vis_guard: float = 9.0
    vis_kili: float = 12.0
    # P4
    clap_min_s: float = 1.0
    clap_max_s: float = 5.0
    # P6
    n_calls: int = 4
    call_period: int = 10

    # ---- derived quantities ----
    @property
    def field_w(self) -> float:
        return self.n_lanes * self.lane_w

    @property
    def field_l(self) -> float:
        return self.n_rows * self.row_d

    @property
    def n_guards(self) -> int:
        return self.n_rows - 1

    @property
    def n_defenders(self) -> int:
        return self.n_guards + 1

    @property
    def n_agents(self) -> int:
        return self.n_crossers + self.n_defenders

    @property
    def n_lines_t(self) -> int:  # transverse lines L0..Lnb
        return self.n_rows + 1

    @property
    def K(self) -> int:  # lines per direction
        return self.n_rows + 1

    @property
    def trip_points(self) -> int:  # 2K line points + 6 bonus  (= 18 by default)
        return 2 * self.K + 6

    @property
    def max_ticks(self) -> int:
        return round(self.innings_s / self.dt)

    @property
    def clap_auto_ticks(self) -> int:
        return round(self.clap_auto_s / self.dt)

    @property
    def clap_min_ticks(self) -> int:
        return round(self.clap_min_s / self.dt)

    @property
    def clap_max_ticks(self) -> int:
        return round(self.clap_max_s / self.dt)

    @property
    def v_max(self) -> float:
        return max(self.v_cross, self.sprint_cross, self.dash_guard, self.dash_kili, self.v_guard, self.v_kili)

    def replace(self, **kw: Any) -> EnvParams:
        return dataclasses.replace(self, **kw)


def rules_hash(p: EnvParams) -> str:
    blob = json.dumps(dataclasses.asdict(p), sort_keys=True).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()[:16]


def _configs_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs" / "rules").is_dir():
            return parent / "configs" / "rules"
    return here.parent / "presets"


def load_params(source: str | Path | dict | None = None, **overrides: Any) -> EnvParams:
    """Load parameters from a preset name (``classic``/``plus``), a YAML path, a dict, or defaults."""
    data: dict[str, Any] = {}
    if isinstance(source, dict):
        data = dict(source)
    elif source is not None:
        path = Path(source)
        if not path.exists():
            path = _configs_dir() / f"{source}.yaml"
        data = yaml.safe_load(path.read_text()) or {}
    data.update(overrides)
    valid = {f.name for f in dataclasses.fields(EnvParams)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"unknown EnvParams keys: {sorted(unknown)}")
    return EnvParams(**data)


CLASSIC = EnvParams()
PLUS = EnvParams(p1_momentum=True, p2_facing=True, p3_vision=False, p4_clap=True)


def sample_field(rng, ranges: dict | None = None, base: EnvParams = CLASSIC) -> EnvParams:
    """P5: sample a field configuration. Shapes are static per environment, so the trainer builds one
    environment per sampled configuration and round-robins between them."""
    r = {"n_lanes": (2, 3), "n_rows": (4, 9), "n_crossers": (3, 6)}
    r.update(ranges or {})
    pick = lambda lo_hi: int(rng.integers(lo_hi[0], lo_hi[1] + 1))
    return base.replace(n_lanes=pick(r["n_lanes"]), n_rows=pick(r["n_rows"]), n_crossers=pick(r["n_crossers"]))
