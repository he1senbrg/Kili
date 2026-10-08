"""Game-balance search with Optuna: sweep speeds, tag reach and field depth so the game is competitive."""
from __future__ import annotations

import optuna

from kilithattu_core import EnvParams, load_params
from kilithattu_agents import ScriptedAttack, ScriptedDefence

from .matches import play_match


def objective_value(m: dict, target_win: float = 0.5) -> float:
    """Lower is better: attack win rate near target, all-out rate inside [0.2, 0.8], some tags and few degenerate stalls."""
    pen = abs(m["win_rate"] - target_win)
    pen += max(0.0, 0.2 - m["all_out_rate"]) + max(0.0, m["all_out_rate"] - 0.8)
    pen += 0.05 * (m["tags_per_innings"] < 0.5)
    return float(pen)


def search(base: EnvParams, n_trials: int = 20, episodes: int = 32, att=None, dfn=None, seed: int = 0, bounds: dict | None = None):
    b = {"v_guard": (2.4, 4.6), "v_kili": (2.8, 5.0), "r_tag": (0.35, 0.7), "row_d": (2.5, 3.5)}
    b.update(bounds or {})

    def obj(trial: optuna.Trial) -> float:
        p = base.replace(**{k: trial.suggest_float(k, *v) for k, v in b.items()})
        a = att or ScriptedAttack(p)
        d = dfn or ScriptedDefence(p)
        m = play_match(p, a, d, episodes, seed + trial.number)
        trial.set_user_attr("metrics", m)
        return objective_value(m)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(obj, n_trials=n_trials)
    return study
