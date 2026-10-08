"""Zero-shot generalisation of a policy to other fields and rule variants."""
from __future__ import annotations

from kilithattu_agents import NetworkPolicy, ScriptedDefence
from kilithattu_agents.networks import NetConfig
from kilithattu_core import EnvParams

from .matches import play_match

DEFAULT_VARIANTS = {
    "classic": {},
    "wider_3_lanes": {"n_lanes": 3},
    "deeper_7_rows": {"n_rows": 7},
    "five_crossers": {"n_crossers": 5},
    "fast_guards": {"v_guard": 3.6},
}


def evaluate_variants(base: EnvParams, actor_params, net: NetConfig, variants: dict | None = None, n: int = 32, seed: int = 0) -> dict:
    out = {}
    for name, kw in (variants or DEFAULT_VARIANTS).items():
        p = base.replace(**kw)
        out[name] = play_match(p, NetworkPolicy("attack", p, net, actor_params), ScriptedDefence(p), n, seed)
    return out
