"""Scripted baselines implementing the Policy protocol (same interface as the RL networks)."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from kilithattu_core import Actions, EnvParams, Obs

from . import crosser, guard, kili


class ScriptedAttack:
    side = "attack"

    def __init__(self, p: EnvParams, aggression: float = 1.0, safety: float = 0.0):
        self.p, self.aggression, self.safety = p, aggression, safety

    def init_carry(self, key):
        return crosser.init_carry(self.p, key, self.aggression)

    def act(self, key, carry, obs: Obs):
        a, carry = crosser.act(self.p, carry, obs, self.safety)
        return a, carry, {}


class ScriptedDefence:
    side = "defence"

    def __init__(self, p: EnvParams, reaction: float = 0.4, anticipation: float = 0.12):
        self.p, self.reaction, self.anticipation = p, reaction, anticipation

    def init_carry(self, key):
        return guard.init_carry(self.p)

    def act(self, key, carry, obs: Obs):
        p = self.p
        view = p.n_crossers  # first defender's view
        gm, gd, gt, carry = guard.act(p, carry, obs, view, self.reaction, self.anticipation)
        km, kd, kt = kili.act(p, obs, view)
        n = p.n_defenders
        z = jnp.zeros((n,), jnp.int32)
        move = jnp.concatenate([gm, km[None]])
        dash = jnp.concatenate([gd, kd[None]])
        turn = jnp.concatenate([gt, kt[None]])
        clap = z.at[-1].set(1)
        return Actions(move, dash, turn, clap, z), carry, {}


__all__ = ["ScriptedAttack", "ScriptedDefence"]
