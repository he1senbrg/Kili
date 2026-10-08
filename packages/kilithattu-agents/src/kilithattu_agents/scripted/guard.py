"""Scripted guards: slide along the line to meet the most threatening crosser (nearest to the line),
anticipating its velocity; with P2 turn toward the threat."""
from __future__ import annotations

import jax.numpy as jnp

from kilithattu_core import EnvParams, Obs

from ._common import unpack


def init_carry(p: EnvParams):
    return {"tx": jnp.full((p.n_guards,), p.field_w / 2)}


def act(p: EnvParams, carry, obs: Obs, view: int, lag: float = 0.4, anticipation: float = 0.12):
    nG = p.n_guards
    u = unpack(obs, p, view)
    c, v, out = u["c_pos"], u["c_vel"], u["out"]
    gy = jnp.arange(1, nG + 1, dtype=jnp.float32) * p.row_d
    in_field = (c[:, 1] >= 0) & (c[:, 1] <= p.field_l) & ~out
    dy = jnp.where(in_field[None, :], jnp.abs(c[None, :, 1] - gy[:, None]), 99.0)
    j = jnp.argmin(dy, 1)
    b = dy[jnp.arange(nG), j]
    want = jnp.where(b < 4.0, c[j, 0] + v[j, 0] * anticipation, p.field_w / 2)
    tx = carry["tx"] + (jnp.clip(want, 0.0, p.field_w) - carry["tx"]) * lag
    delta = tx - u["g_x"]
    move = jnp.where(jnp.abs(delta) < 0.1, 0, jnp.where(delta < 0, 1, 2)).astype(jnp.int32)
    threat = jnp.sign(c[j, 1] - gy)
    turn = (p.p2_facing & (b < 2.5) & (threat != 0) & (threat != u["g_face"])).astype(jnp.int32)
    dash = (p.p1_momentum & (jnp.abs(delta) > 1.0)).astype(jnp.int32)
    return move, dash, turn, {"tx": tx}
