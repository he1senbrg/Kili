"""Scripted Kili: pick the nearest live crosser, head for the line it is about to cross, moving only along
the line graph (turn at junctions)."""
from __future__ import annotations

import jax.numpy as jnp

from kilithattu_core import EnvParams, Obs

from ._common import unpack


def act(p: EnvParams, obs: Obs, view: int, lead: float = 0.15):
    u = unpack(obs, p, view)
    c, v, out, kappa, k = u["c_pos"], u["c_vel"], u["out"], u["kappa"], u["k_pos"]
    in_field = (c[:, 1] >= 0) & (c[:, 1] <= p.field_l) & ~out
    q = jnp.where(in_field, jnp.abs(c[:, 0] - k[0]) + jnp.abs(c[:, 1] - k[1]), 1e9)
    j = jnp.argmin(q)
    has = jnp.any(in_field)
    nxt = jnp.where(kappa[j] < p.K, jnp.ceil(c[j, 1] / p.row_d), jnp.floor(c[j, 1] / p.row_d)) * p.row_d
    tx = jnp.where(has, jnp.clip(c[j, 0] + v[j, 0] * lead, 0.0, p.field_w), p.field_w / 2)
    ty = jnp.where(has, jnp.clip(nxt, 0.0, p.field_l), jnp.round(p.n_rows / 2) * p.row_d)
    snap = max(p.v_kili, p.dash_kili if p.p1_momentum else 0.0) * p.dt
    nl = jnp.clip(jnp.round(k[0] / p.lane_w) * p.lane_w, 0.0, p.field_w)
    nt = jnp.clip(jnp.round(k[1] / p.row_d) * p.row_d, 0.0, p.field_l)
    near_l, near_t = jnp.abs(k[0] - nl) <= snap, jnp.abs(k[1] - nt) <= snap
    want_v = jnp.abs(ty - k[1]) > jnp.abs(tx - k[0])
    vert = jnp.where(jnp.abs(ty - k[1]) > 0.15, jnp.where(ty > k[1], 1, 3), 0)
    horiz = jnp.where(jnp.abs(tx - k[0]) > 0.15, jnp.where(tx > k[0], 2, 4), 0)
    to_l = jnp.where(jnp.abs(nl - k[0]) > 1e-3, jnp.where(nl > k[0], 2, 4), 0)
    to_t = jnp.where(jnp.abs(nt - k[1]) > 1e-3, jnp.where(nt > k[1], 1, 3), 0)
    move = jnp.where(want_v, jnp.where(near_l, vert, to_l), jnp.where(near_t, horiz, to_t)).astype(jnp.int32)
    # turn toward the nearest threat (P2), dash when far from the target (P1)
    threat = jnp.sign(c[j, 1] - k[1])
    face_k = obs.players[view][-1, 15]
    turn = (p.p2_facing & has & (jnp.abs(c[j, 1] - k[1]) < 2.5) & (threat != 0) & (threat != face_k)).astype(jnp.int32)
    dash = (p.p1_momentum & (jnp.abs(ty - k[1]) + jnp.abs(tx - k[0]) > 2.0)).astype(jnp.int32)
    return move, dash, turn
