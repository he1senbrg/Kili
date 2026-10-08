"""Helpers shared by the scripted bots. Bots read true positions from the un-fogged token view of one of
their own agents; they assume full observability (use them without P3)."""
from __future__ import annotations

import jax.numpy as jnp

from kilithattu_core import EnvParams, Obs
from kilithattu_core.spaces import F_ABS, F_FACE, F_KAPPA, F_STATUS, F_VEL

_OCT = jnp.array([3, 2, 1, 8, 7, 6, 5, 4], jnp.int32)  # angle octant (E, NE, N, NW, W, SW, S, SE) -> move id


def unpack(obs: Obs, p: EnvParams, view: int):
    nA, nG = p.n_crossers, p.n_guards
    tok = obs.players[view]
    scale = jnp.array([p.field_w, p.field_l])
    pos = tok[:, F_ABS] * scale
    vel = tok[:, F_VEL] * p.v_max
    out = tok[:nA, F_STATUS][:, 2] > 0.5
    kappa = jnp.round(tok[:nA, F_KAPPA] * 2 * p.K).astype(jnp.int32)
    return {
        "c_pos": pos[:nA], "c_vel": vel[:nA], "kappa": kappa, "out": out, "g_x": pos[nA : nA + nG, 0],
        "g_face": tok[nA : nA + nG, F_FACE], "k_pos": pos[-1], "pre": obs.glob[view, 1] > 0.5,
    }


def unit(x, y):
    m = jnp.sqrt(x * x + y * y)
    return jnp.where((m < 0.12)[..., None], 0.0, jnp.stack([x, y], -1) / jnp.maximum(m, 1e-6)[..., None])


def to_move(vec):
    mag = jnp.linalg.norm(vec, axis=-1)
    idx = jnp.mod(jnp.round(jnp.arctan2(vec[..., 1], vec[..., 0]) / (jnp.pi / 4)).astype(jnp.int32), 8)
    return jnp.where(mag < 0.12, 0, _OCT[idx])
