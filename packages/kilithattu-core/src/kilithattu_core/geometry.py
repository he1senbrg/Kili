"""Field geometry: boxes, line graph for the Kili, safe zones."""
from __future__ import annotations

import jax.numpy as jnp

from .config import EnvParams


def box_index(p: EnvParams, pos):
    """Box id ``lane * n_rows + row`` for positions inside the field (0 <= y <= L), else -1."""
    x, y = pos[..., 0], pos[..., 1]
    j = jnp.clip(jnp.floor(x / p.lane_w), 0, p.n_lanes - 1).astype(jnp.int32)
    k = jnp.clip(jnp.floor(y / p.row_d), 0, p.n_rows - 1).astype(jnp.int32)
    inside = (y >= 0) & (y <= p.field_l)
    return jnp.where(inside, j * p.n_rows + k, -1)


def in_safe_zone(p: EnvParams, pos):
    y = pos[..., 1]
    return (y < 0) | (y > p.field_l)


def nearest_junction_lines(p: EnvParams, pos):
    """Nearest longitudinal x and transverse y (exact multiples of lane_w / row_d)."""
    nl = jnp.clip(jnp.round(pos[..., 0] / p.lane_w) * p.lane_w, 0.0, p.field_w)
    nt = jnp.clip(jnp.round(pos[..., 1] / p.row_d) * p.row_d, 0.0, p.field_l)
    return nl, nt


def kili_snap(p: EnvParams) -> float:
    vmax = max(p.v_kili, p.dash_kili if p.p1_momentum else 0.0)
    return vmax * p.dt


def kili_move_mask(p: EnvParams, pos):
    """(5,) bool: stay, N, E, S, W. A perpendicular move is allowed within ``v*dt`` of a junction."""
    snap = kili_snap(p) + 1e-6
    nl, nt = nearest_junction_lines(p, pos)
    near_l = jnp.abs(pos[0] - nl) <= snap
    near_t = jnp.abs(pos[1] - nt) <= snap
    eps = 1e-4
    can_n = near_l & (pos[1] < p.field_l - eps)
    can_s = near_l & (pos[1] > eps)
    can_e = near_t & (pos[0] < p.field_w - eps)
    can_w = near_t & (pos[0] > eps)
    return jnp.stack([jnp.array(True), can_n, can_e, can_s, can_w])


def line_tokens(p: EnvParams):
    """Static (NL, LINE_F) array: transverse lines L0..Lnb, then longitudinal lines."""
    rows = []
    for k in range(p.n_lines_t):
        rows.append([1.0, 0.0, k * p.row_d / p.field_l, float(1 <= k <= p.n_rows - 1), k / p.n_rows])
    for j in range(p.n_lanes + 1):
        rows.append([0.0, 1.0, j * p.lane_w / p.field_w, 0.0, j / p.n_lanes])
    return jnp.array(rows, jnp.float32)
