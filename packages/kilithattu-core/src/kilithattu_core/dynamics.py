"""Movement: crossers (free 2D), guards (1D on a line), Kili (line graph with junction snapping)."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .config import EnvParams
from .geometry import kili_move_mask, nearest_junction_lines

_D = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]], np.float32)
_n = np.linalg.norm(_D, axis=1, keepdims=True)
DIRS = jnp.asarray(_D / np.where(_n > 0, _n, 1.0))  # stay, N, NE, E, SE, S, SW, W, NW (diagonals normalised)
GUARD_DIR = jnp.array([0.0, -1.0, 1.0])  # stay, W, E
KILI_DX = jnp.array([0.0, 0.0, 1.0, 0.0, -1.0])  # stay, N, E, S, W
KILI_DY = jnp.array([0.0, 1.0, 0.0, -1.0, 0.0])


def _stamina(p: EnvParams, want_fast, moving, stam):
    can = want_fast & (stam >= p.stam_min)
    drain = can & moving
    stam = jnp.clip(stam + jnp.where(drain, -p.stam_drain, p.stam_regen) * p.dt, 0.0, 1.0)
    return can, stam


def update_turn(p: EnvParams, face, turn_t, turn_act):
    """P2: a turn takes ``t_turn`` seconds, flips the facing at the end, and blocks tagging meanwhile."""
    if not p.p2_facing:
        return face, turn_t
    turning = turn_t > 0
    started = (~turning) & (turn_act == 1)
    new_t = jnp.where(turning, turn_t - p.dt, jnp.where(started, p.t_turn, 0.0))
    finished = turning & (new_t <= 1e-6)
    return jnp.where(finished, -face, face), jnp.where(finished, 0.0, new_t)


def crosser_move(p: EnvParams, pos, vel, stam, move, sprint, alive, live):
    d = DIRS[move]
    if p.p1_momentum:
        can, stam = _stamina(p, sprint == 1, move != 0, stam)
        speed = jnp.where(can, p.sprint_cross, p.walk_cross)
        dv = d * speed[:, None] - vel
        mag = jnp.linalg.norm(dv, axis=-1, keepdims=True)
        vel = vel + dv * jnp.minimum(1.0, p.a_max * p.dt / jnp.maximum(mag, 1e-6))
    else:
        vel = d * p.v_cross
    vel = vel * alive[:, None]
    new = pos + vel * p.dt
    hold = (not p.p4_clap) & (~live) & alive  # Classic: crossers are held behind L0 until the clap
    y = jnp.where(hold, jnp.minimum(new[:, 1], -0.01), new[:, 1])
    return jnp.stack([new[:, 0], y], -1), vel, stam


def guard_move(p: EnvParams, x, vx, stam, move, dash, face, turn_t, turn_act):
    d = GUARD_DIR[move]
    if p.p1_momentum:
        can, stam = _stamina(p, dash == 1, move != 0, stam)
        speed = jnp.where(can, p.dash_guard, p.walk_guard)
        lim = p.a_max * p.dt
        vx = vx + jnp.clip(d * speed - vx, -lim, lim)
    else:
        vx = d * p.v_guard
    xn = jnp.clip(x + vx * p.dt, 0.0, p.field_w)
    vx = jnp.where((xn <= 0.0) | (xn >= p.field_w), 0.0, vx)
    face, turn_t = update_turn(p, face, turn_t, turn_act)
    return xn, vx, stam, face, turn_t


def kili_move(p: EnvParams, pos, stam, move, dash, face, turn_t, turn_act):
    move = jnp.where(kili_move_mask(p, pos)[move], move, 0)
    dx, dy = KILI_DX[move], KILI_DY[move]
    if p.p1_momentum:
        can, stam = _stamina(p, dash == 1, move != 0, stam)
        speed = jnp.where(can, p.dash_kili, p.walk_kili)
    else:
        speed = p.v_kili
    v = speed * p.dt
    nl, nt = nearest_junction_lines(p, pos)
    nx = jnp.where(dy != 0, nl, jnp.where(dx != 0, jnp.clip(pos[0] + dx * v, 0.0, p.field_w), pos[0]))
    ny = jnp.where(dx != 0, nt, jnp.where(dy != 0, jnp.clip(pos[1] + dy * v, 0.0, p.field_l), pos[1]))
    face, turn_t = update_turn(p, face, turn_t, turn_act)
    return jnp.stack([nx, ny]), stam, face, turn_t
