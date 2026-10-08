"""Scripted crossers: stage near the next line, run to the side away from the guard, cross when the gap is
large enough (patience shrinks with waiting), avoid fouls and the Kili."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from kilithattu_core import Actions, EnvParams, Obs

from ._common import to_move, unit, unpack


def init_carry(p: EnvParams, key, aggression: float):
    nA = p.n_crossers
    ag = aggression * jax.random.uniform(key, (nA,), minval=0.8, maxval=1.2)
    side = jnp.where(jnp.arange(nA) % 2 == 0, -1.0, 1.0)
    z = jnp.zeros((nA,))
    return {"ag": ag, "w": z, "pin": z, "side": side, "kprev": jnp.zeros((nA,), jnp.int32), "commit": jnp.zeros((nA,), bool)}


def act(p: EnvParams, carry, obs: Obs, safety: float = 0.0) -> tuple[Actions, dict]:
    nA, nG, K, dt = p.n_crossers, p.n_guards, p.K, p.dt
    u = unpack(obs, p, 0)
    c_pos, kappa, out = u["c_pos"], u["kappa"], u["out"]
    x, y = c_pos[:, 0], c_pos[:, 1]
    up = kappa < K
    dirn = jnp.where(up, 1.0, -1.0)
    li = jnp.where(up, kappa, 2 * K - 1 - kappa)
    ly = li.astype(jnp.float32) * p.row_d
    gap = (ly - y) * dirn
    has_g = (li >= 1) & (li <= nG)
    gx = jnp.where(has_g, u["g_x"][jnp.clip(li - 1, 0, nG - 1)], x)
    row = jnp.where(up, li, li - 1)
    inside = (y >= 0) & (y <= p.field_l) & ~out
    orow = jnp.where(inside, jnp.clip(jnp.floor(y / p.row_d), 0, p.n_rows - 1).astype(jnp.int32), -99)
    is_up = kappa >= K
    opp = is_up[None, :] != is_up[:, None]
    blocked = jnp.any((orow[None, :] == row[:, None]) & opp & ~jnp.eye(nA, dtype=bool), 1) & (row >= 0) & (row < p.n_rows)
    kp = u["k_pos"]
    stage_y = ly - dirn * 1.1
    w = jnp.where(kappa != carry["kprev"], 0.0, carry["w"])
    # patience: the longer a crosser waits, the smaller the clearance it demands from guard and Kili
    safe_k = (jnp.abs(kp[0] - x) + jnp.abs(kp[1] - ly)) > jnp.maximum(1.2, 3.2 + safety - 0.6 * w)
    approach = gap > 1.5
    txa = jnp.clip(jnp.where(has_g, jnp.where(gx < x, p.field_w - 1.4, 1.4), x), 0.5, p.field_w - 0.5)
    v_app = unit(txa - x, stage_y - y)
    dx = jnp.abs(gx - x)
    thr = jnp.maximum(0.9, 2.6 * carry["ag"] - 0.5 * w) + safety
    staged = (gap >= 0.75) & (gap <= 1.5)
    commit = carry["commit"] & (kappa == carry["kprev"]) & (gap > 0.0)  # keep running once decided
    go_new = (~blocked) & safe_k & staged & ((~has_g) | (dx > thr))
    go = commit | go_new
    # teamwork: crossers heading for the same line take opposite sides so the guard cannot mirror both
    idx = jnp.arange(nA)
    mates = (li[None, :] == li[:, None]) & (up[None, :] == up[:, None]) & ~out[None, :] & (idx[None, :] != idx[:, None])
    rank = jnp.sum(mates & (idx[None, :] < idx[:, None]), 1)
    paired = jnp.any(mates, 1)
    side_now = jnp.where(paired, jnp.where(rank % 2 == 0, -1.0, 1.0), carry["side"])
    xs = jnp.where(side_now > 0, p.field_w - 0.7, 0.7)
    pinned = (jnp.abs(x - xs) < 0.3) & (dx < thr) & ~go & ~approach
    pin = jnp.where(pinned, carry["pin"] + dt, 0.0)
    flip = pin > 0.45
    side = jnp.where(flip, -side_now, side_now)
    pin = jnp.where(flip, 0.0, pin)
    w = jnp.where(go, 0.0, jnp.where((~blocked) & (~approach), w + dt, w))
    lx = jnp.where((jnp.abs(xs - x) < 0.15) | blocked, 0.0, jnp.sign(xs - x))
    ly2 = jnp.where(gap < 0.75, -dirn * 0.6, 0.0)  # back off only if too close to the line
    v_stage = unit(lx, ly2)
    v_go = jnp.stack([jnp.zeros_like(dirn), dirn], -1)
    vec = jnp.where(go[:, None], v_go, jnp.where(approach[:, None], v_app, v_stage))
    move = jnp.where(u["pre"] | out, 0, to_move(vec))
    z = jnp.zeros((nA,), jnp.int32)
    dash = (go & p.p1_momentum).astype(jnp.int32)
    new_carry = {"ag": carry["ag"], "w": w, "pin": pin, "side": side, "kprev": kappa, "commit": go}
    return Actions(move, dash, z, z, z), new_carry
