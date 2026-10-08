"""Rule resolution: swept tags, line crossings and scoring, fouls."""
from __future__ import annotations

import jax.numpy as jnp

from .config import EnvParams


def swept_hits(a0, a1, b0, b1, r):
    """Continuous tag test. a: crossers (nA,2) start/end, b: defenders (nD,2) start/end -> (nA,nD) bool.

    Closest approach within the tick is at tau* = clip(-(d0.e)/(e.e), 0, 1) for relative motion e.
    """
    d0 = a0[:, None, :] - b0[None, :, :]
    e = (a1 - a0)[:, None, :] - (b1 - b0)[None, :, :]
    ee = jnp.sum(e * e, -1)
    tau = jnp.where(ee > 1e-12, jnp.clip(-jnp.sum(d0 * e, -1) / jnp.maximum(ee, 1e-12), 0.0, 1.0), 0.0)
    dist = jnp.linalg.norm(d0 + tau[..., None] * e, axis=-1)
    return dist <= r


def resolve_crossings(p: EnvParams, kappa, prev_y, new_y, alive):
    """Advance trip progress kappa only when the next expected line is crossed in the expected direction.

    Returns (kappa, status, adv, delivered). Retreating and re-crossing earns nothing, so points cannot be
    farmed; a completed trip (kappa == 2K) pays the +6 bonus (counted by the caller) and resets to 0.
    """
    K = p.K
    up = kappa < K
    li = jnp.where(up, kappa, 2 * K - 1 - kappa)
    ly = li.astype(jnp.float32) * p.row_d
    cross_up = (prev_y < ly) & (new_y >= ly)
    cross_dn = (prev_y > ly) & (new_y <= ly)
    adv = alive & (kappa < 2 * K) & jnp.where(up, cross_up, cross_dn)
    k2 = kappa + adv.astype(jnp.int32)
    delivered = adv & (k2 == 2 * K)
    k2 = jnp.where(delivered, 0, k2)
    status = jnp.where(alive, jnp.where(k2 >= K, 1, 0), 2).astype(jnp.int32)
    return k2, status, adv, delivered


def resolve_fouls(status, box, prev_box, alive):
    """Uppu and pacha in the same box at the end of a tick: the one that entered the box this tick is out;
    if both entered (or neither did) the uppu is out. Returns (n,) bool of crossers put out."""
    uppu = alive & (status == 1)
    pacha = alive & (status == 0)
    same = (box[:, None] == box[None, :]) & (box[:, None] >= 0)
    conflict = uppu[:, None] & pacha[None, :] & same  # [i uppu, j pacha]
    entered = box != prev_box
    pacha_out_pair = entered[None, :] & ~entered[:, None]  # [i, j]: pacha j entered, uppu i did not
    out_pacha = jnp.any(conflict & pacha_out_pair, axis=0)
    out_uppu = jnp.any(conflict & ~pacha_out_pair, axis=1)
    return out_pacha | out_uppu
