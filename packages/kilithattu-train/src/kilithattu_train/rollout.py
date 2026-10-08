"""Vectorised two-team rollouts with auto-reset, optional partner mixing, scripted or learned controllers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from kilithattu_agents.networks import Actor, Critic
from kilithattu_agents.policy import join_actions, log_prob_heads, sample_heads, team_forward, team_slice, team_value
from kilithattu_core import EnvParams, Obs
from kilithattu_core.state import ActionMask


def slice_obs_b(obs: Obs, sl: slice) -> Obs:
    """Slice the agent axis of a batched (E, A, ...) observation."""
    m = obs.mask
    mask = ActionMask(m.move[:, sl], m.dash[:, sl], m.turn[:, sl], m.clap[:, sl], m.call[:, sl])
    return Obs(obs.players[:, sl], obs.lines, obs.glob[:, sl], obs.ego[:, sl], obs.role[:, sl], mask)


@dataclass
class Ctrl:
    """A team controller: ``kind`` is "net" (parameters live in TrainState) or "scripted" (policy object)."""

    side: str
    kind: str
    n: int
    sl: slice
    policy: Any = None  # scripted policy
    learn: bool = False


@struct.dataclass
class RCarry:
    state: Any
    obs: Any
    ca: Any
    cd: Any
    pa: Any  # partner carry for the attack team (or None)
    reset: jax.Array  # (E,) carry-reset flag
    key: jax.Array


@struct.dataclass
class TeamBatch:
    obs: Any  # Obs with (T, E, n, ...) leading dims
    act: Any
    logp: jax.Array  # (T, E, n)
    alive: jax.Array  # (T, E, n)
    val: jax.Array  # (T, E)
    rew: jax.Array  # (T, E)
    carry0: Any
    last_val: jax.Array  # (E,)


@struct.dataclass
class Batch:
    a: TeamBatch
    d: TeamBatch
    cobs: Any
    done: jax.Array  # (T, E)
    reset: jax.Array  # (T, E)
    ep_points: jax.Array  # (T, E) innings points at termination (0 elsewhere)
    ep_trips: jax.Array
    ep_done: jax.Array


def team_act(ctrl: Ctrl, actor: Actor | None, params, key, carry, obs_b: Obs, full_obs: Obs, reset, hidden: int):
    """-> (Actions (E,n), carry, logp (E,n))."""
    E = reset.shape[0]
    keys = jax.random.split(key, E)
    if ctrl.kind == "scripted":
        a, c, _ = jax.vmap(ctrl.policy.act)(keys, carry, full_obs)
        return a, c, jnp.zeros((E, ctrl.n))
    ot = slice_obs_b(obs_b, ctrl.sl)
    rs = jnp.broadcast_to(reset[:, None], (E, ctrl.n))
    carry, logits = jax.vmap(lambda c, o, r: team_forward(actor, params, c, o, r))(carry, ot, rs)
    a, lp, _ = jax.vmap(sample_heads)(keys, logits, ot.mask)
    return a, carry, lp


def init_carry(ctrl: Ctrl, key, E: int, hidden: int):
    if ctrl.kind == "scripted":
        return jax.vmap(ctrl.policy.init_carry)(jax.random.split(key, E))
    return jnp.zeros((E, ctrl.n, hidden))


def make_rollout(env, ca: Ctrl, cd: Ctrl, actor: Actor, critic: Critic, hidden: int, T: int, gamma: float, beta: float,
                 partner_mask=None):
    """Returns ``rollout(params, partner_params, rc) -> (rc, Batch)`` (jit-friendly; ``params`` = {"a":..., "d":...})."""
    reset_v = jax.vmap(env.reset)
    step_v = jax.vmap(env.step)
    crit_v = jax.vmap(env.critic_state)
    pot_v = jax.vmap(env.potential)
    n_a, n_d = ca.n, cd.n
    pm = None if partner_mask is None else jnp.asarray(partner_mask)  # (nA,) True where a partner plays

    def value(cparams, cobs, team):
        return jax.vmap(lambda c: team_value(critic, cparams, c, team))(cobs)

    def step_fn(c: RCarry, _):
        key, k1, k2, k3, k4 = jax.random.split(c.key, 5)
        obs = c.obs
        cobs = crit_v(c.state)
        pa_carry = c.pa
        if ca.kind == "net" and pm is not None:  # mix learner and partner logits slot-wise
            E = c.reset.shape[0]
            ot = slice_obs_b(obs, ca.sl)
            rs = jnp.broadcast_to(c.reset[:, None], (E, n_a))
            fwd = lambda prm, car: jax.vmap(lambda cc, o, r: team_forward(actor, prm, cc, o, r))(car, ot, rs)
            ca_new, lg_l = fwd(params_ref["a"]["actor"], c.ca)
            pa_new, lg_p = fwd(params_ref["partner"], c.pa)
            logits = jax.tree_util.tree_map(lambda l, p: jnp.where(pm[None, :, None], p, l), lg_l, lg_p)
            keys = jax.random.split(k1, E)
            act_a, lp_a, _ = jax.vmap(sample_heads)(keys, logits, ot.mask)
            lp_l = jax.vmap(lambda lg, m, a: log_prob_heads(lg, m, a)[0])(lg_l, ot.mask, act_a)
            lp_a, pa_carry = lp_l, pa_new
        else:
            act_a, ca_new, lp_a = team_act(ca, actor, params_ref["a"]["actor"] if ca.kind == "net" else None, k1, c.ca, obs, obs, c.reset, hidden)
        act_d, cd_new, lp_d = team_act(cd, actor, params_ref["d"]["actor"] if cd.kind == "net" else None, k2, c.cd, obs, obs, c.reset, hidden)
        alive_a = c.state.c_status != 2
        alive_d = jnp.ones((c.reset.shape[0], n_d), bool)
        va = value(params_ref["a"]["critic"], cobs, 0) if ca.kind == "net" else jnp.zeros(c.reset.shape)
        vd = value(params_ref["d"]["critic"], cobs, 1) if cd.kind == "net" else jnp.zeros(c.reset.shape)
        E = c.reset.shape[0]
        sk = jax.random.split(k3, E)
        obs2, st2, rew, done, ev = step_v(sk, c.state, join_actions(act_a, act_d))
        phi, phi2 = pot_v(c.state), pot_v(st2)
        r_a = rew[:, 0] + beta * (gamma * phi2 - phi)
        r_d = -r_a
        ep_pts = jnp.where(done, st2.points, 0.0)
        ep_trips = jnp.where(done, st2.deliveries.astype(jnp.float32), 0.0)
        # auto-reset
        obs_r, st_r = reset_v(jax.random.split(k4, E))
        sel = lambda new, old: jnp.where(done.reshape((E,) + (1,) * (old.ndim - 1)), new, old)
        st3 = jax.tree_util.tree_map(sel, st_r, st2)
        obs3 = jax.tree_util.tree_map(sel, obs_r, obs2)
        out = dict(
            oa=slice_obs_b(obs, ca.sl), od=slice_obs_b(obs, cd.sl), aa=act_a, ad=act_d, lpa=lp_a, lpd=lp_d,
            alive_a=alive_a, alive_d=alive_d, va=va, vd=vd, ra=r_a, rd=r_d, done=done, reset=c.reset, cobs=cobs,
            pts=ep_pts, trips=ep_trips,
        )
        nc = RCarry(st3, obs3, ca_new, cd_new, pa_carry, done, key)
        return nc, out

    params_ref: dict = {}

    def rollout(params, partner_params, rc: RCarry):
        params_ref.clear()
        params_ref.update(params)
        if partner_params is not None:
            params_ref["partner"] = partner_params
        ca0, cd0 = rc.ca, rc.cd
        rc2, o = jax.lax.scan(step_fn, rc, None, length=T)
        cobs_last = crit_v(rc2.state)
        lva = value(params["a"]["critic"], cobs_last, 0) if ca.kind == "net" else jnp.zeros(rc.reset.shape)
        lvd = value(params["d"]["critic"], cobs_last, 1) if cd.kind == "net" else jnp.zeros(rc.reset.shape)
        batch = Batch(
            a=TeamBatch(o["oa"], o["aa"], o["lpa"], o["alive_a"], o["va"], o["ra"], ca0, lva),
            d=TeamBatch(o["od"], o["ad"], o["lpd"], o["alive_d"], o["vd"], o["rd"], cd0, lvd),
            cobs=o["cobs"], done=o["done"], reset=o["reset"], ep_points=o["pts"], ep_trips=o["trips"], ep_done=o["done"],
        )
        return rc2, batch

    return rollout


def init_rcarry(env, ca: Ctrl, cd: Ctrl, key, E: int, hidden: int, partner: bool = False) -> RCarry:
    k0, k1, k2, k3, k4 = jax.random.split(key, 5)
    obs, st = jax.vmap(env.reset)(jax.random.split(k0, E))
    pa = jnp.zeros((E, ca.n, hidden)) if partner else None
    return RCarry(st, obs, init_carry(ca, k1, E, hidden), init_carry(cd, k2, E, hidden), pa, jnp.ones((E,), bool), k3)
