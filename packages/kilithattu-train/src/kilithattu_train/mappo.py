"""MAPPO with a centralised team critic, recurrent actors (sequence minibatches over environments), GAE,
masked-categorical heads, potential-based shaping and per-side learning flags (for self-play and leagues)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import struct

from kilithattu_agents.networks import Actor, Critic, NetConfig
from kilithattu_agents.policy import init_networks, log_prob_heads, team_forward, team_value
from kilithattu_core import KilithattuEnv

from .config import PPOConfig
from .rollout import Batch, Ctrl, RCarry, TeamBatch, init_rcarry, make_rollout


@struct.dataclass
class TrainState:
    params: Any  # {"a": {"actor","critic"} | None, "d": {...} | None}
    opt: Any  # {"a": opt_state | None, "d": ...}
    rc: RCarry
    partner: Any  # partner actor params or None
    step: jax.Array


def gae(rew, val, last_val, done, gamma, lam):
    """All (T, E). ``done[t]`` means the episode ended after step t."""
    def f(carry, x):
        adv_next, v_next = carry
        r, v, d = x
        nd = 1.0 - d.astype(jnp.float32)
        delta = r + gamma * v_next * nd - v
        adv = delta + gamma * lam * nd * adv_next
        return (adv, v), adv
    _, adv = jax.lax.scan(f, (jnp.zeros_like(last_val), last_val), (rew, val, done), reverse=True)
    return adv, adv + val


def make_optimizer(cfg: PPOConfig):
    sched = optax.linear_schedule(cfg.lr, cfg.lr * 0.1, cfg.total_updates * cfg.epochs * cfg.minibatches)
    return optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(sched, eps=1e-5))


class Trainer:
    """One compiled update for a fixed (environment, controllers, learning flags) combination."""

    def __init__(self, env: KilithattuEnv, net: NetConfig, ppo: PPOConfig, ca: Ctrl, cd: Ctrl, partner_mask=None):
        self.env, self.net, self.ppo, self.ca, self.cd = env, net, ppo, ca, cd
        self.partner_mask = partner_mask
        self.actor, self.critic = Actor(net), Critic(net)
        self.opt = make_optimizer(ppo)
        self.rollout = make_rollout(env, ca, cd, self.actor, self.critic, net.hidden, ppo.rollout_len, ppo.gamma, ppo.shaping_beta, partner_mask)
        self.update = jax.jit(self._update)

    # ---------------------------------------------------------------- init
    def init_params(self, key):
        _, _, a, c = init_networks(key, self.env.params, self.net)
        return {"actor": a, "critic": c}

    def init(self, key, params_a=None, params_d=None, partner=None) -> TrainState:
        k1, k2, k3 = jax.random.split(key, 3)
        pa = params_a if params_a is not None else (self.init_params(k1) if self.ca.kind == "net" else None)
        pd = params_d if params_d is not None else (self.init_params(k2) if self.cd.kind == "net" else None)
        opt = {"a": self.opt.init(pa) if (self.ca.learn and pa is not None) else None,
               "d": self.opt.init(pd) if (self.cd.learn and pd is not None) else None}
        rc = init_rcarry(self.env, self.ca, self.cd, k3, self.ppo.num_envs, self.net.hidden, partner is not None)
        return TrainState({"a": pa, "d": pd}, opt, rc, partner, jnp.zeros((), jnp.int32))

    # ---------------------------------------------------------------- loss
    def _seq_logits(self, actor_params, tb: TeamBatch, reset, carry0, idx):
        take = lambda x: x[:, idx]
        obs = jax.tree_util.tree_map(take, tb.obs)
        rs = reset[:, idx]
        n = obs.players.shape[2]

        def f(c, x):
            o, r = x
            rr = jnp.broadcast_to(r[:, None], (r.shape[0], n))
            c2, lg = jax.vmap(lambda cc, oo, rrr: team_forward(self.actor, actor_params, cc, oo, rrr))(c, o, rr)
            return c2, lg

        _, logits = jax.lax.scan(f, carry0[idx], (obs, rs))
        return logits, obs

    def _team_loss(self, params, tb: TeamBatch, adv, ret, cobs, reset, idx, team: int, ent_coef, slot_mask):
        logits, obs = self._seq_logits(params["actor"], tb, reset, tb.carry0, idx)
        act = jax.tree_util.tree_map(lambda x: x[:, idx], tb.act)
        logp, ent = log_prob_heads(logits, obs.mask, act)
        old_logp, alive = tb.logp[:, idx], tb.alive[:, idx]
        w = alive & slot_mask[None, None, :]
        adv_b, ret_b, old_v = adv[:, idx], ret[:, idx], tb.val[:, idx]
        wf = w.astype(jnp.float32)
        denom = jnp.maximum(wf.sum(), 1.0)
        a_in = adv_b[..., None]
        mean = (a_in * wf).sum() / denom
        std = jnp.sqrt((((a_in - mean) ** 2) * wf).sum() / denom + 1e-8)
        a_n = (a_in - mean) / std
        ratio = jnp.exp(logp - old_logp)
        pg = -jnp.minimum(ratio * a_n, jnp.clip(ratio, 1 - self.ppo.clip, 1 + self.ppo.clip) * a_n)
        pg_loss = (pg * wf).sum() / denom
        cb = jax.tree_util.tree_map(lambda x: x[:, idx], cobs)
        v = jax.vmap(jax.vmap(lambda c: team_value(self.critic, params["critic"], c, team)))(cb)
        v_clip = old_v + jnp.clip(v - old_v, -self.ppo.vf_clip, self.ppo.vf_clip)
        v_loss = 0.5 * jnp.maximum((v - ret_b) ** 2, (v_clip - ret_b) ** 2).mean()
        ent_m = (ent * wf).sum() / denom
        loss = pg_loss + self.ppo.vf_coef * v_loss - ent_coef * ent_m
        approx_kl = (((ratio - 1) - (logp - old_logp)) * wf).sum() / denom
        return loss, {"pg": pg_loss, "vf": v_loss, "ent": ent_m, "kl": approx_kl}

    def _learn_team(self, key, params, opt_state, tb: TeamBatch, batch: Batch, team: int, ent_coef, slot_mask):
        cfg = self.ppo
        adv, ret = gae(tb.rew, tb.val, tb.last_val, batch.done, cfg.gamma, cfg.gae_lambda)
        E = tb.rew.shape[1]
        mb = E // cfg.minibatches

        def epoch(carry, k):
            params, opt_state = carry
            perm = jax.random.permutation(k, E)[: mb * cfg.minibatches].reshape(cfg.minibatches, mb)

            def mb_step(c, idx):
                p, o = c
                (loss, aux), g = jax.value_and_grad(self._team_loss, has_aux=True)(p, tb, adv, ret, batch.cobs, batch.reset, idx, team, ent_coef, slot_mask)
                upd, o = self.opt.update(g, o, p)
                return (optax.apply_updates(p, upd), o), aux

            (params, opt_state), aux = jax.lax.scan(mb_step, (params, opt_state), perm)
            return (params, opt_state), jax.tree_util.tree_map(jnp.mean, aux)

        (params, opt_state), aux = jax.lax.scan(epoch, (params, opt_state), jax.random.split(key, cfg.epochs))
        return params, opt_state, jax.tree_util.tree_map(jnp.mean, aux)

    # ---------------------------------------------------------------- update
    def _update(self, ts: TrainState, ent_coef):
        rc, batch = self.rollout(ts.params, ts.partner, ts.rc)
        key = jax.random.fold_in(rc.key, 7)
        ka, kd = jax.random.split(key)
        params, opt = dict(ts.params), dict(ts.opt)
        metrics = {}
        nA = self.ca.n
        slot = jnp.ones((nA,), bool) if self.partner_mask is None else ~jnp.asarray(self.partner_mask)
        if self.ca.learn:
            params["a"], opt["a"], aux = self._learn_team(ka, params["a"], opt["a"], batch.a, batch, 0, ent_coef, slot)
            metrics.update({f"a/{k}": v for k, v in aux.items()})
        if self.cd.learn:
            params["d"], opt["d"], aux = self._learn_team(kd, params["d"], opt["d"], batch.d, batch, 1, ent_coef, jnp.ones((self.cd.n,), bool))
            metrics.update({f"d/{k}": v for k, v in aux.items()})
        n_ep = batch.ep_done.sum()
        metrics["ep/count"] = n_ep
        metrics["ep/points"] = batch.ep_points.sum() / jnp.maximum(n_ep, 1)
        metrics["ep/trips"] = batch.ep_trips.sum() / jnp.maximum(n_ep, 1)
        metrics["rew/attack"] = batch.a.rew.sum() / self.ppo.num_envs
        return ts.replace(params=params, opt=opt, rc=rc, step=ts.step + 1), metrics

    def ent_coef(self, i: int, total: int, scale: float = 1.0) -> float:
        f = min(1.0, i / max(total, 1))
        return float((self.ppo.ent_start + f * (self.ppo.ent_end - self.ppo.ent_start)) * scale)


def run_updates(trainer: Trainer, ts: TrainState, n: int, total: int, start: int = 0, scale: float = 1.0, log=None, log_every: int = 10):
    """Run ``n`` updates, returning (ts, last metrics as python floats)."""
    last = {}
    t0 = time.time()
    for i in range(n):
        ts, m = trainer.update(ts, jnp.float32(trainer.ent_coef(start + i, total, scale)))
        if (i + 1) % log_every == 0 or i == n - 1:
            last = {k: float(v) for k, v in m.items()}
            last["sps"] = trainer.ppo.num_envs * trainer.ppo.rollout_len * (i + 1) / (time.time() - t0)
            if log:
                log(start + i + 1, last)
    return ts, last
