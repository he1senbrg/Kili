"""Behaviour cloning: pre-train an actor on (observation, action) pairs from scripted play or human replays."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from kilithattu_agents.networks import Actor, NetConfig
from kilithattu_agents.policy import join_actions, log_prob_heads, slice_obs, team_forward, team_slice
from kilithattu_core import Actions, EnvParams, Obs, make_env
from kilithattu_core.replay import _to_actions, read_replay
from kilithattu_core.config import load_params


def collect_scripted(p: EnvParams, att, dfn, side: str, n_envs: int, T: int, seed: int = 0):
    """Roll scripted policies and record the observations and actions of ``side`` -> (obs_seq, act_seq), (T, E, n, ...)."""
    env = make_env(p)
    sl = team_slice(p, side)

    def run(key):
        k0, k1, k2, k3 = jax.random.split(key, 4)
        obs, s = env.reset(k0)
        c = (att.init_carry(k1), dfn.init_carry(k2))

        def body(cc, _):
            s, obs, ca, cd, key = cc
            key, ka, kd, ke = jax.random.split(key, 4)
            aa, ca, _ = att.act(ka, ca, obs)
            ad, cd, _ = dfn.act(kd, cd, obs)
            act = join_actions(aa, ad)
            obs2, s2, _, done, _ = env.step(ke, s, act)
            robs, rs = env.reset(ke)
            pick = lambda a, b: jnp.where(done, a, b)
            s2 = jax.tree_util.tree_map(pick, rs, s2)
            obs2 = jax.tree_util.tree_map(pick, robs, obs2)
            out = (slice_obs(obs, sl), jax.tree_util.tree_map(lambda x: x[sl], act), done)
            return (s2, obs2, ca, cd, key), out

        _, out = jax.lax.scan(body, (s, obs, c[0], c[1], k3), None, length=T)
        return out

    o, a, d = jax.jit(jax.vmap(run))(jax.random.split(jax.random.PRNGKey(seed), n_envs))
    swap = lambda x: jnp.swapaxes(x, 0, 1)  # (E,T,...) -> (T,E,...)
    return jax.tree_util.tree_map(swap, o), jax.tree_util.tree_map(swap, a), swap(d)


def replay_dataset(path, side: str):
    """Re-simulate a human replay and extract (obs_seq, act_seq) for ``side`` with a singleton env axis."""
    header, log = read_replay(path)
    p = load_params(header["params"])
    env = make_env(p)
    sl = team_slice(p, side)
    key = jax.random.PRNGKey(header["seed"])
    key, k0 = jax.random.split(key)
    obs, s = env.reset(k0)
    step = jax.jit(env.step)
    os_, as_ = [], []
    for t in range(len(log)):
        key, k = jax.random.split(key)
        a = _to_actions(log[t])
        os_.append(slice_obs(obs, sl))
        as_.append(jax.tree_util.tree_map(lambda x: x[sl], a))
        obs, s, _, done, _ = step(k, s, a)
        if bool(done):
            break
    stack = lambda xs: jax.tree_util.tree_map(lambda *v: jnp.stack(v)[:, None], *xs)
    return stack(os_), stack(as_)


def bc_train(actor_params, cfg: NetConfig, obs_seq: Obs, act_seq: Actions, steps: int = 200, lr: float = 1e-3, minibatch: int | None = None, seed: int = 0):
    """Supervised training with sequence minibatches over the env axis. Returns (params, losses)."""
    actor = Actor(cfg)
    T, E, n = obs_seq.players.shape[:3]
    opt = optax.adam(lr)
    st = opt.init(actor_params)

    def loss_fn(params, idx):
        o = jax.tree_util.tree_map(lambda x: x[:, idx], obs_seq)
        a = jax.tree_util.tree_map(lambda x: x[:, idx], act_seq)
        rs = jnp.zeros((T, idx.shape[0], n), bool)

        def f(c, x):
            ob, r = x
            c2, lg = jax.vmap(lambda cc, oo, rr: team_forward(actor, params, cc, oo, rr))(c, ob, r)
            return c2, lg

        _, logits = jax.lax.scan(f, jnp.zeros((idx.shape[0], n, cfg.hidden)), (o, rs))
        lp, _ = log_prob_heads(logits, o.mask, a)
        return -lp.mean()

    @jax.jit
    def step(params, st, idx):
        l, g = jax.value_and_grad(loss_fn)(params, idx)
        up, st = opt.update(g, st, params)
        return optax.apply_updates(params, up), st, l

    rng = np.random.default_rng(seed)
    mb = minibatch or E
    losses = []
    for _ in range(steps):
        idx = jnp.asarray(rng.permutation(E)[:mb])
        actor_params, st, l = step(actor_params, st, idx)
        losses.append(float(l))
    return actor_params, losses
