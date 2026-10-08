"""Play full innings between any two Policy objects (networks or scripted), vmapped over episodes."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from kilithattu_agents.policy import join_actions
from kilithattu_core import EnvParams, make_env


def play_innings(p: EnvParams, att, dfn, n_episodes: int, seed: int = 0, env=None) -> dict:
    """Returns numpy arrays per episode: points, trips, all_out, ticks (live ticks), tags, fouls."""
    env = env or make_env(p)

    def episode(key):
        k0, k1, k2, k3 = jax.random.split(key, 4)
        obs, s = env.reset(k0)
        c0 = (att.init_carry(k1), dfn.init_carry(k2))

        def body(c, _):
            s, obs, ca, cd, key, done, tag, foul = c
            key, ka, kd, ke = jax.random.split(key, 4)
            aa, ca2, _ = att.act(ka, ca, obs)
            ad, cd2, _ = dfn.act(kd, cd, obs)
            obs2, s2, _, d, ev = env.step(ke, s, join_actions(aa, ad))
            keep = lambda new, old: jax.tree_util.tree_map(lambda a, b: jnp.where(done, b, a), new, old)
            tag = tag + jnp.where(done, 0, ev.tagged.sum())
            foul = foul + jnp.where(done, 0, ev.fouled.sum())
            return (keep(s2, s), keep(obs2, obs), keep(ca2, ca), keep(cd2, cd), key, done | d, tag, foul), None

        init = (s, obs, c0[0], c0[1], k3, jnp.zeros((), bool), jnp.zeros((), jnp.int32), jnp.zeros((), jnp.int32))
        (s, _, _, _, _, _, tag, foul), _ = jax.lax.scan(body, init, None, length=p.max_ticks + p.clap_max_ticks + 2)
        return s.points, s.deliveries, jnp.all(s.c_status == 2), s.t, tag, foul

    f = jax.jit(jax.vmap(episode))
    out = f(jax.random.split(jax.random.PRNGKey(seed), n_episodes))
    return {k: np.asarray(v) for k, v in zip(("points", "trips", "all_out", "ticks", "tags", "fouls"), out)}
