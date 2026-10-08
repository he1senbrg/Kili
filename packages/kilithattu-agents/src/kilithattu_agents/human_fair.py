"""P7 human-fair reflexes as an environment wrapper: observation delay, decision interval, sticky actions.

The delay, interval and stickiness are sampled per episode, so one policy can be trained for (and
evaluated against) a range of reaction speeds. The wrapper has the same API as ``KilithattuEnv``.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import struct

from kilithattu_core import Actions, KilithattuEnv, Obs


@dataclass(frozen=True)
class HumanFairConfig:
    max_delay: int = 5  # ticks (5 ticks = 250 ms)
    delay_range: tuple[int, int] = (0, 5)
    interval_range: tuple[int, int] = (1, 3)  # decision interval in ticks
    sticky_range: tuple[float, float] = (0.0, 0.1)


@struct.dataclass
class HFState:
    inner: object
    buf: object  # ring buffer of Obs, leading axis max_delay + 1
    prev: Actions
    delay: jax.Array
    interval: jax.Array
    sticky: jax.Array


class HumanFairEnv:
    def __init__(self, env: KilithattuEnv, cfg: HumanFairConfig = HumanFairConfig()):
        self.env, self.cfg, self.params = env, cfg, env.params
        self.n_agents, self.n_crossers, self.n_defenders = env.n_agents, env.n_crossers, env.n_defenders

    def reset(self, key):
        k0, k1, k2, k3 = jax.random.split(key, 4)
        obs, inner = self.env.reset(k0)
        c = self.cfg
        buf = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (c.max_delay + 1,) + x.shape), obs)
        z = jnp.zeros((self.n_agents,), jnp.int32)
        st = HFState(inner, buf, Actions(z, z, z, z, z),
                     jax.random.randint(k1, (), c.delay_range[0], c.delay_range[1] + 1),
                     jax.random.randint(k2, (), c.interval_range[0], c.interval_range[1] + 1),
                     jax.random.uniform(k3, (), minval=c.sticky_range[0], maxval=c.sticky_range[1]))
        return obs, st

    def step(self, key, st: HFState, a: Actions):
        k1, k2 = jax.random.split(key)
        tick = st.inner.tick
        hold = (tick % st.interval) != 0
        stick = jax.random.uniform(k1, (self.n_agents,)) < st.sticky
        use_prev = hold | stick
        a_eff = jax.tree_util.tree_map(lambda new, old: jnp.where(use_prev, old, new), a, st.prev)
        obs, inner, r, done, ev = self.env.step(k2, st.inner, a_eff)
        n = self.cfg.max_delay + 1
        buf = jax.tree_util.tree_map(lambda b, x: b.at[(tick + 1) % n].set(x), st.buf, obs)
        delayed = jax.tree_util.tree_map(lambda b: b[(tick + 1 - st.delay) % n], buf)
        # during the first ticks the delayed slot still holds the reset observation, which is correct
        return delayed, st.replace(inner=inner, buf=buf, prev=a_eff), r, done, ev

    def critic_state(self, st: HFState):
        return self.env.critic_state(st.inner)

    def potential(self, st: HFState):
        return self.env.potential(st.inner)
