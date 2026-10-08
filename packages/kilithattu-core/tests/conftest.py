from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import pytest

from kilithattu_core import CLASSIC, Actions, EnvParams, make_env, zeros_actions


@functools.lru_cache(maxsize=None)
def get_env(p: EnvParams):
    env = make_env(p)
    return env, jax.jit(env.step)


class Sim:
    """Small helper to place agents and step with scripted actions."""

    def __init__(self, p: EnvParams = CLASSIC, seed: int = 0):
        self.p = p
        self.env, self.step_fn = get_env(p)
        _, self.s = self.env.reset(jax.random.PRNGKey(seed))
        self.key = jax.random.PRNGKey(seed + 1)
        self.nA, self.nG = p.n_crossers, p.n_guards
        # a neutral scene: all guards at far east, Kili in the far east corner, all crossers parked and out
        self.set(g_x=jnp.full((self.nG,), p.field_w - 0.05), k_pos=jnp.array([p.field_w, p.field_l]), live=jnp.array(True),
                 tick=jnp.array(p.clap_auto_ticks + 1, jnp.int32))
        self.last_ev = None

    def set(self, **kw):
        self.s = self.s.replace(**kw)

    def crosser(self, i, pos, kappa=0, status=None, box=None):
        st = (1 if kappa >= self.p.K else 0) if status is None else status
        self.set(c_pos=self.s.c_pos.at[i].set(jnp.array(pos, jnp.float32)),
                 c_kappa=self.s.c_kappa.at[i].set(kappa), c_status=self.s.c_status.at[i].set(st))
        if box is not None:
            self.set(c_box=self.s.c_box.at[i].set(box))

    def park_others(self, keep):
        st = self.s.c_status
        for i in range(self.nA):
            if i not in keep:
                st = st.at[i].set(2)
        self.set(c_status=st)

    def act(self, c=None, g=None, k=0, **extra):
        a = zeros_actions(self.p.n_agents)
        mv = a.move
        for i, m in (c or {}).items():
            mv = mv.at[i].set(m)
        for i, m in (g or {}).items():
            mv = mv.at[self.nA + i].set(m)
        mv = mv.at[-1].set(k)
        return a.replace(move=mv, **extra)

    def step(self, a=None, n=1):
        for _ in range(n):
            self.key, k = jax.random.split(self.key)
            _, self.s, r, d, ev = self.step_fn(k, self.s, a if a is not None else self.act())
            self.last_ev = ev
            self.r = r
        return self.s


@pytest.fixture
def sim():
    return Sim()
