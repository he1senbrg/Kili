import jax
import jax.numpy as jnp
import numpy as np
from conftest import get_env
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from kilithattu_core import CLASSIC, PLUS, Actions

P_ALL = PLUS.replace(p6_calls=True)


def rollout(p, seed, n=150):
    env, step = get_env(p)
    key = jax.random.PRNGKey(seed)
    _, s = env.reset(key)
    out = []
    for t in range(n):
        key, k1, k2 = jax.random.split(key, 3)
        m = env.action_mask(s)
        logits = lambda mk: jnp.where(mk, 0.0, -1e9)
        ks = jax.random.split(k1, 5)
        a = Actions(*[jax.random.categorical(ks[i], logits(getattr(m, h))) for i, h in enumerate(["move", "dash", "turn", "clap", "call"])])
        _, s2, r, d, ev = step(k2, s, a)
        out.append((s, s2, r, ev))
        s = s2
        if bool(d):
            break
    return out


@settings(max_examples=6, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10_000), cfg=st.sampled_from(["classic", "plus", "all"]))
def test_invariants_random_play(seed, cfg):
    p = {"classic": CLASSIC, "plus": PLUS, "all": P_ALL}[cfg]
    total = 0.0
    for s, s2, r, ev in rollout(p, seed):
        k = np.asarray(s2.k_pos)
        on_graph = (abs(k[0] / p.lane_w - round(k[0] / p.lane_w)) * p.lane_w < 1e-3) or (abs(k[1] / p.row_d - round(k[1] / p.row_d)) * p.row_d < 1e-3)
        assert on_graph, k
        g = np.asarray(s2.g_x)
        assert (g >= -1e-6).all() and (g <= p.field_w + 1e-6).all()
        assert float(r[0]) + float(r[1]) == 0.0
        out_before = np.asarray(s.c_status) == 2
        assert (np.asarray(s2.c_status)[out_before] == 2).all(), "out is absorbing"
        kap0, kap1 = np.asarray(s.c_kappa), np.asarray(s2.c_kappa)
        ok = (kap1 == kap0) | (kap1 == kap0 + 1) | (kap1 == 0) | (np.asarray(s2.c_status) == 2)
        assert ok.all()
        total += float(r[0])
    assert total * p.trip_points >= 0


def test_points_equal_scaled_rewards():
    for s, s2, r, ev in rollout(CLASSIC, 1, 300):
        assert abs(float(s2.points - s.points) / CLASSIC.trip_points - float(r[0])) < 1e-6
