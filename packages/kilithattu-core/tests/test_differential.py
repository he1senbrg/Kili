"""JAX engine vs the plain-Python reference on random rollouts (Classic rules)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import get_env

from kilithattu_core import CLASSIC, Actions, zeros_actions
from kilithattu_core.reference import ReferenceEnv


def run(seed, n_ticks, bias):
    env, step = get_env(CLASSIC)
    key = jax.random.PRNGKey(seed)
    _, s = env.reset(key)
    ref = ReferenceEnv(CLASSIC)
    ref.load(s)
    rng = np.random.default_rng(seed)
    counts = dict(tag=0, cross=0, deliv=0)
    for t in range(n_ticks):
        mv = np.zeros(CLASSIC.n_agents, np.int32)
        mv[:4] = rng.choice([1, 1, 1, 2, 8, 3, 7, 5, 0], 4) if t < bias else rng.choice([5, 5, 5, 4, 6, 3, 7, 0], 4)
        mv[4:8] = rng.integers(0, 3, 4)
        mv[8] = rng.integers(0, 5)
        a = zeros_actions(CLASSIC.n_agents).replace(move=jnp.asarray(mv))
        # the engine masks invalid actions itself; give the reference the already-masked move
        m = env.action_mask(s)
        mv_m = np.where(np.asarray(jnp.take_along_axis(m.move, a.move[:, None], 1)[:, 0]), mv, 0)
        _, s, r, d, ev = step(key, s, a)
        ref.step(mv_m)
        counts["tag"] += int(ev.tagged.sum()); counts["cross"] += int(ev.crossed.sum()); counts["deliv"] += int(ev.delivered.sum())
        np.testing.assert_allclose(np.asarray(s.c_pos), ref.c_pos, atol=2e-4, err_msg=f"c_pos t={t}")
        np.testing.assert_array_equal(np.asarray(s.c_status), ref.c_status, err_msg=f"status t={t}")
        np.testing.assert_array_equal(np.asarray(s.c_kappa), ref.c_kappa, err_msg=f"kappa t={t}")
        np.testing.assert_allclose(np.asarray(s.g_x), ref.g_x, atol=2e-4)
        np.testing.assert_allclose(np.asarray(s.k_pos), ref.k_pos, atol=2e-4)
        assert float(s.points) == pytest.approx(ref.points)
    return counts


@pytest.mark.parametrize("seed", range(12))
def test_matches_reference(seed):
    run(seed, 300, 150)


def test_scenarios_were_actually_exercised():
    tot = dict(tag=0, cross=0, deliv=0)
    for seed in range(100, 108):
        c = run(seed, 300, 150)
        for k in tot: tot[k] += c[k]
    assert tot["tag"] > 0 and tot["cross"] > 0
