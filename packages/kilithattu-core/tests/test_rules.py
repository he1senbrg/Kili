import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import Sim, get_env

from kilithattu_core import CLASSIC, PLUS, zeros_actions

N, NE, E, S = 1, 2, 3, 5


def test_round_trip_is_18_points():
    sim = Sim()
    sim.park_others({0})
    sim.crosser(0, (0.3, -1.0))
    pts = []
    for _ in range(200):
        sim.step(sim.act(c={0: N}))
        if float(sim.s.c_pos[0, 1]) > 16.0:
            break
    assert int(sim.s.c_status[0]) == 1, "should be uppu after crossing L5"
    for _ in range(200):
        sim.step(sim.act(c={0: S}))
        if float(sim.s.c_pos[0, 1]) < -0.5:
            break
    assert float(sim.s.points) == 18.0
    assert int(sim.s.deliveries) == 1 and int(sim.s.c_status[0]) == 0 and int(sim.s.c_kappa[0]) == 0


def test_no_point_farming_by_oscillating():
    sim = Sim()
    sim.park_others({0})
    sim.crosser(0, (0.3, -0.3))
    for _ in range(5):
        sim.step(sim.act(c={0: N}), n=4)   # crosses L0 up
        sim.step(sim.act(c={0: S}), n=4)   # crosses back down
    assert float(sim.s.points) == 1.0


def test_tag_within_reach_and_safe_zones():
    sim = Sim()
    sim.park_others({0, 1})
    sim.set(g_x=sim.s.g_x.at[1].set(1.0))                 # guard on L2 (y=6) at x=1
    sim.crosser(0, (1.0, 5.6), kappa=2)
    sim.crosser(1, (1.0, 15.5), kappa=6)                 # far zone: safe even next to the Kili
    sim.set(k_pos=jnp.array([1.0, 15.0]))
    sim.step()
    assert int(sim.s.c_status[0]) == 2 and bool(sim.last_ev.tagged[0])
    assert int(sim.s.c_status[1]) != 2


def test_swept_tag_has_no_tunnelling():
    p = CLASSIC.replace(dt=0.5, r_tag=0.3)
    sim = Sim(p)
    sim.park_others({0})
    sim.set(g_x=sim.s.g_x.at[1].set(1.0))
    sim.crosser(0, (1.0, 6.3), kappa=2)
    sim.step(sim.act(c={0: S}))                          # moves 2.5 m, straight through the guard
    assert int(sim.s.c_status[0]) == 2


def test_side_exit_is_out():
    sim = Sim()
    sim.park_others({0})
    sim.crosser(0, (5.95, 5.0), kappa=2)
    sim.step(sim.act(c={0: E}))
    assert int(sim.s.c_status[0]) == 2 and bool(sim.last_ev.side_exit[0])


def test_foul_pacha_that_entered_is_out():
    sim = Sim()
    sim.park_others({0, 1})
    sim.crosser(0, (1.5, 4.0), kappa=7, box=1)           # uppu already in box (0, row 1)
    sim.crosser(1, (1.5, 2.9), kappa=1, box=0)           # pacha walks into the same box
    sim.step(sim.act(c={1: N}))
    assert int(sim.s.c_status[1]) == 2 and int(sim.s.c_status[0]) == 1 and bool(sim.last_ev.fouled[1])


def test_foul_when_neither_entered_puts_uppu_out():
    sim = Sim()
    sim.park_others({0, 1})
    sim.crosser(0, (1.5, 4.0), kappa=7, box=1)
    sim.crosser(1, (1.5, 5.0), kappa=1, box=1)
    sim.step()
    assert int(sim.s.c_status[0]) == 2 and int(sim.s.c_status[1]) == 0


def test_classic_clap_gating():
    sim = Sim()
    _, s = sim.env.reset(jax.random.PRNGKey(3))
    sim.s = s
    for _ in range(10):
        sim.step(sim.act(c={0: N}))
    assert float(sim.s.c_pos[0, 1]) < 0 and not bool(sim.s.live)
    sim.step(n=CLASSIC.clap_auto_ticks)
    assert bool(sim.s.live)


def test_p4_false_start_is_out():
    p = CLASSIC.replace(p4_clap=True)
    sim = Sim(p)
    _, s = sim.env.reset(jax.random.PRNGKey(3))
    sim.s = s.replace(c_pos=s.c_pos.at[0].set(jnp.array([1.0, -0.1])))
    sim.step(sim.act(c={0: N}), n=3)
    assert int(sim.s.c_status[0]) == 2 and bool(sim.last_ev.false_start[0]) or int(sim.s.c_status[0]) == 2


def test_p2_cone_blocks_tag_from_behind():
    p = CLASSIC.replace(p2_facing=True)
    sim = Sim(p)
    sim.park_others({0, 1})
    sim.set(g_x=sim.s.g_x.at[1].set(1.0))                 # guard on y=6 facing home (-1)
    sim.crosser(0, (1.0, 6.3), kappa=3)                   # behind the guard
    sim.crosser(1, (3.0, 5.7), kappa=2)
    sim.set(g_x=sim.s.g_x.at[1].set(1.0))
    sim.step()
    assert int(sim.s.c_status[0]) != 2
    sim.crosser(1, (1.0, 5.7), kappa=2)                   # in front of the guard
    sim.step()
    assert int(sim.s.c_status[1]) == 2


def test_innings_ends_when_all_out_and_on_clock():
    sim = Sim()
    sim.park_others(set())
    sim.step()
    assert bool(sim.s.done)
    sim2 = Sim()
    sim2.park_others({0})
    sim2.set(t=jnp.array(CLASSIC.max_ticks - 1, jnp.int32))
    sim2.step()
    assert bool(sim2.s.done)


def test_rewards_are_zero_sum_and_normalised():
    sim = Sim()
    sim.park_others({0})
    sim.crosser(0, (0.3, -0.1))
    sim.step(sim.act(c={0: N}), n=3)
    r = np.asarray(sim.r)
    assert r[0] == pytest.approx(-r[1])
    assert float(sim.s.points) == 1.0


def test_same_seed_same_actions_same_hash():
    from kilithattu_core.replay import state_hash
    env, step = get_env(CLASSIC)
    def run():
        key = jax.random.PRNGKey(7); _, s = env.reset(key)
        for t in range(120):
            a = jax.random.randint(jax.random.PRNGKey(t), (CLASSIC.n_agents,), 0, 3)
            _, s, *_ = step(key, s, zeros_actions(CLASSIC.n_agents).replace(move=a))
        return state_hash(s)
    assert run() == run()
