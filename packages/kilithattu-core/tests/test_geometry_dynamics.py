import jax.numpy as jnp
import numpy as np
from conftest import Sim

from kilithattu_core import CLASSIC
from kilithattu_core.geometry import box_index, kili_move_mask


def test_boxes_and_zones():
    b = lambda x, y: int(box_index(CLASSIC, jnp.array([x, y])))
    assert b(1, -1) == -1 and b(1, 16) == -1
    assert b(1, 1) == 0 and b(4, 1) == 5 and b(1, 14.9) == 4 and b(4, 15.0) == 9
    assert b(3.0, 3.0) == 5 + 1                              # exactly on a line: higher-index box


def test_kili_mask_follows_the_line_graph():
    m = lambda x, y: np.asarray(kili_move_mask(CLASSIC, jnp.array([x, y], jnp.float32)))
    assert m(3, 0).tolist() == [True, True, True, False, True]        # junction on L0: N, E, W
    assert m(1.5, 0).tolist() == [True, False, True, False, True]     # mid-segment on L0: E, W only
    assert m(3, 1.5).tolist() == [True, True, False, True, False]     # mid-segment on centre line: N, S
    assert m(0, 15).tolist() == [True, False, True, True, False]      # corner: E, S


def test_kili_snaps_at_junctions_and_turns():
    sim = Sim()
    sim.set(k_pos=jnp.array([2.9, 0.0]))                               # within v*dt of the centre junction
    sim.step(sim.act(k=1))                                             # N: snaps onto x=3 and climbs
    assert float(sim.s.k_pos[0]) == 3.0 and float(sim.s.k_pos[1]) > 0.0
    sim.step(sim.act(k=1), n=3)                                        # climb well past the snap distance
    sim.step(sim.act(k=2))                                             # E mid-segment is masked -> stays
    assert float(sim.s.k_pos[0]) == 3.0


def test_guard_stays_on_its_line_and_in_bounds():
    sim = Sim()
    for _ in range(200):
        sim.step(sim.act(g={0: 2, 1: 1, 2: 2, 3: 1}))
    x = np.asarray(sim.s.g_x)
    assert (x >= 0).all() and (x <= CLASSIC.field_w).all()


def test_p1_acceleration_limit_and_stamina():
    from kilithattu_core import PLUS
    p = CLASSIC.replace(p1_momentum=True)
    sim = Sim(p)
    sim.park_others({0})
    sim.crosser(0, (1.0, 5.0), kappa=2)
    sim.step(sim.act(c={0: 1}, dash=jnp.zeros(p.n_agents, jnp.int32).at[0].set(1)))
    assert float(jnp.linalg.norm(sim.s.c_vel[0])) <= p.a_max * p.dt + 1e-4
    sim.step(sim.act(c={0: 1}, dash=jnp.zeros(p.n_agents, jnp.int32).at[0].set(1)), n=60)
    assert float(sim.s.c_stam[0]) < 1.0
