import numpy as np
import jax
import pytest

from kilithattu_core import CLASSIC, PLUS, ENV_VERSION, load_params, rules_hash, sample_field
from kilithattu_core.replay import stack_actions, verify_replay, write_replay
from kilithattu_core import zeros_actions


def test_presets_and_hash():
    assert load_params("classic") == CLASSIC
    assert load_params("plus").p1_momentum and load_params("plus").p4_clap
    assert rules_hash(CLASSIC) != rules_hash(PLUS)
    with pytest.raises(ValueError):
        load_params({"nope": 1})


def test_derived_values():
    assert CLASSIC.trip_points == 18 and CLASSIC.max_ticks == 2400 and CLASSIC.n_guards == 4 and CLASSIC.n_agents == 9
    assert CLASSIC.field_w == 6 and CLASSIC.field_l == 15


def test_sample_field_ranges():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = sample_field(rng)
        assert 2 <= p.n_lanes <= 3 and 4 <= p.n_rows <= 9 and 3 <= p.n_crossers <= 6


def test_replay_roundtrip_and_verify(tmp_path):
    acts = []
    for t in range(150):
        a = jax.random.randint(jax.random.PRNGKey(t), (CLASSIC.n_agents,), 0, 3)
        acts.append(zeros_actions(CLASSIC.n_agents).replace(move=a))
    log = stack_actions(acts)
    h = write_replay(tmp_path / "r.kreplay", CLASSIC, seed=5, action_log=log)
    assert h["env_version"] == ENV_VERSION
    assert verify_replay(tmp_path / "r.kreplay")
    # tampering is detected
    log2 = log.copy(); log2[:, :, 0] = (log2[:, :, 0] + 1) % 3
    np.savez_compressed(tmp_path / "r.kreplay" / "actions.npz", actions=log2)
    assert not verify_replay(tmp_path / "r.kreplay")
