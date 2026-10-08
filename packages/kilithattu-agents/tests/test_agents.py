import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kilithattu_agents import (HumanFairConfig, HumanFairEnv, NetConfig, NetworkPolicy, ScriptedAttack, ScriptedDefence,
                               init_networks, join_actions, load_checkpoint, load_policy, save_checkpoint)
from kilithattu_core import CLASSIC, PLUS, make_env, zeros_actions

CFG = NetConfig(d_model=32, layers=1, heads=2, hidden=32)


def test_network_policy_respects_masks_for_both_sides():
    p = PLUS
    env = make_env(p)
    obs, s = env.reset(jax.random.PRNGKey(0))
    _, _, a_params, _ = init_networks(jax.random.PRNGKey(1), p, CFG)
    for side in ("attack", "defence"):
        pol = NetworkPolicy(side, p, CFG, a_params)
        carry = pol.init_carry()
        for t in range(5):
            act, carry, info = pol.act(jax.random.PRNGKey(t), carry, obs)
            sl = pol.sl
            for h in ("move", "dash", "turn", "clap", "call"):
                m = np.asarray(getattr(obs.mask, h))[sl]
                chosen = np.asarray(getattr(act, h))
                assert m[np.arange(len(chosen)), chosen].all(), (side, h)


def test_params_are_field_size_independent():
    _, _, a1, c1 = init_networks(jax.random.PRNGKey(0), CLASSIC, CFG)
    big = CLASSIC.replace(n_lanes=3, n_rows=7, n_crossers=5)
    _, _, a2, c2 = init_networks(jax.random.PRNGKey(0), big, CFG)
    assert jax.tree_util.tree_map(jnp.shape, a1) == jax.tree_util.tree_map(jnp.shape, a2)
    pol = NetworkPolicy("attack", big, CFG, a1)               # params from the small field run on the big one
    obs, _ = make_env(big).reset(jax.random.PRNGKey(0))
    act, _, _ = pol.act(jax.random.PRNGKey(0), pol.init_carry(), obs)
    assert act.move.shape == (5,)


def test_checkpoint_roundtrip_and_compat(tmp_path):
    _, _, a, _ = init_networks(jax.random.PRNGKey(0), CLASSIC, CFG)
    save_checkpoint(tmp_path / "ck", a, CLASSIC, CFG, "attack", run_id="t", step=3)
    a2, cfg2, meta = load_checkpoint(tmp_path / "ck", CLASSIC)
    assert cfg2 == CFG and meta["step"] == 3 and meta["side"] == "attack"
    assert all(jnp.allclose(x, y) for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(a2)))
    with pytest.warns(UserWarning):
        load_checkpoint(tmp_path / "ck", PLUS)               # different rules -> warning
    import json
    m = json.loads((tmp_path / "ck" / "metadata.json").read_text()); m["env_version"] = "9.0.0"
    (tmp_path / "ck" / "metadata.json").write_text(json.dumps(m))
    with pytest.raises(RuntimeError):
        load_checkpoint(tmp_path / "ck", CLASSIC)


@pytest.mark.parametrize("p", [CLASSIC, PLUS])
def test_scripted_bots_play_legal_and_score(p):
    env = make_env(p)
    att, dfn = ScriptedAttack(p), ScriptedDefence(p)
    obs, s = env.reset(jax.random.PRNGKey(0))
    ca, cd = att.init_carry(jax.random.PRNGKey(1)), dfn.init_carry(jax.random.PRNGKey(2))
    step, aa, da = jax.jit(env.step), jax.jit(att.act), jax.jit(dfn.act)
    pts = 0.0
    for t in range(800):
        a1, ca, _ = aa(jax.random.PRNGKey(t), ca, obs)
        a2, cd, _ = da(jax.random.PRNGKey(t + 9), cd, obs)
        act = join_actions(a1, a2)
        for h in ("move", "dash", "turn", "clap", "call"):
            m = np.asarray(getattr(obs.mask, h)); ch = np.asarray(getattr(act, h))
            if h == "move":
                assert m[np.arange(len(ch)), ch].all(), (t, h, ch[~m[np.arange(len(ch)), ch]])
        obs, s, r, d, ev = step(jax.random.PRNGKey(t), s, act)
        if bool(d):
            break
    assert float(s.points) > 0


def test_registry_scripted_refs():
    pol = load_policy("scripted:guard?reaction=0.7", "defence", CLASSIC)
    assert isinstance(pol, ScriptedDefence) and pol.reaction == 0.7
    assert isinstance(load_policy("scripted:crosser?aggression=1.3", "attack", CLASSIC), ScriptedAttack)


def test_human_fair_delays_observations_and_is_jittable():
    env = HumanFairEnv(make_env(CLASSIC), HumanFairConfig(max_delay=3, delay_range=(3, 3), interval_range=(2, 2), sticky_range=(0, 0)))
    obs, st = env.reset(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    a = zeros_actions(CLASSIC.n_agents).replace(move=jnp.full((CLASSIC.n_agents,), 2))
    o1, st, *_ = step(jax.random.PRNGKey(1), st, a)
    assert int(st.delay) == 3 and int(st.interval) == 2
    assert jnp.allclose(o1.players, obs.players)             # first ticks still show the reset observation
    for i in range(6):
        o, st, *_ = step(jax.random.PRNGKey(i), st, a)
    assert not jnp.allclose(o.players, obs.players)
