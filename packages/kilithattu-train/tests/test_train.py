import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kilithattu_agents import NetConfig, ScriptedAttack, ScriptedDefence
from kilithattu_core import CLASSIC
from kilithattu_train.config import PPOConfig, StageCfg, TrainConfig, load_config
from kilithattu_train.league.matchmaker import pfsp_weights
from kilithattu_train.mappo import Trainer, gae, run_updates
from kilithattu_train.rollout import Ctrl

NET = NetConfig(d_model=16, layers=1, heads=2, hidden=16)
PPO = PPOConfig(num_envs=4, rollout_len=8, epochs=1, minibatches=2, total_updates=4)
P2 = CLASSIC.replace(n_crossers=2, innings_s=10.0)


def test_gae_matches_reference():
    rng = np.random.default_rng(0)
    T, E = 7, 3
    rew, val = rng.normal(size=(T, E)), rng.normal(size=(T, E))
    last, done = rng.normal(size=E), rng.random((T, E)) < 0.2
    adv, ret = gae(jnp.asarray(rew, jnp.float32), jnp.asarray(val, jnp.float32), jnp.asarray(last, jnp.float32), jnp.asarray(done), 0.99, 0.9)
    ref = np.zeros((T, E)); nxt = np.zeros(E); vn = last.copy()
    for t in reversed(range(T)):
        nd = 1 - done[t]
        delta = rew[t] + 0.99 * vn * nd - val[t]
        nxt = delta + 0.99 * 0.9 * nd * nxt
        ref[t] = nxt; vn = val[t]
    np.testing.assert_allclose(np.asarray(adv), ref, atol=1e-4)
    np.testing.assert_allclose(np.asarray(ret), ref + val, atol=1e-4)


def test_pfsp_prefers_hard_opponents():
    w = pfsp_weights(np.array([0.9, 0.5, 0.1]), 2.0)
    assert w[2] > w[1] > w[0] and abs(w.sum() - 1) < 1e-9


def test_config_overrides():
    cfg = load_config(None, ["ppo.num_envs=8", "net.d_model=32", "seed=3"])
    assert cfg.ppo.num_envs == 8 and cfg.net.d_model == 32 and cfg.seed == 3


def test_trainer_updates_attack_against_scripted_defence():
    p = P2
    ca = Ctrl("attack", "net", 2, slice(0, 2), learn=True)
    cd = Ctrl("defence", "scripted", p.n_defenders, slice(2, p.n_agents), policy=ScriptedDefence(p))
    tr = Trainer(__import__("kilithattu_core").make_env(p), NET, PPO, ca, cd)
    ts = tr.init(jax.random.PRNGKey(0))
    before = jax.tree_util.tree_leaves(ts.params["a"]["actor"])[0].copy()
    ts, m = run_updates(tr, ts, 2, 4, log=None, log_every=1)
    assert all(np.isfinite(v) for v in m.values())
    assert not jnp.allclose(before, jax.tree_util.tree_leaves(ts.params["a"]["actor"])[0])


@pytest.mark.nightly
def test_bc_loss_decreases():
    from kilithattu_agents.policy import init_networks
    from kilithattu_train.bc import bc_train, collect_scripted
    p = P2
    obs, act, _ = collect_scripted(p, ScriptedAttack(p), ScriptedDefence(p), "attack", n_envs=4, T=64)
    _, _, a, _ = init_networks(jax.random.PRNGKey(0), p, NET)
    a2, losses = bc_train(a, NET, obs, act, steps=25, lr=3e-3)
    assert losses[-1] < losses[0]


@pytest.mark.nightly
def test_league_phase_records_payoffs_and_persists(tmp_path):
    from kilithattu_train.league import League
    from kilithattu_train.logging_utils import RunLogger
    cfg = TrainConfig(run_id="t", out_dir=str(tmp_path), league_phases=1, phase_updates=1, snapshot_every=1, eval_episodes=2, exploiter_updates=1)
    cfg.ppo, cfg.net = PPO, __import__("kilithattu_train.config", fromlist=["NetCfg"]).NetCfg(16, 1, 2, 16)
    log = RunLogger(tmp_path, "t", config={})
    lg = League(cfg, P2, log)
    pool = lg.run(exploiters=False)
    assert len(pool.side("attack")) >= 2 and len(pool.side("defence")) >= 2 and pool.payoff
    assert (tmp_path / "t" / "league" / "league.json").exists()


@pytest.mark.nightly
def test_fcp_partner_mixing_trains(tmp_path):
    from kilithattu_train.config import NetCfg
    from kilithattu_train.logging_utils import RunLogger
    from kilithattu_train.partners import train_fcp, train_partner_pool
    cfg = TrainConfig(run_id="f", out_dir=str(tmp_path), fcp_partners=1, fcp_partner_updates=2, fcp_updates=2, eval_episodes=2)
    cfg.ppo, cfg.net = PPO, NetCfg(16, 1, 2, 16)
    p = CLASSIC.replace(n_crossers=3, innings_s=10.0)
    partners = train_partner_pool(cfg, p)
    assert len(partners) == 2
    learner = train_fcp(cfg, p, partners, phases=1, logger=RunLogger(tmp_path, "f", config={}))
    assert "actor" in learner and (tmp_path / "f" / "attack" / "fcp" / "metadata.json").exists()


@pytest.mark.nightly
def test_curriculum_two_tiny_stages(tmp_path):
    from kilithattu_train.config import NetCfg
    from kilithattu_train.curriculum import run_curriculum
    from kilithattu_train.logging_utils import RunLogger
    cfg = TrainConfig(run_id="c", out_dir=str(tmp_path), eval_episodes=2, log_every=1, ckpt_every=2)
    cfg.ppo, cfg.net = PPO, NetCfg(16, 1, 2, 16)
    stages = [StageCfg("a", env={"n_crossers": 1, "r_tag": 0.0, "innings_s": 6.0}, updates=2), StageCfg("b", env={"n_crossers": 2, "innings_s": 6.0}, updates=2)]
    out = run_curriculum(cfg, stages, RunLogger(tmp_path, "c", config={}))
    assert out["a"] is not None and (tmp_path / "c" / "attack" / "b" / "metadata.json").exists()
