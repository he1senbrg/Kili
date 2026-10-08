"""Fictitious Co-Play (FCP): a diverse partner pool, then a learner trained with randomly sampled partners.

Stage 1 trains ``fcp_partners`` attack policies against the scripted defence with different seeds, entropy
scales and shaping weights, saving an early and a final checkpoint of each. Stage 2 trains one crosser slot
while the other crossers are played by frozen partners sampled each phase (the loss only uses the learner
slot). The result generalises to teammates it never trained with, which is what human-AI teaming needs.
"""
from __future__ import annotations

import copy
from dataclasses import replace

import jax
import numpy as np

from kilithattu_agents import save_checkpoint
from kilithattu_core import EnvParams, make_env

from .config import TrainConfig
from .curriculum import make_ctrl
from .mappo import Trainer, run_updates


def train_partner_pool(cfg: TrainConfig, p: EnvParams, defence: str = "scripted:guard?reaction=0.3", logger=None, init=None):
    """-> list of actor-parameter pytrees (early + final checkpoint per partner)."""
    net = cfg.net.build()
    pool = []
    for i in range(cfg.fcp_partners):
        ppo = copy.deepcopy(cfg.ppo)
        ppo.ent_start *= float(np.random.default_rng(cfg.seed + i).uniform(0.5, 2.0))
        ppo.shaping_beta = float(np.random.default_rng(cfg.seed + 100 + i).uniform(0.0, 0.4))
        tr = Trainer(make_env(p), net, ppo, make_ctrl("learn", "attack", p), make_ctrl(defence, "defence", p))
        ts = tr.init(jax.random.PRNGKey(cfg.seed * 1000 + i), params_a=init)
        half = max(1, cfg.fcp_partner_updates // 2)
        ts, _ = run_updates(tr, ts, half, cfg.fcp_partner_updates, log=None, log_every=half)
        pool.append(ts.params["a"]["actor"])
        ts, _ = run_updates(tr, ts, cfg.fcp_partner_updates - half, cfg.fcp_partner_updates, start=half, log=None, log_every=cfg.fcp_partner_updates)
        pool.append(ts.params["a"]["actor"])
        if logger:
            logger(i + 1, {"fcp/partner": i}, tag="fcp")
    return pool


def train_fcp(cfg: TrainConfig, p: EnvParams, partners: list, defence: str = "scripted:guard?reaction=0.3", logger=None, params_a=None, phases: int | None = None):
    """Train the learner (crosser slot 0) with sampled partners. Returns the learner params."""
    net = cfg.net.build()
    mask = np.array([False] + [True] * (p.n_crossers - 1))
    tr = Trainer(make_env(p), net, cfg.ppo, make_ctrl("learn", "attack", p), make_ctrl(defence, "defence", p), partner_mask=mask)
    rng = np.random.default_rng(cfg.seed)
    key = jax.random.PRNGKey(cfg.seed + 7)
    phases = phases or max(1, cfg.fcp_updates // 10)
    per = max(1, cfg.fcp_updates // phases)
    ts = None
    learner = params_a
    for ph in range(phases):
        partner = partners[int(rng.integers(len(partners)))]
        key, k = jax.random.split(key)
        ts = tr.init(k, params_a=learner, partner=partner)
        if learner is None:
            learner = ts.params["a"]
        ts, m = run_updates(tr, ts, per, cfg.fcp_updates, start=ph * per, log=None, log_every=per)
        learner = ts.params["a"]
        if logger:
            logger((ph + 1) * per, {**m, "fcp/phase": ph}, tag="fcp")
    if logger:
        save_checkpoint(logger.dir / "attack" / "fcp", learner["actor"], p, net, "attack", cfg.run_id, cfg.fcp_updates, critic_params=learner["critic"])
    return learner
