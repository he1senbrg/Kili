"""``kilithattu-train`` command line: curriculum | selfplay | league | fcp | bc."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

from kilithattu_agents import NetworkPolicy, ScriptedAttack, ScriptedDefence, load_checkpoint, save_checkpoint
from kilithattu_agents.policy import init_networks
from kilithattu_core import load_params, make_env

from .bc import bc_train, collect_scripted, replay_dataset
from .config import TrainConfig, load_config
from .curriculum import make_ctrl, run_curriculum
from .logging_utils import RunLogger
from .mappo import Trainer, run_updates


def _cfg(args) -> TrainConfig:
    return load_config(args.config, args.override)


def cmd_curriculum(args):
    cfg = _cfg(args)
    log = RunLogger(cfg.out_dir, cfg.run_id, config=cfg.__dict__, wandb=cfg.wandb)
    run_curriculum(cfg, logger=log)


def cmd_selfplay(args):
    cfg = _cfg(args)
    p = load_params(cfg.rules)
    log = RunLogger(cfg.out_dir, cfg.run_id, config=cfg.__dict__, wandb=cfg.wandb)
    net = cfg.net.build()
    ca, cd = make_ctrl("learn" if cfg.side in ("both", "attack") else "frozen", "attack", p), make_ctrl("learn" if cfg.side in ("both", "defence") else "frozen", "defence", p)
    tr = Trainer(make_env(p), net, cfg.ppo, ca, cd)
    ts = tr.init(jax.random.PRNGKey(cfg.seed))
    ts, _ = run_updates(tr, ts, cfg.ppo.total_updates, cfg.ppo.total_updates, log=lambda i, m: log(i, m), log_every=cfg.log_every)
    for side, k in (("attack", "a"), ("defence", "d")):
        save_checkpoint(log.dir / side / "latest", ts.params[k]["actor"], p, net, side, cfg.run_id, cfg.ppo.total_updates, critic_params=ts.params[k]["critic"])


def cmd_league(args):
    from .league import League

    cfg = _cfg(args)
    p = load_params(cfg.rules)
    log = RunLogger(cfg.out_dir, cfg.run_id, config=cfg.__dict__, wandb=cfg.wandb)
    init_a = load_checkpoint(args.init_attack)[0] if args.init_attack else None
    league = League(cfg, p, log, init_attack=None if init_a is None else {"actor": init_a, "critic": init_networks(jax.random.PRNGKey(0), p, cfg.net.build())[3]})
    league.run(exploiters=not args.no_exploiters)


def cmd_fcp(args):
    from .partners import train_fcp, train_partner_pool

    cfg = _cfg(args)
    p = load_params(cfg.rules)
    log = RunLogger(cfg.out_dir, cfg.run_id, config=cfg.__dict__, wandb=cfg.wandb)
    partners = train_partner_pool(cfg, p, logger=log)
    train_fcp(cfg, p, partners, logger=log)


def cmd_bc(args):
    cfg = _cfg(args)
    p = load_params(cfg.rules)
    net = cfg.net.build()
    log = RunLogger(cfg.out_dir, cfg.run_id, config=cfg.__dict__)
    _, _, actor, _ = init_networks(jax.random.PRNGKey(cfg.seed), p, net)
    if args.replays:
        data = [replay_dataset(r, args.side) for r in args.replays]
        import jax.numpy as jnp
        obs = jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, 1), *[d[0] for d in data]) if len(data) > 1 else data[0][0]
        act = jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, 1), *[d[1] for d in data]) if len(data) > 1 else data[0][1]
    else:
        obs, act, _ = collect_scripted(p, ScriptedAttack(p), ScriptedDefence(p), args.side, n_envs=16, T=256, seed=cfg.seed)
    actor, losses = bc_train(actor, net, obs, act, steps=args.steps)
    log(args.steps, {"bc/loss_first": losses[0], "bc/loss_last": losses[-1]}, tag="bc")
    save_checkpoint(log.dir / args.side / "bc", actor, p, net, args.side, cfg.run_id, args.steps)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kilithattu-train")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("curriculum", cmd_curriculum), ("selfplay", cmd_selfplay), ("league", cmd_league), ("fcp", cmd_fcp), ("bc", cmd_bc)):
        sp = sub.add_parser(name)
        sp.add_argument("--config", default=None)
        sp.add_argument("--override", "-o", nargs="*", default=[], help="OmegaConf dotlist, e.g. ppo.num_envs=128")
        if name == "league":
            sp.add_argument("--init-attack", default=None)
            sp.add_argument("--no-exploiters", action="store_true")
        if name == "bc":
            sp.add_argument("--side", default="attack", choices=["attack", "defence"])
            sp.add_argument("--steps", type=int, default=200)
            sp.add_argument("--replays", nargs="*", default=None)
        sp.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
