"""Curriculum: staged training against scripted opponents; parameters carry over between stages
(they are independent of team size, so the stage may change the number of crossers)."""
from __future__ import annotations

import jax

from kilithattu_agents import NetworkPolicy, load_policy, save_checkpoint
from kilithattu_core import EnvParams, load_params, make_env

from .config import StageCfg, TrainConfig
from .mappo import Trainer, run_updates
from .play import play_innings
from .rollout import Ctrl

DEFAULT_STAGES = [
    StageCfg("s0_cross_empty", env={"n_crossers": 1, "r_tag": 0.0, "innings_s": 24.0}, defence="scripted:guard?reaction=0.1", updates=300, promote_points=12.0),
    StageCfg("s1_one_vs_slow_guards", env={"n_crossers": 1, "innings_s": 40.0}, defence="scripted:guard?reaction=0.15", updates=300, promote_points=8.0),
    StageCfg("s2_two_vs_scripted", env={"n_crossers": 2, "innings_s": 60.0}, defence="scripted:guard?reaction=0.3", updates=400, promote_points=8.0),
    StageCfg("s3_full_vs_scripted", env={}, defence="scripted:guard?reaction=0.4", updates=600, promote_points=12.0),
]


def make_ctrl(spec: str, side: str, p: EnvParams) -> Ctrl:
    sl = slice(0, p.n_crossers) if side == "attack" else slice(p.n_crossers, p.n_agents)
    n = sl.stop - sl.start
    if spec in ("learn", "frozen"):
        return Ctrl(side, "net", n, sl, learn=spec == "learn")
    return Ctrl(side, "scripted", n, sl, policy=load_policy(spec, side, p))


def stage_env(cfg: TrainConfig, st: StageCfg) -> EnvParams:
    return load_params(cfg.rules).replace(**st.env)


def evaluate_stage(p: EnvParams, trainer: Trainer, ts, n_episodes: int, seed: int) -> dict:
    """Deterministic evaluation of the learning attack against the stage defence."""
    att = trainer.ca.policy if trainer.ca.kind == "scripted" else NetworkPolicy("attack", p, trainer.net, ts.params["a"]["actor"], deterministic=False)
    dfn = trainer.cd.policy if trainer.cd.kind == "scripted" else NetworkPolicy("defence", p, trainer.net, ts.params["d"]["actor"])
    r = play_innings(p, att, dfn, n_episodes, seed)
    return {"eval/points": float(r["points"].mean()), "eval/trips": float(r["trips"].mean()), "eval/all_out": float(r["all_out"].mean())}


def run_curriculum(cfg: TrainConfig, stages: list[StageCfg] | None = None, logger=None, params_a=None, params_d=None, key=None):
    """Train the attack (and/or defence) through the stages. Returns final params dict {"a":..., "d":...}."""
    stages = stages or cfg.stages or DEFAULT_STAGES
    key = key if key is not None else jax.random.PRNGKey(cfg.seed)
    net = cfg.net.build()
    done_updates = 0
    total = sum(s.updates for s in stages)
    for si, st in enumerate(stages):
        p = stage_env(cfg, st)
        ca, cd = make_ctrl(st.attack, "attack", p), make_ctrl(st.defence, "defence", p)
        tr = Trainer(make_env(p), net, cfg.ppo, ca, cd)
        key, k = jax.random.split(key)
        ts = tr.init(k, params_a=params_a, params_d=params_d)
        chunk = max(1, min(cfg.ckpt_every, st.updates))
        ran = 0
        best: dict = {}
        while ran < st.updates:
            n = min(chunk, st.updates - ran)
            ts, m = run_updates(tr, ts, n, total, start=done_updates + ran, scale=st.ent_scale, log=(lambda i, m, _s=st: logger(i, {**m, "stage": _s.name})) if logger else None, log_every=cfg.log_every)
            ran += n
            ev = evaluate_stage(p, tr, ts, cfg.eval_episodes, cfg.seed + ran)
            if logger:
                logger(done_updates + ran, {**ev, "stage": st.name}, tag="eval")
            if logger:  # checkpoint at every evaluation, and keep the best by the learner's own objective
                score = ev["eval/points"]
                for side, k, ctrl in (("attack", "a", ca), ("defence", "d", cd)):
                    prm = ts.params[k]
                    if not (ctrl.learn and prm is not None):
                        continue
                    kw = dict(critic_params=prm["critic"])
                    save_checkpoint(logger.dir / side / "latest", prm["actor"], p, net, side, cfg.run_id, done_updates + ran, **kw)
                    better = score > best.get(side, -1e9) if side == "attack" else score < best.get(side, 1e9)
                    if better:  # attack wants more points, defence wants fewer
                        best[side] = score
                        save_checkpoint(logger.dir / side / "best", prm["actor"], p, net, side, cfg.run_id, done_updates + ran, extra={"eval_points": score}, **kw)
            if st.promote_points and ev["eval/points"] >= st.promote_points:
                break
        done_updates += st.updates
        params_a, params_d = ts.params["a"], ts.params["d"]
        if logger and params_a is not None:
            save_checkpoint(logger.dir / "attack" / st.name, params_a["actor"], p, net, "attack", cfg.run_id, done_updates, critic_params=params_a["critic"])
            save_checkpoint(logger.dir / "attack" / "latest", params_a["actor"], p, net, "attack", cfg.run_id, done_updates, critic_params=params_a["critic"])
        if logger and params_d is not None and cd.learn:
            save_checkpoint(logger.dir / "defence" / "latest", params_d["actor"], p, net, "defence", cfg.run_id, done_updates, critic_params=params_d["critic"])
    return {"a": params_a, "d": params_d}
