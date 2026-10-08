"""League learners (main, main-exploiter, league-exploiter) and the league training loop."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

from kilithattu_agents import NetworkPolicy, load_checkpoint
from kilithattu_core import EnvParams, load_params, make_env

from ..config import TrainConfig
from ..mappo import Trainer, run_updates
from ..play import play_innings
from ..rollout import Ctrl
from .matchmaker import sample_opponent
from .pool import LeaguePool, Snapshot


@dataclass
class Agent:
    id: str
    side: str
    kind: str  # main | main_exploiter | league_exploiter
    params: Any
    opt: Any
    updates: int = 0


class League:
    def __init__(self, cfg: TrainConfig, env_params: EnvParams, logger=None, init_attack=None, init_defence=None):
        self.cfg, self.p, self.log = cfg, env_params, logger
        self.net = cfg.net.build()
        self.env = make_env(env_params)
        n, nd = env_params.n_crossers, env_params.n_defenders
        sa, sd = slice(0, n), slice(n, env_params.n_agents)
        self.tr = {
            "attack": Trainer(self.env, self.net, cfg.ppo, Ctrl("attack", "net", n, sa, learn=True), Ctrl("defence", "net", nd, sd, learn=False)),
            "defence": Trainer(self.env, self.net, cfg.ppo, Ctrl("attack", "net", n, sa, learn=False), Ctrl("defence", "net", nd, sd, learn=True)),
        }
        self.rng = np.random.default_rng(cfg.seed)
        self.key = jax.random.PRNGKey(cfg.seed)
        self.pool = LeaguePool()
        self.total_updates = cfg.league_phases * cfg.phase_updates * 2
        self.counter = 0
        self.agents: dict[str, Agent] = {}
        self.main = {"attack": self._new_agent("main_att", "attack", "main", init_attack), "defence": self._new_agent("main_def", "defence", "main", init_defence)}
        for side in ("attack", "defence"):
            self._snapshot(self.main[side], "past")

    # ---- helpers ----------------------------------------------------------------------------------
    def _next_key(self):
        self.key, k = jax.random.split(self.key)
        return k

    def _new_agent(self, aid: str, side: str, kind: str, init=None) -> Agent:
        tr = self.tr[side]
        params = init if init is not None else tr.init_params(self._next_key())
        a = Agent(aid, side, kind, params, tr.opt.init(params))
        self.agents[aid] = a
        return a

    def _snapshot(self, agent: Agent, kind: str | None = None) -> Snapshot:
        self.counter += 1
        s = Snapshot(f"{agent.id}#{self.counter}", agent.side, kind or agent.kind, agent.updates, jax.tree_util.tree_map(lambda x: x, agent.params))
        self.pool.add(s)
        return s

    def _policy(self, side: str, params, det: bool = False) -> NetworkPolicy:
        return NetworkPolicy(side, self.p, self.net, params["actor"], det)

    def _opponents(self, side: str) -> list[Snapshot]:
        return self.pool.side("defence" if side == "attack" else "attack")

    def _p_win_learner(self, agent: Agent, opps: list[Snapshot], latest_id: str | None) -> np.ndarray:
        out = []
        for o in opps:
            w = self.pool.attack_win(latest_id, o.id) if agent.side == "attack" else 1.0 - self.pool.attack_win(o.id, latest_id)
            out.append(w)
        return np.asarray(out)

    def _choose_opponent(self, agent: Agent) -> Snapshot:
        opps = self._opponents(agent.side)
        if agent.kind == "main_exploiter":  # only the current main agent of the other side
            other = self.main["defence" if agent.side == "attack" else "attack"]
            return Snapshot("main_latest", other.side, "main", other.updates, other.params)
        if agent.kind == "main" and self.rng.random() < 0.35:  # self-play against the latest main
            other = self.main["defence" if agent.side == "attack" else "attack"]
            return Snapshot("main_latest", other.side, "main", other.updates, other.params)
        latest = self._latest_id(agent)
        pw = self._p_win_learner(agent, opps, latest)
        oid = sample_opponent(self.rng, [o.id for o in opps], pw, self.cfg.pfsp_power)
        return self.pool.snaps[oid]

    def _latest_id(self, agent: Agent) -> str | None:
        mine = [s for s in self.pool.side(agent.side) if s.id.startswith(agent.id + "#")]
        return mine[-1].id if mine else None

    # ---- one training phase for one agent ---------------------------------------------------------------
    def train_agent(self, agent: Agent, n_updates: int) -> dict:
        tr = self.tr[agent.side]
        opp = self._choose_opponent(agent)
        pa, pd = (agent.params, opp.params) if agent.side == "attack" else (opp.params, agent.params)
        ts = tr.init(self._next_key(), params_a=pa, params_d=pd)
        key_opt = "a" if agent.side == "attack" else "d"
        ts = ts.replace(opt={**ts.opt, key_opt: agent.opt})
        ts, m = run_updates(tr, ts, n_updates, self.total_updates, start=agent.updates, log=None, log_every=max(1, n_updates))
        agent.params, agent.opt = ts.params[key_opt], ts.opt[key_opt]
        agent.updates += n_updates
        # evaluate against the chosen opponent and update the payoff table
        att_p, def_p = (agent.params, opp.params) if agent.side == "attack" else (opp.params, agent.params)
        r = play_innings(self.p, self._policy("attack", att_p), self._policy("defence", def_p), self.cfg.eval_episodes, int(self.rng.integers(1 << 30)), env=self.env)
        win = float((r["points"] >= self.cfg.win_points).mean())
        me = self._snapshot(agent, "past") if False else None
        latest = self._latest_id(agent) or agent.id
        if agent.side == "attack":
            self.pool.record(latest, opp.id, win, float(r["points"].mean()))
        else:
            self.pool.record(opp.id, latest, win, float(r["points"].mean()))
        return {**m, "opp": opp.id, "win_attack": win, "points": float(r["points"].mean())}

    # ---- main loop ---------------------------------------------------------------------------------------
    def run(self, exploiters: bool = True, resume_hook=None) -> LeaguePool:
        cfg = self.cfg
        for phase in range(cfg.league_phases):
            for side in ("attack", "defence"):
                m = self.train_agent(self.main[side], cfg.phase_updates)
                if self.log:
                    self.log(phase * 2 + (side == "defence") + 1, {"league/side": side, "league/opp": m["opp"], "league/win_attack": m["win_attack"], "eval/points": m["points"]}, tag="league")
            if exploiters:
                self._exploiters(phase)
            if (phase + 1) % cfg.snapshot_every == 0:
                for side in ("attack", "defence"):
                    self._snapshot(self.main[side], "main")
            if self.log and hasattr(self.log, "dir"):
                self.pool.save(self.log.dir, self.p, self.net, cfg.run_id)
        return self.pool

    def _exploiters(self, phase: int) -> None:
        cfg = self.cfg
        for side in ("attack", "defence"):
            for kind in ("main_exploiter", "league_exploiter"):
                aid = f"{kind}_{side[:3]}"
                ag = self.agents.get(aid) or self._new_agent(aid, side, kind)
                self.train_agent(ag, max(1, cfg.phase_updates // 2))
                if ag.updates >= cfg.exploiter_updates:
                    # promote into the pool and restart from scratch so exploiters stay diverse
                    self._snapshot(ag, kind)
                    fresh = self.tr[side].init_params(self._next_key())
                    ag.params, ag.opt, ag.updates = fresh, self.tr[side].opt.init(fresh), 0
