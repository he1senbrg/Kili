"""Headless game session: owns the environment, the AI policies, human seat overrides and the action log."""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from kilithattu_agents import join_actions
from kilithattu_core import Actions, EnvParams, make_env
from kilithattu_core.replay import _to_actions, stack_actions, write_replay


@dataclass
class Seat:
    side: str  # "attack" | "defence"
    index: int  # index within the team (defence: last index is the Kili)


class Session:
    """Fixed-timestep simulation. ``tick(inputs)`` advances 1/20 s. ``inputs`` maps agent id -> dict of head values."""

    def __init__(self, p: EnvParams, att, dfn, human: Seat | None = None, seed: int = 0, replay_log: np.ndarray | None = None):
        self.p, self.att, self.dfn, self.human, self.seed = p, att, dfn, human, seed
        self.env = make_env(p)
        self._step = jax.jit(self.env.step)
        self._aa, self._da = jax.jit(att.act), jax.jit(dfn.act)
        self.replay_log = replay_log
        self.reset()

    # -------------------------------------------------------------------------------------------------
    def reset(self) -> None:
        key = jax.random.PRNGKey(self.seed)
        self.key, k0, k1, k2 = jax.random.split(key, 4)
        self.obs, self.state = self.env.reset(k0)
        self.ca, self.cd = self.att.init_carry(k1), self.dfn.init_carry(k2)
        self.prev = self.snapshot()
        self.log: list[Actions] = []
        self.events: list[dict] = []
        self.t = 0

    def snapshot(self) -> dict:
        s = self.state
        return {"c": np.asarray(s.c_pos), "g": np.asarray(s.g_x), "k": np.asarray(s.k_pos)}

    @property
    def human_agent(self) -> int | None:
        if self.human is None:
            return None
        return self.human.index if self.human.side == "attack" else self.p.n_crossers + self.human.index

    def human_mask(self):
        """Action mask row of the human's agent (for input buffering such as the Kili's turn queue)."""
        a = self.human_agent
        m = self.obs.mask
        return None if a is None else {h: np.asarray(getattr(m, h)[a]) for h in ("move", "dash", "turn", "clap", "call")}

    def tick(self, human_action: dict | None = None) -> list[dict]:
        if self.replay_log is not None:
            act = _to_actions(self.replay_log[min(self.t, len(self.replay_log) - 1)])
        else:
            self.key, ka, kd, ke = jax.random.split(self.key, 4)
            aa, self.ca, _ = self._aa(ka, self.ca, self.obs)
            ad, self.cd, _ = self._da(kd, self.cd, self.obs)
            act = join_actions(aa, ad)
            if human_action is not None and self.human_agent is not None:
                a = self.human_agent
                act = act.replace(**{h: getattr(act, h).at[a].set(int(human_action.get(h, 0))) for h in ("move", "dash", "turn", "clap", "call")})
        self.prev = self.snapshot()
        self.key, ke = jax.random.split(self.key)
        self.obs, self.state, _, done, ev = self._step(ke, self.state, act)
        self.log.append(act)
        self.t += 1
        out = []
        for name in ("tagged", "fouled", "side_exit", "false_start", "crossed", "delivered"):
            for i in np.flatnonzero(np.asarray(getattr(ev, name))):
                out.append({"type": name, "crosser": int(i), "pos": np.asarray(self.state.c_pos[i]).copy()})
        if bool(ev.clapped):
            out.append({"type": "clap"})
        self.events.extend(out)
        return out

    @property
    def done(self) -> bool:
        return bool(self.state.done)

    def save_replay(self, path, controllers: dict | None = None) -> dict:
        return write_replay(path, self.p, self.seed, stack_actions(self.log), controllers)
