"""Policy protocol, masked factorised-categorical heads, and the network-backed team policy."""
from __future__ import annotations

from typing import Any, Protocol

import jax
import jax.numpy as jnp

from kilithattu_core import Actions, EnvParams, Obs
from kilithattu_core.state import ActionMask

from .networks import Actor, Critic, NetConfig

HEADS = ("move", "dash", "turn", "clap", "call")
NEG = -1e9


class Policy(Protocol):
    """Run one team in ONE environment (vmap across environments). RL networks and scripted bots both implement it."""

    side: str  # "attack" | "defence"

    def init_carry(self, key: Any) -> Any: ...
    def act(self, key: Any, carry: Any, obs: Obs) -> tuple[Actions, Any, dict]: ...


def team_slice(p: EnvParams, side: str) -> slice:
    return slice(0, p.n_crossers) if side == "attack" else slice(p.n_crossers, p.n_agents)


def slice_obs(obs: Obs, sl: slice) -> Obs:
    m = obs.mask
    mask = ActionMask(m.move[sl], m.dash[sl], m.turn[sl], m.clap[sl], m.call[sl])
    return Obs(obs.players[sl], obs.lines, obs.glob[sl], obs.ego[sl], obs.role[sl], mask)


def join_actions(att: Actions, dfn: Actions) -> Actions:
    return jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b], -1), att, dfn)


def masked(logits, mask):
    return jnp.where(mask, logits, NEG)


def sample_heads(key, logits: dict, mask: ActionMask, deterministic: bool = False):
    """Sample every head. Returns (Actions, summed log-prob, summed entropy), over the leading agent axis."""
    keys = jax.random.split(key, len(HEADS))
    acts, lps, ents = {}, 0.0, 0.0
    for k, h in zip(keys, HEADS):
        lg = masked(logits[h], getattr(mask, h))
        a = jnp.argmax(lg, -1) if deterministic else jax.random.categorical(k, lg)
        logp = jax.nn.log_softmax(lg)
        acts[h] = a.astype(jnp.int32)
        lps = lps + jnp.take_along_axis(logp, a[..., None], -1)[..., 0]
        ents = ents - jnp.sum(jnp.exp(logp) * jnp.where(getattr(mask, h), logp, 0.0), -1)
    return Actions(**acts), lps, ents


def log_prob_heads(logits: dict, mask: ActionMask, a: Actions):
    lps, ents = 0.0, 0.0
    for h in HEADS:
        lg = masked(logits[h], getattr(mask, h))
        logp = jax.nn.log_softmax(lg)
        lps = lps + jnp.take_along_axis(logp, getattr(a, h)[..., None].astype(jnp.int32), -1)[..., 0]
        ents = ents - jnp.sum(jnp.exp(logp) * jnp.where(getattr(mask, h), logp, 0.0), -1)
    return lps, ents


def team_forward(actor: Actor, params, carry, obs_t: Obs, reset):
    """carry (n,H), obs for n agents, reset (n,) bool -> (carry, logits dict of (n, size))."""
    f = lambda c, pl, gl, e, r, rs: actor.apply(params, c, pl, obs_t.lines, gl, e, r, rs)
    return jax.vmap(f)(carry, obs_t.players, obs_t.glob, obs_t.ego, obs_t.role, reset)


def team_value(critic: Critic, params, cobs, team: int):
    return critic.apply(params, cobs.players[team], cobs.lines[team], cobs.glob[team])


def init_networks(key, env_params: EnvParams, cfg: NetConfig):
    """Initialise actor and critic parameters from a real observation (shapes are field-size independent)."""
    from kilithattu_core import make_env

    env = make_env(env_params)
    obs, s = env.reset(key)
    cobs = env.critic_state(s)
    actor, critic = Actor(cfg), Critic(cfg)
    k1, k2 = jax.random.split(key)
    a_params = actor.init(k1, jnp.zeros((cfg.hidden,)), obs.players[0], obs.lines, obs.glob[0], obs.ego[0], obs.role[0], jnp.array(False))
    c_params = critic.init(k2, cobs.players[0], cobs.lines[0], cobs.glob[0])
    return actor, critic, a_params, c_params


class NetworkPolicy:
    """A trained team policy for ONE environment. ``params`` are the actor parameters."""

    def __init__(self, side: str, env_params: EnvParams, cfg: NetConfig, params, deterministic: bool = False):
        self.side, self.p, self.cfg, self.params, self.det = side, env_params, cfg, params, deterministic
        self.actor = Actor(cfg)
        self.sl = team_slice(env_params, side)
        self.n = self.sl.stop - self.sl.start

    def init_carry(self, key=None):
        return jnp.zeros((self.n, self.cfg.hidden))

    def act(self, key, carry, obs: Obs, reset=None):
        ot = slice_obs(obs, self.sl)
        reset = jnp.zeros((self.n,), bool) if reset is None else reset
        carry, logits = team_forward(self.actor, self.params, carry, ot, reset)
        a, lp, ent = sample_heads(key, logits, ot.mask, self.det)
        return a, carry, {"logp": lp, "entropy": ent}
