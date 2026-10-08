"""Pytree dataclasses: environment state, actions, observations, events."""
from __future__ import annotations

import chex
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class EnvState:
    # crossers
    c_pos: chex.Array  # (nA, 2)
    c_vel: chex.Array  # (nA, 2)
    c_status: chex.Array  # (nA,) 0 pacha, 1 uppu, 2 out
    c_kappa: chex.Array  # (nA,) trip progress 0..2K
    c_trips: chex.Array  # (nA,)
    c_stam: chex.Array  # (nA,)
    c_box: chex.Array  # (nA,) box id at the end of the previous tick, -1 outside
    c_wait: chex.Array  # (nA,) live ticks since the crosser last made progress (observed as a patience cue)
    # guards
    g_x: chex.Array  # (nG,)
    g_vx: chex.Array  # (nG,)
    g_face: chex.Array  # (nG,) +1 far, -1 home
    g_turn: chex.Array  # (nG,) seconds of turn left
    g_stam: chex.Array  # (nG,)
    # kili
    k_pos: chex.Array  # (2,)
    k_face: chex.Array  # ()
    k_turn: chex.Array  # ()
    k_stam: chex.Array  # ()
    # global
    tick: chex.Array  # () ticks since reset (pre-clap included)
    t: chex.Array  # () live ticks since the clap
    live: chex.Array  # () bool, clap happened
    done: chex.Array  # () bool
    points: chex.Array  # () attack points this innings
    deliveries: chex.Array  # () salt deliveries
    last_call: chex.Array  # (2,) last call per team


@struct.dataclass
class Actions:
    """One entry per agent. Agent order: crossers, guards, Kili."""

    move: chex.Array  # (A,) int
    dash: chex.Array  # (A,) sprint / dash
    turn: chex.Array  # (A,) turn (defenders, P2)
    clap: chex.Array  # (A,) clap (Kili, P4)
    call: chex.Array  # (A,) team call (P6)


@struct.dataclass
class ActionMask:
    move: chex.Array  # (A, 9)
    dash: chex.Array  # (A, 2)
    turn: chex.Array  # (A, 2)
    clap: chex.Array  # (A, 2)
    call: chex.Array  # (A, K+1)


@struct.dataclass
class Obs:
    players: chex.Array  # (A, P, PLAYER_F)
    lines: chex.Array  # (NL, LINE_F)
    glob: chex.Array  # (A, GLOBAL_F)
    ego: chex.Array  # (A,) index of the agent's own token
    role: chex.Array  # (A,)
    mask: ActionMask


@struct.dataclass
class CriticObs:
    players: chex.Array  # (2, P, PLAYER_F) per team perspective, no fog
    lines: chex.Array  # (2, NL, LINE_F)
    glob: chex.Array  # (2, GLOBAL_F)


@struct.dataclass
class Events:
    tagged: chex.Array  # (nA,) bool
    fouled: chex.Array
    side_exit: chex.Array
    false_start: chex.Array
    crossed: chex.Array  # line points scored this tick per crosser
    delivered: chex.Array  # salt delivered this tick per crosser
    clapped: chex.Array  # () bool


def zeros_actions(n_agents: int) -> Actions:
    z = jnp.zeros((n_agents,), jnp.int32)
    return Actions(z, z, z, z, z)
