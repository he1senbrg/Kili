"""Observation tokens, fog of war (P3), critic observations and action masks."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .config import EnvParams
from .geometry import kili_move_mask, line_tokens
from .spaces import LINE_F, N_MOVE, PLAYER_F, ROLE_CROSSER, ROLE_GUARD, ROLE_KILI, global_f
from .state import ActionMask, CriticObs, EnvState, Obs


class Observer:
    def __init__(self, p: EnvParams):
        self.p = p
        nA, nG = p.n_crossers, p.n_guards
        self.nA, self.nG = nA, nG
        self.P = nA + nG + 1
        self.A = self.P
        role = [ROLE_CROSSER] * nA + [ROLE_GUARD] * nG + [ROLE_KILI]
        self.role = jnp.array(role, jnp.int32)
        self.team = jnp.array([0] * nA + [1] * (nG + 1), jnp.int32)
        self.guard_y = jnp.arange(1, nG + 1, dtype=jnp.float32) * p.row_d
        self.lines = line_tokens(p)
        self.scale = jnp.array([p.field_w, p.field_l], jnp.float32)
        self.radius = jnp.where(self.role == ROLE_CROSSER, p.vis_cross, jnp.where(self.role == ROLE_GUARD, p.vis_guard, p.vis_kili))
        assert self.lines.shape[1] == LINE_F

    # ---- helpers -------------------------------------------------------------------------------------
    def positions(self, s: EnvState):
        g = jnp.stack([s.g_x, self.guard_y], -1)
        return jnp.concatenate([s.c_pos, g, s.k_pos[None]], 0)

    def velocities(self, s: EnvState):
        g = jnp.stack([s.g_vx, jnp.zeros_like(s.g_vx)], -1)
        return jnp.concatenate([s.c_vel, g, jnp.zeros((1, 2))], 0)

    def tokens(self, s: EnvState, ego, my_team, fog: bool, reveal_all: bool):
        p, nA, nG = self.p, self.nA, self.nG
        pos, vel = self.positions(s), self.velocities(s)
        idx = jnp.arange(self.P)
        is_self = idx == ego
        same = self.team == my_team
        is_team = same & ~is_self
        is_opp = ~same
        role1h = jax.nn.one_hot(self.role, 3)
        status = jnp.concatenate([s.c_status, -jnp.ones((nG + 1,), jnp.int32)])
        st1h = jnp.where((status >= 0)[:, None], jax.nn.one_hot(jnp.maximum(status, 0), 3), 0.0)
        has_ego = (ego >= 0).astype(jnp.float32)
        ego_pos = pos[jnp.maximum(ego, 0)] * has_ego
        rel = (pos - ego_pos) / self.scale * has_ego
        ab = pos / self.scale
        vn = vel / p.v_max
        z = jnp.zeros
        face = jnp.concatenate([z(nA), s.g_face, s.k_face[None]]) * float(p.p2_facing)
        stam = jnp.concatenate([s.c_stam, s.g_stam, s.k_stam[None]]) * float(p.p1_momentum)
        stam = stam * jnp.where(same | reveal_all, 1.0, 0.0)
        kap = jnp.concatenate([s.c_kappa / (2.0 * p.K), z(nG + 1)])
        # slot 18: defenders' turn timer (P2); for crossers the same slot carries a patience cue (20 s saturates)
        turn = jnp.concatenate([jnp.minimum(s.c_wait / 400.0, 1.0), s.g_turn / p.t_turn * float(p.p2_facing), s.k_turn[None] / p.t_turn * float(p.p2_facing)])
        if p.p3_vision and fog:
            dist = jnp.linalg.norm(pos - ego_pos, axis=-1)
            vis = same | (dist <= self.radius[jnp.maximum(ego, 0)])
        else:
            vis = jnp.ones((self.P,), bool)
        vf = vis.astype(jnp.float32)[:, None]
        num = jnp.concatenate([st1h, rel, ab, vn, face[:, None], stam[:, None], kap[:, None], turn[:, None]], -1) * vf
        flags = jnp.stack([is_self, is_team, is_opp], -1).astype(jnp.float32)
        tok = jnp.concatenate([flags, role1h, num, vf], -1)
        assert tok.shape[-1] == PLAYER_F, tok.shape
        return tok

    def glob(self, s: EnvState, team):
        p = self.p
        flags = jnp.array([p.p1_momentum, p.p2_facing, p.p3_vision, p.p4_clap, p.p6_calls], jnp.float32)
        head = jnp.stack([1.0 - s.t / p.max_ticks, (~s.live).astype(jnp.float32), s.points / (p.n_crossers * p.trip_points)])
        call = jax.nn.one_hot(s.last_call[team], p.n_calls + 1)
        g = jnp.concatenate([head, flags, call])
        assert g.shape[-1] == global_f(p.n_calls)
        return g

    # ---- public --------------------------------------------------------------------------------------
    def observe(self, s: EnvState) -> Obs:
        egos = jnp.arange(self.A)
        players = jax.vmap(lambda e: self.tokens(s, e, self.team[e], True, False))(egos)
        glob = jax.vmap(lambda e: self.glob(s, self.team[e]))(egos)
        return Obs(players=players, lines=self.lines, glob=glob, ego=egos, role=self.role, mask=self.masks(s))

    def critic_obs(self, s: EnvState) -> CriticObs:
        pl = jnp.stack([self.tokens(s, jnp.array(-1), t, False, True) for t in (0, 1)])
        gl = jnp.stack([self.glob(s, t) for t in (0, 1)])
        return CriticObs(players=pl, lines=jnp.stack([self.lines, self.lines]), glob=gl)

    def masks(self, s: EnvState) -> ActionMask:
        p, nA, nG, A = self.p, self.nA, self.nG, self.A
        out = jnp.concatenate([s.c_status == 2, jnp.zeros((nG + 1,), bool)])
        stay_only = jnp.zeros((N_MOVE,), bool).at[0].set(True)
        cross_m = jnp.where((s.c_status == 2)[:, None], stay_only[None], jnp.ones((nA, N_MOVE), bool))
        guard_m = jnp.tile(jnp.zeros((N_MOVE,), bool).at[:3].set(True)[None], (nG, 1))
        kili_m = jnp.zeros((N_MOVE,), bool).at[:5].set(kili_move_mask(p, s.k_pos))[None]
        move = jnp.concatenate([cross_m, guard_m, kili_m], 0)
        stam = jnp.concatenate([s.c_stam, s.g_stam, s.k_stam[None]])
        fast = jnp.logical_and(p.p1_momentum, (stam >= p.stam_min) & ~out)
        dash = jnp.stack([jnp.ones((A,), bool), fast], -1)
        is_def = self.team == 1
        turning = jnp.concatenate([jnp.zeros((nA,), bool), s.g_turn > 0, (s.k_turn > 0)[None]])
        turn = jnp.stack([jnp.ones((A,), bool), jnp.logical_and(p.p2_facing, is_def & ~turning)], -1)
        can_clap = p.p4_clap & (~s.live) & (s.tick >= p.clap_min_ticks)
        clap = jnp.stack([jnp.ones((A,), bool), (self.role == ROLE_KILI) & can_clap], -1)
        can_call = p.p6_calls & (s.tick % p.call_period == 0)
        call = jnp.zeros((A, p.n_calls + 1), bool).at[:, 0].set(True)
        call = call | (can_call & ~out)[:, None]
        return ActionMask(move=move, dash=dash, turn=turn, clap=clap, call=call)
