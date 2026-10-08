"""The Kilithattu environment: a pure, jittable, vmappable function of (key, state, actions)."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import EnvParams
from .dynamics import crosser_move, guard_move, kili_move
from .geometry import box_index, in_safe_zone
from .observe import Observer
from .rules import resolve_crossings, resolve_fouls, swept_hits
from .state import Actions, ActionMask, CriticObs, EnvState, Events, Obs


class KilithattuEnv:
    """Two-team zero-sum POSG. Agent order everywhere: crossers (attack), guards, Kili (defence).

    ``step`` returns ``(obs, state, rewards, done, events)`` with ``rewards = [r_attack, r_defence]`` and
    ``r_defence = -r_attack`` (normalised so a full round trip is worth +1).
    """

    def __init__(self, params: EnvParams):
        self.params = params
        self.obs = Observer(params)
        self.n_agents = params.n_agents
        self.n_crossers = params.n_crossers
        self.n_defenders = params.n_defenders

    # ---------------------------------------------------------------------------------------------------
    def reset(self, key) -> tuple[Obs, EnvState]:
        p = self.params
        nA, nG = p.n_crossers, p.n_guards
        k1, k2, k3, k4 = jax.random.split(key, 4)
        base = (jnp.arange(nA) + 0.5) / nA * p.field_w
        jitter = jax.random.uniform(k1, (nA,), minval=-0.3, maxval=0.3) * (p.field_w / nA)
        cx = jnp.clip(base + jitter, 0.2, p.field_w - 0.2)
        cy = -jax.random.uniform(k2, (nA,), minval=0.3, maxval=p.zone - 0.2)
        gx = jax.random.uniform(k3, (nG,), minval=0.5, maxval=p.field_w - 0.5)
        kx = jnp.clip(jnp.round(p.field_w / 2 / p.lane_w) * p.lane_w, 0.0, p.field_w)
        s = EnvState(
            c_pos=jnp.stack([cx, cy], -1), c_vel=jnp.zeros((nA, 2)), c_status=jnp.zeros((nA,), jnp.int32),
            c_kappa=jnp.zeros((nA,), jnp.int32), c_trips=jnp.zeros((nA,), jnp.int32), c_stam=jnp.ones((nA,)),
            c_box=-jnp.ones((nA,), jnp.int32), c_wait=jnp.zeros((nA,), jnp.int32),
            g_x=gx, g_vx=jnp.zeros((nG,)), g_face=-jnp.ones((nG,)), g_turn=jnp.zeros((nG,)), g_stam=jnp.ones((nG,)),
            k_pos=jnp.array([kx, 0.0]), k_face=-jnp.ones(()), k_turn=jnp.zeros(()), k_stam=jnp.ones(()),
            tick=jnp.zeros((), jnp.int32), t=jnp.zeros((), jnp.int32), live=jnp.zeros((), bool),
            done=jnp.zeros((), bool), points=jnp.zeros(()), deliveries=jnp.zeros((), jnp.int32),
            last_call=jnp.zeros((2,), jnp.int32),
        )
        return self.obs.observe(s), s

    def observe(self, s: EnvState) -> Obs:
        return self.obs.observe(s)

    def critic_state(self, s: EnvState) -> CriticObs:
        return self.obs.critic_obs(s)

    def action_mask(self, s: EnvState) -> ActionMask:
        return self.obs.masks(s)

    def potential(self, s: EnvState):
        """Shaping potential Phi(s) in [0,1]; exactly 0 in terminal states (policy-invariant shaping)."""
        p = self.params
        y = jnp.clip(s.c_pos[:, 1], 0.0, p.field_l)
        prog = jnp.where(s.c_kappa < p.K, y, p.field_l + (p.field_l - y)) / (2 * p.field_l)
        phi = jnp.where(s.c_status == 2, 0.0, prog).mean()
        return jnp.where(s.done, 0.0, phi)

    # ---------------------------------------------------------------------------------------------------
    def _mask_actions(self, s: EnvState, a: Actions) -> Actions:
        m = self.obs.masks(s)
        ok = lambda mk, x: jnp.take_along_axis(mk, x[:, None], 1)[:, 0]
        z = lambda mk, x: jnp.where(ok(mk, x), x, 0).astype(jnp.int32)
        return Actions(z(m.move, a.move), z(m.dash, a.dash), z(m.turn, a.turn), z(m.clap, a.clap), z(m.call, a.call))

    def step(self, key, s: EnvState, a: Actions):
        p = self.params
        nA, nG = p.n_crossers, p.n_guards
        a = self._mask_actions(s, a)
        c_act = jax.tree_util.tree_map(lambda x: x[:nA], a)
        g_act = jax.tree_util.tree_map(lambda x: x[nA : nA + nG], a)
        k_act = jax.tree_util.tree_map(lambda x: x[-1], a)
        alive0 = s.c_status != 2

        # 1. clap (Classic: automatic; P4: the Kili chooses inside a window, forced at its end)
        if p.p4_clap:
            clap_req = ((k_act.clap == 1) & (s.tick >= p.clap_min_ticks)) | (s.tick >= p.clap_max_ticks)
        else:
            clap_req = s.tick >= p.clap_auto_ticks
        clap_now = (~s.live) & clap_req
        live = s.live | clap_now

        # 2. team calls (P6)
        if p.p6_calls:
            def team_call(calls):
                return jnp.max(calls)
            do_call = (s.tick % p.call_period) == 0
            last_call = jnp.where(do_call, jnp.stack([team_call(a.call[:nA]), team_call(a.call[nA:])]), s.last_call)
        else:
            last_call = s.last_call

        # 3. movement
        c_pos, c_vel, c_stam = crosser_move(p, s.c_pos, s.c_vel, s.c_stam, c_act.move, c_act.dash, alive0, live)
        g_x, g_vx, g_stam, g_face, g_turn = guard_move(p, s.g_x, s.g_vx, s.g_stam, g_act.move, g_act.dash, s.g_face, s.g_turn, g_act.turn)
        k_pos, k_stam, k_face, k_turn = kili_move(p, s.k_pos, s.k_stam, k_act.move, k_act.dash, s.k_face, s.k_turn, k_act.turn)

        # 4. constraints: side exit, false start (P4)
        in_field = (c_pos[:, 1] >= 0) & (c_pos[:, 1] <= p.field_l)
        side = alive0 & in_field & ((c_pos[:, 0] < -p.side_tol) | (c_pos[:, 0] > p.field_w + p.side_tol))
        false_start = alive0 & (c_pos[:, 1] >= 0) & (~live) if p.p4_clap else jnp.zeros((nA,), bool)

        # 5. swept tags (resolved before crossings)
        gy = self.obs.guard_y
        d0 = jnp.concatenate([jnp.stack([s.g_x, gy], -1), s.k_pos[None]], 0)
        d1 = jnp.concatenate([jnp.stack([g_x, gy], -1), k_pos[None]], 0)
        hit = swept_hits(s.c_pos, c_pos, d0, d1, p.r_tag)
        if p2 := p.p2_facing:
            face = jnp.concatenate([g_face, k_face[None]])
            turning = jnp.concatenate([g_turn > 0, (k_turn > 0)[None]])
            cone = (c_pos[:, None, 1] - d1[None, :, 1]) * face[None, :] >= 0
            hit = hit & cone & (~turning)[None, :]
        tagged = alive0 & ~in_safe_zone(p, c_pos) & jnp.any(hit, axis=1)
        out1 = tagged | side | false_start
        alive1 = alive0 & ~out1

        # 6. line crossings for survivors
        kappa, status, adv, delivered = resolve_crossings(p, s.c_kappa, s.c_pos[:, 1], c_pos[:, 1], alive1)
        status = jnp.where(out1 | ~alive0, 2, status)

        # 7. fouls
        box = box_index(p, c_pos)
        fouled = resolve_fouls(status, box, s.c_box, status != 2)
        status = jnp.where(fouled, 2, status)
        alive2 = status != 2

        # 8. bookkeeping, reward, termination
        line_pts = adv.sum()
        deliv = delivered.sum()
        delta = (line_pts + 6 * deliv).astype(jnp.float32)
        r_att = delta / p.trip_points
        t = s.t + s.live.astype(jnp.int32)
        wait = jnp.where(adv | ~alive2, 0, s.c_wait + s.live.astype(jnp.int32))
        done = (~jnp.any(alive2)) | (t >= p.max_ticks)
        ns = EnvState(
            c_pos=c_pos, c_vel=c_vel, c_status=status, c_kappa=kappa, c_trips=s.c_trips + delivered.astype(jnp.int32),
            c_stam=c_stam, c_box=jnp.where(alive2, box, -1), c_wait=wait,
            g_x=g_x, g_vx=g_vx, g_face=g_face, g_turn=g_turn, g_stam=g_stam,
            k_pos=k_pos, k_face=k_face, k_turn=k_turn, k_stam=k_stam,
            tick=s.tick + 1, t=t, live=live, done=done, points=s.points + delta, deliveries=s.deliveries + deliv,
            last_call=last_call,
        )
        ev = Events(tagged=tagged, fouled=fouled, side_exit=side, false_start=false_start, crossed=adv, delivered=delivered, clapped=clap_now)
        rewards = jnp.stack([r_att, -r_att])
        return self.obs.observe(ns), ns, rewards, done, ev


def make_env(params: EnvParams | None = None) -> KilithattuEnv:
    return KilithattuEnv(params or EnvParams())
