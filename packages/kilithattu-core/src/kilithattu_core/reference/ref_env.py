"""Loop-based re-implementation of the Classic rules used for differential testing of the JAX engine.

It is intentionally written differently (explicit loops, no vectorisation) so that a bug has to be made
twice to go unnoticed. Float32 arithmetic mirrors the JAX engine closely.
"""
from __future__ import annotations

import numpy as np

from ..config import EnvParams

f32 = np.float32
_DIRS = [(0, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]


class ReferenceEnv:
    def __init__(self, p: EnvParams):
        assert not (p.p1_momentum or p.p2_facing or p.p4_clap or p.p6_calls), "reference covers Classic rules"
        self.p = p

    def load(self, s) -> None:
        g = lambda x: np.array(x, dtype=f32 if np.asarray(x).dtype.kind == "f" else np.asarray(x).dtype)
        self.c_pos, self.c_status, self.c_kappa = g(s.c_pos), g(s.c_status), g(s.c_kappa)
        self.c_box, self.g_x, self.k_pos = g(s.c_box), g(s.g_x), g(s.k_pos)
        self.tick, self.t, self.live, self.points = int(s.tick), int(s.t), bool(s.live), float(s.points)
        self.deliveries = int(s.deliveries)

    # --- helpers ------------------------------------------------------------------------------------
    def _box(self, pos):
        p = self.p
        if not (0 <= pos[1] <= p.field_l):
            return -1
        j = int(np.clip(np.floor(pos[0] / p.lane_w), 0, p.n_lanes - 1))
        k = int(np.clip(np.floor(pos[1] / p.row_d), 0, p.n_rows - 1))
        return j * p.n_rows + k

    def _kili_mask(self, pos):
        p = self.p
        snap = f32(p.v_kili) * f32(p.dt) + 1e-6
        nl = np.clip(np.round(pos[0] / f32(p.lane_w)) * f32(p.lane_w), 0, p.field_w)
        nt = np.clip(np.round(pos[1] / f32(p.row_d)) * f32(p.row_d), 0, p.field_l)
        near_l, near_t = abs(pos[0] - nl) <= snap, abs(pos[1] - nt) <= snap
        return [True, near_l and pos[1] < p.field_l - 1e-4, near_t and pos[0] < p.field_w - 1e-4,
                near_l and pos[1] > 1e-4, near_t and pos[0] > 1e-4], nl, nt

    @staticmethod
    def _hit(a0, a1, b0, b1, r):
        d0 = a0 - b0
        e = (a1 - a0) - (b1 - b0)
        ee = float(e @ e)
        tau = float(np.clip(-(d0 @ e) / max(ee, 1e-12), 0, 1)) if ee > 1e-12 else 0.0
        return float(np.linalg.norm(d0 + f32(tau) * e)) <= r

    # --- one tick ------------------------------------------------------------------------------------
    def step(self, move) -> None:
        p = self.p
        nA, nG = p.n_crossers, p.n_guards
        dt = f32(p.dt)
        alive0 = self.c_status != 2
        self._was_live = self.live
        if not self.live and self.tick >= p.clap_auto_ticks:
            self.live = True
        old_c, old_g, old_k = self.c_pos.copy(), self.g_x.copy(), self.k_pos.copy()
        # crossers
        for i in range(nA):
            if not alive0[i]:
                continue
            m = move[i]
            d = np.array(_DIRS[m], dtype=f32)
            n = np.linalg.norm(d)
            d = d / n if n > 0 else d
            new = self.c_pos[i] + d * f32(p.v_cross) * dt
            if not self.live:
                new[1] = min(new[1], f32(-0.01))
            self.c_pos[i] = new
        # guards
        for j in range(nG):
            m = move[nA + j]
            m = m if m in (0, 1, 2) else 0
            d = (0.0, -1.0, 1.0)[m]
            self.g_x[j] = np.clip(self.g_x[j] + f32(d) * f32(p.v_guard) * dt, 0, p.field_w)
        # kili
        mask, nl, nt = self._kili_mask(self.k_pos)
        m = move[-1]
        m = m if m < 5 and mask[m] else 0
        dx, dy = (0, 0, 1, 0, -1)[m], (0, 1, 0, -1, 0)[m]
        v = f32(p.v_kili) * dt
        kx, ky = self.k_pos
        nx = nl if dy != 0 else (np.clip(kx + dx * v, 0, p.field_w) if dx != 0 else kx)
        ny = nt if dx != 0 else (np.clip(ky + dy * v, 0, p.field_l) if dy != 0 else ky)
        self.k_pos = np.array([nx, ny], dtype=f32)
        # constraints and tags
        out1 = np.zeros(nA, bool)
        gy = [(j + 1) * p.row_d for j in range(nG)]
        for i in range(nA):
            if not alive0[i]:
                continue
            x, y = self.c_pos[i]
            if 0 <= y <= p.field_l and (x < -p.side_tol or x > p.field_w + p.side_tol):
                out1[i] = True
            if y < 0 or y > p.field_l:
                continue
            for j in range(nG):
                if self._hit(old_c[i], self.c_pos[i], np.array([old_g[j], gy[j]], f32), np.array([self.g_x[j], gy[j]], f32), p.r_tag):
                    out1[i] = True
            if self._hit(old_c[i], self.c_pos[i], old_k, self.k_pos, p.r_tag):
                out1[i] = True
        alive1 = alive0 & ~out1
        # crossings
        K = p.K
        status = np.where(alive0 & ~out1, self.c_status, 2).astype(self.c_status.dtype)
        line_pts, deliv = 0, 0
        for i in range(nA):
            if not alive1[i]:
                status[i] = 2
                continue
            k = int(self.c_kappa[i])
            up = k < K
            li = k if up else 2 * K - 1 - k
            ly = f32(li * p.row_d)
            py, ny_ = old_c[i][1], self.c_pos[i][1]
            crossed = (py < ly and ny_ >= ly) if up else (py > ly and ny_ <= ly)
            if crossed and k < 2 * K:
                k += 1
                line_pts += 1
                if k == 2 * K:
                    deliv += 1
                    k = 0
            self.c_kappa[i] = k
            status[i] = 1 if k >= K else 0
        # fouls
        boxes = [self._box(self.c_pos[i]) for i in range(nA)]
        entered = [boxes[i] != int(self.c_box[i]) for i in range(nA)]
        foul_out = set()
        for i in range(nA):
            for j in range(nA):
                if status[i] == 1 and status[j] == 0 and boxes[i] == boxes[j] and boxes[i] >= 0:
                    foul_out.add(j if (entered[j] and not entered[i]) else i)
        for i in foul_out:
            status[i] = 2
        self.c_status = status
        self.c_box = np.array([boxes[i] if status[i] != 2 else -1 for i in range(nA)], dtype=self.c_box.dtype)
        self.points += line_pts + 6 * deliv
        self.deliveries += deliv
        self.t += int(self._was_live)
        self.tick += 1
