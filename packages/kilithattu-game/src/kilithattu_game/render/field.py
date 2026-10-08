"""Pygame rendering of the field, players, effects and HUD. Colour-blind safe by shape and ring, not hue alone."""
from __future__ import annotations

import math

import numpy as np
import pygame

PAL = {
    "bg": (14, 29, 21), "ground_a": (47, 107, 63), "ground_b": (53, 120, 69), "zone": (86, 100, 52), "chalk": (241, 228, 189),
    "pacha": (134, 211, 111), "pacha_edge": (23, 61, 31), "uppu": (246, 242, 228), "uppu_edge": (122, 122, 106), "salt": (154, 180, 196),
    "guard": (162, 58, 47), "kili": (125, 31, 26), "gold": (224, 173, 74), "foul": (226, 87, 76), "text": (241, 228, 189),
}
PAL_CB = {**PAL, "pacha": (86, 180, 233), "guard": (213, 94, 0), "kili": (153, 62, 0), "foul": (204, 121, 167)}  # Okabe-Ito based


class View:
    def __init__(self, surface: pygame.Surface, p, colorblind: bool = False, reduce_motion: bool = False):
        self.p, self.surf = p, surface
        self.pal = PAL_CB if colorblind else PAL
        self.reduce = reduce_motion
        self.font = pygame.font.Font(None, 28)
        self.big = pygame.font.Font(None, 56)
        self.small = pygame.font.Font(None, 20)
        self.floats: list[dict] = []
        self.layout()

    def layout(self) -> None:
        w, h = self.surf.get_size()
        p = self.p
        self.margin_y = 56
        total_h = p.field_l + 2 * p.zone + 1.0
        self.s = min((h - self.margin_y - 10) / total_h, (w * 0.62) / (p.field_w + 1.6))
        self.ox = (w - p.field_w * self.s) / 2 - 80
        self.oy = self.margin_y + (p.zone + 0.5) * self.s + p.field_l * self.s

    def P(self, x: float, y: float) -> tuple[int, int]:
        return int(self.ox + x * self.s), int(self.oy - y * self.s)

    def add_float(self, text: str, x: float, y: float, color) -> None:
        self.floats.append({"t": 0.0, "text": text, "x": x, "y": y, "c": color})

    # -----------------------------------------------------------------------------------------------
    def draw(self, sess, alpha: float, dt: float, seat_agent: int | None) -> None:
        p, s, pal, surf = self.p, self.s, self.pal, self.surf
        st, prev = sess.state, sess.prev
        surf.fill(pal["bg"])
        # zones and boxes
        for y0, y1 in ((-p.zone, 0), (p.field_l, p.field_l + p.zone)):
            pygame.draw.rect(surf, pal["zone"], pygame.Rect(*self.P(0, y1), p.field_w * s, (y1 - y0) * s))
        for j in range(p.n_lanes):
            for k in range(p.n_rows):
                col = pal["ground_a"] if (j + k) % 2 == 0 else pal["ground_b"]
                pygame.draw.rect(surf, col, pygame.Rect(*self.P(j * p.lane_w, (k + 1) * p.row_d), p.lane_w * s + 1, p.row_d * s + 1))
        # foul tint: boxes holding both statuses, plus warning boxes for the human crosser
        status, kappa, pos = np.asarray(st.c_status), np.asarray(st.c_kappa), np.asarray(st.c_pos)
        boxes: dict[tuple[int, int], set[int]] = {}
        for i in range(p.n_crossers):
            if status[i] != 2 and 0 <= pos[i, 1] <= p.field_l:
                b = (min(int(pos[i, 0] // p.lane_w), p.n_lanes - 1), min(int(pos[i, 1] // p.row_d), p.n_rows - 1))
                boxes.setdefault(b, set()).add(int(status[i]))
        opp = None
        if seat_agent is not None and seat_agent < p.n_crossers and status[seat_agent] != 2:
            opp = 1 - int(status[seat_agent])
        for (j, k), sts in boxes.items():
            if len(sts) > 1 or (opp is not None and opp in sts):
                ov = pygame.Surface((int(p.lane_w * s), int(p.row_d * s)), pygame.SRCALPHA)
                ov.fill((*pal["foul"], 130 if len(sts) > 1 else 60))
                surf.blit(ov, self.P(j * p.lane_w, (k + 1) * p.row_d))
        # chalk lines
        w = max(2, int(s * 0.07))
        for j in range(p.n_lanes + 1):
            pygame.draw.line(surf, pal["chalk"], self.P(j * p.lane_w, 0), self.P(j * p.lane_w, p.field_l), w)
        for k in range(p.n_rows + 1):
            pygame.draw.line(surf, pal["chalk"], self.P(0, k * p.row_d), self.P(p.field_w, k * p.row_d), w)
        t = self.small.render("home", True, pal["text"]); surf.blit(t, (self.P(p.field_w / 2, -1.0)[0] - t.get_width() // 2, self.P(0, -1.0)[1]))
        t = self.small.render("far side", True, pal["text"]); surf.blit(t, (self.P(p.field_w / 2, p.field_l + 1.0)[0] - t.get_width() // 2, self.P(0, p.field_l + 1.0)[1]))
        lerp = lambda a, b: a + (b - a) * alpha
        # defenders
        gy = [(j + 1) * p.row_d for j in range(p.n_guards)]
        for j in range(p.n_guards):
            x = lerp(prev["g"][j], float(st.g_x[j])); c = self.P(x, gy[j])
            pygame.draw.circle(surf, pal["chalk"], c, int(p.r_tag * s), 1)
            pygame.draw.rect(surf, pal["guard"], pygame.Rect(c[0] - int(.36 * s), c[1] - int(.36 * s), int(.72 * s), int(.72 * s)))
            if seat_agent == p.n_crossers + j:
                self._you(c, 0.7)
        kx, ky = lerp(prev["k"][0], float(st.k_pos[0])), lerp(prev["k"][1], float(st.k_pos[1])); kc = self.P(kx, ky)
        pygame.draw.circle(surf, pal["chalk"], kc, int(p.r_tag * s), 1)
        pygame.draw.circle(surf, pal["kili"], kc, int(.5 * s)); pygame.draw.circle(surf, pal["gold"], kc, int(.5 * s), 3)
        k_t = self.small.render("K", True, pal["text"]); surf.blit(k_t, (kc[0] - k_t.get_width() // 2, kc[1] - k_t.get_height() // 2))
        if seat_agent == p.n_agents - 1:
            self._you(kc, 0.75)
        # crossers
        for i in range(p.n_crossers):
            c = self.P(lerp(prev["c"][i][0], pos[i, 0]), lerp(prev["c"][i][1], pos[i, 1]))
            r = int(.4 * s)
            if status[i] == 2:
                pygame.draw.circle(surf, (110, 120, 110), c, r, 2)
                pygame.draw.line(surf, (200, 200, 200), (c[0] - 5, c[1] - 5), (c[0] + 5, c[1] + 5), 2)
                pygame.draw.line(surf, (200, 200, 200), (c[0] + 5, c[1] - 5), (c[0] - 5, c[1] + 5), 2)
                continue
            if status[i] == 1:
                pygame.draw.circle(surf, pal["uppu"], c, r); pygame.draw.circle(surf, pal["uppu_edge"], c, r, 2)
                d = int(.18 * s); pygame.draw.polygon(surf, pal["salt"], [(c[0], c[1] - d), (c[0] + d // 1.5, c[1]), (c[0], c[1] + d), (c[0] - d // 1.5, c[1])])
            else:
                pygame.draw.circle(surf, pal["pacha"], c, r); pygame.draw.circle(surf, pal["pacha_edge"], c, r, 2)
            if seat_agent == i:
                self._you(c, 0.65)
        # floating texts
        for f in self.floats:
            f["t"] += dt
        self.floats = [f for f in self.floats if f["t"] < 1.2]
        for f in self.floats:
            x, y = self.P(f["x"], f["y"] + 0.8 + (0 if self.reduce else f["t"] * 1.4))
            t = self.font.render(f["text"], True, f["c"]); t.set_alpha(int(255 * (1 - f["t"] / 1.2))); surf.blit(t, (x - t.get_width() // 2, y))
        self._hud(sess)

    def _you(self, c, r) -> None:
        pulse = 0 if self.reduce else math.sin(pygame.time.get_ticks() / 160) * 0.04
        pygame.draw.circle(self.surf, self.pal["gold"], c, int((r + pulse) * self.s), 3)

    def _hud(self, sess) -> None:
        p, st, surf = self.p, sess.state, self.surf
        W = surf.get_width()
        pygame.draw.rect(surf, (8, 20, 13), pygame.Rect(0, 0, W, 44))
        live = bool(st.live)
        left = max(0.0, (p.max_ticks - int(st.t)) * p.dt)
        surf.blit(self.big.render(f"{int(st.points)} pts", True, self.pal["text"]), (14, 6))
        txt = f"{int(left)} s" if live else "get ready"
        t = self.big.render(txt, True, self.pal["text"]); surf.blit(t, (W - t.get_width() - 14, 6))
        alive = int((np.asarray(st.c_status) != 2).sum())
        t = self.font.render(f"crossers in play: {alive}   salt delivered: {int(st.deliveries)}", True, self.pal["text"]); surf.blit(t, (W // 2 - t.get_width() // 2, 12))
