"""Kilithattu Kali: the pygame client. ``kilithattu --side attack --difficulty medium``."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from kilithattu_core import load_params
from kilithattu_core.replay import read_replay

from .controllers import team_policies
from .input.keyboard import HumanInput
from .session import Seat, Session

SIDES = {"attack": "attack", "guard": "defence", "kili": "defence", "watch": None}


def build_session(args) -> tuple[Session, HumanInput | None]:
    p = load_params(args.rules)
    att, dfn = team_policies(p, args.difficulty)
    if args.side == "watch":
        return Session(p, att, dfn, None, args.seed), None
    if args.side == "attack":
        seat, hi = Seat("attack", args.index), HumanInput("attack", False)
    elif args.side == "guard":
        seat, hi = Seat("defence", args.index), HumanInput("defence", False)
    else:
        seat, hi = Seat("defence", p.n_defenders - 1), HumanInput("defence", True)
    return Session(p, att, dfn, seat, args.seed), hi


def run_headless(sess: Session, hi: HumanInput | None, frames: int) -> dict:
    """Simulation without a window (used by tests and CI): the human seat idles."""
    for _ in range(frames):
        sess.tick({"move": 0} if hi else None)
        if sess.done:
            break
    return {"ticks": sess.t, "points": float(sess.state.points), "done": sess.done}


def run_window(args) -> None:
    import pygame

    from .render.field import View

    pygame.init()
    screen = pygame.display.set_mode((960, 780), pygame.RESIZABLE)
    pygame.display.set_caption("Kilithattu Kali")
    clock = pygame.time.Clock()
    sess, hi = build_session(args)
    if args.replay:
        header, log = read_replay(args.replay)
        sess = Session(load_params(header["params"]), *team_policies(load_params(header["params"]), args.difficulty), None, header["seed"], replay_log=log)
        hi = None
    view = View(screen, sess.p, args.colorblind, args.reduce_motion)
    held: set[str] = set()
    acc = 0.0
    last = time.perf_counter()
    names = {pygame.K_w: "w", pygame.K_a: "a", pygame.K_s: "s", pygame.K_d: "d", pygame.K_UP: "up", pygame.K_DOWN: "down",
             pygame.K_LEFT: "left", pygame.K_RIGHT: "right", pygame.K_LSHIFT: "shift", pygame.K_RSHIFT: "shift", pygame.K_SPACE: "space", pygame.K_e: "e"}
    running, saved, frames = True, False, 0
    while running:
        frames += 1
        if args.max_frames and frames > args.max_frames:
            break
        pressed: set[str] = set()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
            elif ev.type == pygame.VIDEORESIZE:
                view.layout()
            elif ev.type == pygame.KEYDOWN:
                if ev.key in names:
                    held.add(names[ev.key]); pressed.add(names[ev.key])
                elif ev.key == pygame.K_TAB and sess.human and sess.human.side == "attack":
                    alive = [i for i in range(sess.p.n_crossers) if int(sess.state.c_status[i]) != 2]
                    if alive:
                        nxt = [i for i in alive if i > sess.human.index] or alive
                        sess.human = Seat("attack", nxt[0])
                elif ev.key == pygame.K_r and sess.done:
                    sess.seed += 1; sess.reset(); saved = False
            elif ev.type == pygame.KEYUP and ev.key in names:
                held.discard(names[ev.key])
        now = time.perf_counter()
        acc += min(now - last, 0.1); last = now
        while acc >= sess.p.dt and not sess.done:
            act = hi.action(held, pressed, sess.human_mask()) if hi and sess.human else None
            pressed = set()
            for e in sess.tick(act):
                x, y = (e["pos"] if "pos" in e else (sess.p.field_w / 2, 0.0))
                msg = {"crossed": ("+1", view.pal["pacha"]), "delivered": ("Salt +6", view.pal["gold"]), "tagged": ("Tagged!", view.pal["foul"]),
                       "fouled": ("Foul!", view.pal["foul"]), "side_exit": ("Out!", view.pal["foul"]), "clap": ("Clap!", view.pal["gold"])}.get(e["type"])
                if msg:
                    view.add_float(msg[0], float(x), float(y), msg[1])
            acc -= sess.p.dt
        seat_agent = sess.human_agent
        view.draw(sess, min(1.0, acc / sess.p.dt), clock.get_time() / 1000, seat_agent)
        if sess.done:
            if not saved and not args.replay:
                out = Path(args.replay_dir); out.mkdir(parents=True, exist_ok=True)
                sess.save_replay(out / f"game_{int(time.time())}.kreplay", {"side": args.side, "difficulty": args.difficulty}); saved = True
            t = view.big.render(f"Innings over: {int(sess.state.points)} points   (R: again, Esc: quit)", True, view.pal["text"])
            screen.blit(t, (screen.get_width() // 2 - t.get_width() // 2, screen.get_height() // 2))
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="kilithattu", description="Play Kilithattu Kali against trained or scripted bots")
    ap.add_argument("--side", choices=list(SIDES), default="attack", help="crosser (attack), guard, Kili, or watch the bots")
    ap.add_argument("--index", type=int, default=0, help="which crosser or guard you control")
    ap.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    ap.add_argument("--rules", default="classic", help="classic | plus | path to a rules yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replay", default=None, help="watch a .kreplay folder")
    ap.add_argument("--replay-dir", default="replays")
    ap.add_argument("--colorblind", action="store_true")
    ap.add_argument("--reduce-motion", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0, help="exit the window after N frames (CI smoke test)")
    ap.add_argument("--headless-ticks", type=int, default=0, help="run N ticks without a window and print the result (CI)")
    return ap


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    if args.headless_ticks:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        sess, hi = build_session(args)
        print(run_headless(sess, hi, args.headless_ticks))
        return
    run_window(args)


if __name__ == "__main__":
    main()
