import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

from kilithattu_core import CLASSIC
from kilithattu_core.replay import verify_replay
from kilithattu_game.app import main
from kilithattu_game.input.keyboard import HumanInput, axes
from kilithattu_game.session import Seat, Session


def test_kili_turn_buffering():
    hi = HumanInput("defence", True)
    mask_vertical_only = np.array([True, True, False, True, False])      # mid-segment on a longitudinal line
    mask_junction = np.array([True, True, True, True, True])
    assert hi.kili_move(0, 1, mask_vertical_only) == 1                    # N
    assert hi.kili_move(1, 0, mask_vertical_only) == 1                    # E not possible yet: keep going N
    assert hi.kili_move(1, 0, mask_junction) == 2                         # junction reached: turn E
    assert hi.kili_move(0, 0, mask_junction) == 0


def test_axes_and_modes():
    assert axes({"w", "d"}) == (1, 1) and axes({"left"}) == (-1, 0)
    a = HumanInput("attack", False).action({"d", "w", "shift"}, set(), None)
    assert a["move"] == 2 and a["dash"] == 1
    g = HumanInput("defence", False).action({"a"}, {"e"}, None)
    assert g["move"] == 1 and g["turn"] == 1
    k = HumanInput("defence", True).action(set(), {"space"}, {"move": np.ones(5, bool)})
    assert k["clap"] == 1


def test_session_human_override_and_replay_verifies(tmp_path):
    from kilithattu_game.controllers import team_policies
    p = CLASSIC.replace(n_crossers=2, innings_s=8.0)
    att, dfn = team_policies(p, "easy")
    s = Session(p, att, dfn, Seat("attack", 0), seed=3)
    y0 = float(s.state.c_pos[0, 1])
    for _ in range(80):
        s.tick({"move": 1})                                               # human crosser holds "north"
        if s.done:
            break
    assert s.t > 0 and (float(s.state.c_pos[0, 1]) > y0 + 1.0 or int(s.state.c_status[0]) == 2)
    s.save_replay(tmp_path / "g.kreplay", {"side": "attack"})
    assert verify_replay(tmp_path / "g.kreplay")
    watch = Session(p, att, dfn, None, seed=3, replay_log=np.load(tmp_path / "g.kreplay" / "actions.npz")["actions"])
    for _ in range(s.t):
        watch.tick()
    np.testing.assert_allclose(np.asarray(watch.state.c_pos), np.asarray(s.state.c_pos), atol=1e-5)


def test_cli_headless_and_window_smoke(capsys):
    main(["--side", "watch", "--headless-ticks", "120", "--rules", "classic"])
    assert "ticks" in capsys.readouterr().out
    main(["--side", "kili", "--difficulty", "easy", "--max-frames", "20"])    # opens a dummy-driver window
