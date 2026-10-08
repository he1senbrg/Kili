import numpy as np
import pytest

from kilithattu_core import CLASSIC
from kilithattu_eval.human_study import paired_comparison, sus_score
from kilithattu_eval.ratings import elo_from_matrix, nash_average


def test_nash_of_rock_paper_scissors_is_uniform():
    rps = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], float)
    mix, value, _ = nash_average(rps)
    np.testing.assert_allclose(mix, 1 / 3, atol=1e-6)
    assert abs(value) < 1e-6


def test_elo_orders_a_transitive_population():
    s = np.array([[0, .5, .9], [-.5, 0, .5], [-.9, -.5, 0]])
    e = elo_from_matrix(s)
    assert e[0] > e[1] > e[2] and abs(e.mean() - 1000) < 1e-6


def test_sus_and_paired_stats():
    assert sus_score([5, 1, 5, 1, 5, 1, 5, 1, 5, 1]) == 100.0
    assert sus_score([3] * 10) == 50.0
    r = paired_comparison([5, 6, 7, 8, 9], [1, 2, 3, 4, 5])
    assert r["mean_diff"] == 4.0 and r["p"] < 0.1


@pytest.mark.nightly
def test_match_tournament_and_balance():
    from kilithattu_agents import ScriptedAttack, ScriptedDefence
    from kilithattu_eval.balance import search
    from kilithattu_eval.matches import play_match
    from kilithattu_eval.tournament import round_robin
    p = CLASSIC.replace(n_crossers=2, innings_s=20.0)
    m = play_match(p, ScriptedAttack(p), ScriptedDefence(p), 4)
    assert 0 <= m["win_rate"] <= 1 and m["innings"] == 4
    agents = {"easy": {"attack": ScriptedAttack(p, 0.7), "defence": ScriptedDefence(p, 0.1)},
              "hard": {"attack": ScriptedAttack(p, 1.3), "defence": ScriptedDefence(p, 0.6)}}
    res = round_robin(p, agents, n=4)
    assert res["score"].shape == (2, 2) and abs(res["score"][0, 1] + res["score"][1, 0]) < 1e-9
    study = search(p, n_trials=2, episodes=4)
    assert len(study.trials) == 2
