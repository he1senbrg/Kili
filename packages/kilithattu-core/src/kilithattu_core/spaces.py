"""Static shapes of tokens and action heads."""
from __future__ import annotations

PLAYER_F = 20  # features per player token
LINE_F = 5  # features per line token
N_MOVE = 9  # widest move head (crosser); guard uses 3, Kili 5
HEADS = ("move", "dash", "turn", "clap", "call")
HEAD_SIZES = {"move": N_MOVE, "dash": 2, "turn": 2, "clap": 2}  # call = n_calls + 1 (depends on params)
ROLE_CROSSER, ROLE_GUARD, ROLE_KILI = 0, 1, 2
TEAM_ATTACK, TEAM_DEFENCE = 0, 1
STATUS_PACHA, STATUS_UPPU, STATUS_OUT = 0, 1, 2


def global_f(n_calls: int) -> int:
    return 8 + n_calls + 1

# player-token feature indices (see Observer.tokens)
F_FLAGS = slice(0, 3)  # is_self, is_team, is_opp
F_ROLE = slice(3, 6)
F_STATUS = slice(6, 9)  # pacha, uppu, out (crossers only)
F_REL = slice(9, 11)
F_ABS = slice(11, 13)
F_VEL = slice(13, 15)
F_FACE, F_STAM, F_KAPPA, F_TURN, F_VIS = 15, 16, 17, 18, 19
