"""Keyboard to action mapping for crossers, guards and the Kili (with turn buffering on the line graph)."""
from __future__ import annotations

# move ids: crosser 0 stay, 1 N, 2 NE, 3 E, 4 SE, 5 S, 6 SW, 7 W, 8 NW; guard 0 stay, 1 W, 2 E; Kili 0 stay, 1 N, 2 E, 3 S, 4 W
_CROSSER = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3, (1, -1): 4, (0, -1): 5, (-1, -1): 6, (-1, 0): 7, (-1, 1): 8}
_KILI = {(0, 1): 1, (1, 0): 2, (0, -1): 3, (-1, 0): 4}


def axes(keys) -> tuple[int, int]:
    """keys: set of lower-case key names currently held."""
    dx = (1 if keys & {"d", "right"} else 0) - (1 if keys & {"a", "left"} else 0)
    dy = (1 if keys & {"w", "up"} else 0) - (1 if keys & {"s", "down"} else 0)
    return dx, dy


class HumanInput:
    def __init__(self, side: str, is_kili: bool):
        self.side, self.is_kili = side, is_kili
        self.last_kili = 0
        self.turn_pending = False

    def action(self, keys: set[str], pressed: set[str], mask: dict | None) -> dict:
        dx, dy = axes(keys)
        dash = int("shift" in keys)
        turn = int("e" in pressed)
        clap = int("space" in pressed)
        if self.side == "attack":
            move = _CROSSER[(dx, dy)]
        elif not self.is_kili:
            move = 1 if dx < 0 else 2 if dx > 0 else 0
        else:
            move = self.kili_move(dx, dy, mask["move"] if mask is not None else None)
        return {"move": move, "dash": dash, "turn": turn, "clap": clap, "call": 0}

    def kili_move(self, dx: int, dy: int, mask) -> int:
        """Buffered turning: if the desired direction has no adjacent segment yet, keep the current direction
        until the next junction where the turn becomes available."""
        if (dx, dy) == (0, 0):
            self.last_kili = 0
            return 0
        if dx != 0 and dy != 0:  # diagonal input: prefer the axis that is currently possible, else vertical
            cand = [_KILI[(dx, 0)], _KILI[(0, dy)]]
        else:
            cand = [_KILI[(dx, dy)]]
        for c in cand:
            if mask is None or mask[c]:
                self.last_kili = c
                return c
        if self.last_kili and (mask is None or mask[self.last_kili]):
            return self.last_kili
        return 0
