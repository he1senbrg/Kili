"""Typed per-tick events and helpers to turn them into a readable log."""
from __future__ import annotations

import numpy as np

from .state import Events

EVENT_NAMES = ("tagged", "fouled", "side_exit", "false_start", "crossed", "delivered")


def decode(ev: Events, tick: int = 0) -> list[dict]:
    """Events of one tick as ``[{"tick", "type", "crosser"}]`` (plus a ``clap`` entry)."""
    out = []
    for name in EVENT_NAMES:
        for i in np.flatnonzero(np.asarray(getattr(ev, name))):
            out.append({"tick": tick, "type": name, "crosser": int(i)})
    if bool(ev.clapped):
        out.append({"tick": tick, "type": "clap"})
    return out
