"""Controller factory: build team policies from the difficulty ladder (scripted or trained)."""
from __future__ import annotations

from kilithattu_agents import load_policy


def team_policies(p, difficulty: str, ladder: dict | None = None):
    """-> (attack policy, defence policy) at the requested difficulty level."""
    return load_policy(f"main@{difficulty}", "attack", p, ladder=ladder), load_policy(f"main@{difficulty}", "defence", p, ladder=ladder)
