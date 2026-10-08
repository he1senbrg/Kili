"""Fixed-timestep accumulator: simulate at 20 Hz, render at display rate with interpolation."""
from __future__ import annotations


class FixedStep:
    def __init__(self, dt: float, max_frame: float = 0.1):
        self.dt, self.max_frame, self.acc = dt, max_frame, 0.0

    def advance(self, frame_time: float) -> int:
        """Add wall-clock time, return how many simulation ticks to run now."""
        self.acc += min(frame_time, self.max_frame)
        n = int(self.acc // self.dt)
        self.acc -= n * self.dt
        return n

    @property
    def alpha(self) -> float:
        return self.acc / self.dt
