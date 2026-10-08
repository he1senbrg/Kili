"""Tiny synthesised sound effects (no asset files). Silently disabled when no audio device exists."""
from __future__ import annotations

import numpy as np


class Sfx:
    def __init__(self, enabled: bool = True):
        self.sounds: dict[str, object] = {}
        self.enabled = False
        if not enabled:
            return
        try:
            import pygame

            pygame.mixer.init(frequency=22050, size=-16, channels=1)
            def tone(freq: float, ms: int, vol: float = 0.25):
                t = np.linspace(0, ms / 1000, int(22050 * ms / 1000), False)
                wave = (np.sin(2 * np.pi * freq * t) * np.exp(-t * 8) * vol * 32767).astype(np.int16)
                return pygame.sndarray.make_sound(wave)
            self.sounds = {"clap": tone(180, 120, 0.5), "cross": tone(660, 70), "salt": tone(880, 180), "tag": tone(110, 220, 0.4), "foul": tone(140, 260, 0.4)}
            self.enabled = True
        except Exception:
            self.enabled = False

    def play(self, name: str) -> None:
        if self.enabled and name in self.sounds:
            self.sounds[name].play()
