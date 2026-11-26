"""Easing table in the game's ``EaseInfor`` order (Ligatura / official).

The order is quad -> cubic -> quart -> quint, with zero/one at 13/14 and the
circular family at 15/16.  RizPlayer uses a different table with the sine
family inserted at 1-3, which shifts 4-12 by three and makes moving canvases
and hold bodies land several dozen pixels away from the real game.
"""

from __future__ import annotations

import math


def _clamp01(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x


def ease(kind: int, value: float) -> float:
    x = _clamp01(value)
    if kind == 1:
        return x * x
    if kind == 2:
        return 1.0 - (1.0 - x) * (1.0 - x)
    if kind == 3:
        return 2.0 * x * x if x < 0.5 else 1.0 - (-2.0 * x + 2.0) ** 2 / 2.0
    if kind == 4:
        return x * x * x
    if kind == 5:
        return 1.0 - (1.0 - x) ** 3
    if kind == 6:
        return 4.0 * x * x * x if x < 0.5 else 1.0 - (-2.0 * x + 2.0) ** 3 / 2.0
    if kind == 7:
        return x ** 4
    if kind == 8:
        return 1.0 - (1.0 - x) ** 4
    if kind == 9:
        return 8.0 * x ** 4 if x < 0.5 else 1.0 - (-2.0 * x + 2.0) ** 4 / 2.0
    if kind == 10:
        return x ** 5
    if kind == 11:
        return 1.0 - (1.0 - x) ** 5
    if kind == 12:
        return 16.0 * x ** 5 if x < 0.5 else 1.0 - (-2.0 * x + 2.0) ** 5 / 2.0
    if kind == 13:
        return 0.0
    if kind == 14:
        return 1.0
    if kind == 15:
        return 1.0 - math.sqrt(max(0.0, 1.0 - x * x))
    if kind == 16:
        return math.sqrt(max(0.0, 1.0 - (x - 1.0) ** 2))
    if kind == 17:
        return math.sin(x * math.pi / 2.0)
    if kind == 18:
        return 1.0 - math.cos(x * math.pi / 2.0)
    return x
