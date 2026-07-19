"""Circle generator: candidate point list for a circle primitive."""

from __future__ import annotations

from typing import List, Tuple

import math

Point = Tuple[float, float]


def circle_points(cx: float, cy: float, r: float, n: int = 64) -> List[Point]:
    """Sample `n` points around a circle of radius `r` centred at (cx, cy)."""
    return [(cx + r * math.cos(2 * math.pi * i / n),
             cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


def circle_from_bbox(x0, y0, x1, y1):
    """Return (cx, cy, r) of the circle inscribed in a bounding box."""
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    r = (min(x1 - x0, y1 - y0)) / 2.0
    return cx, cy, r
