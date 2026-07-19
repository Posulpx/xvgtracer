"""Ellipse generator: candidate point list for an axis-aligned ellipse."""

from __future__ import annotations

from typing import List, Tuple

import math

Point = Tuple[float, float]


def ellipse_points(cx: float, cy: float, rx: float, ry: float,
                   n: int = 64) -> List[Point]:
    """Sample `n` points around an ellipse centred at (cx, cy)."""
    return [(cx + rx * math.cos(2 * math.pi * i / n),
             cy + ry * math.sin(2 * math.pi * i / n)) for i in range(n)]


def ellipse_from_bbox(x0, y0, x1, y1):
    """Return (cx, cy, rx, ry) of the ellipse inscribed in a bounding box."""
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    rx = (x1 - x0) / 2.0
    ry = (y1 - y0) / 2.0
    return cx, cy, rx, ry
