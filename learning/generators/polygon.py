"""Polygon generator: candidate point lists for polygon-style primitives.

Provides a generic regular/sampled polygon as well as specialised builders for
the triangle and star primitives used by the learning system.
"""

from __future__ import annotations

from typing import List, Tuple

import math

Point = Tuple[float, float]


def polygon_points(points: List[Point], close: bool = True) -> List[Point]:
    """Return a (closed) ring from an explicit point list."""
    pts = [tuple(p) for p in points]
    if close and pts and pts[0] != pts[-1]:
        pts = pts + [pts[0]]
    return pts


def triangle_points(cx, cy, rx, ry) -> List[Point]:
    """Upright triangle inscribed in a bbox half-extents (rx, ry) about centre."""
    return [(cx, cy - ry), (cx - rx, cy + ry), (cx + rx, cy + ry),
            (cx, cy - ry)]


def regular_polygon(cx, cy, r, sides: int, rotation: float = -math.pi / 2):
    """Sample a regular `sides`-gon of circumradius `r` centred at (cx, cy)."""
    return [(cx + r * math.cos(rotation + 2 * math.pi * i / sides),
             cy + r * math.sin(rotation + 2 * math.pi * i / sides))
            for i in range(sides)]


def star_points(cx, cy, r_outer, r_inner, points: int = 5,
                rotation: float = -math.pi / 2) -> List[Point]:
    """Sample a `points`-point star (alternating outer/inner radii)."""
    pts: List[Point] = []
    for i in range(points * 2):
        r = r_outer if i % 2 == 0 else r_inner
        a = rotation + math.pi * i / points
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    pts.append(pts[0])
    return pts
