"""Rectangle generator: candidate point list for an axis-aligned rectangle."""

from __future__ import annotations

from typing import List, Tuple

Point = Tuple[float, float]


def rectangle_points(x0, y0, x1, y1) -> List[Point]:
    """Four corners of an axis-aligned rectangle (closed ring)."""
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


def rounded_rectangle_points(x0, y0, x1, y1, r: float,
                             steps_per_corner: int = 8) -> List[Point]:
    """Approximate a rounded rectangle with arc samples (closed ring)."""
    w, h = x1 - x0, y1 - y0
    r = min(r, w / 2.0, h / 2.0)
    if r <= 0.5:
        return rectangle_points(x0, y0, x1, y1)
    import math
    # (center_x, center_y, start_angle_rad, end_angle_rad) for each corner
    corners = [
        (x1 - r, y0 + r, math.pi * 1.5, math.pi * 2.0),
        (x1 - r, y1 - r, 0.0, math.pi * 0.5),
        (x0 + r, y1 - r, math.pi * 0.5, math.pi),
        (x0 + r, y0 + r, math.pi, math.pi * 1.5),
    ]
    pts: List[Point] = [(x0 + r, y0)]
    for ccx, ccy, a0, a1 in corners:
        for a in [a0 + (a1 - a0) * i / steps_per_corner for i in range(steps_per_corner + 1)]:
            pts.append((ccx + r * math.cos(a), ccy + r * math.sin(a)))
    pts.append((x0 + r, y0))
    return pts
