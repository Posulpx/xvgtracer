"""Bezier reconstructor: emit SVG path `d` from cubic Bezier control points.

A bezier primitive carries an explicit `points` list: the first point is the
start anchor, then control/anchor triples (cubic segments). When only sampled
points are available (no explicit control points), a Catmull-Rom pass is used to
generate smooth cubic segments through the samples.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def _catmull_rom_to_bezier(p0, p1, p2, p3):
    """Convert a Catmull-Rom segment (p0..p3) to a cubic Bezier (c1, c2, p2)."""
    c1 = (p1[0] + (p2[0] - p0[0]) / 6.0, p1[1] + (p2[1] - p0[1]) / 6.0)
    c2 = (p2[0] - (p3[0] - p1[0]) / 6.0, p2[1] - (p3[1] - p1[1]) / 6.0)
    return c1, c2, p2


def bezier_d(points: List[Tuple[float, float]]) -> str:
    """Build a `d` string from explicit cubic control points.

    `points` = [start, (c1, c2, anchor), (c1, c2, anchor), ...]. Each group
    after the first is a (c1, c2, anchor) triple.
    """
    if len(points) < 2:
        return ""
    d = f"M{points[0][0]:.1f},{points[0][1]:.1f} "
    for i in range(1, len(points), 3):
        seg = points[i:i + 3]
        if len(seg) == 3:
            c1, c2, anchor = seg
            d += (f"C{c1[0]:.1f},{c1[1]:.1f} "
                  f"{c2[0]:.1f},{c2[1]:.1f} "
                  f"{anchor[0]:.1f},{anchor[1]:.1f} ")
    return d.strip()


def smooth_polyline_d(points: List[Tuple[float, float]]) -> str:
    """Build a smooth cubic `d` through sample `points` via Catmull-Rom."""
    pts = [tuple(p) for p in points]
    if len(pts) < 3:
        return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts) if pts else ""
    ext = [pts[0]] + pts + [pts[-1]]
    d = f"M{pts[0][0]:.1f},{pts[0][1]:.1f} "
    for i in range(1, len(ext) - 2):
        c1, c2, anchor = _catmull_rom_to_bezier(
            ext[i - 1], ext[i], ext[i + 1], ext[i + 2])
        d += (f"C{c1[0]:.1f},{c1[1]:.1f} "
              f"{c2[0]:.1f},{c2[1]:.1f} "
              f"{anchor[0]:.1f},{anchor[1]:.1f} ")
    return d.strip()


def reconstruct(node: Dict) -> str:
    """Dispatch a bezier node to `d` (explicit controls, else smooth samples)."""
    t = node["type"]
    if t != "bezier":
        return ""
    pts = node["params"].get("points", [])
    if node["params"].get("explicit_controls"):
        return bezier_d(pts)
    return smooth_polyline_d(pts)
