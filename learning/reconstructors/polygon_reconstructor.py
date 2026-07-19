"""Polygon reconstructor: emit SVG path `d` for polygon-style primitives.

Handles triangle, polygon, and star primitives (explicit point lists). Also
exposes a helper to turn a contour into a polygon `d` for the fallback case.
"""

from __future__ import annotations

from typing import Dict, List


def polygon_d(points: List, close: bool = True) -> str:
    """Build an SVG `d` polyline from an explicit point list."""
    pts = [tuple(p) for p in points]
    if not pts:
        return ""
    if close and pts[0] != pts[-1]:
        pts = pts + [pts[0]]
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + " Z"


def reconstruct(node: Dict) -> str:
    """Dispatch a polygon-style node (triangle/polygon/star) to `d`."""
    t = node["type"]
    if t in ("triangle", "polygon", "star"):
        return polygon_d(node["params"]["points"])
    return ""
