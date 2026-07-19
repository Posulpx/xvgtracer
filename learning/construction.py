"""Construction lines for non-primitive (custom polygon) shapes.

A custom polygon is what the classifier produces when a silhouette does NOT
qualify as any primitive: a faithful traced outline. Its *straight* edges are
meaningful structural sides (as opposed to the short facets that approximate a
curved run). This module extracts those straight edges and extends each a few
pixels past both endpoints, so the extensions can be inspected for **convergence
points** — where two extended edges meet reveals an implied corner/vertex that
the traced outline rounded or clipped.

The output mirrors the `collisions` overlay: a flat list of line dicts that the
SVG renderer draws in a dedicated `<g id="construction">` group.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

Point = Tuple[float, float]


def _straight_edges(pts: List[Point], min_len: float = 6.0):
    """Return the polygon edges that read as genuine straight sides.

    A custom polygon collapses straight runs to single edges; only the tiniest
    segments (a few px) are quantization jitter rather than real sides. So an
    edge qualifies as a straight *side* purely on an ABSOLUTE minimum length
    (``min_len`` px). Using an absolute threshold -- not a fraction of the
    perimeter or of the longest edge -- means EVERY genuine side of a many-sided
    polygon is evaluated, including the shorter ones (e.g. a star's near-tip
    sides), not just the couple of longest edges.

    Returns a list of ``(a, b, vi, vj)`` where ``vi``/``vj`` are the vertex
    indices of the edge endpoints (used to skip trivial adjacent-edge
    convergences at a shared corner).
    """
    n = len(pts)
    if n < 2:
        return []
    closed = pts + [pts[0]] if pts[0] != pts[-1] else list(pts)
    m = len(closed) - 1
    edges = []
    for i in range(m):
        a, b = closed[i], closed[i + 1]
        if math.hypot(b[0] - a[0], b[1] - a[1]) >= min_len:
            edges.append((a, b, i, (i + 1) % m))
    return edges


def _extend(a: Point, b: Point, pad: float) -> Tuple[Point, Point]:
    """Extend segment a->b by ``pad`` pixels past each endpoint."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return a, b
    ux, uy = dx / L, dy / L
    return ((a[0] - ux * pad, a[1] - uy * pad),
            (b[0] + ux * pad, b[1] + uy * pad))


def _seg_intersection(p1: Point, p2: Point, p3: Point, p4: Point):
    """Intersection point of the infinite lines through p1p2 and p3p4, or None."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return (px, py)


def _snap_to_corner(p: Point, corners: List[Point], tol: float) -> Point:
    """Return the nearest confirmed corner to p within ``tol``, else p itself.

    Confirmed corners are the contour curvature maxima (the human-validated
    ideal line-to-line endpoints). Anchoring a side's endpoint to the nearest
    such corner makes construction lines run corner-to-corner rather than from
    the RDP-rounded traced vertex.
    """
    if not corners:
        return p
    best, bd = p, tol
    for cx, cy in corners:
        d = math.hypot(cx - p[0], cy - p[1])
        if d <= bd:
            bd, best = d, (cx, cy)
    return best


def construction_lines(shapes: List[Dict], pad: float = 2.0,
                       converge_dist: float = 2.0,
                       corners_by_layer: Dict[int, List[Point]] = None,
                       snap_tol: float = 6.0) -> List[Dict]:
    """Build construction-line overlays for every non-primitive shape.

    For each shape flagged ``custom`` (a traced polygon), each straight side's
    endpoints are first ANCHORED to the nearest confirmed contour corner (the
    human-validated ideal line-to-line endpoints, passed via
    ``corners_by_layer``) within ``snap_tol`` px, then the side is extended by
    ``pad`` px past both ends. This makes lines run corner-to-corner so their
    intersections land on the true endpoints. Where two extended lines cross
    close to their tips (within ``converge_dist``) a convergence marker is
    emitted.

    Shapes carrying an explicit ``spline_nodes`` ring (an overlapping circle
    rebuilt as a node-spline) additionally get their spline skeleton drawn: a
    ``spline`` segment between each consecutive node and a ``node`` marker at each
    node, so the spline's construction is visible alongside the polygon lines.

    Returns a list of dicts:
      ``{"kind": "line", "layer_index", "x1","y1","x2","y2"}``
      ``{"kind": "convergence", "layer_index", "x","y"}``
      ``{"kind": "spline", "layer_index", "x1","y1","x2","y2"}``
      ``{"kind": "node", "layer_index", "x","y"}``
    """
    corners_by_layer = corners_by_layer or {}
    out: List[Dict] = []
    for s in shapes:
        nodes = s.get("params", {}).get("spline_nodes")
        if nodes and len(nodes) >= 2:
            li = s.get("layer_index", 0)
            m = len(nodes)
            for i in range(m):
                x0, y0 = nodes[i]
                x1, y1 = nodes[(i + 1) % m]
                out.append({
                    "kind": "spline", "layer_index": li,
                    "x1": round(float(x0), 1), "y1": round(float(y0), 1),
                    "x2": round(float(x1), 1), "y2": round(float(y1), 1),
                })
            for x, y in nodes:
                out.append({
                    "kind": "node", "layer_index": li,
                    "x": round(float(x), 1), "y": round(float(y), 1),
                })
        if not s.get("custom"):
            continue
        pts = [(float(x), float(y)) for x, y in s.get("params", {}).get("points", [])]
        if len(pts) < 2:
            continue
        li = s.get("layer_index", 0)
        corners = corners_by_layer.get(li, [])
        edges = _straight_edges(pts)
        # anchor each side's endpoints to the confirmed corners, then extend
        extended = []
        for a, b, vi, vj in edges:
            a = _snap_to_corner(a, corners, snap_tol)
            b = _snap_to_corner(b, corners, snap_tol)
            extended.append((_extend(a, b, pad), vi, vj))
        for (a, b), _vi, _vj in extended:
            out.append({
                "kind": "line",
                "layer_index": li,
                "x1": round(a[0], 1), "y1": round(a[1], 1),
                "x2": round(b[0], 1), "y2": round(b[1], 1),
            })
        # convergence: pairs of extended lines whose intersection sits near the
        # extended tips of both. Skip ADJACENT edges (sharing a polygon vertex):
        # those trivially meet at their existing shared corner and reveal nothing.
        for i in range(len(extended)):
            for j in range(i + 1, len(extended)):
                (a1, b1), vi1, vj1 = extended[i]
                (a2, b2), vi2, vj2 = extended[j]
                if {vi1, vj1} & {vi2, vj2}:
                    continue
                ip = _seg_intersection(a1, b1, a2, b2)
                if ip is None:
                    continue
                if (_near_tip(ip, a1, b1, converge_dist)
                        and _near_tip(ip, a2, b2, converge_dist)):
                    out.append({
                        "kind": "convergence",
                        "layer_index": li,
                        "x": round(ip[0], 1), "y": round(ip[1], 1),
                    })
    return out


def _near_tip(p: Point, a: Point, b: Point, tol: float) -> bool:
    """True if p lies on/just beyond the a-b segment within ``tol`` of a tip.

    Used to keep only convergences that the *extensions* reach -- an intersection
    far outside the padded segment is not an observable convergence.
    """
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return math.hypot(p[0] - a[0], p[1] - a[1]) <= tol
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2
    if -tol / math.sqrt(L2) <= t <= 1 + tol / math.sqrt(L2):
        return True
    return False
