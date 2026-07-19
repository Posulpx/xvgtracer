"""Corner sources of truth for pointy corners.

The quantized-image collision detector only fires where several colors converge,
so it is blind to a sharp tip that sits against plain background and biases the
apex inward (its point is the mean of hit pixels, not the extremum). This module
surfaces *independent* corner sources so they can be compared side by side --
deliberately WITHOUT merging them into a single answer yet.

Sources implemented here:

  * ``contour`` -- curvature (turning-angle) maxima measured directly on a
    shape's RAW mask contour. This sees tips-against-background that collisions
    miss; the apex is a literal sharp turn of the boundary.
  * ``extended`` -- the intersection of two extended straight SIDES of a custom
    polygon (the construction-line convergence). Two long straight edges pin the
    implied apex precisely even when the traced tip pixels are rounded off, so
    this is often more accurate at a pointy corner than the traced vertex.

Each source returns a flat list of labeled dicts the SVG renderer draws in its
own overlay group.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# Source A: contour curvature maxima
# ---------------------------------------------------------------------------

def _turning_angle(a: Point, b: Point, c: Point) -> float:
    """Exterior turning angle (deg) of the path a->b->c at b. 0 = straight."""
    v1x, v1y = b[0] - a[0], b[1] - a[1]
    v2x, v2y = c[0] - b[0], c[1] - b[1]
    L1 = math.hypot(v1x, v1y)
    L2 = math.hypot(v2x, v2y)
    if L1 < 1e-9 or L2 < 1e-9:
        return 0.0
    dot = (v1x * v2x + v1y * v2y) / (L1 * L2)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def contour_corners_scored(contour: List[Point], span: int = 6,
                           min_turn: float = 45.0) -> List[Tuple[Point, float]]:
    """Sharp turning-angle maxima on a raw contour ring, with their turn score.

    For each vertex we measure the turning angle between the chord arriving from
    ``span`` points back and the chord leaving ``span`` points ahead. Using a
    window (not adjacent pixels) makes the measure robust to staircase jitter:
    a stair step turns ~90 degrees locally but the windowed chords stay nearly
    collinear, so only genuine corners exceed ``min_turn``. Non-maximum
    suppression keeps one point per corner. Returns ``[((x, y), turn_deg), ...]``
    so callers can rank corners (the sharpest turns are the real vertices).
    """
    ring = contour[:-1] if len(contour) > 1 and contour[0] == contour[-1] else list(contour)
    n = len(ring)
    if n < 2 * span + 1:
        return []
    turns = []
    for i in range(n):
        a = ring[(i - span) % n]
        b = ring[i]
        c = ring[(i + span) % n]
        turns.append(_turning_angle(a, b, c))
    # non-maximum suppression within +/- span; keep local maxima above min_turn
    corners = []
    for i in range(n):
        t = turns[i]
        if t < min_turn:
            continue
        is_max = True
        for j in range(1, span + 1):
            if turns[(i - j) % n] > t or turns[(i + j) % n] > t:
                is_max = False
                break
        if is_max:
            corners.append(((round(ring[i][0], 1), round(ring[i][1], 1)), round(t, 1)))
    return corners


def contour_corners(contour: List[Point], span: int = 6,
                    min_turn: float = 45.0) -> List[Point]:
    """Sharp turning-angle maxima on a raw contour ring (points only)."""
    return [pt for pt, _t in contour_corners_scored(contour, span, min_turn)]


def strongest_corners(contour: List[Point], k: int, span: int = 6,
                      min_turn: float = 45.0) -> List[Point]:
    """Return the ``k`` sharpest contour corners in contour (ring) order.

    Ranks all detected corners by turning angle, keeps the top ``k`` (a triangle
    has k=3), then restores their original boundary order so the polygon winds
    correctly. This discards weaker false corners caused by staircase jitter on
    straight edges, keeping only the true vertices.
    """
    scored = contour_corners_scored(contour, span, min_turn)
    if len(scored) <= k:
        return [pt for pt, _t in scored]
    top = sorted(scored, key=lambda s: -s[1])[:k]
    top_set = {pt for pt, _t in top}
    # restore boundary order
    return [pt for pt, _t in scored if pt in top_set]


# ---------------------------------------------------------------------------
# Source B: extended straight-side intersections (implied apex)
# ---------------------------------------------------------------------------

def _line_intersection(p1: Point, p2: Point, p3: Point, p4: Point):
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


def extended_side_corners(shapes: List[Dict], min_len: float = 6.0,
                          max_reach: float = 8.0) -> List[Dict]:
    """Implied apexes from intersecting the extended straight sides.

    For each custom polygon, take its straight sides (>= ``min_len`` px) and, for
    every ADJACENT pair (sharing a polygon vertex), intersect their infinite
    lines. The intersection is the geometric apex the two edges imply. It is kept
    only when it lands within ``max_reach`` px of the shared traced vertex -- i.e.
    the trace rounded the tip only slightly, so the implied apex is trustworthy.
    """
    from .construction import _straight_edges
    out: List[Dict] = []
    for s in shapes:
        if not s.get("custom"):
            continue
        pts = [(float(x), float(y)) for x, y in s.get("params", {}).get("points", [])]
        if len(pts) < 3:
            continue
        li = s.get("layer_index", 0)
        edges = _straight_edges(pts, min_len=min_len)
        by_vertex: Dict[int, List] = {}
        for a, b, vi, vj in edges:
            by_vertex.setdefault(vi, []).append((a, b))
            by_vertex.setdefault(vj, []).append((a, b))
        for v, segs in by_vertex.items():
            if len(segs) < 2:
                continue
            (a1, b1), (a2, b2) = segs[0], segs[1]
            ip = _line_intersection(a1, b1, a2, b2)
            if ip is None:
                continue
            vx, vy = pts[v]
            if math.hypot(ip[0] - vx, ip[1] - vy) <= max_reach:
                out.append({
                    "kind": "extended",
                    "layer_index": li,
                    "x": round(ip[0], 1), "y": round(ip[1], 1),
                    "traced_x": round(vx, 1), "traced_y": round(vy, 1),
                })
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def corner_sources(shapes: List[Dict], masks: Dict[int, "object"] = None) -> Dict[str, List[Dict]]:
    """Collect all corner sources of truth as separate labeled lists.

    ``masks`` maps a shape's ``id()`` to its binary mask so per-shape contour
    curvature can be measured. If not supplied, the ``contour`` source is empty
    (the caller may not have retained masks). Returns
    ``{"contour": [...], "extended": [...]}``.
    """
    contour_out: List[Dict] = []
    if masks:
        from .detectors.contour_detector import contour_vertices
        from .detectors.edge_detector import repair_mask
        for s in shapes:
            m = masks.get(id(s))
            if m is None:
                continue
            ring = contour_vertices(repair_mask(m), fill_holes=True)
            if not ring:
                continue
            li = s.get("layer_index", 0)
            for (x, y) in contour_corners(ring):
                contour_out.append({
                    "kind": "contour",
                    "layer_index": li,
                    "x": x, "y": y,
                })
    return {
        "contour": contour_out,
        "extended": extended_side_corners(shapes),
    }


# ---------------------------------------------------------------------------
# Corner-driven geometry refinement
# ---------------------------------------------------------------------------

# Polygonal primitives whose vertex count is known, so we can rebuild them from
# exactly that many of the sharpest contour corners.
_FIXED_VERTEX_SHAPES = {"triangle": 3}

# When re-selecting a custom polygon's corners at a smaller span (to recover
# genuine corners the default span suppresses), accept the new corner set only
# if its mask IoU is within this tolerance of the current traced outline's.
_CUSTOM_REFINE_IOU_TOL = 0.01


def _centroid(pts: List[Point]) -> Point:
    n = max(1, len(pts))
    return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)


def star_corners(contour: List[Point], min_turn: float = 45.0,
                 span: int = 6) -> List[Point]:
    """Vertices of an N-armed star: the sharpest tips + valleys, alternating.

    A star has ``N`` outer tips (convex, far from the centroid) alternating with
    ``N`` inner valleys (concave, near the centroid). A plain turning-angle
    filter can admit an extra point on a slightly bowed arm, breaking the clean
    tip/valley alternation. Here we:

      1. take all curvature maxima above ``min_turn``;
      2. split them into tips vs valleys by comparing each corner's distance to
         the shape centroid against the median distance;
      3. keep the ``N`` sharpest of each (``N`` = min(#tips, #valleys)),
         restored to boundary order so tip and valley alternate.

    Returns ``[]`` when the corner set is not star-like (fewer than 3 tips or
    valleys), so the caller can fall back to the faithful outline.
    """
    scored = contour_corners_scored(contour, span, min_turn)
    if len(scored) < 6:
        return []
    cx, cy = _centroid([pt for pt, _t in scored])
    dists = [math.hypot(pt[0] - cx, pt[1] - cy) for pt, _t in scored]
    med = sorted(dists)[len(dists) // 2]
    tips = [(pt, t) for (pt, t), d in zip(scored, dists) if d >= med]
    valleys = [(pt, t) for (pt, t), d in zip(scored, dists) if d < med]
    n = min(len(tips), len(valleys))
    if n < 3:
        return []
    keep = set()
    for group in (tips, valleys):
        for pt, _t in sorted(group, key=lambda s: -s[1])[:n]:
            keep.add(pt)
    return [pt for pt, _t in scored if pt in keep]


def corners_by_threshold(contour: List[Point], min_turn: float = 45.0,
                         span: int = 6) -> List[Point]:
    """All contour curvature maxima above ``min_turn``, in boundary order.

    Unlike :func:`strongest_corners` (fixed count) this keeps every real corner
    a shape has -- e.g. an N-armed star keeps all its tips and valleys -- while
    staircase jitter on straight runs stays below the threshold and is dropped.
    An equal tip/valley split is deliberately NOT enforced: real stars can be
    asymmetric, so forcing balance would discard genuine corners.
    """
    return [pt for pt, _t in contour_corners_scored(contour, span, min_turn)]


def refine_shapes_from_corners(shapes: List[Dict],
                               masks: Dict[int, "object"],
                               custom_min_turn: float = 45.0) -> List[Dict]:
    """Rebuild polygonal shapes from their validated contour corners.

    * Fixed-vertex primitives (triangle -> 3): rebuilt from the N sharpest
      contour curvature maxima -- the human-validated ideal line-to-line
      endpoints -- rather than the radius/IoU primitive fit.
    * Custom polygons (traced silhouettes with no fixed vertex count, e.g. a
      3-armed star): rebuilt from EVERY contour corner whose turning angle
      exceeds ``custom_min_turn``, so the shape keeps exactly as many real
      corners (tips + valleys) as it has, with the rounded/jittery in-between
      contour points removed.

    Curved primitives (circle/ellipse) and rects are left untouched. Sets
    ``corner_refined`` on shapes that change.
    """
    if not masks:
        return shapes
    from .detectors.contour_detector import contour_vertices
    from .detectors.edge_detector import repair_mask
    from .renderers import rasterize_polygon

    def _poly_iou(pts, mask):
        if len(pts) < 3:
            return 0.0
        h, w = mask.shape
        r = rasterize_polygon([(float(x), float(y)) for x, y in pts] + [pts[0]], w, h)
        union = (r | mask).sum()
        return float((r & mask).sum()) / union if union else 0.0

    for s in shapes:
        t = s.get("type")
        is_custom = bool(s.get("custom"))
        k = _FIXED_VERTEX_SHAPES.get(t)
        if not k and not is_custom:
            continue
        m = masks.get(id(s))
        if m is None:
            continue
        ring = contour_vertices(repair_mask(m), fill_holes=True)
        if not ring:
            continue
        if k:
            corners = strongest_corners(ring, k)
            if len(corners) != k:
                continue
        else:
            # IoU-guarded: the default span can non-max-suppress genuine corners
            # on fine-featured silhouettes (e.g. the thin slots of a comb),
            # collapsing rectangular notches into V-points. Try progressively
            # smaller spans and keep the corner set only if it does not lose
            # fidelity against the mask relative to the current traced outline.
            base_pts = s.get("params", {}).get("points", [])
            base_iou = _poly_iou(base_pts, m) if base_pts else 0.0
            corners = None
            best_iou = base_iou - _CUSTOM_REFINE_IOU_TOL
            for span in (6, 4, 3, 2):
                cand = corners_by_threshold(ring, custom_min_turn, span=span)
                if len(cand) < 3:
                    continue
                iou = _poly_iou(cand, m)
                if iou >= best_iou:
                    best_iou = iou
                    corners = cand
            if not corners:
                continue
        s.setdefault("params", {})["points"] = [[float(x), float(y)] for x, y in corners]
        s["corner_refined"] = True
    return shapes
