"""Snap reconstructed shape nodes onto collision (convergence) points.

The quantized image yields collision points where colors meet, but the ideal
reconstruction (circle/ellipse/rect/polygon …) often misses those junctions by
several pixels. To keep the vector output faithful to the observed junctions, we
deform each converging shape so its boundary passes through the collision point.

The snap is type-aware:
  * polygon / star / triangle / custom -> move the nearest vertex onto the point
    (or, if no vertex is close, insert the point into the ring).
  * circle -> a STANDALONE circle stays a clean parametric primitive; an
    OVERLAPPING circle (one sharing convergence points with a tandem shape) is
    rebuilt as an explicit node-spline whose ring nodes are the cardinal anchors
    PLUS one node placed exactly on each shared convergence point, so its boundary
    visibly touches its neighbours. Only circles overlapping others become
    splines; curves involved in overlaps must be node-splines, not primitives.
  * ellipse -> shift the centre along the centre->point axis so the boundary
    reaches the point (radius preserved).
  * rect / rounded_rect -> grow the bounding box on the nearest edge to include
    the point.
"""

from __future__ import annotations

import math
from typing import Dict, List


def _snap_polygon_points(points: List[List[float]], targets,
                         insert_if_far: float = 8.0) -> List[List[float]]:
    """Snap a point ring onto several target points in one stable pass.

    Each target is projected onto the nearest polygon EDGE and inserted there
    (splitting the edge), which guarantees the point lies exactly on the
    boundary. If a target is already within `insert_if_far` of an existing
    vertex, that vertex is moved onto it instead. Targets are processed in
    angular order around the ring centroid for stable winding.
    """
    if not points or not targets:
        return points
    n = len(points)
    cx = sum(x for x, _ in points) / n
    cy = sum(y for _, y in points) / n
    order = sorted(targets, key=lambda t: math.atan2(t[1] - cy, t[0] - cx))
    claimed = set()  # coordinate tuples already snapped to a target (no stealing)
    for px, py in order:
        # closest existing (unclaimed) vertex within insert_if_far -> move it
        vbest, vd, vbest_i = -1, float("inf"), -1
        for i, (x, y) in enumerate(points):
            if (round(x, 2), round(y, 2)) in claimed:
                continue
            d = math.hypot(x - px, y - py)
            if d < vd:
                vd, vbest, vbest_i = d, [x, y], i
        if vbest_i >= 0 and vd <= insert_if_far:
            points[vbest_i] = [px, py]
            claimed.add((round(px, 2), round(py, 2)))
            continue
        # otherwise split the nearest edge by inserting the target itself, so
        # the boundary is pulled onto (px, py) rather than just onto the edge.
        ebest, ed = 0, float("inf")
        for i in range(len(points)):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % len(points)]
            dx, dy = x2 - x1, y2 - y1
            L2 = dx * dx + dy * dy
            if L2 < 1e-9:
                d = math.hypot(px - x1, py - y1)
            else:
                t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L2))
                d = math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))
            if d < ed:
                ed, ebest = d, i
        points.insert((ebest + 1) % (len(points) + 1), [px, py])
        claimed.add((round(px, 2), round(py, 2)))
    return points


# A convergence node this close to an existing ring node is treated as the same
# node (its angle collapses onto the cardinal), avoiding a degenerate zero-span
# segment in the spline.
_NODE_MERGE_ANG = math.radians(8.0)


def _snap_circle_nodes(params: Dict, targets) -> Dict:
    """Convert an OVERLAPPING circle into an explicit node-spline ring.

    A standalone circle stays a clean parametric primitive. But once it overlaps
    another shape it must pass EXACTLY through the shared convergence points,
    which can sit anywhere on the rim -- not just on the E/S/W/N axes the 4-anchor
    spline exposes. So we build a node ring = the four cardinal anchors PLUS one
    node placed exactly on each target, all sorted by angle around the centre.
    Between nodes the curve keeps circular tangents (see
    ``circle_nodes_spline_d``), so it stays round except where it is pinned to a
    junction. Cardinal nodes whose angle nearly coincides with a target are
    dropped in favour of the target (no degenerate segments).
    """
    cx, cy, r = params["cx"], params["cy"], params["r"]
    # Base ring: any nodes already pinned by a previous snap, else the four
    # cardinal anchors. This lets the circle accumulate targets from multiple
    # passes (e.g. 3-color junctions AND overlapping-shape corners) instead of
    # each pass overwriting the last.
    existing = params.get("spline_nodes")
    if existing:
        base = [(float(x), float(y)) for x, y in existing]
    else:
        offs = params.get("anchor_offsets") or [0.0, 0.0, 0.0, 0.0]
        base = [
            (cx + (r + offs[0]), cy),          # E
            (cx, cy + (r + offs[1])),          # S
            (cx - (r + offs[2]), cy),          # W
            (cx, cy - (r + offs[3])),          # N
        ]
    tgt_ang = [math.atan2(ty - cy, tx - cx) for tx, ty in targets]
    kept_base = []
    for (nx, ny) in base:
        na = math.atan2(ny - cy, nx - cx)
        if any(abs(_ang_diff(na, ta)) <= _NODE_MERGE_ANG for ta in tgt_ang):
            continue  # a target sits right here; the target node replaces it
        kept_base.append((nx, ny))
    nodes = kept_base + [(float(tx), float(ty)) for tx, ty in targets]
    if len(nodes) < 3:
        return params
    nodes.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    params["spline_nodes"] = [[round(x, 2), round(y, 2)] for x, y in nodes]
    params["as_spline"] = True
    return params


def _ang_diff(a: float, b: float) -> float:
    """Smallest signed difference between two angles (radians)."""
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def _snap_ellipse(params: Dict, px: float, py: float) -> Dict:
    cx, cy = params["cx"], params["cy"]
    rx, ry = params["rx"], params["ry"]
    dx, dy = px - cx, py - cy
    # normalised distance along the ellipse (approx); shift centre to reach point
    ang = math.atan2(dy, dx)
    ex = rx * math.cos(ang)
    ey = ry * math.sin(ang)
    d = math.hypot(ex, ey)
    if d < 1e-6:
        return params
    shift = (1.0 - d / math.hypot(dx, dy))
    params["cx"] = round(cx + dx * shift, 2)
    params["cy"] = round(cy + dy * shift, 2)
    return params


def _snap_rect(params: Dict, px: float, py: float) -> Dict:
    x, y = params["x"], params["y"]
    x2, y2 = x + params["w"], y + params["h"]
    # grow the bounding box on whichever edge is nearest the point
    if abs(px - x) <= abs(px - x2):
        x = min(x, px)
    else:
        x2 = max(x2, px)
    if abs(py - y) <= abs(py - y2):
        y = min(y, py)
    else:
        y2 = max(y2, py)
    params["x"] = round(x, 2)
    params["y"] = round(y, 2)
    params["w"] = round(x2 - x, 2)
    params["h"] = round(y2 - y, 2)
    return params


def snap_shape_to_point(node: Dict, targets) -> Dict:
    """Deform a single shape node so its boundary passes through `targets`.

    `targets` is a list of (x, y) points the shape should reach (typically the
    collision points converging on this shape). Snapping is type-aware:

      * polygon / star / triangle / bezier -> vertices are pulled onto the
        targets (aggressive; these are traced boundaries).
      * circle / ellipse / rect / rounded_rect -> only snapped when a target is
        already near the ideal boundary (within `prim_tol` px). Ideal primitives
        are not dragged across the image to reach a distant quantized corner,
        which would destroy their clean form.
    """
    t = node["type"]
    p = node.setdefault("params", {})
    prim_tol = 8.0
    if t in ("polygon", "star", "triangle", "bezier"):
        p["points"] = _snap_polygon_points(p.get("points", []), targets)
    elif t == "circle" and targets:
        # overlapping circle: gather targets near the rim and rebuild the boundary
        # as an explicit node-spline that passes through EACH of them (centre &
        # radius otherwise preserved). A standalone circle reaches here with no
        # near-rim targets and stays the clean parametric primitive.
        near = [(tx, ty) for tx, ty in targets
                if abs(math.hypot(tx - p["cx"], ty - p["cy"]) - p["r"]) <= prim_tol]
        if near:
            _snap_circle_nodes(p, near)
    elif t == "ellipse" and targets:
        near = False
        for tx, ty in targets:
            ang = math.atan2(ty - p["cy"], tx - p["cx"])
            r = math.hypot(p["rx"] * math.cos(ang), p["ry"] * math.sin(ang))
            if abs(math.hypot(tx - p["cx"], ty - p["cy"]) - r) <= prim_tol:
                near = True
                break
        if near:
            _snap_ellipse(p, *targets[0])
    elif t in ("rect", "rounded_rect") and targets:
        # only grow the rect for targets actually on/near its ideal boundary
        x, y, x2, y2 = p["x"], p["y"], p["x"] + p["w"], p["y"] + p["h"]
        for tx, ty in targets:
            on_edge = (abs(tx - x) <= prim_tol or abs(tx - x2) <= prim_tol or
                       abs(ty - y) <= prim_tol or abs(ty - y2) <= prim_tol)
            if on_edge:
                _snap_rect(p, tx, ty)
    node["snapped"] = True
    return node


# Parametric primitives whose fit is this good are already flawless. For types
# whose snap MOVES THE CENTRE (ellipse/rect/rounded_rect) that would shift the
# whole curve ~0.5px and open gaps against neighbouring traced shapes, so a
# near-perfect one is exempt -- keep its aligned centre and let neighbours meet
# it. Circles are NOT exempt: an overlapping circle is rebuilt as a node-spline
# pinned to its convergence points (centre/radius otherwise preserved), which
# only bulges locally toward each junction and leaves the rest of the rim round.
_HIGH_FIT_IOU = 0.95
_CENTRE_MOVE_PARAMETRIC = ("ellipse", "rect", "rounded_rect")


def snap_shapes_to_collisions(shapes: List[Dict], collisions: List[Dict],
                              tolerance: float = 8.0) -> List[Dict]:
    """Snap each converging shape's boundary onto the collision points.

    A collision point snaps a shape only if it lies within `tolerance` pixels of
    that shape's *actual boundary* (not just its bbox) — so a small island is not
    dragged onto collisions that belong to a neighbouring shape sharing its
    color/footprint. Targets are grouped per shape and snapped in one stable
    pass. High-fit parametric primitives are exempt (see ``_HIGH_FIT_IOU``).
    Returns the shapes.
    """
    by_color = {}
    for s in shapes:
        by_color.setdefault(tuple(s.get("color", [])), []).append(s)

    targets_per_shape: Dict[int, List] = {}
    for c in collisions:
        n = c.get("n_colors", 1)
        if n < 3:
            # 2-color edges are just where two shapes touch. Snapping both shapes
            # to every point along such an edge drags a straight edge onto the
            # neighbour's curve (e.g. the red star's straight side gets bent to
            # follow the yellow circle). These are contact markers, not corners —
            # leave them out of the deformation.
            continue
        for col in c.get("colors", []):
            for s in by_color.get(tuple(col), []):
                if s.get("is_background"):
                    continue
                if (s.get("type") in _CENTRE_MOVE_PARAMETRIC
                        and (s.get("fit_iou") or 0.0) >= _HIGH_FIT_IOU):
                    # flawless centre-move primitive -> don't deform (would shift
                    # the whole curve). Circles snap locally, so they pass.
                    continue
                if _boundary_distance(s, c["x"], c["y"]) <= tolerance:
                    targets_per_shape.setdefault(id(s), []).append((c["x"], c["y"]))

    for s in shapes:
        tg = targets_per_shape.get(id(s))
        if tg:
            snap_shape_to_point(s, tg)
    return shapes


def snap_circles_to_overlap_corners(shapes: List[Dict],
                                    masks: Dict[int, "object"],
                                    rim_tol: float = 3.0) -> List[Dict]:
    """Pin an overlapping circle's rim to the CORNERS of shapes it overlaps.

    A circle intersecting another shape shares only a *2-colour* contact edge
    with it (no third colour), so the collision snapper -- which deliberately
    ignores 2-colour edges -- never pulls the circle onto that neighbour. But the
    neighbour's genuine CORNERS (its validated contour curvature maxima) that fall
    right on the circle should visibly touch the rim.

    For each circle, this finds every other shape whose bbox overlaps the circle,
    computes that shape's contour corners, and for any corner lying within
    ``rim_tol`` px of the circle's ideal boundary, adds a spline node placed
    exactly on that corner. This is corner-driven (independent of collision
    colour count) and reuses ``_snap_circle_nodes`` so nodes accumulate with any
    junction nodes already pinned. Circle centre & radius are preserved; the rim
    only bulges locally to meet each corner.
    """
    if not masks:
        return shapes
    from .detectors.contour_detector import contour_vertices
    from .detectors.edge_detector import repair_mask
    from .corners import contour_corners

    circles = [s for s in shapes
               if s.get("type") == "circle" and not s.get("is_background")]
    if not circles:
        return shapes

    # cache each shape's contour corners (computed once)
    corner_cache: Dict[int, List] = {}

    def corners_of(shape) -> List:
        key = id(shape)
        if key in corner_cache:
            return corner_cache[key]
        m = masks.get(key)
        pts: List = []
        if m is not None:
            ring = contour_vertices(repair_mask(m), fill_holes=True)
            if ring:
                pts = list(contour_corners(ring))
        corner_cache[key] = pts
        return pts

    for circ in circles:
        p = circ["params"]
        cx, cy, r = p["cx"], p["cy"], p["r"]
        cfp = _footprint(circ)
        targets = []
        for other in shapes:
            if other is circ or other.get("is_background"):
                continue
            ofp = _footprint(other)
            if not ofp or not cfp or not _bbox_overlap(cfp, ofp):
                continue
            for (x, y) in corners_of(other):
                if abs(math.hypot(x - cx, y - cy) - r) <= rim_tol:
                    targets.append((float(x), float(y)))
        if targets:
            _snap_circle_nodes(p, targets)
            circ["snapped"] = True
    return shapes


def _bbox_overlap(a, b) -> bool:
    """True if two (x0,y0,x1,y1) boxes overlap."""
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _boundary_distance(node: Dict, px: float, py: float) -> float:
    """Approx distance from a point to a shape's boundary (used for the gate)."""
    p = node.get("params", {})
    t = node["type"]
    if t in ("polygon", "star", "triangle", "bezier") and p.get("points"):
        return min(math.hypot(x - px, y - py) for x, y in p["points"])
    if t == "circle":
        return abs(math.hypot(px - p["cx"], py - p["cy"]) - p["r"])
    if t == "ellipse":
        dx, dy = px - p["cx"], py - p["cy"]
        ang = math.atan2(dy, dx)
        r = math.hypot(p["rx"] * math.cos(ang), p["ry"] * math.sin(ang))
        return abs(math.hypot(dx, dy) - r)
    if t in ("rect", "rounded_rect"):
        x, y, x2, y2 = p["x"], p["y"], p["x"] + p["w"], p["y"] + p["h"]
        if x <= px <= x2 and y <= py <= y2:
            return 0.0
        dx = max(x - px, 0, px - x2)
        dy = max(y - py, 0, py - y2)
        return math.hypot(dx, dy)
    b = node.get("bbox")
    if b:
        x, y, x2, y2 = b
        if x <= px <= x2 and y <= py <= y2:
            return 0.0
        dx = max(x - px, 0, px - x2)
        dy = max(y - py, 0, py - y2)
        return math.hypot(dx, dy)
    return float("inf")


def _footprint(node: Dict):
    """Return (x0, y0, x1, y1) bounding box of a shape node."""
    p = node.get("params", {})
    t = node["type"]
    if t in ("polygon", "star", "triangle", "bezier") and p.get("points"):
        xs = [x for x, _ in p["points"]]
        ys = [y for _, y in p["points"]]
        return min(xs), min(ys), max(xs), max(ys)
    if t in ("circle", "ellipse"):
        if t == "circle":
            rx = ry = p["r"]
        else:
            rx, ry = p["rx"], p["ry"]
        return p["cx"] - rx, p["cy"] - ry, p["cx"] + rx, p["cy"] + ry
    if t in ("rect", "rounded_rect"):
        return p["x"], p["y"], p["x"] + p["w"], p["y"] + p["h"]
    b = node.get("bbox")
    if b:
        return tuple(b)
    return None


def _within_footprint(node: Dict, px: float, py: float, margin: float = 8.0) -> bool:
    """True if (px, py) is inside the shape's bbox (or within `margin`)."""
    fp = _footprint(node)
    if not fp:
        return False
    x0, y0, x1, y1 = fp
    return (x0 - margin) <= px <= (x1 + margin) and (y0 - margin) <= py <= (y1 + margin)
