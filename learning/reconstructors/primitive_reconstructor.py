"""Primitive reconstructor: emit ideal SVG path `d` for solid primitives.

Handles point, circle, ellipse, arc, line, rect, rounded_rect. Polygon-style
primitives (triangle/polygon/star) and bezier are handled by their dedicated
reconstructors in this package. Composites are expanded by emitting each child.
"""

from __future__ import annotations

import math
from typing import Dict


def point_d(params: Dict) -> str:
    r = params.get("r", 1.5)
    cx, cy = params["x"], params["y"]
    return (f"M{cx - r:.1f},{cy:.1f} a{r:.1f},{r:.1f} 0 1,0 {2*r:.1f},0 "
            f"a{r:.1f},{r:.1f} 0 1,0 {-2*r:.1f},0 Z")


# Cubic-Bezier circle constant: control-handle length as a fraction of the
# radius that best approximates a quarter circle (max radial error ~0.027%).
_KAPPA = 0.5522847498307936


def circle_spline_d(params: Dict) -> str:
    """Circle as a CLOSED 4-node cubic Bezier (anchors at E, S, W, N).

    Four anchor points sit on the axes; each quarter is one cubic segment with
    tangential handles of length ``kappa*r``. Unlike the SVG ``a`` arc this
    exposes 4 editable anchors, so ``params['anchor_offsets']`` (a 4-list of
    radial deltas for [E, S, W, N]) can push individual anchors toward a
    junction/clip edge WITHOUT moving the centre -- closing a local gap while the
    rest of the circle stays put. Handle lengths scale with each anchor's own
    (possibly offset) radius so the curve stays smooth. Offsets default to 0
    (a perfect circle identical to the arc form to within kappa error).
    """
    cx, cy = params["cx"], params["cy"]
    r = params["r"]
    offs = params.get("anchor_offsets") or [0.0, 0.0, 0.0, 0.0]
    # anchors E, S, W, N (clockwise in SVG's y-down space) with per-anchor radius
    rE, rS, rW, rN = (r + offs[0], r + offs[1], r + offs[2], r + offs[3])
    E = (cx + rE, cy)
    S = (cx, cy + rS)
    W = (cx - rW, cy)
    N = (cx, cy - rN)
    # per-segment handle length uses the mean radius of the segment's endpoints
    def h(ra, rb):
        return _KAPPA * (ra + rb) / 2.0
    hES, hSW, hWN, hNE = h(rE, rS), h(rS, rW), h(rW, rN), h(rN, rE)
    d = (f"M{E[0]:.2f},{E[1]:.2f} "
         # E -> S (handles: down from E, right from S)
         f"C{E[0]:.2f},{E[1] + hES:.2f} {S[0] + hES:.2f},{S[1]:.2f} {S[0]:.2f},{S[1]:.2f} "
         # S -> W (handles: left from S, down from W)
         f"C{S[0] - hSW:.2f},{S[1]:.2f} {W[0]:.2f},{W[1] + hSW:.2f} {W[0]:.2f},{W[1]:.2f} "
         # W -> N (handles: up from W, left from N)
         f"C{W[0]:.2f},{W[1] - hWN:.2f} {N[0] - hWN:.2f},{N[1]:.2f} {N[0]:.2f},{N[1]:.2f} "
         # N -> E (handles: right from N, up from E)
         f"C{N[0] + hNE:.2f},{N[1]:.2f} {E[0]:.2f},{E[1] - hNE:.2f} {E[0]:.2f},{E[1]:.2f} "
         "Z")
    return d


def circle_nodes_spline_d(params: Dict) -> str:
    """Circle/curve as a closed cubic Bezier through an ARBITRARY node ring.

    Used when a circle OVERLAPS another shape: it can no longer stay a clean
    parametric primitive because its boundary must pass exactly through the shared
    convergence points, which may sit anywhere (not just on the cardinal axes).

    ``params['spline_nodes']`` is a list of ``[x, y]`` anchor points already in
    angular order around the centre. Consecutive nodes are joined by a cubic
    segment whose handles are tangential to the circle at each node (perpendicular
    to the centre->node radius), with length ``kappa`` scaled by the actual arc
    span between the pair (``4/3 * tan(dtheta/4) * r``) so uneven spacing still
    reads as a smooth round arc. This keeps the shape circular between junctions
    while pinning it EXACTLY onto every convergence node.
    """
    nodes = params.get("spline_nodes") or []
    if len(nodes) < 3:
        return circle_spline_d(params)
    cx, cy = params["cx"], params["cy"]
    n = len(nodes)
    # per-node outward radius and unit tangent (CW in y-down space: (-dy, dx)->
    # we take (dy, -dx)? choose direction consistent with node ordering below)
    ang = [math.atan2(y - cy, x - cx) for x, y in nodes]
    d = f"M{nodes[0][0]:.2f},{nodes[0][1]:.2f} "
    for i in range(n):
        x0, y0 = nodes[i]
        x1, y1 = nodes[(i + 1) % n]
        r0 = math.hypot(x0 - cx, y0 - cy) or 1e-6
        r1 = math.hypot(x1 - cx, y1 - cy) or 1e-6
        # signed angular span from node i to i+1 (wrap to a positive small step)
        dtheta = ang[(i + 1) % n] - ang[i]
        while dtheta <= 0:
            dtheta += 2 * math.pi
        while dtheta > 2 * math.pi:
            dtheta -= 2 * math.pi
        k = (4.0 / 3.0) * math.tan(dtheta / 4.0)
        # tangent direction = derivative of (cos,sin) at each angle, scaled by r*k
        t0x, t0y = -math.sin(ang[i]), math.cos(ang[i])
        t1x, t1y = -math.sin(ang[(i + 1) % n]), math.cos(ang[(i + 1) % n])
        c1x, c1y = x0 + t0x * r0 * k, y0 + t0y * r0 * k
        c2x, c2y = x1 - t1x * r1 * k, y1 - t1y * r1 * k
        d += (f"C{c1x:.2f},{c1y:.2f} {c2x:.2f},{c2y:.2f} "
              f"{x1:.2f},{y1:.2f} ")
    return d + "Z"


def circle_d(params: Dict) -> str:
    # Circles are emitted as a closed 4-node cubic spline so their N/E/S/W
    # anchors are individually editable/snappable. Pass as_spline=False for the
    # legacy SVG arc form. When overlapping another shape the circle carries an
    # explicit ``spline_nodes`` ring that also passes through convergence points.
    if params.get("spline_nodes"):
        return circle_nodes_spline_d(params)
    if params.get("as_spline", True):
        return circle_spline_d(params)
    r = params["r"]
    cx, cy = params["cx"], params["cy"]
    return (f"M{cx - r:.1f},{cy:.1f} "
            f"a{r:.1f},{r:.1f} 0 1,0 {2*r:.1f},0 "
            f"a{r:.1f},{r:.1f} 0 1,0 {-2*r:.1f},0 Z")


def ellipse_d(params: Dict) -> str:
    rx, ry = params["rx"], params["ry"]
    cx, cy = params["cx"], params["cy"]
    return (f"M{cx - rx:.1f},{cy:.1f} "
            f"a{rx:.1f},{ry:.1f} 0 1,0 {2*rx:.1f},0 "
            f"a{rx:.1f},{ry:.1f} 0 1,0 {-2*rx:.1f},0 Z")


def arc_d(params: Dict) -> str:
    cx, cy, rx, ry = params["cx"], params["cy"], params["rx"], params["ry"]
    a0 = params.get("start", 0.0)
    a1 = params.get("end", 2 * math.pi)
    x0, y0 = cx + rx * math.cos(a0), cy + ry * math.sin(a0)
    x1, y1 = cx + rx * math.cos(a1), cy + ry * math.sin(a1)
    large = 1 if abs(a1 - a0) > math.pi else 0
    sweep = 1 if a1 > a0 else 0
    return (f"M{x0:.1f},{y0:.1f} "
            f"A{rx:.1f},{ry:.1f} 0 {large},{sweep} {x1:.1f},{y1:.1f}")


def line_d(params: Dict) -> str:
    return f"M{params['x0']:.1f},{params['y0']:.1f} L{params['x1']:.1f},{params['y1']:.1f}"


def rect_d(params: Dict) -> str:
    x, y, w, h = params["x"], params["y"], params["w"], params["h"]
    return (f"M{x:.1f},{y:.1f} "
            f"L{x + w:.1f},{y:.1f} "
            f"L{x + w:.1f},{y + h:.1f} "
            f"L{x:.1f},{y + h:.1f} Z")


def rounded_rect_d(params: Dict) -> str:
    from ..generators.rectangle import rounded_rectangle_points
    x, y, w, h = params["x"], params["y"], params["w"], params["h"]
    r = min(params.get("r", 0), w / 2, h / 2)
    pts = rounded_rectangle_points(x, y, x + w, y + h, r)
    d = "M" + " L".join(f"{px:.1f},{py:.1f}" for px, py in pts) + " Z"
    return d


def reconstruct(node: Dict) -> str:
    """Dispatch a primitive node to its `d` builder."""
    t = node["type"]
    p = node["params"]
    if t == "point":
        return point_d(p)
    if t == "circle":
        return circle_d(p)
    if t == "ellipse":
        return ellipse_d(p)
    if t == "arc":
        return arc_d(p)
    if t == "line":
        return line_d(p)
    if t == "rect":
        return rect_d(p)
    if t == "rounded_rect":
        return rounded_rect_d(p)
    if t == "lens":
        from ..circle_boolean import lens_d
        return lens_d(p)
    if t == "lune":
        from ..circle_boolean import lune_d
        return lune_d(p)
    if t == "occluded_circle":
        from ..occluded_circle import occluded_circle_d
        return occluded_circle_d(p)
    return ""
