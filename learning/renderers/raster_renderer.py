"""Raster renderer: rasterise vector primitives/polygons into boolean masks.

Provides the shared rasterisation backend used by IoU/overlap metrics and by the
shape-to-points conversion. Uses PIL's polygon fill, which matches the SVG fill
semantics.
"""

from __future__ import annotations

from typing import List, Tuple

import math
import numpy as np
from PIL import Image, ImageDraw

Point = Tuple[float, float]


def rasterize_polygon(points: List[Point], w: int, h: int) -> np.ndarray:
    """Rasterise a polygon (list of (x, y)) into a boolean mask of shape (h, w)."""
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    d.polygon([(float(x), float(y)) for x, y in points], fill=255)
    return np.asarray(img) > 0


def _apply_transform(points: List[Point], transform: dict) -> List[Point]:
    """Apply a node Transform (translate/rotate/scale/mirror) to a point list.

    Mirrors the order used by ``svg_renderer.transform_to_svg`` so a rasterised
    footprint lands where the SVG actually draws it. Without this, a rotated
    primitive (e.g. a rotated ellipse whose params sit in its un-rotated local
    frame) would be rasterised at the wrong location, corrupting overlap/IoU
    metrics and spuriously colliding with unrelated shapes.
    """
    if not transform:
        return points
    pts = [(float(x), float(y)) for x, y in points]
    if "translate" in transform:
        tx, ty = transform["translate"]
        pts = [(x + tx, y + ty) for x, y in pts]
    if "rotate" in transform:
        ang = math.radians(transform["rotate"])
        cx = transform.get("rotate_cx", 0.0)
        cy = transform.get("rotate_cy", 0.0)
        ca, sa = math.cos(ang), math.sin(ang)
        pts = [((x - cx) * ca - (y - cy) * sa + cx,
                (x - cx) * sa + (y - cy) * ca + cy) for x, y in pts]
    if "scale" in transform:
        sx, sy = transform["scale"]
        pts = [(x * sx, y * sy) for x, y in pts]
    if transform.get("mirror") == "x":
        pts = [(-x, y) for x, y in pts]
    elif transform.get("mirror") == "y":
        pts = [(x, -y) for x, y in pts]
    return pts


def rasterize_node(node: dict, w: int, h: int) -> np.ndarray:
    """Rasterise a primitive/composite node to a boolean mask.

    Delegates to node->points conversion. Any Transform on the node
    (translate/rotate/scale/mirror) is applied so the footprint matches the SVG
    draw position. For composite nodes the boolean op is realised by the caller
    (svg_renderer) via fill-rule; here we rasterise the union of children as a
    conservative footprint.
    """
    from ..generators import (
        circle_points, ellipse_points, rectangle_points,
        rounded_rectangle_points, polygon_points,
    )
    t = node["type"]
    p = node.get("params", {})
    if t == "circle":
        pts = circle_points(p["cx"], p["cy"], p["r"])
    elif t == "ellipse":
        pts = ellipse_points(p["cx"], p["cy"], p["rx"], p["ry"])
    elif t == "rect":
        pts = rectangle_points(p["x"], p["y"], p["x"] + p["w"], p["y"] + p["h"])
    elif t == "rounded_rect":
        pts = rounded_rectangle_points(p["x"], p["y"], p["x"] + p["w"],
                                      p["y"] + p["h"], p.get("r", 0))
    elif t == "point":
        pts = circle_points(p["x"], p["y"], p.get("r", 1.5), n=16)
    elif t == "arc":
        a0, a1 = p.get("start", 0.0), p.get("end", 2 * math.pi)
        n = max(8, int(abs(a1 - a0) / (math.pi / 16)))
        pts = [(p["cx"] + p["rx"] * math.cos(a0 + (a1 - a0) * i / n),
                p["cy"] + p["ry"] * math.sin(a0 + (a1 - a0) * i / n))
               for i in range(n + 1)]
    elif t == "lens":
        c0 = p["circles"][0]
        c1 = p["circles"][1]
        yy, xx = np.mgrid[0:h, 0:w]
        m0 = (xx - c0[0]) ** 2 + (yy - c0[1]) ** 2 <= c0[2] ** 2
        m1 = (xx - c1[0]) ** 2 + (yy - c1[1]) ** 2 <= c1[2] ** 2
        return np.logical_and(m0, m1)
    elif t == "lune":
        c = p["circle"]
        cut = p["cut"]
        yy, xx = np.mgrid[0:h, 0:w]
        mc = (xx - c[0]) ** 2 + (yy - c[1]) ** 2 <= c[2] ** 2
        mk = (xx - cut[0]) ** 2 + (yy - cut[1]) ** 2 <= cut[2] ** 2
        return np.logical_and(mc, np.logical_not(mk))
    elif t == "occluded_circle":
        pts = polygon_points(p.get("points", []))
    elif t in ("triangle", "polygon", "star", "bezier"):
        pts = polygon_points(p.get("points", []))
    elif t in ("union", "difference", "intersection", "xor"):
        masks = [rasterize_node(c, w, h) for c in node.get("children", [])]
        if not masks:
            return np.zeros((h, w), bool)
        out = masks[0].copy()
        for m in masks[1:]:
            out = np.logical_or(out, m)
        return out
    else:
        return np.zeros((h, w), bool)
    pts = _apply_transform(pts, node.get("transform"))
    return rasterize_polygon(pts, w, h)
