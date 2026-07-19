"""Composite construction: turn detected overlaps into composite nodes.

Given the foreground shape indices and the pairwise overlap records produced by
:func:`learning.learner.detect_overlaps`, this module builds the composite node
hierarchy (difference / union) and annotates each overlap with the applied op and
the bottom/top shape indices.
"""

from __future__ import annotations

from typing import Dict, List


def _curve_quality(node: Dict) -> float:
    """Higher = cleaner / more ideal primitive.

    Prefers parametric primitives (circle/ellipse/rect...) over traced polygons,
    and uses fit_iou as a tiebreak. Used to decide which shape is "on top" so a
    perfect curve wins when two shapes intersect.
    """
    t = node.get("type", "")
    base = 1.0 if t in ("circle", "ellipse", "rect", "rounded_rect",
                         "triangle", "arc", "line", "point", "bezier", "star") else 0.0
    iou = node.get("fit_iou") or 0.0
    return base + iou


def build_composites(shapes: List[Dict], fg: List[int],
                     overlaps: List[Dict]) -> List[Dict]:
    """Build composite nodes from overlap records.

    `fg` maps local overlap indices (0..len(fg)-1) back to global `shapes`
    indices. Each overlap gains ``applied``, ``op``, ``bottom``, ``top`` fields.
    Returns a list of composite node dicts.

    For ``difference`` (intersecting shapes), the shape with the cleaner curve
    (higher :func:`_curve_quality`) is placed on top / made visible, so perfect
    circles/ellipses win over traced polygons when regions merge.
    """
    composites: List[Dict] = []
    for o in overlaps:
        a = fg[o["a"]]
        b = fg[o["b"]]
        o["a"] = a
        o["b"] = b
        if o["mode"] == "keep_topmost":
            op = "difference"
        else:
            op = "union"
        # bottom = lower layer index (drawn first / underneath)
        bottom, top = min(a, b), max(a, b)
        if op == "difference":
            # promote the cleaner curve to the visible/top slot
            if _curve_quality(shapes[top]) < _curve_quality(shapes[bottom]):
                bottom, top = top, bottom
        o["applied"] = op
        o["op"] = op
        o["bottom"] = bottom
        o["top"] = top
        # The visible fill color: for difference the topmost shape punches the
        # hole and is drawn on top (its color shows), for union the bottom/base
        # shape defines the fill.
        visible_idx = top if op == "difference" else bottom
        # Tag the exact shapes owned by this composite so the renderer can skip
        # their standalone re-draw by IDENTITY -- not by layer_index, which
        # disjoint islands (e.g. a clipped star's small fragment) may share.
        shapes[bottom]["in_composite"] = True
        shapes[top]["in_composite"] = True
        composites.append({
            "kind": "composite",
            "type": op,
            "children": [shapes[bottom], shapes[top]],
            "visible": visible_idx,
            "bbox": shapes[bottom].get("bbox"),
            "fill_rule": "evenodd" if op == "difference" else "nonzero",
        })
    return composites
