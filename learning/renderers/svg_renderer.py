"""SVG renderer: turn a learned model (node hierarchy) into an SVG document.

Primitives are emitted as individual <g> groups (z-order = list order) with their
reconstructed `d` path. Composite nodes are realised via child paths plus the
parent `fill-rule` (evenodd for difference). Transform nodes are emitted as
`transform=` attributes.
"""

from __future__ import annotations

import math
from typing import Dict, Optional


def transform_to_svg(transform: Optional[Dict]) -> str:
    """Render a Transform node as an SVG transform attribute string."""
    if not transform:
        return ""
    parts = []
    if "translate" in transform:
        tx, ty = transform["translate"]
        parts.append(f"translate({tx} {ty})")
    if "rotate" in transform:
        ang = transform["rotate"]
        cx = transform.get("rotate_cx", 0)
        cy = transform.get("rotate_cy", 0)
        parts.append(f"rotate({ang} {cx} {cy})")
    if "scale" in transform:
        sx, sy = transform["scale"]
        parts.append(f"scale({sx} {sy})")
    if "skew" in transform:
        kx, ky = transform["skew"]
        parts.append(f"skewX({kx}) skewY({ky})")
    if transform.get("mirror") == "x":
        parts.append("matrix(-1 0 0 1 0 0)")
    elif transform.get("mirror") == "y":
        parts.append("matrix(1 0 0 -1 0 0)")
    return " ".join(parts)


def _node_d(node: Dict) -> str:
    """Reconstruct the `d` string for any node via the reconstructor package."""
    from ..reconstructors import (
        reconstruct_primitive,
        reconstruct_polygon,
        reconstruct_bezier,
    )
    t = node["type"]
    if t in ("union", "difference", "intersection", "xor"):
        return " ".join(_node_d(c) for c in node.get("children", []))
    if t in ("triangle", "polygon", "star"):
        return reconstruct_polygon(node)
    if t == "bezier":
        return reconstruct_bezier(node)
    return reconstruct_primitive(node)


def model_to_svg(model: Dict, background: bool = True) -> str:
    """Render a learned model back to a layered SVG using rebuilt primitives."""
    w, h = model["width"], model["height"]
    svg = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
           f'xmlns="http://www.w3.org/2000/svg">']
    svg.append('<metadata>XVGTracer learned model: '
               'identify -> rebuild -> postprocess '
               '(primitive/composite/transform)</metadata>')
    if background and model["shapes"]:
        bg = model["shapes"][0]["color"]
        svg.append(f'<rect width="100%" height="100%" fill="rgb{tuple(bg)}" />')

    composite_members = _composite_member_ids(model)
    for s in model["shapes"]:
        li = s.get("layer_index", 0)
        # Skip standalone re-draw of shapes that are wholly owned by a composite;
        # the composite block renders them (with correct colour) instead. This
        # avoids a double draw where the composite's flat group colour would
        # otherwise cover the shape's own colour. Ownership is decided PER SHAPE
        # (identity / an ``in_composite`` tag), never by layer_index: multiple
        # disjoint islands can share a layer_index (e.g. a star split into a big
        # body + a small clipped fragment), and only the island actually inside
        # the composite must be skipped -- the other islands still need their own
        # standalone draw.
        if s.get("in_composite") or id(s) in composite_members:
            continue
        fill = f"rgb({s['color'][0]},{s['color'][1]},{s['color'][2]})"
        tr = transform_to_svg(s.get("transform"))
        tr_attr = f' transform="{tr}"' if tr else ""
        svg.append(f'<g id="shape_{li}" data-type="{s["type"]}" '
                   f'data-kind="{s.get("kind", "primitive")}" '
                   f'class="xvg-shape" fill="{fill}" '
                   f'fill-rule="nonzero"{tr_attr}>')
        svg.append(f'  <path d="{s["rebuilt_d"]}" />')
        svg.append("</g>")

    for k, c in enumerate(model.get("composites", [])):
        # Render each child as its own coloured, z-ordered path (bottom first,
        # top last). The top shape overpaints, so both colours stay visible and
        # the overlap shows the top colour -- the natural "union"/"difference"
        # look. No flat group colour, no evenodd hole-to-background trick.
        svg.append(f'<g id="composite_{k}" data-type="{c["type"]}" '
                   f'class="xvg-composite">')
        for ch in c["children"]:
            fill = f"rgb({ch['color'][0]},{ch['color'][1]},{ch['color'][2]})"
            d = _node_d(ch)
            tr = transform_to_svg(ch.get("transform"))
            tr_attr = f' transform="{tr}"' if tr else ""
            svg.append(f'  <path d="{d}" fill="{fill}" '
                       f'fill-rule="nonzero"{tr_attr} />')
        svg.append("</g>")

    # Collision points: hollow-circle markers at color-region junctions. About
    # 3px quantized diameter (r=1.5), drawn on top. 3+ color convergences are
    # red, 2-color edges blue. Hollow (no fill) so they read as junction markers.
    collisions = model.get("collisions")
    if collisions:
        svg.append('<g id="collisions" class="xvg-collisions" '
                   'fill="none" stroke-width="0.2">')
        for c in collisions:
            n = c.get("n_colors", 3)
            stroke = "rgb(220,30,30)" if n >= 3 else "rgb(30,120,220)"
            cx, cy = c["x"], c["y"]
            svg.append(f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="1.5" '
                       f'stroke="{stroke}" data-n-colors="{n}" '
                       f'data-colors="{c.get("colors", [])}" />')
        svg.append("</g>")

    # Construction lines: straight edges of non-primitive (custom) shapes,
    # extended a few px past both ends, plus convergence markers where extended
    # edges meet. Dashed thin green lines; convergence = small hollow square.
    construction = model.get("construction")
    if construction:
        svg.append('<g id="construction" class="xvg-construction" '
                   'fill="none" stroke="rgb(20,160,90)" stroke-width="0.2" '
                   'stroke-dasharray="2,1.5">')
        for c in construction:
            if c.get("kind") == "line":
                svg.append(f'  <line x1="{c["x1"]:.1f}" y1="{c["y1"]:.1f}" '
                           f'x2="{c["x2"]:.1f}" y2="{c["y2"]:.1f}" '
                           f'data-layer="{c.get("layer_index", 0)}" />')
            elif c.get("kind") == "convergence":
                x, y = c["x"], c["y"]
                svg.append(f'  <rect x="{x - 1.5:.1f}" y="{y - 1.5:.1f}" '
                           f'width="3" height="3" stroke="rgb(200,120,20)" '
                           f'stroke-dasharray="none" '
                           f'data-layer="{c.get("layer_index", 0)}" />')
            elif c.get("kind") == "spline":
                # spline skeleton: node-to-node guide chord for a node-spline
                # (e.g. an overlapping circle). Solid teal, distinct from the
                # dashed green polygon extension lines.
                svg.append(f'  <line x1="{c["x1"]:.1f}" y1="{c["y1"]:.1f}" '
                           f'x2="{c["x2"]:.1f}" y2="{c["y2"]:.1f}" '
                           f'stroke="rgb(0,150,170)" stroke-dasharray="none" '
                           f'data-layer="{c.get("layer_index", 0)}" />')
            elif c.get("kind") == "node":
                # spline anchor node marker: small hollow diamond.
                x, y = c["x"], c["y"]
                svg.append(f'  <path d="M{x:.1f},{y - 1.6:.1f} '
                           f'L{x + 1.6:.1f},{y:.1f} L{x:.1f},{y + 1.6:.1f} '
                           f'L{x - 1.6:.1f},{y:.1f} Z" stroke="rgb(0,150,170)" '
                           f'stroke-dasharray="none" '
                           f'data-layer="{c.get("layer_index", 0)}" />')
        svg.append("</g>")

    # Corner sources of truth, drawn as SEPARATE labeled overlays so each source
    # can be compared independently (no merging). Distinct shapes/colors:
    #   contour curvature maxima -> magenta cross (+)
    #   extended-side apex        -> purple diamond, with a hair line to the
    #                                traced vertex it corrects
    corners = model.get("corners") or {}
    cc = corners.get("contour") or []
    ce = corners.get("extended") or []
    if cc or ce:
        svg.append('<g id="corners" class="xvg-corners" fill="none">')
        if cc:
            svg.append('  <g id="corners-contour" stroke="rgb(220,20,180)" '
                       'stroke-width="0.25">')
            for c in cc:
                x, y = c["x"], c["y"]
                svg.append(f'    <path d="M{x - 2:.1f},{y:.1f} L{x + 2:.1f},{y:.1f} '
                           f'M{x:.1f},{y - 2:.1f} L{x:.1f},{y + 2:.1f}" '
                           f'data-layer="{c.get("layer_index", 0)}" />')
            svg.append("  </g>")
        if ce:
            svg.append('  <g id="corners-extended" stroke="rgb(130,40,200)" '
                       'stroke-width="0.25">')
            for c in ce:
                x, y = c["x"], c["y"]
                tx, ty = c.get("traced_x", x), c.get("traced_y", y)
                svg.append(f'    <path d="M{x:.1f},{y - 2.2:.1f} L{x + 2.2:.1f},{y:.1f} '
                           f'L{x:.1f},{y + 2.2:.1f} L{x - 2.2:.1f},{y:.1f} Z" '
                           f'data-layer="{c.get("layer_index", 0)}" />')
                if (tx, ty) != (x, y):
                    svg.append(f'    <line x1="{x:.1f}" y1="{y:.1f}" '
                               f'x2="{tx:.1f}" y2="{ty:.1f}" '
                               f'stroke-dasharray="1,1" stroke-width="0.15" />')
            svg.append("  </g>")
        svg.append("</g>")

    svg.append("</svg>")
    return "\n".join(svg)


def _composite_member_ids(model: Dict) -> set:
    """Object ids of shapes owned by a composite (rendered by the composite block).

    Composite children are the SAME shape dict objects as in ``model['shapes']``
    (see :func:`learning.composites.build_composites`), so identity matching is
    exact and never over-skips a disjoint island that merely shares a layer.
    """
    ids = set()
    for c in model.get("composites", []):
        for ch in c.get("children", []):
            ids.add(id(ch))
    return ids
