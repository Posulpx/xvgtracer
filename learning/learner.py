"""XVGTracer learning system (orchestrator).

This module wires together the package: it quantizes the image into per-color
masks, classifies each mask into a primitive node (shape_registry.classify_mask),
rebuilds ideal SVG `d` strings (reconstructors / svg_renderer), and detects
overlaps to build composite nodes.

Package layout:
    metrics/        iou, contour_error, hausdorff
    detectors/      contour_detector, edge_detector, component_detector
    generators/     circle, rectangle, ellipse, polygon
    reconstructors/ primitive, polygon, bezier
    renderers/      raster_renderer, svg_renderer
    shape_registry  primitive vocabulary + classify-by-fit engine
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional

import numpy as np

from xvgtracer import quantize, _binary_mask_for_color

from .classifiers import classify_mask, build_transform
from .shape_registry import register, PRIMITIVES, COMPOSITES, TRANSFORMS
from .composites import build_composites
from .renderers import model_to_svg, rasterize_node
from .detectors import is_background, components_merged, collision_points


# ---------------------------------------------------------------------------
# Public re-exports used by api.py / CLI
# ---------------------------------------------------------------------------

__all__ = [
    "learn_image",
    "model_to_svg",
    "save_model",
    "load_model",
    "identify_shape",
    "rebuild_shape",
    "detect_overlaps",
]


# ---------------------------------------------------------------------------
# Stage 1+2: identify each mask and rebuild its path
# ---------------------------------------------------------------------------

def identify_shape(mask: np.ndarray) -> Dict:
    """Classify a binary mask into a primitive node.

    Delegates to :func:`learning.classifiers.classify_mask`, which already
    attaches a rotate Transform for rotated rects/ellipses. `build_transform` is
    only used as a fallback when no transform was set (kept as a single owner of
    transform logic in the classifiers package).
    """
    node = classify_mask(mask)
    if node.get("transform") is None:
        node["transform"] = build_transform(node, mask)
    return node


def rebuild_shape(node: Dict) -> str:
    """Emit an ideal SVG path `d` for a primitive/composite node.

    Delegates to the SVG renderer's node->d builder so there is a single source
    of truth for type->reconstructor dispatch.
    """
    from .renderers.svg_renderer import _node_d
    return _node_d(node)


# ---------------------------------------------------------------------------
# Stage 3: overlaps -> composites
# ---------------------------------------------------------------------------

def detect_overlaps(shapes: List[Dict], w: int, h: int,
                    min_overlap: float = 0.02) -> List[Dict]:
    """Detect pairwise overlaps between identified shapes (rasterised IoU)."""
    rasters = []
    for s in shapes:
        rasters.append(rasterize_node(s, w, h))
    overlaps = []
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            a, b = rasters[i], rasters[j]
            inter = np.logical_and(a, b).sum()
            if inter == 0:
                continue
            union = np.logical_or(a, b).sum()
            iou = inter / max(union, 1)
            if iou < min_overlap:
                continue
            contained = inter / max(min(a.sum(), b.sum()), 1) > 0.8
            mode = "keep_topmost" if not contained else "clip_smaller_inside"
            overlaps.append({
                "a": i, "b": j,
                "iou": round(float(iou), 3),
                "mode": mode,
                "contained": bool(contained),
            })
    return overlaps


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def learn_image(image_path: str, num_colors: int = 8,
                merge_gap: int = 0, min_colors: int = 2,
                boundary_only: bool = True,
                collision_merge_dist: Optional[float] = None,
                min_component_area: Optional[int] = None) -> Dict:
    """Run the full learning pipeline on an image and return a JSON-ready dict."""
    rgb_pil, colors = quantize(image_path, num_colors)
    rgb = np.asarray(rgb_pil)
    rgb_arr = rgb
    h, w = rgb.shape[0], rgb.shape[1]
    # Fringe-island floor: drop connected components smaller than this. Defaults
    # to a fraction of the image area with a hard minimum so sub-pixel anti-alias
    # seams never become spurious primitives, while small real shapes survive.
    if min_component_area is None:
        min_component_area = max(32, int(w * h * 5e-5))

    # Collision-merge distance scales with image size: near a near-tangent seam
    # (e.g. a lens tip) many pixels along the boundary each see all converging
    # colours, producing a *swarm* of hits that is really one geometric junction.
    # On a large image that swarm spans dozens of px, so a fixed 1px merge leaves
    # a smear of magenta crosses. Scale to ~1.5% of the image diagonal (hard floor
    # 1px) so small images stay tight while large ones collapse the swarm to a dot.
    if collision_merge_dist is None:
        diag = (w * w + h * h) ** 0.5
        collision_merge_dist = max(1.0, diag * 0.03)

    shapes = []
    shape_masks = {}  # id(shape) -> its binary component mask (for corner sources)
    for ci, color in enumerate(colors):
        mask = _binary_mask_for_color(rgb, color)
        if not mask.any():
            continue
        # A single color can occupy several disjoint regions (e.g. a drawn star
        # whose anti-aliased arm split off into a separate fragment). Treat each
        # connected component as its own silhouette so detached pieces are not
        # silently merged into — or dropped by — the longest-contour extraction.
        # `merge_gap` re-joins fragments that sit within tolerance of each other
        # (a split-off sharp tip), so the pointy sides of a star stay one shape.
        for comp in components_merged(mask, merge_gap=merge_gap):
            # Drop tiny fringe islands: anti-aliasing seams between adjacent
            # shapes can leave a handful of quantized pixels that otherwise
            # become spurious sub-pixel primitives (e.g. a 29px sliver -> a
            # 0.5px "ellipse"). Floor scaled to image size, with a hard minimum.
            if comp.sum() < min_component_area:
                continue
            spec = identify_shape(comp)
            register(spec, color, ci, comp.sum())
            shapes.append(spec)
            shape_masks[id(spec)] = comp

    # mark background layers and exclude them from overlap analysis
    total = w * h
    for s in shapes:
        s["is_background"] = is_background(rasterize_node(s, w, h), total)
    fg = [i for i, s in enumerate(shapes) if not s["is_background"]]

    # Collision points: where color regions meet (the structural junctions of
    # the image). Computed from the *quantized* image. The canvas (white backdrop)
    # is INCLUDED in the convergence count. 3-color convergences are prioritised.
    bg_color = tuple(int(c) for c in colors[0]) if colors else None
    collisions = collision_points(rgb_arr, min_colors=min_colors,
                                  background_color=bg_color,
                                  include_background=True,
                                  boundary_only=boundary_only,
                                  merge_dist=collision_merge_dist)

    # Snap reconstructed nodes onto the collision points so the vector output is
    # faithful to the observed junctions (otherwise ideal shapes miss them by px).
    from .snap import snap_shapes_to_collisions
    snap_shapes_to_collisions(shapes, collisions, tolerance=8.0)

    # Two-circle boolean recovery: an ellipse that actually tapers to points is a
    # lens (intersection of two circles); its neighbouring faceted-polygon lunes
    # are each a circle minus that lens. Recover the generating circles so all
    # three regions reconstruct from true arcs instead of a wrong ellipse / facets.
    from .circle_boolean import recover_two_circle_booleans
    recover_two_circle_booleans(shapes, shape_masks, w, h)

    # Corner-driven refinement: rebuild fixed-vertex polygons (e.g. triangle)
    # from their validated contour corners -- the ideal line-to-line endpoints --
    # instead of the radius/IoU primitive fit that can sit a couple px off.
    from .corners import refine_shapes_from_corners
    refine_shapes_from_corners(shapes, shape_masks)

    # Pin overlapping circles onto the CORNERS of shapes they overlap. Circle
    # intersections are only 2-colour contact edges (no third colour), so the
    # collision snapper skips them; but a neighbour's real corners sitting on the
    # rim should touch it. Corner-driven, adds spline nodes on those corners.
    from .snap import snap_circles_to_overlap_corners
    snap_circles_to_overlap_corners(shapes, shape_masks)

    # stage 2: rebuild d strings (after snapping + corner refinement, so they
    # honour the junctions and the true corners)
    for s in shapes:
        s["rebuilt_d"] = rebuild_shape(s)

    # stage 3: overlaps -> composite nodes
    overlaps = detect_overlaps([shapes[i] for i in fg], w, h)
    composites = build_composites(shapes, fg, overlaps)

    # Corner sources of truth (kept SEPARATE, not merged): contour curvature
    # maxima (the human-validated ideal line-to-line endpoints; sees tips-
    # against-background the collision detector misses) and extended straight-
    # side intersections (implied apex, robust to tip pixel rounding).
    from .corners import corner_sources
    corners = corner_sources(shapes, shape_masks)

    # Confirmed corners (contour curvature maxima) grouped by layer, used to
    # anchor construction-line endpoints to the true line-to-line endpoints.
    corners_by_layer: Dict[int, list] = {}
    for c in corners.get("contour", []):
        corners_by_layer.setdefault(c["layer_index"], []).append((c["x"], c["y"]))

    # Construction lines: for non-primitive (custom polygon) shapes, anchor each
    # straight side's endpoints to the nearest confirmed corner, then extend a
    # few px past both ends so lines run corner-to-corner and their convergence
    # points land on the true endpoints.
    from .construction import construction_lines
    construction = construction_lines(shapes, corners_by_layer=corners_by_layer)

    # The canvas/backdrop is itself a silhouette: counting it, the image holds
    # N distinct shapes (7 foreground + 1 background = 8 for drawn_shapes.png).
    # Surface that explicitly so callers don't silently lose the backdrop.
    n_bg = sum(1 for s in shapes if s.get("is_background"))
    summary = {
        "n_silhouettes": len(shapes),
        "n_foreground": len(fg),
        "n_background": n_bg,
        "types": [s["type"] for s in shapes],
        "n_collisions": len(collisions),
    }

    return {
        "source": image_path,
        "width": w,
        "height": h,
        "num_colors": num_colors,
        "summary": summary,
        "hierarchy": {
            "primitives": PRIMITIVES,
            "composites": COMPOSITES,
            "transforms": TRANSFORMS,
        },
        "shapes": shapes,
        "composites": composites,
        "overlaps": overlaps,
        "collisions": collisions,
        "construction": construction,
        "corners": corners,
        "pipeline": ["identify", "rebuild", "postprocess"],
    }


def save_model(model: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)


def load_model(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else "tests/clean_shapes.png"
    nc = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    mg = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    mc = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    m = learn_image(img, nc, merge_gap=mg, min_colors=mc)
    out = img.rsplit(".", 1)[0] + "_learned.json"
    save_model(m, out)
    svg_out = img.rsplit(".", 1)[0] + "_learned.svg"
    open(svg_out, "w", encoding="utf-8").write(model_to_svg(m))
    sm = m["summary"]
    print(f"Learned {sm['n_silhouettes']} silhouettes "
          f"({sm['n_foreground']} foreground + {sm['n_background']} background), "
          f"{len(m['overlaps'])} overlaps")
    print(f"  -> {out}")
    print(f"  -> {svg_out}")
    for s in m["shapes"]:
        print(f"  layer {s['layer_index']} {s['type']:8s} {tuple(s['color'])} "
              f"area={s['mask_area']}")
    for o in m["overlaps"]:
        print(f"  overlap {o['a']}<->{o['b']} iou={o['iou']} mode={o['mode']}")
