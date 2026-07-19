"""Shape registry: vocabulary + primitive->reconstructor map + fit helper.

This module is intentionally *data*, not logic. It declares the supported
primitive / composite / transform vocabulary and the mapping from each primitive
to the reconstructor that builds its SVG `d`. It also exposes a small shared
`fit_candidates` routine used by the classifier to rank candidate primitives by
IoU against a mask.

All mask-classification logic lives in :mod:`learning.classifiers.mask_classifier`.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .renderers import rasterize_polygon
from .metrics import iou_masks

Point = Tuple[float, float]

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

PRIMITIVES = ["point", "line", "circle", "ellipse", "rect", "rounded_rect",
              "triangle", "polygon", "bezier", "arc"]
COMPOSITES = ["union", "difference", "intersection", "xor"]
TRANSFORMS = ["translate", "rotate", "scale", "skew", "mirror"]

# Map each primitive type to the reconstructor function that emits its `d`.
# (populated lazily to avoid import cycles at module load)
_PRIMITIVE_RECONSTRUCTORS: Dict[str, Callable[[Dict], str]] = {}


def _register_reconstructors() -> None:
    if _PRIMITIVE_RECONSTRUCTORS:
        return
    from .reconstructors import (
        reconstruct_primitive,
        reconstruct_polygon,
        reconstruct_bezier,
    )
    for t in ("point", "circle", "ellipse", "arc", "line", "rect", "rounded_rect"):
        _PRIMITIVE_RECONSTRUCTORS[t] = reconstruct_primitive
    for t in ("triangle", "polygon", "star"):
        _PRIMITIVE_RECONSTRUCTORS[t] = reconstruct_polygon
    _PRIMITIVE_RECONSTRUCTORS["bezier"] = reconstruct_bezier


def reconstructor_for(node_type: str) -> Optional[Callable[[Dict], str]]:
    """Return the reconstructor callable for a primitive node type, or None."""
    _register_reconstructors()
    return _PRIMITIVE_RECONSTRUCTORS.get(node_type)


# ---------------------------------------------------------------------------
# Shared fitting routine
# ---------------------------------------------------------------------------

def fit_candidates(builders: Dict[str, Callable[..., List[Point]]],
                   mask: np.ndarray, w: int, h: int,
                   *args) -> Optional[Tuple[str, float, List[Point]]]:
    """Rank candidate primitives by IoU against `mask`.

    `builders` maps a primitive name to a callable returning a point list; the
    callable is invoked with `args` (typically the mask extent box + centroid).
    Returns ``(best_name, best_iou, best_points)`` or ``None`` if no candidate
    produces points.
    """
    best: Optional[Tuple[str, float, List[Point]]] = None
    for name, builder in builders.items():
        try:
            pts = builder(*args)
        except Exception:
            continue
        if not pts:
            continue
        r = rasterize_polygon(pts, w, h)
        iou = iou_masks(r, mask)
        if best is None or iou > best[1]:
            best = (name, iou, pts)
    return best


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------

def register(node: Dict, color, layer_index: int, mask_area: int) -> Dict:
    """Attach bookkeeping fields (color, layer_index, mask_area) to a node."""
    node["color"] = [int(c) for c in color]
    node["layer_index"] = layer_index
    node["mask_area"] = int(mask_area)
    return node
