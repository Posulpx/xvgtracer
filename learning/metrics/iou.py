"""Intersection-over-Union metric for masks and polygon sets."""

from __future__ import annotations

import numpy as np


def iou_masks(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two boolean masks (same shape)."""
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def iou_polygons(pts_a, pts_b, w: int, h: int, raster) -> float:
    """IoU between two polygon point lists, rasterised via the supplied `raster`.

    `raster(points, w, h) -> bool ndarray` is injected so the metrics layer stays
    independent of the rasterisation backend (see renderers.raster_renderer).
    """
    ra = raster(pts_a, w, h)
    rb = raster(pts_b, w, h)
    return iou_masks(ra, rb)


def coverage(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of `a` that is also inside `b` (a ⊆ b measure)."""
    inter = float(np.logical_and(a, b).sum())
    return inter / float(a.sum()) if a.sum() > 0 else 0.0
