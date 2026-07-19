"""Contour detection: extract the outer boundary ring of a binary mask.

Wraps skimage marching squares. The contour is returned as a list of (x, y)
vertices (skimage yields (row, col), which we swap to (x, y) for drawing). For
anti-aliased masks the boundary can fragment; small gaps are closed and holes
filled so a single closed outer ring is produced.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

Point = Tuple[float, float]


def contour_vertices(mask: np.ndarray, fill_holes: bool = True,
                     close_gaps: bool = True) -> List[Point]:
    """Return the outer contour of `mask` as a closed list of (x, y) vertices.

    Returns [] for an empty mask. The ring is closed (first == last).
    """
    from skimage import measure
    if not mask.any():
        return []
    m = mask
    if close_gaps:
        try:
            from scipy.ndimage import binary_closing
            m = binary_closing(m, iterations=2)
        except Exception:
            m = mask
    if fill_holes:
        try:
            from scipy.ndimage import binary_fill_holes
            m = binary_fill_holes(m)
        except Exception:
            pass
    contours = measure.find_contours(m.astype(float), 0.5)
    if not contours:
        return []
    # longest contour = outer boundary; skimage returns (row, col) -> swap to (x, y)
    c = max(contours, key=len)
    pts = [(float(y), float(x)) for x, y in c]
    if pts and pts[0] != pts[-1]:
        pts.append(pts[0])
    return pts


def all_contours(mask: np.ndarray, level: float = 0.5):
    """Return every contour (including holes) as (x, y) point lists.

    Useful for component/edge analysis where interior structure matters.
    """
    from skimage import measure
    if not mask.any():
        return []
    contours = measure.find_contours(mask.astype(float), level)
    return [[(float(y), float(x)) for x, y in c] for c in contours]
