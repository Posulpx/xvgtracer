"""Component detection: connected regions, extent, and background detection.

Operates on the raw mask pixels rather than a single marching-squares contour,
which is robust to fragmented boundaries.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

Box = Tuple[float, float, float, float]


def mask_extent(mask: np.ndarray) -> Optional[Box]:
    """Return (x0, y0, x1, y1) bounding box from mask pixels.

    Returns None for an empty mask. (x1, y1) are exclusive (one past last).
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return (float(xs.min()), float(ys.min()),
            float(xs.max() + 1), float(ys.max() + 1))


def components(mask: np.ndarray, min_size: int = 16) -> List[np.ndarray]:
    """Return a list of boolean masks, one per connected component.

    Components smaller than `min_size` pixels are dropped as noise.
    """
    from scipy.ndimage import label
    lab, n = label(mask)
    out = []
    for k in range(1, n + 1):
        comp = lab == k
        if comp.sum() >= min_size:
            out.append(comp)
    return out


def components_merged(mask: np.ndarray, merge_gap: int = 0,
                     min_size: int = 16) -> List[np.ndarray]:
    """Connected components, with nearby fragments merged across a gap.

    A drawn star whose sharp tip / arm was split off by anti-aliasing (or a
    hairline gap) shows up as two disjoint components that *visually* belong to
    the same silhouette. `merge_gap` is the tolerance (in pixels) within which
    such fragments are rejoined before splitting into silhouettes. Only the
    connectivity is merged — the returned masks keep the *original* pixels, so
    the shape is not artificially fattened.

    With ``merge_gap == 0`` this is identical to :func:`components`.
    """
    if merge_gap <= 0:
        return components(mask, min_size=min_size)
    from scipy.ndimage import label, binary_dilation
    # Dilate just enough to bridge the gap, then label to find which original
    # components are neighbours; the actual pixels come from the original mask.
    conn = binary_dilation(mask, iterations=merge_gap)
    lab, n = label(conn)
    out = []
    for k in range(1, n + 1):
        comp = mask & (lab == k)
        if comp.sum() >= min_size:
            out.append(comp)
    return out


def is_background(mask: np.ndarray, total_pixels: int,
                  threshold: float = 0.6) -> bool:
    """True if the mask covers more than `threshold` of the frame area."""
    return (int(mask.sum()) / max(total_pixels, 1)) > threshold


def centroid_of_mask(mask: np.ndarray) -> Tuple[float, float]:
    """Pixel-centroid (cx, cy) of the mask."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0.0, 0.0)
    return (float(xs.mean()), float(ys.mean()))
