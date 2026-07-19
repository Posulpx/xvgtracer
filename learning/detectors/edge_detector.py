"""Edge detection: prepare a mask for robust contour extraction.

Anti-aliased shapes quantized to a palette develop 1-2px gaps along the boundary
and noisy edges. This module repairs (gap-close + hole-fill) and smooths masks,
and can produce gradient/edge maps for downstream analysis.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def repair_mask(mask: np.ndarray) -> np.ndarray:
    """Close small edge gaps and fill interior holes in a binary mask."""
    try:
        from scipy.ndimage import binary_closing, binary_fill_holes
        m = binary_closing(mask, iterations=2)
        m = binary_fill_holes(m)
        return m
    except Exception:
        return mask


def smooth_mask(mask: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Light Gaussian smoothing of a binary mask to remove contour jaggedness."""
    im = Image.fromarray((mask.astype(np.uint8)) * 255).filter(
        ImageFilter.GaussianBlur(sigma))
    return np.asarray(im) > 127


def sobel_edges(mask: np.ndarray) -> np.ndarray:
    """Return a normalised [0,1] gradient-magnitude edge map of the mask."""
    from scipy.ndimage import sobel
    gx = sobel(mask.astype(float), axis=0)
    gy = sobel(mask.astype(float), axis=1)
    mag = np.hypot(gx, gy)
    mmax = mag.max()
    return mag / mmax if mmax > 0 else mag


def edge_density(mask: np.ndarray) -> float:
    """Fraction of mask pixels that lie on an edge (proxy for complexity)."""
    edges = sobel_edges(mask) > 0.1
    return float(edges.sum()) / float(max(mask.sum(), 1))
