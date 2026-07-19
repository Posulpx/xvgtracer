"""Hausdorff distance metric between two point sets (contours).

Used to quantify the worst-case disagreement between a reconstructed primitive
and the reference contour. The directed Hausdorff is the max over A of the min
distance to B; the (undirected) Hausdorff is the max of both directions.
"""

from __future__ import annotations

import numpy as np


def _to_array(pts, dtype=float):
    return np.asarray([[p[0], p[1]] for p in pts], dtype=dtype)


def directed_hausdorff(a: list, b: list) -> float:
    """Max over points in `a` of the min distance to any point in `b`."""
    if not a or not b:
        return float("inf")
    pa = _to_array(a)
    pb = _to_array(b)
    d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
    return float(d.min(axis=1).max())


def hausdorff(a: list, b: list) -> float:
    """Undirected Hausdorff distance between point sets `a` and `b`."""
    return max(directed_hausdorff(a, b), directed_hausdorff(b, a))
