"""Contour error metrics: compares a reconstructed contour to a reference.

These measure how faithfully a generated/reconstructed primitive tracks the
source mask boundary (the contour extracted by detectors.contour_detector).
"""

from __future__ import annotations

import numpy as np


def _to_array(pts, dtype=float):
    return np.asarray([[p[0], p[1]] for p in pts], dtype=dtype)


def mean_contour_error(reconstructed: list, reference: list) -> float:
    """Average nearest-neighbour distance from reconstructed points to reference.

    Lower is better. Both are point lists [(x, y), ...].
    """
    if not reconstructed or not reference:
        return float("inf")
    rc = _to_array(reconstructed)
    rr = _to_array(reference)
    # for each reconstructed point, distance to nearest reference point
    d = np.linalg.norm(rc[:, None, :] - rr[None, :, :], axis=2)
    return float(d.min(axis=1).mean())


def rms_contour_error(reconstructed: list, reference: list) -> float:
    """RMS of nearest-neighbour distances reconstructed -> reference."""
    if not reconstructed or not reference:
        return float("inf")
    rc = _to_array(reconstructed)
    rr = _to_array(reference)
    d = np.linalg.norm(rc[:, None, :] - rr[None, :, :], axis=2)
    return float(np.sqrt((d.min(axis=1) ** 2).mean()))


def max_contour_error(reconstructed: list, reference: list) -> float:
    """Worst-case nearest-neighbour distance reconstructed -> reference."""
    if not reconstructed or not reference:
        return float("inf")
    rc = _to_array(reconstructed)
    rr = _to_array(reference)
    d = np.linalg.norm(rc[:, None, :] - rr[None, :, :], axis=2)
    return float(d.min(axis=1).max())
