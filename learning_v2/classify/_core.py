"""Whole-shape family classification."""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from .._image import contour_vertices, repair_mask
from ..conv import build_lc_path
from ..rdp import _closed_ring, _simplified_signature, _smooth_ring


_RDP_EPS_FACTOR = 0.012
_RDP_EPS_MIN = 1.0
_MAX_TURN_SMOOTH = 0.5
_MAX_TURN_VAR_EARLY = 0.008
_N_SIGN_POLY = 2
_N_STRONG_POLY = 2
_N_STRONG_EARLY = 2
_PEAK_THRESH = 0.35
_STRONG_THRESH = 0.8
_STAR_VAR_MIN = 0.05
_STAR_PEAK_MIN = 6
_BLOB_PEAK_MIN = 8
_BLOB_VAR_MAX = 0.05
_MIN_RING = 8
_MIN_AREA = 50
_LUM_MIN = 12
_LUM_MAX = 243


def classify_shape(mask: np.ndarray) -> Dict[str, Any]:
    contour = contour_vertices(repair_mask(mask), fill_holes=True)
    raw = _closed_ring(contour)
    ring = _smooth_ring(raw, win=3)
    n = len(ring)
    if n < _MIN_RING:
        return {"family": "polygon", "ring": ring,
                "lc": build_lc_path(raw)}
    xs = [p[0] for p in raw]
    ys = [p[1] for p in raw]
    diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    eps_family = max(_RDP_EPS_MIN, _RDP_EPS_FACTOR * diag)
    _, sig_simp = _simplified_signature(raw, eps_family)
    mag_simp = np.abs(sig_simp)
    global_var = float(np.var(mag_simp)) if len(mag_simp) else 0.0
    n_peaks = int(np.sum(mag_simp > _PEAK_THRESH))
    max_turn = float(np.max(mag_simp)) if len(mag_simp) else 0.0
    signs = [1 if s > 0 else -1 for s in sig_simp]
    n_sign = sum(1 for i in range(len(signs))
                 if signs[i] != signs[(i + 1) % len(signs)])
    n_strong = int(np.sum(mag_simp > _STRONG_THRESH))
    if max_turn < _MAX_TURN_SMOOTH:
        family = "smooth"
    elif n_sign <= _N_SIGN_POLY:
        if n_strong >= _N_STRONG_POLY:
            family = "polygon"
        else:
            n_low = int(np.sum(mag_simp < 0.4))
            n_high = int(np.sum(mag_simp > 0.5))
            family = "polygon" if n_low >= 2 and n_high >= 2 else "smooth"
    elif global_var >= _STAR_VAR_MIN and n_peaks >= _STAR_PEAK_MIN:
        family = "star"
    elif n_peaks >= _BLOB_PEAK_MIN and global_var < _BLOB_VAR_MAX:
        family = "blob"
    else:
        family = "polygon"
    if family == "smooth":
        lc = {"points": ring[::max(1, len(ring)//24)] + [ring[0]],
              "seg_types": ["curve"], "convergence": []}
    else:
        lc = build_lc_path(raw)
        if n_sign <= _N_SIGN_POLY and n_strong < _N_STRONG_EARLY and len(lc.get("seg_types", [])) > 0:
            lc["seg_types"] = ["curve"] * len(lc["seg_types"])
    return {"family": family, "ring": ring,
            "lc": lc, "global_var": global_var,
            "n_peaks": n_peaks}
