"""Convergence-point detection and line/curve segmentation."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from .._image import Point
from ..rdp import _simplified_signature


_ANGLE_THRESH = 0.35
_HIGH_THRESH = 0.5
_MIN_DIST_SIG = 1
_MIN_DIST_PX_FACTOR = 0.14
_MIN_DIST_PX_MIN = 5.0
_LINE_DEV = 0.12
_STRENGTH_CUT = 0.8


def find_convergence_points(ring: List[Point]) -> List[Tuple[int, float]]:
    n = len(ring)
    if n < 6:
        return []
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    eps = max(1.0, 0.012 * diag)
    simp, sig = _simplified_signature(ring, eps)
    if len(sig) == 0:
        return []
    asig = np.abs(sig)
    if float(np.max(asig)) < 0.5 and float(np.var(asig)) < 0.008:
        return []
    m = len(sig)
    extended = list(abs(sig)) + list(abs(sig))
    kept_raw: List[Tuple[int, float]] = []
    i = 0
    while i < 2 * m:
        if extended[i] <= _ANGLE_THRESH:
            i += 1
            continue
        run_start = i
        while i < 2 * m and extended[i] > _ANGLE_THRESH:
            i += 1
        run_end = i - 1
        has_high = any(extended[j] > _HIGH_THRESH for j in range(run_start, run_end + 1))
        if has_high:
            to_keep = list(range(run_start, run_end + 1))
        else:
            to_keep = [max(range(run_start, run_end + 1), key=lambda j: extended[j])]
        for j in to_keep:
            modest = j % m
            if all(min(abs(modest - (k % m)), m - abs(modest - (k % m))) >= _MIN_DIST_SIG
                   for k, _ in kept_raw):
                kept_raw.append((modest, float(abs(sig[modest]))))
    kept_simp = sorted(set(k for k, _ in kept_raw))
    orig_cand = []
    for si in kept_simp:
        sp = simp[si]
        best = min(range(n), key=lambda k: (ring[k][0] - sp[0]) ** 2 + (ring[k][1] - sp[1]) ** 2)
        strength = max(abs(sig[si]) for ks, a in kept_raw if ks == si for b in [a])
        orig_cand.append((best, ring[best][0], ring[best][1], float(strength)))
    min_dist_px = max(_MIN_DIST_PX_MIN, _MIN_DIST_PX_FACTOR * diag)
    kept: List[Tuple[int, float, float, float]] = []
    for best, px, py, st in sorted(orig_cand, key=lambda t: t[0]):
        merged = False
        for ki, (k2, px2, py2, st2) in enumerate(kept):
            if math.hypot(px - px2, py - py2) < min_dist_px:
                if st > st2:
                    kept[ki] = (best, px, py, st)
                merged = True
                break
        if not merged:
            kept.append((best, px, py, st))
    return sorted((k[0], k[3]) for k in kept)


def classify_segment(ring: List[Point], i0: int, i1: int) -> str:
    n = len(ring)
    if i1 >= i0:
        idx = list(range(i0, i1 + 1))
    else:
        idx = list(range(i0, n)) + list(range(0, i1 + 1))
    sub = [ring[k] for k in idx]
    if len(sub) < 3:
        return "line"
    p0, p2 = sub[0], sub[-1]
    chord = math.hypot(p2[0] - p0[0], p2[1] - p0[1])
    if chord < 10.0:
        return "curve"
    if chord < 1e-6:
        return "line"
    dx, dy = p2[0] - p0[0], p2[1] - p0[1]
    dev = 0.0
    for p in sub[1:-1]:
        d = abs((p[0] - p0[0]) * dy - (p[1] - p0[1]) * dx) / chord
        if d > dev:
            dev = d
    return "line" if (dev / chord) < _LINE_DEV else "curve"


def build_lc_path(ring: List[Point]) -> Dict[str, Any]:
    raw = find_convergence_points(ring)
    cps = [p[0] for p in raw]
    if len(cps) < 2:
        n = len(ring)
        return {"points": [ring[i] for i in range(0, n, max(1, n // 24))],
                "seg_types": ["curve"], "convergence": cps}
    strengths = dict(raw)
    if len(cps) >= 4:
        strong_idx = [i for i, cp in enumerate(cps) if strengths.get(cp, 0) > _STRENGTH_CUT]
        if len(strong_idx) == 2:
            cps = [cps[strong_idx[0]], cps[strong_idx[1]]]
        elif len(cps) > 6 and len(strong_idx) > 0 and len(strong_idx) < len(cps) // 2:
            cps = [cps[i] for i in strong_idx]
    seg_types = []
    for k in range(len(cps)):
        i0 = cps[k]
        i1 = cps[(k + 1) % len(cps)]
        seg_types.append(classify_segment(ring, i0, i1))
    for k in range(len(seg_types)):
        prev = (k - 1) % len(seg_types)
        nxt = (k + 1) % len(seg_types)
        if seg_types[k] == "line" and seg_types[prev] == "curve" and seg_types[nxt] == "curve":
            seg_types[k] = "curve"
    return {"points": [ring[i] for i in cps],
            "seg_types": seg_types, "convergence": cps}
