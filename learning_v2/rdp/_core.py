"""RDP simplification and turning-signature computation."""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from .._image import Point


def _closed_ring(contour: List[Point]) -> List[Point]:
    if not contour:
        return []
    ring = list(contour)
    if ring[0] == ring[-1]:
        ring = ring[:-1]
    return ring


def _smooth_ring(ring: List[Point], win: int = 3) -> List[Point]:
    n = len(ring)
    if n < 5:
        return list(ring)
    out = []
    for i in range(n):
        xs = ys = 0.0
        cnt = 0
        for k in range(-win, win + 1):
            j = (i + k) % n
            xs += ring[j][0]
            ys += ring[j][1]
            cnt += 1
        out.append((xs / cnt, ys / cnt))
    return out


def _tangent_angle(p0: Point, p1: Point) -> float:
    return math.atan2(p1[1] - p0[1], p1[0] - p0[0])


def _curvature_signature(ring: List[Point], step: int = 1) -> np.ndarray:
    n = len(ring)
    if n < 3:
        return np.zeros(max(0, n))
    sig = np.zeros(n)
    for i in range(n):
        a = ring[(i - step) % n]
        b = ring[i]
        c = ring[(i + step) % n]
        t1 = _tangent_angle(a, b)
        t2 = _tangent_angle(b, c)
        d = t2 - t1
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        sig[i] = d
    return sig


def _rdp_simplify(ring: List[Point], eps: float) -> List[Point]:
    n = len(ring)
    if n < 3:
        return list(ring)

    def recurse(i0: int, i1: int):
        if i1 <= i0 + 1:
            return [i0]
        ax, ay = ring[i0]
        bx, by = ring[i1]
        dx, dy = bx - ax, by - ay
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-9:
            return [i0]
        max_d = 0.0
        split = i0
        for i in range(i0 + 1, i1):
            px, py = ring[i]
            d = abs((px - ax) * dy - (py - ay) * dx) / seg_len
            if d > max_d:
                max_d = d
                split = i
        if max_d <= eps:
            return [i0]
        return recurse(i0, split) + recurse(split, i1)

    idx = recurse(0, n - 1)
    return [ring[i] for i in idx]


def _simplified_signature(ring: List[Point],
                          eps: float) -> Tuple[List[Point], np.ndarray]:
    n = len(ring)
    if n < 4:
        return list(ring), np.zeros(max(0, n))
    simp = _rdp_simplify(ring, eps)
    if len(simp) < 3:
        simp = list(ring)
    sig = _curvature_signature(_closed_ring(simp))
    return simp, sig
