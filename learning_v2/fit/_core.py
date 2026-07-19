"""Curve fitting: circle, ellipse, arc, cubic bezier, Catmull-Rom spline."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from .._image import Point


def _fit_arc_3pt(p0: Point, p1: Point, p2: Point):
    (x0, y0), (x1, y1), (x2, y2) = p0, p1, p2
    d = 2.0 * (x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1))
    if abs(d) < 1e-9:
        return None
    ux = ((x0 * x0 + y0 * y0) * (y1 - y2) + (x1 * x1 + y1 * y1) * (y2 - y0)
          + (x2 * x2 + y2 * y2) * (y0 - y1)) / d
    uy = ((x0 * x0 + y0 * y0) * (x2 - x1) + (x1 * x1 + y1 * y1) * (x0 - x2)
          + (x2 * x2 + y2 * y2) * (x1 - x0)) / d
    r = math.hypot(x0 - ux, y0 - uy)
    return (ux, uy, r)


def _fit_ellipse_lsq(pts: List[Point]):
    m = len(pts)
    if m < 5:
        return None
    xs = np.array([p[0] for p in pts], float)
    ys = np.array([p[1] for p in pts], float)
    cx, cy = xs.mean(), ys.mean()
    u, v = xs - cx, ys - cy
    cov = np.cov(u, v)
    evals, evecs = np.linalg.eigh(cov)
    rx = math.sqrt(2.0 * abs(evals[1]))
    ry = math.sqrt(2.0 * abs(evals[0]))
    angle = math.atan2(evecs[1, 1], evecs[0, 1])
    if rx < ry:
        rx, ry = ry, rx
        angle = math.atan2(evecs[1, 0], evecs[0, 0])
    return (cx, cy, rx, ry, angle)


def _fit_circle_lsq(pts: List[Point]):
    m = len(pts)
    if m < 3:
        return None
    xs = np.array([p[0] for p in pts], float)
    ys = np.array([p[1] for p in pts], float)
    xm, ym = xs.mean(), ys.mean()
    u = xs - xm
    v = ys - ym
    Sxx = (u * u).sum()
    Syy = (v * v).sum()
    Sxy = (u * v).sum()
    Suu = (u * u * u).sum()
    Svv = (v * v * v).sum()
    Suv = (u * u * v).sum()
    Svuv = (u * v * v).sum()
    A = np.array([[Sxx, Sxy], [Sxy, Syy]], float)
    B = 0.5 * np.array([Suu + Suv, Suv + Svv], float)
    det = np.linalg.det(A)
    if abs(det) < 1e-12:
        return None
    uc, vc = np.linalg.solve(A, B)
    cx, cy = xm + uc, ym + vc
    r = math.sqrt(uc * uc + vc * vc + (Sxx + Syy) / m)
    if not (math.isfinite(r) and r > 0):
        return None
    return (cx, cy, r)


def _segment_arc(ring: List[Point], i0: int, i1: int):
    n = len(ring)
    if i1 >= i0:
        idx = list(range(i0, i1 + 1))
    else:
        idx = list(range(i0, n)) + list(range(0, i1 + 1))
    if len(idx) < 3:
        return None
    sub = [ring[k] for k in idx]
    p0, p2 = sub[0], sub[-1]
    arc = _fit_arc_3pt(sub[0], sub[len(sub) // 2], sub[-1])
    if arc is None:
        return None
    cx, cy, r = arc
    pm = sub[len(sub) // 2]
    cross = (pm[0] - p0[0]) * (p2[1] - p0[1]) - (pm[1] - p0[1]) * (p2[0] - p0[0])
    sweep = 1 if cross > 0 else 0
    cross_c = (cx - p0[0]) * (p2[1] - p0[1]) - (cy - p0[1]) * (p2[0] - p0[0])
    large = 1 if cross * cross_c > 0 else 0
    return (p0, p2, r, r, large, sweep)


def _spline_path(pts: List[Point]) -> str:
    m = len(pts)
    if m < 3:
        return ""
    d = "M %.1f %.1f" % (pts[0][0], pts[0][1])
    for k in range(m):
        p0 = pts[(k - 1) % m]
        p1 = pts[k]
        p2 = pts[(k + 1) % m]
        p3 = pts[(k + 2) % m]
        c1x = p1[0] + (p2[0] - p0[0]) / 6.0
        c1y = p1[1] + (p2[1] - p0[1]) / 6.0
        c2x = p2[0] - (p3[0] - p1[0]) / 6.0
        c2y = p2[1] - (p3[1] - p1[1]) / 6.0
        d += " C %.1f %.1f %.1f %.1f %.1f %.1f" % (c1x, c1y, c2x, c2y, p2[0], p2[1])
    d += " Z"
    return d


def _fit_bezier_lsq(ring: List[Point], i0: int, i1: int
                    ) -> Optional[Tuple[float, float, float, float]]:
    N = len(ring)
    if i1 < i0:
        i1 += N
    idx = np.arange(i0, i1 + 1) % N
    pts = np.array([ring[i] for i in idx])
    if len(pts) < 3:
        return None
    p0, p3 = pts[0], pts[-1]
    chords = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
    chord_len = np.sum(chords)
    if chord_len < 1e-8:
        return None
    t = np.zeros(len(pts))
    cum = 0.0
    for i in range(1, len(pts)):
        cum += chords[i - 1]
        t[i] = cum / chord_len
    A = np.column_stack([3 * (1 - t)**2 * t, 3 * (1 - t) * t**2])
    bx = pts[:, 0] - ((1 - t)**3 * p0[0] + t**3 * p3[0])
    by = pts[:, 1] - ((1 - t)**3 * p0[1] + t**3 * p3[1])
    try:
        cp, _, _, _ = np.linalg.lstsq(A, np.column_stack([bx, by]), rcond=None)
    except np.linalg.LinAlgError:
        return None
    return (float(cp[0, 0]), float(cp[0, 1]),
            float(cp[1, 0]), float(cp[1, 1]))
