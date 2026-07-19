"""Occluded-circle recovery.

A circle that is partly hidden by other shapes reads, after colour quantization,
as an irregular region: its VISIBLE boundary is one (or more) circular arcs, while
the hidden side is a straight-ish chord where the occluding shape cut across it.
A plain primitive fit fails (the bbox-circle IoU is low because the occluded part
is missing), so the classifier would fall back to a FACETED polygon -- turning a
smooth arc into a staircase of line segments.

This module detects that case and reconstructs the region as an ``occluded_circle``
node: the underlying circle (cx, cy, r) plus the traced boundary ring. The
``d``-builder then emits SVG **arc** commands along the parts of the ring that lie
on the fitted circle and straight **line** commands across the occluder chords, so
the curved edges stay smooth while the cut edges stay straight.

Detection is robust to the occlusion chords via a bbox-constrained RANSAC circle
fit, and gated so genuine ellipses / non-circular blobs are NOT converted:
  * a large fraction of boundary points must be circle inliers,
  * the inliers must span a wide ANGULAR range (an ellipse only matches a circle
    near two vertices, so its coverage is low),
  * the reconstructed arc-path must beat the faithful polygon's fit.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

Circle = Tuple[float, float, float]


def _circle_through_3(p1, p2, p3) -> Optional[Circle]:
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-6:
        return None
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return ux, uy, math.hypot(ax - ux, ay - uy)


def _ransac_circle(pts: List[Tuple[float, float]],
                   x0, y0, x1, y1,
                   iters: int = 800) -> Optional[Tuple[float, Circle, np.ndarray]]:
    """RANSAC circle fit robust to occlusion chords.

    Returns ``(inlier_fraction, (cx, cy, r), inlier_mask)`` for the circle with
    the most boundary inliers, constrained so the radius/centre stay near the
    region's bounding box (rejecting the giant near-collinear circles that fit a
    locally straight staircase edge).
    """
    if len(pts) < 3:
        return None
    P = np.asarray(pts, dtype=float)
    n = len(P)
    diag = math.hypot(x1 - x0, y1 - y0)
    if diag <= 0:
        return None
    rmin, rmax = diag * 0.15, diag * 0.75
    cx0, cy0 = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    max_centre_off = diag * 0.4
    best: Optional[Tuple[float, Circle, np.ndarray]] = None
    rng = random.Random(0)
    for _ in range(iters):
        i, j, k = rng.sample(range(n), 3)
        c = _circle_through_3(P[i], P[j], P[k])
        if c is None:
            continue
        cx, cy, r = c
        if r < rmin or r > rmax:
            continue
        if math.hypot(cx - cx0, cy - cy0) > max_centre_off:
            continue
        dist = np.abs(np.hypot(P[:, 0] - cx, P[:, 1] - cy) - r)
        tol = max(2.0, 0.04 * r)
        inl = dist < tol
        ninl = int(inl.sum())
        if best is None or ninl > best[0]:
            best = (ninl, (cx, cy, r), inl)
    if best is None:
        return None
    ninl, circ, inl = best
    # refine centre/radius on the inlier set (algebraic Kasa fit)
    Q = P[inl]
    if len(Q) >= 3:
        A = np.c_[2 * Q[:, 0], 2 * Q[:, 1], np.ones(len(Q))]
        b = Q[:, 0] ** 2 + Q[:, 1] ** 2
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        rcx, rcy = sol[0], sol[1]
        rr = math.sqrt(max(1e-6, sol[2] + rcx * rcx + rcy * rcy))
        if rmin <= rr <= rmax and math.hypot(rcx - cx0, rcy - cy0) <= max_centre_off:
            circ = (rcx, rcy, rr)
            dist = np.abs(np.hypot(P[:, 0] - circ[0], P[:, 1] - circ[1]) - circ[2])
            inl = dist < max(2.0, 0.04 * circ[2])
    return float(inl.sum()) / n, circ, inl


def _angular_coverage(pts: np.ndarray, inl: np.ndarray, cx, cy, bins: int = 36) -> float:
    """Fraction of angular bins around (cx,cy) that contain an inlier point."""
    if inl.sum() == 0:
        return 0.0
    ang = np.arctan2(pts[inl, 1] - cy, pts[inl, 0] - cx)
    idx = ((ang + math.pi) / (2 * math.pi) * bins).astype(int) % bins
    return len(np.unique(idx)) / float(bins)


def fit_occluded_circle(mask: np.ndarray,
                        contour: List[Tuple[float, float]],
                        x0, y0, x1, y1) -> Optional[Dict]:
    """Return an ``occluded_circle`` node if the mask is a partly-hidden circle.

    ``contour`` is the traced boundary ring (closed or open). Returns ``None`` if
    the region is not a clipped circle (leaving the caller to fall back to a
    faithful polygon).
    """
    if not contour or len(contour) < 8:
        return None
    ring = contour[:-1] if contour[-1] == contour[0] else list(contour)
    if len(ring) < 8:
        return None
    fit = _ransac_circle(ring, x0, y0, x1, y1)
    if fit is None:
        return None
    inl_frac, (cx, cy, r), inl = fit
    P = np.asarray(ring, dtype=float)
    cover = _angular_coverage(P, inl, cx, cy)

    # Gate: a real clipped circle has many inliers spread over a wide arc AND its
    # visible mask sits inside the fitted disk. A genuine ellipse matches the
    # circle only near two vertices (low coverage); a random blob has few inliers.
    if inl_frac < 0.40 or cover < 0.45:
        return None
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    contain = (mask & disk).sum() / max(1, int(mask.sum()))
    if contain < 0.90:
        return None
    # The fitted disk must not be wildly larger than the visible mask (an ellipse
    # inscribed in a big circle would pass containment but waste most of the disk).
    disk_iou = (mask & disk).sum() / max(1, int((mask | disk).sum()))
    if disk_iou < 0.45:
        return None

    # Regular-polygon rejection (hexagon/pentagon): a genuine N-gon inscribed in
    # its circumcircle fills a large fraction of that disk (hexagon ~0.83, pentagon
    # ~0.90), whereas a *clipped* circle is only partly present so its disk IoU is
    # much lower. So if the fitted disk already accounts for most of the mask
    # (disk_iou high), the shape is a convex polygon on its circumcircle, not a
    # circle hidden by occluders -> reject so it stays a faithful polygon. This
    # cleanly separates a hexagon (disk_iou ~0.93) from a clipped circle (~0.5-0.7).
    if disk_iou >= 0.80:
        return None

    # Simplify the ring: keep on-circle (arc) vertices dense enough to read as a
    # smooth arc, but collapse the staircase along the straight occluder chords so
    # they emit a few clean line segments rather than dozens of 1px jitters.
    simp = _simplify_ring(ring, cx, cy, r, eps=max(1.0, 0.02 * r))

    node = {
        "kind": "primitive",
        "type": "occluded_circle",
        "params": {
            "cx": round(float(cx), 2),
            "cy": round(float(cy), 2),
            "r": round(float(r), 2),
            "points": [[round(float(px), 1), round(float(py), 1)] for px, py in simp],
        },
        "transform": None,
        "boundary_eval": None,
        "bbox": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
        "centroid": [round(float(cx), 1), round(float(cy), 1)],
        "area": int(mask.sum()),
        "fit_iou": round(float(disk_iou), 3),
        "custom": True,
    }
    return node


def _rdp(points, eps):
    """Ramer-Douglas-Peucker on an open polyline."""
    if len(points) < 3:
        return list(points)
    a = points[0]
    b = points[-1]
    dx, dy = b[0] - a[0], b[1] - a[1]
    seg = math.hypot(dx, dy)
    dmax, idx = 0.0, 0
    for i in range(1, len(points) - 1):
        px, py = points[i]
        if seg < 1e-9:
            dist = math.hypot(px - a[0], py - a[1])
        else:
            dist = abs(dy * px - dx * py + b[0] * a[1] - b[1] * a[0]) / seg
        if dist > dmax:
            dmax, idx = dist, i
    if dmax > eps:
        left = _rdp(points[:idx + 1], eps)
        right = _rdp(points[idx:], eps)
        return left[:-1] + right
    return [a, b]


def _rasterize_d(d: str, w: int, h: int) -> np.ndarray:
    """Rasterise an SVG path `d` (M/L/A/Z) into a boolean mask via point sampling.

    Used to measure the true fidelity of the arc/L path (so arcs that bulge
    outside a regular polygon count against it), independent of any straight-edge
    approximation. Arcs are sampled into dense points so the filled region matches
    what the SVG renderer draws.
    """
    import re
    from .renderers.raster_renderer import rasterize_polygon
    toks = re.findall(r"[MLAZmlaz]|-?\d+(?:\.\d+)?", d)
    pts: List[Tuple[float, float]] = []
    cur = None
    i = 0
    while i < len(toks):
        cmd = toks[i].upper()
        if cmd == "M":
            cur = (float(toks[i + 1]), float(toks[i + 2]))
            pts.append(cur)
            i += 3
        elif cmd == "L":
            cur = (float(toks[i + 1]), float(toks[i + 2]))
            pts.append(cur)
            i += 3
        elif cmd == "A":
            r = float(toks[i + 1])
            large = int(toks[i + 4])
            sweep = int(toks[i + 5])
            ex, ey = float(toks[i + 6]), float(toks[i + 7])
            sx, sy = cur
            dxc, dyc = ex - sx, ey - sy
            L = math.hypot(dxc, dyc)
            if L < 1e-6 or r < L / 2.0:
                pts.append((ex, ey)); cur = (ex, ey); i += 8; continue
            # centre via perpendicular bisector (robust, unambiguous)
            mx, my = (sx + ex) / 2.0, (sy + ey) / 2.0
            hh = math.sqrt(max(0.0, r * r - (L / 2.0) ** 2))
            px, py = -dyc / L, dxc / L  # unit perpendicular to chord
            # choose side by sweep flag
            sgn = 1 if sweep == 1 else -1
            ccx = mx + sgn * h * px
            ccy = my + sgn * h * py
            a0 = math.atan2(sy - ccy, sx - ccx)
            a1 = math.atan2(ey - ccy, ex - ccx)
            da = a1 - a0
            if sweep == 1 and da < 0:
                da += 2 * math.pi
            elif sweep == 0 and da > 0:
                da -= 2 * math.pi
            nseg = max(6, int(abs(da) * r))
            for s in range(1, nseg + 1):
                a = a0 + da * (s / nseg)
                pts.append((ccx + r * math.cos(a), ccy + r * math.sin(a)))
            cur = (ex, ey)
            i += 8
        else:
            i += 1
    if len(pts) < 3:
        return np.zeros((int(h), int(w)), bool)
    pts = [(float(x), float(y)) for x, y in pts]
    return rasterize_polygon(pts, int(w), int(h))



def _simplify_ring(ring, cx, cy, r, eps):
    """Simplify a closed ring, preserving on-circle arc detail.

    Split the ring into maximal runs of on-circle vs off-circle (chord) vertices.
    Off-circle runs are RDP-simplified (straight chords -> few points); on-circle
    runs are kept dense (so the arc coalescer sees a smooth curve).
    """
    tol = max(2.0, 0.06 * r)
    n = len(ring)
    if n < 4:
        return list(ring)

    def on(p):
        return abs(math.hypot(p[0] - cx, p[1] - cy) - r) <= tol

    flags = [on(p) for p in ring]
    out = []
    i = 0
    while i < n:
        j = i
        while j < n and flags[j] == flags[i]:
            j += 1
        run = ring[i:j]
        if flags[i]:
            out.extend(run)
        else:
            seg = [ring[i - 1]] + run + [ring[j % n]]
            simp = _rdp(seg, eps)
            out.extend(simp[1:-1])
        i = j
    # dedupe consecutive
    dd = []
    for p in out:
        if not dd or (abs(dd[-1][0] - p[0]) > 1e-6 or abs(dd[-1][1] - p[1]) > 1e-6):
            dd.append(p)
    return dd if len(dd) >= 3 else list(ring)


def occluded_circle_d(params: Dict) -> str:
    """Emit an SVG path: arcs along the on-circle boundary, lines across chords.

    Walk the traced ring. For each edge between consecutive vertices, if BOTH
    endpoints lie on the fitted circle (within tolerance) the edge is a genuine
    circular arc -> emit an SVG ``A`` command. Otherwise the edge crosses (or lies
    inside) an occluder -> emit a straight ``L``. This keeps curved edges smooth
    while cut edges stay straight, faithfully reproducing the visible silhouette.
    """
    cx, cy, r = params["cx"], params["cy"], params["r"]
    pts = params.get("points") or []
    if len(pts) < 3:
        return ""
    tol = max(2.0, 0.06 * r)

    def on_circle(p):
        return abs(math.hypot(p[0] - cx, p[1] - cy) - r) <= tol

    def signed_span(a, b):
        aa = math.atan2(a[1] - cy, a[0] - cx)
        ab = math.atan2(b[1] - cy, b[0] - cx)
        dt = ab - aa
        while dt <= -math.pi:
            dt += 2 * math.pi
        while dt > math.pi:
            dt -= 2 * math.pi
        return dt

    n = len(pts)
    d = f"M{pts[0][0]:.2f},{pts[0][1]:.2f} "
    i = 0
    while i < n:
        a = pts[i]
        b = pts[(i + 1) % n]
        if on_circle(a) and on_circle(b):
            # Coalesce a whole RUN of consecutive on-circle edges into a single
            # arc command (all lie on the same circle) so we emit one smooth arc
            # per visible segment instead of hundreds of 1px arclets. Split if the
            # accumulated sweep would exceed a semicircle (SVG large-arc flag).
            start = a
            total = 0.0
            j = i
            while j < n:
                p = pts[j]
                q = pts[(j + 1) % n]
                if not (on_circle(p) and on_circle(q)):
                    break
                seg = signed_span(p, q)
                if abs(total + seg) >= math.pi:
                    break
                total += seg
                j += 1
            end = pts[j % n]
            sweep = 1 if total > 0 else 0
            large = 1 if abs(total) > math.pi else 0
            d += f"A{r:.2f},{r:.2f} 0 {large},{sweep} {end[0]:.2f},{end[1]:.2f} "
            i = j if j > i else i + 1
        else:
            d += f"L{b[0]:.2f},{b[1]:.2f} "
            i += 1
    return d + "Z"
