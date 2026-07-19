from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._image import (
    Point,
    components_merged,
    contour_vertices,
    mask_for_color,
    quantize,
    repair_mask,
)

# ==============================================================================
# Contour smoothing + curvature signature
# ==============================================================================


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


# ==============================================================================
# RDP simplification
# ==============================================================================


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
    out = [ring[i] for i in idx]
    return out


def _simplified_signature(ring: List[Point], eps: float) -> Tuple[List[Point], np.ndarray]:
    n = len(ring)
    if n < 4:
        return list(ring), np.zeros(max(0, n))
    simp = _rdp_simplify(ring, eps)
    if len(simp) < 3:
        simp = list(ring)
    sig = _curvature_signature(_closed_ring(simp))
    return simp, sig


# ==============================================================================
# Convergence points
# ==============================================================================


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
    angle_thresh = 0.35
    high_thresh = 0.5
    kept_raw: List[Tuple[int, float]] = []
    i = 0
    while i < 2 * m:
        if extended[i] <= angle_thresh:
            i += 1
            continue
        run_start = i
        while i < 2 * m and extended[i] > angle_thresh:
            i += 1
        run_end = i - 1
        has_high = any(extended[j] > high_thresh for j in range(run_start, run_end + 1))
        if has_high:
            to_keep = list(range(run_start, run_end + 1))
        else:
            to_keep = [max(range(run_start, run_end + 1), key=lambda j: extended[j])]
        for j in to_keep:
            modest = j % m
            if all(min(abs(modest - (k % m)), m - abs(modest - (k % m))) >= 1
                   for k, _ in kept_raw):
                kept_raw.append((modest, float(abs(sig[modest]))))
    kept_simp = sorted(set(k for k, _ in kept_raw))
    orig_cand = []
    for si in kept_simp:
        sp = simp[si]
        best = min(range(n), key=lambda k: (ring[k][0] - sp[0]) ** 2 + (ring[k][1] - sp[1]) ** 2)
        strength = max(abs(sig[si]) for ks, a in kept_raw if ks == si for b in [a])
        orig_cand.append((best, ring[best][0], ring[best][1], float(strength)))
    min_dist_px = max(5.0, 0.14 * diag)
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


# ==============================================================================
# Per-segment line/curve decision
# ==============================================================================


def classify_segment(ring: List[Point], i0: int, i1: int,
                     line_dev: float = 0.12) -> str:
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
    if chord < 1e-6:
        return "line"
    dx, dy = p2[0] - p0[0], p2[1] - p0[1]
    dev = 0.0
    for p in sub[1:-1]:
        d = abs((p[0] - p0[0]) * dy - (p[1] - p0[1]) * dx) / chord
        if d > dev:
            dev = d
    return "line" if (dev / chord) < line_dev else "curve"


# ==============================================================================
# Build line/curve path from convergence points
# ==============================================================================


def build_lc_path(ring: List[Point]) -> Dict[str, Any]:
    raw = find_convergence_points(ring)
    cps = [p[0] for p in raw]
    if len(cps) < 2:
        n = len(ring)
        return {"points": [ring[i] for i in range(0, n, max(1, n // 24))],
                "seg_types": ["curve"], "convergence": cps}
    strengths = dict(raw)
    if len(cps) >= 4:
        strong = [i for i, cp in enumerate(cps) if strengths.get(cp, 0) > 0.8]
        if len(strong) == 2 and len(cps) <= 5:
            cps = [cps[strong[0]], cps[strong[1]]]
    seg_types = []
    for k in range(len(cps)):
        i0 = cps[k]
        i1 = cps[(k + 1) % len(cps)]
        seg_types.append(classify_segment(ring, i0, i1))
    return {"points": [ring[i] for i in cps],
            "seg_types": seg_types, "convergence": cps}


# ==============================================================================
# Whole-shape family decision
# ==============================================================================


def classify_shape(mask: np.ndarray) -> Dict[str, Any]:
    contour = contour_vertices(repair_mask(mask), fill_holes=True)
    raw = _closed_ring(contour)
    ring = _smooth_ring(raw, win=3)
    n = len(ring)
    if n < 8:
        return {"family": "polygon", "ring": ring,
                "lc": build_lc_path(raw)}
    # Global family decision uses the RDP-simplified ring's turning signature.
    # RDP removes all staircase noise, leaving only the true coarse geometry.
    #
    # Three families: smooth, polygon, star.
    #   - smooth: no sharp corners OR a convex shape where only a minority of
    #     RDP vertices exceed 0.8 rad (ellipse with varying curvature).
    #   - polygon: convex shape where MOST RDP vertices have sharp turns.
    #   - star: alternating turn signs + high variance.
    xs = [p[0] for p in raw]
    ys = [p[1] for p in raw]
    diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    eps_family = max(1.0, 0.012 * diag)
    _, sig_simp = _simplified_signature(raw, eps_family)
    mag_simp = np.abs(sig_simp)
    global_var = float(np.var(mag_simp)) if len(mag_simp) else 0.0
    n_peaks = int(np.sum(mag_simp > 0.35))
    max_turn = float(np.max(mag_simp)) if len(mag_simp) else 0.0
    # count sign alternations in the RDP turn sequence
    signs = [1 if s > 0 else -1 for s in sig_simp]
    n_sign = sum(1 for i in range(len(signs))
                 if signs[i] != signs[(i + 1) % len(signs)])
    n_strong = int(np.sum(mag_simp > 0.8))
    if max_turn < 0.5:
        family = "smooth"
    elif n_sign <= 2:
        if n_strong >= 2:
            family = "polygon"
        else:
            n_low = int(np.sum(mag_simp < 0.4))
            n_high = int(np.sum(mag_simp > 0.5))
            family = "polygon" if n_low >= 2 and n_high >= 2 else "smooth"
    elif global_var >= 0.05 and n_peaks >= 6:
        family = "star"
    elif n_peaks >= 8 and global_var < 0.05:
        family = "blob"
    else:
        family = "polygon"
    if family == "smooth":
        lc = {"points": ring[::max(1, len(ring)//24)] + [ring[0]],
              "seg_types": ["curve"], "convergence": []}
    else:
        lc = build_lc_path(raw)
        # For shapes with uniform curvature direction (all RDP turns same sign)
        # and few strong corners, force all segments to curve for smooth reproduction.
        if n_sign <= 2 and n_strong < 2 and len(lc.get("seg_types", [])) > 0:
            lc["seg_types"] = ["curve"] * len(lc["seg_types"])
    return {"family": family, "ring": ring,
            "lc": lc, "global_var": global_var,
            "n_peaks": n_peaks}


# ==============================================================================
# Driver
# ==============================================================================


def run_image(path: str, num_colors: int = 8) -> List[Dict[str, Any]]:
    img, cols = quantize(path, num_colors)
    rgb = np.asarray(img)
    out = []
    for col in cols:
        c = mask_for_color(rgb, col)
        comps = components_merged(c, merge_gap=0)
        if not comps:
            continue
        mask = max(comps, key=lambda m: m.sum())
        if mask.sum() < 50:
            continue
        lum = sum(col) / 3.0
        if lum < 12 or lum > 243:
            continue
        res = classify_shape(mask)
        res["color"] = tuple(int(v) for v in col)
        out.append(res)
    return out


# ==============================================================================
# Arc fitting + SVG path emission
# ==============================================================================


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


def _fit_bezier_lsq(ring: List[Point], i0: int, i1: int) -> Optional[Tuple[float, float, float, float]]:
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


def to_path(ring: List[Point], lc: Dict[str, Any], family: str = "") -> str:
    pts = lc["points"]
    segs = lc["seg_types"]
    conv = lc.get("convergence", [])
    if len(pts) < 2:
        return ""
    if len(conv) == 0:
        ell = _fit_ellipse_lsq(ring)
        if ell is not None:
            cx, cy, rx, ry, angle = ell
            if max(rx, ry) / min(rx, ry) > 1.05:
                ang = (math.degrees(angle) + 360.0) % 360.0
                sx = cx - rx * math.cos(angle)
                sy = cy - rx * math.sin(angle)
                ex = cx + rx * math.cos(angle)
                ey = cy + rx * math.sin(angle)
                d = ("M %.1f %.1f A %.1f %.1f %.1f 1 0 %.1f %.1f "
                     "A %.1f %.1f %.1f 1 0 %.1f %.1f Z") % (
                    sx, sy, rx, ry, ang, ex, ey, rx, ry, ang, sx, sy)
                return d
        circle = _fit_circle_lsq(ring)
        if circle is not None:
            cx, cy, r = circle
            d = ("M %.1f %.1f A %.1f %.1f 0 1 1 %.1f %.1f "
                 "A %.1f %.1f 0 1 1 %.1f %.1f Z") % (
                cx + r, cy, r, r, cx - r, cy, r, r, cx + r, cy)
            return d
    if family == "blob":
        n = len(ring)
        step = max(1, n // 24)
        sub = [ring[i] for i in range(0, n, step)]
        sp = _spline_path(sub)
        if sp:
            return sp
    use_conv = len(conv) == len(pts)
    d = "M %.1f %.1f" % (pts[0][0], pts[0][1])
    for k in range(len(pts)):
        b = pts[(k + 1) % len(pts)]
        typ = segs[k] if k < len(segs) else "curve"
        if typ == "line":
            d += " L %.1f %.1f" % (b[0], b[1])
        else:
            if use_conv:
                i0 = conv[k]
                i1 = conv[(k + 1) % len(pts)]
            else:
                n = len(ring)
                i0 = int(k * n / len(pts))
                i1 = int((k + 1) * n / len(pts)) - 1
                if i1 <= i0:
                    i1 = i0 + max(1, n // len(pts))
            # For long curve segments (>20 contour points), split at midpoint
            N = len(ring)
            i0c, i1c = i0, i1
            if i1c < i0c:
                i1c += N
            cnt_len = i1c - i0c + 1
            if cnt_len > 20:
                mid = (i0c + i1c) // 2
                mid_idx = mid % N
                sub = [(i0, mid_idx), (mid_idx, i1)]
                for si0, si1 in sub:
                    ctrl = _fit_bezier_lsq(ring, si0, si1)
                    sb = ring[si1]
                    if ctrl is not None:
                        c1x, c1y, c2x, c2y = ctrl
                        d += " C %.1f %.1f %.1f %.1f %.1f %.1f" % (c1x, c1y, c2x, c2y, sb[0], sb[1])
                    else:
                        d += " L %.1f %.1f" % (sb[0], sb[1])
            else:
                ctrl = _fit_bezier_lsq(ring, i0, i1)
                if ctrl is not None:
                    c1x, c1y, c2x, c2y = ctrl
                    d += " C %.1f %.1f %.1f %.1f %.1f %.1f" % (c1x, c1y, c2x, c2y, b[0], b[1])
                else:
                    d += " L %.1f %.1f" % (b[0], b[1])
    d += " Z"
    return d


def render_svg(path: str, num_colors: int = 8, out: Optional[str] = None) -> str:
    results = run_image(path, num_colors)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="800" viewBox="0 0 800 800" shape-rendering="geometricPrecision">']
    for r in results:
        ring = r["ring"]
        lc = r["lc"]
        d = to_path(ring, lc, family=r["family"])
        if not d:
            continue
        col = r["color"]
        fill = "rgb(%d,%d,%d)" % col
        parts.append('  <path d="%s" fill="%s" stroke="%s" stroke-width="1"/>' % (d, fill, fill))
    parts.append("</svg>")
    svg = "\n".join(parts)
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(svg)
    return svg


def _seg_summary(lc: Dict[str, Any]) -> str:
    return "".join("L" if t == "line" else "C" for t in lc["seg_types"])
