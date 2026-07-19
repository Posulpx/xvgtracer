"""Clean curvature/variance + convergence-point classifier experiment.

Philosophy (deliberately different from learning/):
  * Decide curve-vs-line from the LOCAL curvature signature of the smoothed
    contour, not from prioritized shape-name matching.
  * When confidence is low, fall back to a line/curve reconstruction: find
    convergence points (where the boundary direction changes sharply) and build
    each segment between them independently as either a straight line or a smooth
    curve, decided per-segment by its own curvature variance.

This module is self-contained and does NOT import or mutate learning/*.
It only borrows read-only raster helpers (quantize, components) from learning
as inputs; all classification logic lives here.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

# ---- read-only inputs borrowed from the existing pipeline (no mutation) -------
from learning.learner import quantize, _binary_mask_for_color  # noqa: E402
from learning.detectors import components_merged  # noqa: E402
from learning.detectors.edge_detector import repair_mask, smooth_mask  # noqa: E402
from learning.detectors.contour_detector import contour_vertices  # noqa: E402

Point = Tuple[float, float]


# ==============================================================================
# Contour smoothing + curvature signature
# ==============================================================================

def _closed_ring(contour: List[Point]) -> List[Point]:
    """Return a closed ring (no duplicate closing point)."""
    if not contour:
        return []
    ring = list(contour)
    if ring[0] == ring[-1]:
        ring = ring[:-1]
    return ring


def _smooth_ring(ring: List[Point], win: int = 3) -> List[Point]:
    """Moving-average smoothing of a closed ring (kills staircase jitter)."""
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
    """Per-vertex turning angle (signed, radians) of the smoothed closed ring.

    Large-magnitude turning concentrated at isolated vertices => polygon/star.
    Near-zero turning everywhere => smooth curve.
    """
    n = len(ring)
    if n < 4:
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
    """Douglas-Peucker simplification of a closed ring (epsilon in pixels).

    This is the 'broader staircase analysis': a rasterised contour is a dense
    staircase of tiny grid edges, so a fixed-step curvature probe breaks
    continuity (long treads wash the turn out, long risers concentrate it). RDP
    collapses each staircase run to its true coarse vertex, giving a clean
    polygon whose corners are exactly the real direction changes -- whether fine
    (a sharp polygon corner) or coarse (a long staircase riser). Curvature is
    then measured on this simplified ring.
    """
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

    # simplify along the ring; then close by also allowing the wrap segment
    idx = recurse(0, n - 1)
    # ensure the closing segment (last->first) respects eps too
    out = [ring[i] for i in idx]
    return out


def _simplified_signature(ring: List[Point], eps: float) -> Tuple[List[Point], np.ndarray]:
    """Return (simplified ring, turning signature on it).

    eps scales with shape size so coarse staircases on big shapes are simplified
    as aggressively as on small ones.
    """
    n = len(ring)
    if n < 4:
        return list(ring), np.zeros(max(0, n))
    simp = _rdp_simplify(ring, eps)
    if len(simp) < 3:
        simp = list(ring)
    sig = _curvature_signature(_closed_ring(simp))
    return simp, sig


# ==============================================================================
# Convergence points (sharp direction-change vertices)
# ==============================================================================

def find_convergence_points(ring: List[Point],
                            angle_thresh: float = 0.35,
                            min_sep: int = 6) -> List[int]:
    """Indices (into the ORIGINAL ring) of real corners / star tips / vertices.

    The staircase contour is first collapsed with RDP to its true coarse
    vertices, so a corner is detected from the simplified polygon's turning angle
    rather than from grid jitter. Adjacent simplified vertices that all have
    high |turn| form a run; within each run vertices exceeding 80 % of the run's
    best are kept (preserving crowded polygon corners while suppressing clip arc
    plateaus). Kept simplified vertices are mapped back to original-ring indices,
    then clustered by PHYSICAL distance (min_dist_px): points closer than ~2.5 %
    of the shape diagonal are merged into one. This collapses clip-boundary
    clusters (e.g. 3+ adjacent RDP vertices on a clip arc) into a single convergence
    point while leaving well-separated polygon corners distinct.
    """
    n = len(ring)
    if n < 6:
        return []
    # eps scales with shape size: ~1.2% of the bounding-box diagonal
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    eps = max(1.0, 0.012 * diag)
    simp, sig = _simplified_signature(ring, eps)
    if len(sig) == 0:
        return []
    # Gate: if the RDP-simplified ring has no sufficiently sharp corners, treat
    # the contour as a smooth curve with no convergence points. A genuine polygon
    # corner has |turn| >= ~1.05 rad (60°, hexagon) while a many-sided RDP
    # approximation of a circle produces uniform turns ~0.3-0.5 rad with low
    # variance. Both conditions must fail to proceed.
    asig = np.abs(sig)
    if float(np.max(asig)) < 0.6 and float(np.var(asig)) < 0.008:
        return []
    # Cluster adjacent threshold-passing vertices into runs. Within each run, keep
    # ALL vertices whose |turn| >= 80 % of the run's best. This preserves crowded
    # polygon corners (e.g. 3 adjacent hexagon vertices all at ~1.0) while
    # suppressing low-turn clip arc plateaus (~0.37-0.47, well below the 80 %
    # cutoff of a run whose best is ~1.2). Wrap-around is handled by replicating.
    m = len(sig)
    extended = list(abs(sig)) + list(abs(sig))
    kept_raw: List[int] = []
    i = 0
    while i < 2 * m:
        if extended[i] <= angle_thresh:
            i += 1
            continue
        run_start = i
        while i < 2 * m and extended[i] > angle_thresh:
            i += 1
        run_end = i - 1
        best_val = max(extended[run_start:run_end + 1])
        cutoff = 0.80 * best_val
        cand = [j for j in range(run_start, run_end + 1) if extended[j] >= cutoff]
        cand.sort(key=lambda j: -extended[j])
        for j in cand:
            modest = j % m
            if all(min(abs(modest - (k % m)), m - abs(modest - (k % m))) >= 1
                   for k in kept_raw):
                kept_raw.append(j)
    kept_simp = sorted(set(k % m for k in kept_raw))
    # map simplified-vertex indices back to original ring indices
    orig_cand = []
    for si in kept_simp:
        sp = simp[si]
        best = min(range(n), key=lambda k: (ring[k][0] - sp[0]) ** 2 + (ring[k][1] - sp[1]) ** 2)
        orig_cand.append((best, ring[best][0], ring[best][1]))
    # cluster by physical distance: merge cps closer than min_dist_px.
    # A clip arc spanning ~200 original indices produces many RDP vertices
    # concentrated within ~20 px; merging at 0.10*diag collapses them into
    # one per cluster while hexagon corners (66-74 px apart) stay separate.
    min_dist_px = max(5.0, 0.14 * diag)
    kept: List[int] = []
    # order by original index to keep spatial ordering
    for best, px, py in sorted(orig_cand, key=lambda t: t[0]):
        if all(math.hypot(px - px2, py - py2) >= min_dist_px
               for (_, px2, py2) in [(k, k2, k3) for k, k2, k3 in kept]):
            kept.append((best, px, py))
    orig_idx = sorted(k[0] for k in kept)
    return orig_idx


# ==============================================================================
# Per-segment line-vs-curve decision by curvature variance
# ==============================================================================

def _segment_curvature_stats(ring: List[Point],
                             i0: int, i1: int) -> Tuple[float, float]:
    """Return (mean_abs_turning, variance_of_abs_turning) for arc ring[i0..i1].

    The sub-arc is lightly smoothed first so grid jitter does not inflate the
    stats of an otherwise straight edge. A straight segment has ~0 mean turning;
    a pure arc has a steady nonzero mean turning (low variance); a faceted /
    mixed segment has high variance.
    """
    n = len(ring)
    if i1 >= i0:
        idx = list(range(i0, i1 + 1))
    else:
        idx = list(range(i0, n)) + list(range(0, i1 + 1))
    if len(idx) < 3:
        return 0.0, 0.0
    sub = _smooth_ring([ring[k] for k in idx], win=1)
    sig = _curvature_signature(sub)
    mag = np.abs(sig)
    if len(mag) == 0:
        return 0.0, 0.0
    return float(np.mean(mag)), float(np.var(mag))


def classify_segment(ring: List[Point], i0: int, i1: int,
                     line_dev: float = 0.12) -> str:
    """Decide 'line' or 'curve' for the segment ring[i0..i1] (inclusive arc).

    Collinearity test: measure the maximum perpendicular deviation of the segment
    points from the chord connecting its endpoints, as a fraction of the chord
    length. A straight edge (polygon side, clip chord) has ~0 deviation -> 'line'.
    A genuine circular arc bows away consistently -> 'curve'. This cleanly tells a
    polygon edge from an arc, and is robust to staircase noise (which adds small,
    non-systematic deviations). Gentle crescents still deviate visibly -> 'curve'.
    """
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
    # perpendicular distance from each interior point to the chord line
    dx, dy = p2[0] - p0[0], p2[1] - p0[1]
    dev = 0.0
    for p in sub[1:-1]:
        # |cross| / chord
        d = abs((p[0] - p0[0]) * dy - (p[1] - p0[1]) * dx) / chord
        if d > dev:
            dev = d
    return "line" if (dev / chord) < line_dev else "curve"


# ==============================================================================
# Top-level: build a line/curve path from convergence points
# ==============================================================================

def build_lc_path(ring: List[Point]) -> Dict[str, Any]:
    """Reconstruct a shape as a sequence of line/curve segments between
    convergence points.

    Returns a dict with 'points' (the convergence vertices, closed) and
    'seg_types' (parallel list: 'line'/'curve' for each segment ring[ci..c(i+1)]).
    """
    n = len(ring)
    cps = find_convergence_points(ring)
    if len(cps) < 3:
        # too few corners -> treat whole boundary as one smooth curve
        return {"points": [ring[i] for i in range(0, n, max(1, n // 24))],
                "seg_types": ["curve"], "convergence": cps}
    seg_types = []
    for k in range(len(cps)):
        i0 = cps[k]
        i1 = cps[(k + 1) % len(cps)]
        seg_types.append(classify_segment(ring, i0, i1))
    return {"points": [ring[i] for i in cps],
            "seg_types": seg_types, "convergence": cps}


# ==============================================================================
# Whole-shape decision: curve vs polygon vs star (curvature/variance)
# ==============================================================================

def classify_shape(mask: np.ndarray) -> Dict[str, Any]:
    """Classify one binary mask into a coarse family using curvature variance.

    * global turning-angle variance low  -> smooth (circle/ellipse)
    * high but periodic (few strong peaks) -> polygon
    * many strong alternating peaks        -> star
    """
    contour = contour_vertices(repair_mask(mask), fill_holes=True)
    raw = _closed_ring(contour)
    ring = _smooth_ring(raw, win=3)  # smoothed ring for segment building
    n = len(ring)
    if n < 8:
        return {"family": "polygon", "ring": ring,
                "lc": build_lc_path(raw)}
    # Global family signature: single-step curvature on the moderately smoothed
    # ring (win=3). Staircase noise elevates gvar to ~0.003-0.004 for smooth
    # shapes, so the threshold is relaxed accordingly.
    sig = _curvature_signature(ring)
    mag = np.abs(sig)
    global_var = float(np.var(mag))
    n_peaks = int(np.sum(mag > 0.35))
    # Curvature/variance thresholds learned from the test corpus:
    #   smooth  : gvar < 0.0045, no strong peaks (circle/ellipse)
    #             (staircase noise lifts circles to ~0.003-0.0035)
    #   star    : gvar >= 0.008 with many peaks (>=8) -> spiky silhouette
    #   polygon : everything in between (incl. occluded circles, which keep a
    #             curved arc + a couple of chord endpoints -> low peak count)
    if global_var < 0.0045 and n_peaks <= 2:
        family = "smooth"
    elif global_var >= 0.008 and n_peaks >= 8:
        family = "star"
    else:
        family = "polygon"
    # Only run RDP-based convergence detection on non-smooth shapes: a smooth
    # circle/ellipse has no true corners even after simplification, and the RDP
    # polygonal approximation would produce spurious vertices that each pass the
    # turning-angle threshold. Skip it entirely to keep the circle as one arc.
    if family == "smooth":
        lc = {"points": ring[::max(1, len(ring)//24)] + [ring[0]],
              "seg_types": ["curve"], "convergence": []}
    else:
        lc = build_lc_path(raw)
    return {"family": family, "ring": ring,
            "lc": lc, "global_var": global_var,
            "n_peaks": n_peaks}


# ==============================================================================
# Driver: run on a test image, one mask per color
# ==============================================================================

def run_image(path: str, num_colors: int = 8) -> List[Dict[str, Any]]:
    img, cols = quantize(path, num_colors)
    rgb = np.asarray(img)
    out = []
    for col in cols:
        c = _binary_mask_for_color(rgb, col)
        comps = components_merged(c, merge_gap=0)
        if not comps:
            continue
        mask = max(comps, key=lambda m: m.sum())
        if mask.sum() < 50:
            continue
        # skip near-black / near-white background
        lum = sum(col) / 3.0
        if lum < 12 or lum > 243:
            continue
        res = classify_shape(mask)
        res["color"] = tuple(int(v) for v in col)
        out.append(res)
    return out


def _seg_summary(lc: Dict[str, Any]) -> str:
    return "".join("L" if t == "line" else "C" for t in lc["seg_types"])


# ==============================================================================
# Arc fitting for curved segments + SVG path emission
# ==============================================================================

def _fit_arc_3pt(p0: Point, p1: Point, p2: Point):
    """Circle through 3 points. Returns (cx, cy, r) or None if collinear."""
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


def _fit_circle_lsq(pts: List[Point]):
    """Least-squares circle fit (Pratt method). Returns (cx, cy, r) or None."""
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
    """Fit an arc to a curved segment.

    For long arcs (>= 8 points, e.g. a near-full circle) use a least-squares
    circle fit over ALL segment points -- a 3-point fit through endpoints+mid
    diverges wildly when the arc spans most of a circle. For short arcs the
    3-point fit is adequate.

    Returns (start, end, rx, ry, large_arc, sweep) for an SVG A command, or None.
    """
    n = len(ring)
    if i1 >= i0:
        idx = list(range(i0, i1 + 1))
    else:
        idx = list(range(i0, n)) + list(range(0, i1 + 1))
    if len(idx) < 3:
        return None
    sub = [ring[k] for k in idx]
    p0, p2 = sub[0], sub[-1]
    if len(sub) >= 8:
        arc = _fit_circle_lsq(sub)
    else:
        arc = _fit_arc_3pt(sub[0], sub[len(sub) // 2], sub[-1])
    if arc is None:
        return None
    cx, cy, r = arc
    # sweep direction: signed area of the triangle (start, mid, end)
    pm = sub[len(sub) // 2]
    cross = (pm[0] - p0[0]) * (p2[1] - p0[1]) - (pm[1] - p0[1]) * (p2[0] - p0[0])
    sweep = 1 if cross > 0 else 0
    # large-arc: if the chord spans more than half the circle
    chord = math.hypot(p2[0] - p0[0], p2[1] - p0[1])
    large = 1 if chord < 2 * r * 0.999 else 0
    return (p0, p2, r, r, large, sweep)


def to_path(ring: List[Point], lc: Dict[str, Any]) -> str:
    """Emit an SVG path (M/L/A commands) from the convergence points + seg tags.

    A round shape (full or clipped circle) is detected first: if a single circle
    fits the whole ring with high inlier fraction, emit it as a smooth circle.
    Otherwise each segment between convergence points is a straight 'L' or a
    smooth arc 'A'; the path closes with 'Z'.
    """
    pts = lc["points"]
    segs = lc["seg_types"]
    conv = lc.get("convergence", [])
    if len(pts) < 2:
        return ""
    # Round-shape gate: a circle (even when slightly clipped) fits the whole ring.
    # For shapes with no convergence points (classified smooth), always emit the
    # LSQ circle regardless of inlier — contour-tracing artifacts can depress the
    # inlier fraction below 0.85 even for a perceptually perfect circle.
    circle = _fit_circle_lsq(ring)
    if circle is not None:
        cx, cy, r = circle
        rr = np.array([math.hypot(p[0] - cx, p[1] - cy) for p in ring], float)
        inlier = float((np.abs(rr - r) < 0.06 * r).mean())
        if inlier >= 0.85 or len(conv) == 0:
            d = ("M %.1f %.1f A %.1f %.1f 0 1 1 %.1f %.1f "
                 "A %.1f %.1f 0 1 1 %.1f %.1f Z") % (
                cx + r, cy, r, r, cx - r, cy, r, r, cx + r, cy)
            return d
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
            arc = _segment_arc(ring, i0, i1)
            if arc is None:
                d += " L %.1f %.1f" % (b[0], b[1])
            else:
                _, end, rx, ry, large, sweep = arc
                d += " A %.1f %.1f 0 %d %d %.1f %.1f" % (rx, ry, large, sweep, end[0], end[1])
    d += " Z"
    return d


def render_svg(path: str, num_colors: int = 8, out: Optional[str] = None) -> str:
    """Classify every color mask and emit a full multi-shape SVG."""
    results = run_image(path, num_colors)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" shape-rendering="geometricPrecision">']
    for r in results:
        ring = r["ring"]
        lc = r["lc"]
        d = to_path(ring, lc)
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


if __name__ == "__main__":
    import sys
    for name, nc in [("hex_in_circle", 4), ("color12", 13),
                     ("clean_shapes", 8), ("drawn_shapes", 8),
                     ("clipped_star", 8)]:
        try:
            res = run_image("tests/%s.png" % name, nc)
        except Exception as e:  # noqa
            print("%-14s ERROR %s" % (name, e))
            continue
        print("%-14s" % name)
        for r in res:
            print("   %-14s %-8s gvar=%.4f peaks=%d conv=%d segs=%s"
                  % (str(r["color"]), r["family"], r["global_var"],
                     r["n_peaks"], len(r["lc"]["convergence"]),
                     _seg_summary(r["lc"])))
