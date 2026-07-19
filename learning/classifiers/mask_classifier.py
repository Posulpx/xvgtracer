"""Mask classifier: turn a binary mask into a primitive node.

This module owns all mask-classification logic (previously inside
:mod:`learning.shape_registry`):

  * contour simplification (RDP) and star detection,
  * orientation estimation + rotated-primitive fitting,
  * the `classify_mask` pipeline (repair -> extent -> best-fit IoU -> star ->
    rotated -> polygon fallback),
  * `build_transform` to attach a `rotate` Transform to rotated rects/ellipses.

It uses :func:`learning.shape_registry.fit_candidates` as the shared ranking
helper so adding a primitive is a one-line addition to the candidate map.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..detectors import (
    contour_vertices,
    repair_mask,
    smooth_mask,
    mask_extent,
    centroid_of_mask,
)
from ..generators import (
    circle_points,
    ellipse_points,
    rectangle_points,
    rounded_rectangle_points,
    triangle_points,
)
from ..renderers import rasterize_polygon
from ..metrics import iou_masks
from ..shape_registry import fit_candidates

Point = Tuple[float, float]

# Boundary-frame alignment. `mask_extent` returns a cell box (min, max+1) whose
# center sits +0.5px off the boundary the contour tracer (skimage find_contours
# at the 0.5 iso-level) produces. Traced polygons (custom shapes, corners, clip
# edges) live in that boundary frame, so a parametric circle/ellipse/rect built
# from the cell box renders ~0.5px shifted and opens gaps against neighbouring
# traced shapes -- visible on small sources. We keep classification/detection in
# pixel-index space (star/triangle radial math is sensitive to the centre) and
# apply this -0.5 correction ONLY to the emitted parametric params, so the drawn
# primitive shares the traced boundary frame without disturbing any fit logic.
_BOUNDARY_ADJ = 0.5

# Candidate primitives tried during fitting (name -> builder).
# Each builder takes (x0, y0, x1, y1, cx, cy) and returns a point list.
def _cand_circle(x0, y0, x1, y1, cx, cy):
    rx = (x1 - x0) / 2.0
    ry = (y1 - y0) / 2.0
    r = (rx + ry) / 2.0
    return circle_points(cx, cy, r)

def _cand_ellipse(x0, y0, x1, y1, cx, cy):
    rx = (x1 - x0) / 2.0
    ry = (y1 - y0) / 2.0
    return ellipse_points(cx, cy, rx, ry)

def _cand_rect(x0, y0, x1, y1, cx, cy):
    return rectangle_points(x0, y0, x1, y1)

def _cand_rounded_rect(x0, y0, x1, y1, cx, cy):
    r = min(x1 - x0, y1 - y0) * 0.18
    return rounded_rectangle_points(x0, y0, x1, y1, r)

def _cand_triangle(x0, y0, x1, y1, cx, cy):
    rx = (x1 - x0) / 2.0
    ry = (y1 - y0) / 2.0
    return triangle_points(cx, cy, rx, ry)

CANDIDATE_BUILDERS = {
    "circle": _cand_circle,
    "ellipse": _cand_ellipse,
    "rect": _cand_rect,
    "rounded_rect": _cand_rounded_rect,
    "triangle": _cand_triangle,
}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _rdp(points: List[Point], epsilon: float) -> List[Point]:
    """Ramer-Douglas-Peucker polyline simplification (closed-contour aware)."""
    if len(points) < 3:
        return list(points)
    start, end = points[0], points[-1]
    dx, dy = end[0] - start[0], end[1] - start[1]
    norm = math.hypot(dx, dy)
    if norm == 0:
        return [start, end]
    dmax, idx = 0.0, 0
    for i in range(1, len(points) - 1):
        x, y = points[i]
        d = abs(dy * x - dx * y + end[0] * start[1] - end[1] * start[0]) / norm
        if d > dmax:
            dmax, idx = d, i
    if dmax > epsilon:
        return _rdp(points[:idx + 1], epsilon)[:-1] + _rdp(points[idx:], epsilon)
    return [start, end]


def _simplify_ring(verts: List[Point], epsilon: float = 2.0) -> List[Point]:
    if len(verts) < 3:
        return list(verts)
    simplified = _rdp(verts[:-1], epsilon)
    if simplified and simplified[0] != simplified[-1]:
        simplified.append(simplified[0])
    return simplified


def _resampled_radii(pts: List[Point], center: Point, n_bins: int = 72) -> List[float]:
    """Sample the contour at evenly spaced angles around `center`.

    Robust to RDP collapsing sharp tips: we bin the contour by polar angle and
    take the *max* radius in each bin, so a star's tips always survive.
    """
    cx, cy = center
    radii = [0.0] * n_bins
    for x, y in pts:
        ang = (math.atan2(y - cy, x - cx) + math.pi) % (2 * math.pi)
        b = int(ang / (2 * math.pi) * n_bins) % n_bins
        d = math.hypot(x - cx, y - cy)
        if d > radii[b]:
            radii[b] = d
    return radii


def _detect_star(pts: List[Point], center: Point, mask: Optional[np.ndarray] = None) -> bool:
    """Star detection via strictly-alternating radial tips/valleys + contrast.

    Uses angle-binned radii (max radius per bin) so sharp tips survive contour
    simplification. A genuine star shows tips and valleys that *strictly
    alternate* (never two maxima or two minima adjacent), which separates it
    from noisy rounded-rect/triangle contours.

    A star whose arms are truncated by an overlapping shape (e.g. a yellow
    circle cutting clean through an arm) loses the radial symmetry this check
    enforces: the alternation breaks, the tip/valley balance collapses, or the
    reconstructed star can no longer re-rasterise onto the *actual* mask. Such
    a clipped silhouette is NOT a star — it must fall through to custom polygon
    reconstruction. We therefore require strict alternation, near-even tip/valley
    counts, and (when the mask is available) a high re-rasterised IoU.
    """
    if len(pts) < 8:
        return False
    radii = _resampled_radii(pts, center)
    n = len(radii)
    # classify each bin as peak (1), valley (-1), or flat (0)
    cls = []
    for i in range(n):
        a = radii[(i - 1) % n]
        b = radii[i]
        c = radii[(i + 1) % n]
        if b > a and b >= c:
            cls.append(1)
        elif b < a and b <= c:
            cls.append(-1)
        else:
            cls.append(0)
    # count directional changes (alternations) across non-flat bins
    seq = [v for v in cls if v != 0]
    if len(seq) < 8:
        return False
    alternations = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
    # require near-perfect alternation: almost every step flips direction
    if alternations < len(seq) - 2:
        return False
    # tips and valleys must be (near) balanced — a clipped star has more
    # valleys than tips (or vice-versa), breaking the strict alternation count.
    n_peaks = sum(1 for v in seq if v == 1)
    n_valleys = sum(1 for v in seq if v == -1)
    if abs(n_peaks - n_valleys) > 1:
        return False
    rmin, rmax = min(radii), max(radii)
    contrast = (rmax - rmin) / max(rmax, 1e-6)
    if contrast <= 0.30:
        return False
    # A genuine N-point star needs at least N well-defined tips (local maxima
    # well above the valley floor), not just one overlap-notch on a circle.
    thr = rmin + 0.25 * (rmax - rmin)
    n_tips = sum(1 for i in range(len(radii))
                 if radii[i] >= radii[(i - 1) % len(radii)]
                 and radii[i] >= radii[(i + 1) % len(radii)]
                 and radii[i] > thr)
    if n_tips < 5:
        return False
    # Hard guard: only accept the star if its reconstructed vertices re-rasterise
    # onto the ACTUAL mask. A truncated silhouette (arms cut by an overlap) fails
    # this and is correctly rejected in favour of custom polygon reconstruction.
    if mask is not None:
        best_iou = -1.0
        h, w = mask.shape
        for extractor in (_star_vertices, _star_vertices_angle):
            sv = extractor(pts, center, n_tips=max(5, min(n_tips, 8)))
            if not sv or len(sv) < 2 * 3:
                continue
            r = rasterize_polygon(sv + [sv[0]], w, h)
            iou = iou_masks(r, mask)
            if iou > best_iou:
                best_iou = iou
        if best_iou < 0.80:
            return False
    # Concavity guard: a genuine star has genuinely CONCAVE valley vertices — that
    # is what makes it a star rather than a regular polygon whose "valleys" are
    # just convex edge midpoints. A convex polygon (even one clipped by an
    # overlapping shape) has only convex vertices, so it is NOT a star. We test the
    # EXTRACTED star vertices (clean tips/valleys), not the raw staircase contour,
    # else grid jitter would read as spurious concave vertices.
    sv = None
    for extractor in (_star_vertices, _star_vertices_angle):
        cand = extractor(pts, center, n_tips=max(5, min(n_tips, 8)))
        if cand and len(cand) >= 6:
            sv = cand
            break
    if sv is None or not _has_concave_vertex(sv, center):
        return False
    # Convex-polygon guard: a regular/convex N-gon (hexagon, pentagon) also yields
    # an alternating radius profile that passes the star test, but it is NOT a star.
    # Such a polygon sits almost entirely within its circumcircle (disk IoU high,
    # e.g. hexagon ~0.93), whereas a genuine star or a clipped circle leaves most of
    # its circumcircle empty. If the mask's own circumcircle (fitted from the full
    # contour around the centroid) accounts for most of the mask, reject the star in
    # favour of a polygon. Use the full contour (not the extracted sv, which can be
    # distorted by clipping) so the true circumcircle is captured.
    if mask is not None and pts:
        import numpy as _np
        px = _np.array([p[0] for p in pts], float)
        py = _np.array([p[1] for p in pts], float)
        scx, scy = float(px.mean()), float(py.mean())
        sr = float(_np.sqrt(((px - scx) ** 2 + (py - scy) ** 2).max())) + 1e-6
        h, w = mask.shape
        yy, xx = _np.mgrid[0:h, 0:w]
        disk = (xx - scx) ** 2 + (yy - scy) ** 2 <= sr * sr
        disk_iou = (mask & disk).sum() / max(1, int((mask | disk).sum()))
        if disk_iou >= 0.80:
            return False
    return True


def _interior_angle(a, b, c) -> float:
    """Interior angle (radians) at vertex b of the polyline a->b->c (CCW order)."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    ang = math.atan2(abs(cross), dot)
    if cross < 0:  # reflex / concave when the turn is clockwise in y-down space
        ang = 2 * math.pi - ang
    return ang


def _has_concave_vertex(pts: List[Point], center: Point) -> bool:
    """True if the (closed) ring has at least one clearly concave (reflex) vertex.

    Used to separate genuine stars (concave valleys) from regular/convex polygons
    whose alternating radius profile could otherwise mimic a star. A reflex vertex
    is one whose turn direction is OPPOSITE the polygon's overall winding (so the
    test is independent of whether the ring is ordered CW or CCW in y-down space).
    """
    if len(pts) < 4:
        return False
    ring = pts[:]
    if ring[0] != ring[-1]:
        ring = ring + [ring[0]]
    n = len(ring)
    # overall winding from the polygon's signed area
    area = 0.0
    for i in range(n - 1):
        x0, y0 = ring[i]
        x1, y1 = ring[i + 1]
        area += x0 * y1 - x1 * y0
    winding = 1.0 if area >= 0 else -1.0
    for i in range(1, n - 1):
        a, b, c = ring[i - 1], ring[i], ring[i + 1]
        cross = (a[0] - b[0]) * (c[1] - b[1]) - (a[1] - b[1]) * (c[0] - b[0])
        if winding * cross < -1e-6:  # turn opposite the overall winding -> reflex
            if _interior_angle(a, b, c) > math.pi + 0.20:
                return True
    return False


def _star_vertices(pts: List[Point], center: Point, n_tips: int = 5) -> Optional[List[Point]]:
    """Extract the exact tip/valley vertices of a star from its contour.

    Tips are the N strongest radius maxima (de-duplicated by angular spacing so
    a fat arm can't yield two tips), detected on an angle-binned radius profile
    for stability. Each valley is the *minimum radius over a window* centred on
    the midpoint between consecutive tips, sampled from the same profile. This
    yields a clean parametric star and works for both regular and slightly
    irregular stars.
    """
    cx, cy = center
    radii = _resampled_radii(pts, center, n_bins=72)
    n = len(radii)
    # candidate tips: local maxima of the binned radius profile
    cands = [(radii[i], i) for i in range(n)
             if radii[i] >= radii[(i - 1) % n] and radii[i] >= radii[(i + 1) % n]
             and radii[i] > 0]
    cands.sort(reverse=True)
    min_sep = n // (n_tips + 1)
    tips = []
    for r, i in cands:
        ok = True
        for _, j in tips:
            d = min(abs(i - j), n - abs(i - j))
            if d < min_sep:
                ok = False
                break
        if ok:
            tips.append((r, i))
        if len(tips) == n_tips:
            break
    if len(tips) < n_tips:
        return None
    tip_idx = sorted(i for _, i in tips)

    tip_angles = [(ti + 0.5) / n * 2 * math.pi - math.pi for ti in tip_idx]
    verts: List[Point] = []
    m = len(tip_idx)
    for k in range(m):
        ti = tip_idx[k]
        nxt = tip_idx[(k + 1) % m]
        # valley angle = midpoint between consecutive tips (shortest arc)
        da = ((tip_angles[(k + 1) % m] - tip_angles[k]) + 2 * math.pi) % (2 * math.pi)
        va = tip_angles[k] + da / 2.0
        vbin = int(((va + math.pi) / (2 * math.pi) * n)) % n
        # windowed minimum around the valley angle for a true inner radius
        lo, hi = (vbin - 2) % n, (vbin + 3) % n
        if lo < hi:
            window = radii[lo:hi]
        else:
            window = radii[lo:] + radii[:hi]
        vr = min(window) if window else radii[vbin]
        if vr < 1e-6:
            vr = radii[ti] * 0.5
        ta = (ti + 0.5) / n * 2 * math.pi - math.pi
        verts.append((cx + radii[ti] * math.cos(ta), cy + radii[ti] * math.sin(ta)))
        verts.append((cx + vr * math.cos(va), cy + vr * math.sin(va)))
    if len(verts) < n_tips * 2:
        return None
    return verts


def _star_vertices_angle(pts: List[Point], center: Point, n_tips: int = 5) -> Optional[List[Point]]:
    """Angle-space star vertex extractor (alternative to the binned profile).

    Works directly on contour points: tips are local radius maxima enforced to be
    well-spaced, valleys are the min-radius contour point in the gap between two
    tips. Handles irregular/skewed stars better than the binned method but is
    more sensitive to noise; `classify_mask` keeps whichever reconstruction
    re-rasterises with higher IoU.
    """
    cx, cy = center
    n = len(pts)
    ang = [math.atan2(y - cy, x - cx) for (x, y) in pts]
    rad = [math.hypot(x - cx, y - cy) for (x, y) in pts]
    cands = [i for i in range(n)
             if rad[i] >= rad[(i - 1) % n] and rad[i] >= rad[(i + 1) % n]
             and rad[i] > 0]
    cands.sort(key=lambda i: rad[i], reverse=True)
    min_sep = math.radians(30)
    tips = []
    for i in cands:
        if all(abs(((ang[i] - ang[j] + math.pi) % (2 * math.pi)) - math.pi) >= min_sep
               for j in tips):
            tips.append(i)
        if len(tips) == n_tips:
            break
    if len(tips) < n_tips:
        return None
    tips.sort(key=lambda i: ang[i])

    def gap(a0, a1):
        span = (a1 - a0) % (2 * math.pi)
        return [i for i in range(n) if ((ang[i] - a0) % (2 * math.pi)) < span]

    verts: List[Point] = []
    m = len(tips)
    for k in range(m):
        ti = tips[k]
        nxt = tips[(k + 1) % m]
        g = gap(ang[ti], ang[nxt])
        vi = min(g, key=lambda i: rad[i]) if g else ti
        verts.append((pts[ti][0], pts[ti][1]))
        verts.append((pts[vi][0], pts[vi][1]))
    if len(verts) < n_tips * 2:
        return None
    return verts


def _triangle_corners(pts: List[Point], center: Point) -> Optional[List[Point]]:
    """If the contour is roughly triangular, return its 3 corner points.

    Bins radii by angle; the three angular sectors with the *largest* radii
    (the corners) are located and their peaks returned. Returns None if the
    contour is not well-approximated by 3 corners.
    """
    radii = _resampled_radii(pts, center)
    n = len(radii)
    # find local maxima of radius (corners stick out)
    peaks = []
    for i in range(n):
        a = radii[(i - 1) % n]
        b = radii[i]
        c = radii[(i + 1) % n]
        if b >= a and b >= c and b > 0:
            peaks.append((b, i))
    peaks.sort(reverse=True)
    if len(peaks) < 3:
        return None
    # take the 3 strongest peaks as corners
    idxs = sorted(i for _, i in peaks[:3])
    corners = []
    for i in idxs:
        ang = (i + 0.5) / n * 2 * math.pi - math.pi
        r = radii[i]
        corners.append((center[0] + r * math.cos(ang),
                        center[1] + r * math.sin(ang)))
    # sanity: corners must span a wide spread (not all clustered)
    if max(radii) - min(radii) < 1e-6:
        return None
    return corners


def _fit_triangle(pts: List[Point], center: Point, mask: np.ndarray,
                  w: int, h: int) -> Optional[List[Point]]:
    """Return 3 corner points if the contour is well-fit by a triangle.

    Confirms via re-rasterized IoU so stars (whose '3 corners' fit poorly) are
    rejected. Returns None otherwise.
    """
    corners = _triangle_corners(pts, center)
    if not corners:
        return None
    ring = corners + [corners[0]]
    r = rasterize_polygon(ring, w, h)
    if iou_masks(r, mask) < 0.6:
        return None
    return corners


def estimate_orientation(mask: np.ndarray) -> Optional[float]:
    """Min-area-rectangle orientation of a mask, in [0, 180)."""
    from scipy.ndimage import rotate as ndi_rotate
    if not mask.any():
        return None
    best_area = None
    best_deg = 0.0
    for deg in range(0, 180, 2):
        rot = ndi_rotate(mask.astype(float), deg, order=0, reshape=True,
                         mode="constant", cval=0)
        rm = rot > 0.5
        if not rm.any():
            continue
        x0, y0, x1, y1 = mask_extent(rm)
        area = (x1 - x0) * (y1 - y0)
        if best_area is None or area < best_area:
            best_area = area
            best_deg = float(deg)
    return best_deg % 180.0


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_mask(mask: np.ndarray, threshold: float = 0.85) -> Dict:
    """Fit candidate primitives to `mask` and return a primitive node dict.

    Thin wrapper around :func:`_classify_mask_impl` that guarantees every node
    carries a `boundary_eval` (straight/curve summary of its contour), even for
    the early-return unknown/background cases.
    """
    node = _classify_mask_impl(mask, threshold)
    if node.get("boundary_eval") is None:
        try:
            repaired = repair_mask(mask)
            contour = contour_vertices(repaired, fill_holes=True)
            if contour:
                from ..evaluators import evaluate_boundary
                node["boundary_eval"] = evaluate_boundary(contour)
        except Exception:
            pass
    return node


def _classify_mask_impl(mask: np.ndarray, threshold: float = 0.85) -> Dict:
    """Fit candidate primitives to `mask` and return a primitive node dict.

    Pipeline: repair mask -> pixel extent -> best-fit IoU over candidates ->
    circle/ellipse/rect/triangle, else star, else rotated primitive, else
    polygon fallback. Attaches a `rotate` Transform for rotated rect/ellipse.
    """
    extent = mask_extent(mask)
    if extent is None:
        return {"kind": "primitive", "type": "unknown", "params": {},
                "transform": None, "bbox": None, "centroid": None,
                "area": 0, "fit_iou": None}
    x0, y0, x1, y1 = extent
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0

    repaired = repair_mask(mask)
    contour = contour_vertices(repaired, fill_holes=True)
    if not contour:
        # entire-frame mask (e.g. background) -> bbox rect
        return {"kind": "primitive", "type": "rect", "transform": None,
                "bbox": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
                "centroid": [round(cx, 1), round(cy, 1)],
                "area": int(mask.sum()), "fit_iou": None,
                "params": {"x": round(x0, 1), "y": round(y0, 1),
                           "w": round(x1 - x0, 1), "h": round(y1 - y0, 1)}}

    sm = contour_vertices(smooth_mask(repaired), fill_holes=True)
    if sm:
        contour_sm = _simplify_ring(sm[:-1], 2.0)
    else:
        contour_sm = _simplify_ring(contour[:-1], 2.0)
    if contour_sm and contour_sm[0] != contour_sm[-1]:
        contour_sm.append(contour_sm[0])

    # Low-level boundary evaluation (straight incl. staircase vs curve) is
    # attached uniformly by the `classify_mask` wrapper as `boundary_eval`, so
    # every node — including early-return cases — carries it.

    # Triangle detection: a clean 3-corner fit (confirmed by re-rasterised IoU)
    # must run before star, since a star's contour also yields 3 spurious
    # "corners" but fits a triangle very poorly.
    h, w = mask.shape
    tri = _fit_triangle(contour, (cx, cy), repaired, w, h) if contour else None

    if tri is not None:
        node = _node_base(x0, y0, x1, y1, cx, cy, mask)
        node["type"] = "triangle"
        node["params"] = {"points": [[round(px, 1), round(py, 1)]
                                     for px, py in tri]}
        return node

    # Star detection runs on the RAW (un-simplified) contour so sharp tips are
    # not collapsed by RDP. Must precede the candidate-fit return. The mask is
    # passed so a truncated/clipped silhouette (arms cut by an overlap) is
    # rejected as a star and falls through to custom polygon reconstruction.
    if contour and _detect_star(contour, (cx, cy), repaired):
        node = _node_base(x0, y0, x1, y1, cx, cy, mask)
        node["type"] = "star"
        # try both vertex extractors and keep the one that re-rasterises best
        best_sv = None
        best_iou = -1.0
        for extractor in (_star_vertices, _star_vertices_angle):
            sv = extractor(contour, (cx, cy))
            if not sv or len(sv) < 2 * 3:
                continue
            r = rasterize_polygon(sv + [sv[0]], w, h)
            iou = iou_masks(r, repaired)
            if iou > best_iou:
                best_iou, best_sv = iou, sv
        if best_sv:
            node["params"] = {"points": [[round(px, 1), round(py, 1)]
                                         for px, py in best_sv]}
        else:
            node["params"] = {"points": [[round(px, 1), round(py, 1)]
                                         for px, py in contour_sm[:-1]]}
        return node

    h, w = mask.shape
    fit_mask = repaired
    best = fit_candidates(CANDIDATE_BUILDERS, fit_mask, w, h,
                          x0, y0, x1, y1, cx, cy)

    node = _node_base(x0, y0, x1, y1, cx, cy, mask)

    if best is not None and best[1] > threshold:
        name, iou, pts = best
        node["fit_iou"] = round(iou, 3)
        if name in ("circle", "ellipse"):
            rx, ry = (x1 - x0) / 2.0, (y1 - y0) / 2.0
            aspect = max(rx, ry) / max(min(rx, ry), 1e-6)
            acx, acy = cx - _BOUNDARY_ADJ, cy - _BOUNDARY_ADJ
            # A regular polygon (hexagon, pentagon...) has a fill ratio close to a
            # circle/ellipse and can clear the IoU threshold, wrongly committing to
            # a curve. But it has N genuine sharp corners a curve does not. Before
            # accepting a circle/ellipse, test a corner-based polygon: if it has a
            # small, stable set of corners that fit MARKEDLY better, it is a
            # polygon (spurious occlusion-notch corners fit terribly, so this does
            # not fire on partly-hidden real circles).
            poly = _fit_corner_polygon(contour, w, h, fit_mask, iou)
            if poly is not None:
                return poly
            # A peanut/figure-8 of two overlapping circles fits a single ellipse
            # only moderately (the elongated axis is right but the waist bulges).
            # Before committing to an ellipse, try a two-circle UNION; if it fits
            # markedly better, emit that instead.
            twin = _fit_two_circle_union(fit_mask, x0, y0, x1, y1, iou)
            if twin is not None:
                return twin
            if aspect < 1.15:
                node["type"] = "circle"
                node["params"] = {"cx": round(acx, 1), "cy": round(acy, 1),
                                  "r": round((rx + ry) / 2.0, 1)}
            else:
                node["type"] = "ellipse"
                node["params"] = {"cx": round(acx, 1), "cy": round(acy, 1),
                                  "rx": round(rx, 1), "ry": round(ry, 1)}
            return node
        if name == "rounded_rect":
            # Simplicity preference: a plain rect is the simpler primitive. Only
            # keep rounded_rect if it beats a plain rect by a real margin --
            # otherwise a sharp-cornered rect whose edges were nibbled by
            # anti-aliasing/occlusion would spuriously read as rounded.
            rect_pts = _cand_rect(x0, y0, x1, y1, cx, cy)
            rect_iou = iou_masks(rasterize_polygon(rect_pts, w, h), fit_mask) \
                if rect_pts else 0.0
            if iou - rect_iou < _ROUNDED_OVER_RECT_GAIN:
                node["type"] = "rect"
                node["fit_iou"] = round(rect_iou, 3)
                node["params"] = {"x": round(x0 - _BOUNDARY_ADJ, 1),
                                  "y": round(y0 - _BOUNDARY_ADJ, 1),
                                  "w": round(x1 - x0, 1), "h": round(y1 - y0, 1)}
                return node
            node["type"] = "rounded_rect"
            node["params"] = {"x": round(x0 - _BOUNDARY_ADJ, 1),
                              "y": round(y0 - _BOUNDARY_ADJ, 1),
                              "w": round(x1 - x0, 1), "h": round(y1 - y0, 1),
                              "r": round(min(x1 - x0, y1 - y0) * 0.18, 1)}
            return node
        if name == "rect":
            node["type"] = "rect"
            node["params"] = {"x": round(x0 - _BOUNDARY_ADJ, 1),
                              "y": round(y0 - _BOUNDARY_ADJ, 1),
                              "w": round(x1 - x0, 1), "h": round(y1 - y0, 1)}
            return node
        if name == "triangle":
            # prefer the clean 3-corner fit over the raw (possibly notched) pts
            corners = _triangle_corners(contour, (cx, cy))
            if corners:
                node["type"] = "triangle"
                node["params"] = {"points": [[round(px, 1), round(py, 1)]
                                             for px, py in corners]}
                return node
            node["type"] = "triangle"
            node["params"] = {"points": [[round(px, 1), round(py, 1)]
                                         for px, py in pts]}
            return node

    # rotated primitive (rect/ellipse) -> axis-aligned primitive + rotate transform
    rot = _fit_rotated_primitive(fit_mask, x0, y0, x1, y1, cx, cy)
    if rot is not None:
        return rot

    # Occluded circle: a partly-hidden circle whose bbox-circle IoU was too low to
    # pass as a primitive, but whose visible boundary is a circular arc. Recover
    # the underlying circle (RANSAC, robust to occluder chords) so the curved edges
    # reconstruct as smooth arcs instead of a faceted polygon. Runs in the fallback
    # (after the clean-primitive/star/polygon checks) and itself rejects regular
    # polygons via a path-fidelity test, so a hexagon stays a polygon while a
    # clipped circle becomes a smooth arc path.
    from ..occluded_circle import fit_occluded_circle
    occ = fit_occluded_circle(fit_mask, contour, x0, y0, x1, y1)
    if occ is not None:
        return occ

    # ----------------------------------------------------------------------
    # Custom shape reconstruction.
    #
    # The shape did not qualify as ANY primitive (circle/ellipse/rect/
    # rounded_rect/triangle/star) nor as a rotated one. By the project's
    # discipline, "not a primitive => it MUST be a polygon". But rather than a
    # naive RDP dump of the staircase contour, we reconstruct the shape as a
    # FAITHFUL traced polygon: smooth the boundary to kill grid jitter, then
    # simplify it with an adaptive epsilon so the result reads as the real
    # (possibly clipped/irregular) silhouette — e.g. a star whose arms were cut
    # off by an overlapping yellow circle becomes the genuine clipped outline,
    # NOT a mis-identified ideal star.
    # ----------------------------------------------------------------------
    return _custom_polygon_node(mask, contour, x0, y0, x1, y1, cx, cy)


def _custom_polygon_node(mask, contour, x0, y0, x1, y1, cx, cy) -> Dict:
    """Reconstruct a non-primitive mask as a faithful traced polygon.

    Uses the smoothed contour (grid-jitter removed) simplified with an epsilon
    scaled to the shape size, so corners survive while staircase jitter is
    suppressed. Marks the node `custom: True` so renderers/editors know this is
    a reconstructed boundary rather than an ideal primitive.
    """
    node = _node_base(x0, y0, x1, y1, cx, cy, mask)
    node["type"] = "polygon"
    node["custom"] = True

    ring = None
    # prefer the smoothed contour; fall back to raw if smoothing produced nothing
    sm = contour_vertices(smooth_mask(repair_mask(mask)), fill_holes=True)
    if sm:
        ring = sm
    elif contour:
        ring = contour

    if not ring or len(ring) < 3:
        ring = contour if contour else [(cx, cy)]

    # adaptive simplification: larger shapes tolerate a larger epsilon
    diag = math.hypot(x1 - x0, y1 - y0)
    eps = max(1.0, diag * 0.01)
    simplified = _simplify_ring(ring[:-1] if ring[-1] == ring[0] else ring, eps)
    if simplified and simplified[0] != simplified[-1]:
        simplified.append(simplified[0])
    pts = simplified[:-1] if simplified and simplified[0] == simplified[-1] else simplified
    if len(pts) < 3:
        pts = ring[:-1] if ring[-1] == ring[0] else ring

    node["params"] = {"points": [[round(px, 1), round(py, 1)] for px, py in pts]}
    node["fit_iou"] = None
    return node


def _node_base(x0, y0, x1, y1, cx, cy, mask) -> Dict:
    return {
        "kind": "primitive",
        "type": "polygon",
        "params": {},
        "transform": None,
        "boundary_eval": None,
        "bbox": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
        "centroid": [round(cx, 1), round(cy, 1)],
        "area": int(mask.sum()),
        "fit_iou": None,
    }


# A corner-based polygon is preferred over an accepted circle/ellipse only if it
# has a small, stable corner count and fits by at least this margin better --
# separating a real regular polygon (hexagon/pentagon) from a genuine curve whose
# occlusion notches yield spurious, poorly-fitting "corners".
_POLY_OVER_CURVE_GAIN = 0.03
_POLY_MAX_CORNERS = 10

# A rounded_rect is kept over a plain rect only if it fits by at least this much
# better (else a sharp rect whose edges were nibbled reads as spuriously rounded).
_ROUNDED_OVER_RECT_GAIN = 0.02


def _fit_corner_polygon(contour, w, h, mask, curve_iou: float) -> Optional[Dict]:
    """Return a polygon node if the mask is really an N-gon mis-fit as a curve.

    A regular polygon can clear the circle/ellipse IoU threshold (its fill ratio
    is close), but it has N genuine sharp corners. We extract corners at the
    default span, require a modest count (a true curve yields 0-2 real corners,
    or many spurious ones that rasterise terribly), and accept the polygon only
    if it re-rasterises at least ``_POLY_OVER_CURVE_GAIN`` above the curve fit.
    """
    if not contour:
        return None
    from ..corners import corners_by_threshold
    corners = corners_by_threshold(contour, 45.0, span=6)
    n = len(corners)
    if n < 3 or n > _POLY_MAX_CORNERS:
        return None
    r = rasterize_polygon([(float(x), float(y)) for x, y in corners] + [corners[0]],
                          w, h)
    poly_iou = iou_masks(r, mask)
    if poly_iou < curve_iou + _POLY_OVER_CURVE_GAIN:
        return None
    ys, xs = np.where(mask)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    node = _node_base(x0, y0, x1, y1, cx, cy, mask)
    node["type"] = "polygon"
    node["custom"] = True
    node["params"] = {"points": [[round(float(px), 1), round(float(py), 1)]
                                 for px, py in corners]}
    node["fit_iou"] = round(poly_iou, 3)
    return node


# A two-circle union is only accepted if it beats the single-primitive IoU by at
# least this margin (avoids splitting a genuine ellipse into two circles).
_TWIN_MIN_GAIN = 0.05
_TWIN_MIN_IOU = 0.90


def _fit_two_circle_union(mask, x0, y0, x1, y1, base_iou: float) -> Optional[Dict]:
    """Detect a peanut/figure-8 silhouette = UNION of two equal overlapping circles.

    Such a shape reads as one connected blob but has a characteristic *waist*:
    scanning columns (or rows) along the long axis, the cross-section dips in the
    middle between two lobes. A single ellipse can't capture that dip, so it fits
    only moderately; two equal circles placed on the two lobes fit far better.

    Strategy (fast, deterministic -- no brute force): pick the long axis, set the
    radius from the SHORT axis half-extent (each circle spans the short side),
    place the two centres one radius in from each long-axis end, then refine the
    centre separation, radius and cross-axis centre over a small local grid.
    Returns a ``union`` node of two circle children if the union beats the single
    fit by ``_TWIN_MIN_GAIN`` and clears ``_TWIN_MIN_IOU``; else ``None``.
    """
    W = x1 - x0
    H = y1 - y0
    horizontal = W >= H
    long_len = W if horizontal else H
    short_len = H if horizontal else W
    # a single circle would have long==short; a two-circle peanut is elongated
    if long_len < short_len * 1.25:
        return None

    # Work on the bbox crop only (with a small pad) so each candidate raster is
    # cheap. Coordinates below are in crop space; converted back at the end.
    pad = 3
    ix0 = int(max(0, math.floor(x0) - pad))
    iy0 = int(max(0, math.floor(y0) - pad))
    ix1 = int(min(mask.shape[1], math.ceil(x1) + pad))
    iy1 = int(min(mask.shape[0], math.ceil(y1) + pad))
    crop = mask[iy0:iy1, ix0:ix1]
    yy, xx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]
    crop_area = int(crop.sum())

    def score(c0, c1, r, mid):
        if horizontal:
            m0 = (xx - c0) ** 2 + (yy - mid) ** 2 <= r * r
            m1 = (xx - c1) ** 2 + (yy - mid) ** 2 <= r * r
        else:
            m0 = (xx - mid) ** 2 + (yy - c0) ** 2 <= r * r
            m1 = (xx - mid) ** 2 + (yy - c1) ** 2 <= r * r
        u = m0 | m1
        inter = np.logical_and(u, crop).sum()
        uni = crop_area + int(u.sum()) - inter
        return inter / uni if uni else 0.0

    r0 = short_len / 2.0
    mid0 = (y0 + y1) / 2.0 - iy0 if horizontal else (x0 + x1) / 2.0 - ix0
    lo = (x0 - ix0) if horizontal else (y0 - iy0)
    hi = (x1 - ix0) if horizontal else (y1 - iy0)
    a0, b0 = lo + r0, hi - r0

    # coarse grid then one local refinement (fast, avoids a huge 4-D sweep)
    best = None

    def sweep(ranges, step_a, step_b, step_r, step_m):
        nonlocal best
        (aC, bC, rC, mC) = ranges
        for r in np.arange(rC - step_r * 3, rC + step_r * 3 + 1e-9, step_r):
            if r < 4:
                continue
            for da in np.arange(-step_a * 3, step_a * 3 + 1e-9, step_a):
                for db in np.arange(-step_b * 3, step_b * 3 + 1e-9, step_b):
                    for dm in np.arange(-step_m * 2, step_m * 2 + 1e-9, step_m):
                        io = score(aC + da, bC + db, r, mC + dm)
                        if best is None or io > best[0]:
                            best = (io, aC + da, bC + db, r, mC + dm)

    sweep((a0, b0, r0, mid0), 3.0, 3.0, 2.0, 2.0)
    if best is not None:
        _, ba, bb, br, bm = best
        sweep((ba, bb, br, bm), 1.0, 1.0, 0.5, 1.0)
    if best is None:
        return None
    iou, ca, cb, r, mid = best
    # convert crop-space back to image space
    if horizontal:
        ca += ix0; cb += ix0; mid += iy0
    else:
        ca += iy0; cb += iy0; mid += ix0
    if iou < _TWIN_MIN_IOU or iou < base_iou + _TWIN_MIN_GAIN:
        return None
    if horizontal:
        c0 = (ca - _BOUNDARY_ADJ, mid - _BOUNDARY_ADJ)
        c1 = (cb - _BOUNDARY_ADJ, mid - _BOUNDARY_ADJ)
    else:
        c0 = (mid - _BOUNDARY_ADJ, ca - _BOUNDARY_ADJ)
        c1 = (mid - _BOUNDARY_ADJ, cb - _BOUNDARY_ADJ)
    rr = round(r, 1)

    def circ_child(c):
        return {"kind": "primitive", "type": "circle",
                "params": {"cx": round(float(c[0]), 1),
                           "cy": round(float(c[1]), 1), "r": float(rr)},
                "transform": None}

    return {
        "kind": "composite",
        "type": "union",
        "children": [circ_child(c0), circ_child(c1)],
        "params": {},
        "transform": None,
        "bbox": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
        "centroid": [round((x0 + x1) / 2.0, 1), round((y0 + y1) / 2.0, 1)],
        "area": int(mask.sum()),
        "fit_iou": round(iou, 3),
    }


def _fit_rotated_primitive(mask, x0, y0, x1, y1, cx, cy) -> Optional[Dict]:
    """If the shape is a rotated rectangle/ellipse, return it as primitive+transform."""
    ang = estimate_orientation(mask)
    if ang is None:
        return None
    from scipy.ndimage import rotate as ndi_rotate
    best = None
    for deg in (ang, ang - 90.0):
        if abs(deg) < 6.0:
            continue
        rot = ndi_rotate(mask.astype(float), -deg, order=0, reshape=True,
                         mode="constant", cval=0)
        rm = rot > 0.5
        if not rm.any():
            continue
        rx0, ry0, rx1, ry1 = mask_extent(rm)
        rcx, rcy = (rx0 + rx1) / 2.0, (ry0 + ry1) / 2.0
        w, h = rx1 - rx0, ry1 - ry0
        rH, rW = rm.shape
        local = None
        for name, builder in (("rect", rectangle_points),
                              ("ellipse", lambda a, b, c, d, e, f:
                               ellipse_points(e, f, (c - a) / 2.0, (d - b) / 2.0))):
            if name == "rect":
                pts = rectangle_points(rx0, ry0, rx1, ry1)
            else:
                pts = ellipse_points(rcx, rcy, w / 2.0, h / 2.0)
            r = rasterize_polygon(pts, rW, rH)
            iou = iou_masks(r, rm)
            if local is None or iou > local[1]:
                local = (name, iou, rx0, ry0, w, h, rcx, rcy, deg)
        if local is not None and (best is None or local[1] > best[1]):
            best = local
    if best is None or best[1] < 0.85:
        return None
    name, iou, rx0, ry0, w, h, rcx, rcy, deg = best
    # ``rcx, rcy`` live in the ROTATED image's own (reshaped) coordinate frame,
    # which has a different origin than the source image. Emitting them directly
    # and then rotating around the source centroid puts the shape in the wrong
    # place. Instead build axis-aligned params centred on the SOURCE centroid
    # (cx, cy): rotating THOSE by ``deg`` around (cx, cy) reproduces the shape,
    # because the fit measured the un-rotated size (w, h) about that same centre.
    if name == "rect":
        params = {"x": round(cx - w / 2.0, 1), "y": round(cy - h / 2.0, 1),
                  "w": round(w, 1), "h": round(h, 1)}
    else:
        params = {"cx": round(cx, 1), "cy": round(cy, 1),
                  "rx": round(w / 2.0, 1), "ry": round(h / 2.0, 1)}
    return {
        "kind": "primitive",
        "type": name,
        "params": params,
        # The fit DE-rotates the mask by ``-deg`` (ndi_rotate) to measure the
        # axis-aligned size, so re-applying the primitive requires the INVERSE
        # rotation (``-deg`` in SVG's y-down space) about the source centroid to
        # place it back over the original silhouette.
        "transform": {"rotate": round(-deg, 1), "rotate_cx": round(cx, 1),
                      "rotate_cy": round(cy, 1)},
        "bbox": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
        "centroid": [round(cx, 1), round(cy, 1)],
        "area": int(mask.sum()),
        "fit_iou": round(iou, 3),
    }


def build_transform(node: Dict, mask: np.ndarray) -> Optional[Dict]:
    """Attach a rotate Transform to a rect/ellipse that is clearly rotated.

    IoU-guarded: `estimate_orientation` (min-area-rect brute search) can report a
    diagonal angle for a silhouette that is actually axis-aligned (e.g. a slightly
    irregular upright ellipse whose min-area rect lands off-axis). Applying that
    rotation would *worsen* the fit, so we only keep the transform when the
    rotated node rasterizes closer to the mask than the untransformed node.
    """
    t = node["type"]
    if t not in ("rect", "rounded_rect", "ellipse"):
        return None
    ang = estimate_orientation(mask)
    if ang is None:
        return None
    if min(ang % 90.0, 90.0 - (ang % 90.0)) <= 6.0:
        return None
    cx, cy = node["centroid"]
    cand = {"rotate": round(ang, 1), "rotate_cx": cx, "rotate_cy": cy}
    from ..renderers import rasterize_node
    h, w = mask.shape

    def _iou(n):
        r = rasterize_node(n, w, h)
        u = (r | mask).sum()
        return float((r & mask).sum()) / u if u else 0.0

    base = _iou(node)
    rotated = dict(node)
    rotated["transform"] = cand
    if _iou(rotated) <= base + 1e-3:
        return None
    return cand
