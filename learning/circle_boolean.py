"""Two-circle boolean recovery.

When two circles overlap, a colour-quantized image splits the arrangement into
THREE regions that the per-region classifier fits in isolation and gets wrong:

  * the shared **intersection** (a lens / vesica) is mis-fit as an ellipse -- but
    its top/bottom taper to sharp points, so it is really two circular arcs;
  * each remaining **lune** (a circle with a lens-shaped bite removed) is not any
    primitive, so it falls back to a faceted polygon -- but its boundary is two
    circular arcs (the outer full-circle arc + the inner lens arc).

This module recovers the two generating circles from the region masks and rewrites
the region nodes to arc-based ``lens`` / ``lune`` nodes so every boundary is a
true circular arc instead of a facet or a wrong ellipse.

Recovery strategy (mask-driven, no assumption about which colour is which):
  1. Find a node classified as ``ellipse`` whose mask tapers to points top&bottom
     or left&right -- a lens candidate.
  2. The two neighbouring regions that each SHARE a long boundary with the lens
     are its lunes. lune_i + lens should reconstitute a full circle.
  3. Fit a circle to (lune_i | lens); confirm both fits are strong and the two
     circles' intersection reproduces the lens.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

Circle = Tuple[float, float, float]  # (cx, cy, r)


def _fit_circle_to_mask(mask: np.ndarray) -> Optional[Circle]:
    """Best-fit circle to a solid disk-ish mask by IoU.

    A least-squares (Kasa) fit on boundary pixels is biased when the mask's
    boundary is not a full clean circle -- e.g. a full disk reconstituted as
    ``lune | lens`` still carries a slightly straight seam that pulls an
    algebraic fit off. Instead we seed from the bounding box (centre = bbox
    centre, r = mean half-extent) and refine centre & radius by a small local
    grid search that maximises IoU against the mask. Returns ``(cx, cy, r)``.
    """
    ys, xs = np.where(mask)
    if len(xs) < 8:
        return None
    h, w = mask.shape
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx0 = (x0 + x1) / 2.0
    cy0 = (y0 + y1) / 2.0
    r0 = ((x1 - x0) + (y1 - y0)) / 4.0

    yy, xx = np.mgrid[0:h, 0:w]

    def iou_of(cx, cy, r):
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        u = np.logical_or(disk, mask).sum()
        return float(np.logical_and(disk, mask).sum()) / u if u else 0.0

    best = (cx0, cy0, r0)
    best_iou = iou_of(cx0, cy0, r0)
    step = max(2.0, r0 * 0.06)
    for _ in range(6):  # coarse-to-fine
        improved = False
        for dcx in (-step, 0, step):
            for dcy in (-step, 0, step):
                for dr in (-step, 0, step):
                    cx, cy, r = cx0 + dcx, cy0 + dcy, r0 + dr
                    if r <= 1:
                        continue
                    io = iou_of(cx, cy, r)
                    if io > best_iou:
                        best_iou = io
                        best = (cx, cy, r)
                        improved = True
        cx0, cy0, r0 = best
        if not improved:
            step /= 2.0
            if step < 0.5:
                break
    return best


def _circle_mask(circle: Circle, w: int, h: int) -> np.ndarray:
    cx, cy, r = circle
    yy, xx = np.mgrid[0:h, 0:w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()) / u if u else 0.0


def _tapers_to_points(mask: np.ndarray) -> bool:
    """True if the silhouette narrows to (near) points at both ends of its long
    axis -- the signature of a lens, distinguishing it from an ellipse."""
    ys, xs = np.where(mask)
    if len(xs) < 8:
        return False
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    if bh >= bw:  # tall -> ends are top & bottom rows
        top = (mask[y0] | mask[min(y0 + 1, y1)]).sum()
        bot = (mask[y1] | mask[max(y1 - 1, y0)]).sum()
        span = bw
    else:         # wide -> ends are left & right columns
        top = (mask[:, x0] | mask[:, min(x0 + 1, x1)]).sum()
        bot = (mask[:, x1] | mask[:, max(x1 - 1, x0)]).sum()
        span = bh
    # a lens end is a near-point (< 10% of the cross span); an ellipse end is a
    # wide rounded cap
    return top < 0.12 * span and bot < 0.12 * span


def recover_two_circle_booleans(shapes: List[Dict],
                                masks: Dict[int, np.ndarray],
                                w: int, h: int,
                                min_iou: float = 0.97) -> List[Dict]:
    """Rewrite lens/lune region nodes to arc-based boolean nodes in-place.

    For every ellipse node whose mask actually tapers to points, try to recover
    the two generating circles from its neighbouring regions. On success:
      * the lens node becomes ``type='lens'`` with the two circles in
        ``params['circles']`` (arc reconstruction, smooth);
      * each lune node becomes ``type='lune'`` with its own outer circle +
        the lens' other circle (arc reconstruction, smooth).

    Returns the (mutated) shapes list. No-op if recovery is not confident.
    """
    if not masks:
        return shapes

    for lens in shapes:
        if lens.get("type") != "ellipse":
            continue
        lens_mask = masks.get(id(lens))
        if lens_mask is None or not _tapers_to_points(lens_mask):
            continue

        # neighbouring regions: dilate the lens and see which other shapes' masks
        # it touches heavily (they share the lens' arc boundary)
        from scipy.ndimage import binary_dilation
        halo = binary_dilation(lens_mask, iterations=3) & ~lens_mask
        neighbours = []
        for s in shapes:
            if s is lens or s.get("is_background"):
                continue
            m = masks.get(id(s))
            if m is None:
                continue
            shared = np.logical_and(halo, m).sum()
            if shared > 0.15 * halo.sum():
                neighbours.append((s, m, shared))
        if len(neighbours) < 2:
            continue
        neighbours.sort(key=lambda t: -t[2])
        (lu0, m0, _), (lu1, m1, _) = neighbours[0], neighbours[1]

        # each lune + lens should reconstitute a full circle
        c0 = _fit_circle_to_mask(np.logical_or(m0, lens_mask))
        c1 = _fit_circle_to_mask(np.logical_or(m1, lens_mask))
        if not c0 or not c1:
            continue
        cm0 = _circle_mask(c0, w, h)
        cm1 = _circle_mask(c1, w, h)
        iou0 = _iou(cm0, np.logical_or(m0, lens_mask))
        iou1 = _iou(cm1, np.logical_or(m1, lens_mask))
        if iou0 < min_iou or iou1 < min_iou:
            continue
        # the two circles' intersection must reproduce the lens
        lens_iou = _iou(np.logical_and(cm0, cm1), lens_mask)
        if lens_iou < min_iou - 0.02:
            continue

        # ---- commit: rewrite the three region nodes to arc-based booleans ----
        lens["type"] = "lens"
        lens["params"] = {"circles": [list(map(float, c0)), list(map(float, c1))]}
        lens["fit_iou"] = round(lens_iou, 4)
        lens["recovered"] = "two_circle_intersection"
        lens.pop("transform", None)

        # lune0 = circle c0 minus the lens (bite from circle c1)
        lu0["type"] = "lune"
        lu0["params"] = {"circle": list(map(float, c0)),
                         "cut": list(map(float, c1))}
        lu0["fit_iou"] = round(iou0, 4)
        lu0["recovered"] = "circle_minus_lens"
        lu0.pop("transform", None)

        lu1["type"] = "lune"
        lu1["params"] = {"circle": list(map(float, c1)),
                         "cut": list(map(float, c0))}
        lu1["fit_iou"] = round(iou1, 4)
        lu1["recovered"] = "circle_minus_lens"
        lu1.pop("transform", None)

    return shapes


# ---------------------------------------------------------------------------
# arc-based path geometry (shared by reconstructor + rasteriser)
# ---------------------------------------------------------------------------

def circle_intersections(c0: Circle, c1: Circle) -> Optional[Tuple[Tuple[float, float],
                                                                   Tuple[float, float]]]:
    """The two points where circles c0 and c1 cross, or None if they do not."""
    x0, y0, r0 = c0
    x1, y1, r1 = c1
    dx, dy = x1 - x0, y1 - y0
    d = math.hypot(dx, dy)
    if d == 0 or d > r0 + r1 or d < abs(r0 - r1):
        return None
    a = (r0 * r0 - r1 * r1 + d * d) / (2 * d)
    hsq = r0 * r0 - a * a
    if hsq < 0:
        return None
    hh = math.sqrt(hsq)
    xm = x0 + a * dx / d
    ym = y0 + a * dy / d
    px, py = -dy / d * hh, dx / d * hh
    return ((xm + px, ym + py), (xm - px, ym - py))


def _ordered_crossings(cA: Circle, cB: Circle):
    """Crossing points of cA & cB, ordered so that going A->B along cA's arc that
    bulges AWAY from cB is a positive (sweep=1) minor arc.

    Returns ((ax,ay),(bx,by)) or None. Ordering is made deterministic by the sign
    of the cross product of (cB-cA) with (P-cA): the point on the +perp side of
    the centre line comes first.
    """
    pts = circle_intersections(cA, cB)
    if not pts:
        return None
    (p0, p1) = pts
    ux, uy = cB[0] - cA[0], cB[1] - cA[1]
    cross0 = ux * (p0[1] - cA[1]) - uy * (p0[0] - cA[0])
    if cross0 < 0:
        p0, p1 = p1, p0
    return p0, p1


def lens_d(params: Dict) -> str:
    """Lens (intersection of two circles) as two SVG arcs between the crossings.

    Flags verified against masks: with crossings ordered A(+perp)->B(-perp),
    both boundary arcs are MINOR arcs (large=0) with opposite sweeps so the two
    convex caps meet at the tips.
    """
    c0 = tuple(params["circles"][0])
    c1 = tuple(params["circles"][1])
    oc = _ordered_crossings(c0, c1)
    if not oc:
        cx, cy, r = c0
        return (f"M{cx - r:.1f},{cy:.1f} a{r:.1f},{r:.1f} 0 1,0 {2*r:.1f},0 "
                f"a{r:.1f},{r:.1f} 0 1,0 {-2*r:.1f},0 Z")
    (ax, ay), (bx, by) = oc
    r0, r1 = c0[2], c1[2]
    return (f"M{ax:.2f},{ay:.2f} "
            f"A{r0:.2f},{r0:.2f} 0 0,0 {bx:.2f},{by:.2f} "
            f"A{r1:.2f},{r1:.2f} 0 0,0 {ax:.2f},{ay:.2f} Z")


def lune_d(params: Dict) -> str:
    """Lune (circle with a lens-shaped bite) = outer MAJOR arc + inner MINOR arc.

    ``params['circle']`` is the lune's own full circle; ``params['cut']`` is the
    other circle whose intersection carves the bite. Flags verified against
    masks: with crossings ordered A(+perp of own->cut line)->B, the outer arc is
    the MAJOR arc of the own circle (large=1, sweep=1) and the inner (cut) arc is
    the MINOR arc curving inward (large=0, sweep=0).
    """
    c = tuple(params["circle"])
    cut = tuple(params["cut"])
    oc = _ordered_crossings(c, cut)
    cx, cy, r = c
    if not oc:
        return (f"M{cx - r:.1f},{cy:.1f} a{r:.1f},{r:.1f} 0 1,0 {2*r:.1f},0 "
                f"a{r:.1f},{r:.1f} 0 1,0 {-2*r:.1f},0 Z")
    (ax, ay), (bx, by) = oc
    rc = cut[2]
    return (f"M{ax:.2f},{ay:.2f} "
            f"A{r:.2f},{r:.2f} 0 1,1 {bx:.2f},{by:.2f} "
            f"A{rc:.2f},{rc:.2f} 0 0,0 {ax:.2f},{ay:.2f} Z")
