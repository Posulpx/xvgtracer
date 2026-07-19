"""Collision-point detection: where color regions meet in the quantized image.

A *collision point* is a pixel location where several distinct quantized colors
converge. These are the meaningful junctions of an image (corners where shapes
meet). We detect them from the quantized RGB array by scanning each pixel's
local neighbourhood: if the window contains at least `min_colors` distinct
(non-background) colors, that pixel is a convergence point. Hits are clustered
so a single junction yields one dot, not a smear of adjacent pixels.

Priority is given to higher-order convergences (3 colors before 2), since those
are the strongest structural junctions.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

Point = Tuple[float, float]


def _label_colors(rgb: np.ndarray, background_color=None) -> np.ndarray:
    """Map the quantized RGB image to an integer label per pixel.

    The background color (if given) is labelled -1 so it is ignored when
    counting distinct *foreground* colors at a junction.
    """
    h, w = rgb.shape[:2]
    pix = rgb.reshape(-1, 3)
    uniq, inv = np.unique(pix, axis=0, return_inverse=True)
    labels = inv.reshape(h, w)
    if background_color is not None:
        bg = np.array(background_color, dtype=uniq.dtype)
        bg_idx = None
        for i, u in enumerate(uniq):
            if np.array_equal(u, bg):
                bg_idx = i
                break
        if bg_idx is not None:
            labels = np.where(labels == bg_idx, -1, labels)
    return labels


def _merge_close_points(points: List[Dict], merge_dist: float) -> List[Dict]:
    """Fuse output collision points whose centres lie within ``merge_dist`` px.

    The in-detection clustering groups raw pixel hits around a running centroid,
    which can still leave two *distinct* final junctions closer than intended on
    a small/quantized source. This is a final agglomerative pass over the emitted
    points: repeatedly merge the closest pair within ``merge_dist`` until none
    remain. A merged point takes the area-weighted centroid of its members, the
    union of their converging colors, and the max ``n_colors``.
    """
    if merge_dist <= 0 or len(points) < 2:
        return points
    # working records: [x, y, weight, n_colors, {color-tuple: rgb-list}]
    recs = []
    for p in points:
        cmap = {tuple(c): c for c in p.get("colors", [])}
        recs.append([p["x"], p["y"], 1.0, p["n_colors"], cmap])
    merged = True
    while merged and len(recs) > 1:
        merged = False
        best = None
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                d = math.hypot(recs[i][0] - recs[j][0], recs[i][1] - recs[j][1])
                if d <= merge_dist and (best is None or d < best[0]):
                    best = (d, i, j)
        if best is not None:
            _, i, j = best
            a, b = recs[i], recs[j]
            wa, wb = a[2], b[2]
            w = wa + wb
            a[0] = (a[0] * wa + b[0] * wb) / w
            a[1] = (a[1] * wa + b[1] * wb) / w
            a[2] = w
            a[4] = {**a[4], **b[4]}
            a[3] = max(a[3], b[3], len(a[4]))
            recs.pop(j)
            merged = True
    out = []
    for x, y, _w, n, cmap in recs:
        cols = list(cmap.values())
        out.append({"x": round(x, 2), "y": round(y, 2),
                    "n_colors": int(max(n, len(cols))), "colors": cols})
    out.sort(key=lambda d: (-d["n_colors"], -len(d["colors"]), d["y"], d["x"]))
    return out


def collision_points(rgb: np.ndarray,
                     min_colors: int = 3,
                     window: int = 1,
                     cluster_radius: float = 3.0,
                     merge_dist: float = 1.0,
                     background_color=None,
                     include_background: bool = True,
                     boundary_only: bool = False,
                     min_region_frac: float = 0.15) -> List[Dict]:
    """Find convergence points where >= `min_colors` distinct colors meet.

    Scans a (2*window+1)^2 neighbourhood around each pixel. Returns a list of
    dicts ``{"x", "y", "n_colors", "colors"}`` sorted by descending ``n_colors``
    (so 3-color junctions come first), then by descending pixel count. Nearby
    hits are merged into a single dot within ``cluster_radius`` during detection;
    a final pass then fuses any two emitted points within ``merge_dist`` px
    (default 1.0, adjustable) so quantization does not leave duplicate junctions a
    pixel apart. ``colors`` holds the actual RGB triples that converge at the
    point (not internal indices).

    By default the background/canvas color is *included* in the convergence
    count, so a foreground shape meeting the white backdrop counts as a junction
    (e.g. a red shape against white canvas = a 2-color collision). Pass
    ``include_background=False`` to ignore the backdrop. With ``min_colors=3``
    this reports exactly the 3-color convergence points you asked to prioritize;
    pass 2 to also include plain 2-color edges.

    If ``boundary_only=True``, a hit is kept only when the centre pixel's own
    color differs from at least one of its 4-neighbours — i.e. the point truly
    sits *on* a region boundary, not just inside a region whose window happens
    to catch a neighbouring color at its edge. This removes "internal" points
    that are actually inside a single shape's filled area.
    """
    _bg = background_color if (include_background and background_color is not None) else None
    labels = _label_colors(rgb, _bg)
    h, w = labels.shape
    win = max(1, int(window))

    # Map label index -> original RGB triple (for reporting real colors).
    pix = rgb.reshape(-1, 3)
    uniq, inv = np.unique(pix, axis=0, return_inverse=True)
    label_to_rgb = {int(i): [int(v) for v in u] for i, u in enumerate(uniq)}

    # Pad so border pixels can be sampled without index errors.
    pad = win
    padded = np.full((h + 2 * pad, w + 2 * pad), -2, dtype=labels.dtype)
    padded[pad:pad + h, pad:pad + w] = labels

    hits = []  # (y, x, n_colors)
    for y in range(h):
        for x in range(w):
            block = padded[y:y + 2 * win + 1, x:x + 2 * win + 1]
            vals, counts = np.unique(block, return_counts=True)
            fg = vals >= 0
            # A colour only *converges* here if it occupies a real share of the
            # window, not a 1px anti-alias transition sliver. Along a smooth curved
            # edge, quantization leaves a thin intermediate band that would
            # otherwise inflate the colour count all along the arc (spurious
            # "corners"). Require each counted colour to fill >= min_region_frac of
            # the window, so only true multi-region convergences (real corners /
            # tips) survive.
            total = block.size
            thresh = max(1.0, min_region_frac * total)
            substantial = fg & (counts >= thresh)
            n = int(np.sum(substantial))
            if n < min_colors:
                continue
            if boundary_only:
                # require the centre pixel to actually sit on a boundary:
                # its own color must differ from at least one 4-neighbour.
                c0 = labels[y, x]
                on_edge = False
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] != c0:
                        on_edge = True
                        break
                if not on_edge:
                    continue
            hits.append((y, x, n))

    if not hits:
        return []

    # cluster nearby hits; the cluster's centre becomes the float centroid of
    # all its member pixels (sub-pixel accurate), so each indicator is centred
    # on the true convergence rather than snapped to one hit pixel.
    kept = []  # [sum_y, sum_x, count, n_colors, set_of_color_indices]
    for y, x, n in sorted(hits, key=lambda t: -t[2]):
        # gather the distinct color indices present in this hit's window
        block = labels[max(0, y - win):y + win + 1, max(0, x - win):x + win + 1]
        idxs = tuple(sorted(int(c) for c in np.unique(block) if c >= 0))
        merged = False
        for k in kept:
            if abs(k[0] / k[2] - y) <= cluster_radius and \
               abs(k[1] / k[2] - x) <= cluster_radius:
                k[0] += y
                k[1] += x
                k[2] += 1
                # prefer the highest n_colors / most colors for the dot's record
                if n > k[3] or (n == k[3] and len(idxs) > len(k[4])):
                    k[3], k[4] = n, idxs
                else:
                    k[4] = tuple(sorted(set(k[4]) | set(idxs)))
                merged = True
                break
        if not merged:
            kept.append([y, x, 1, n, idxs])

    out = []
    for sum_y, sum_x, cnt, n, idxs in kept:
        cols = [label_to_rgb[i] for i in idxs if i in label_to_rgb]
        out.append({
            # centroid = mean of member pixel coords; +0.5 centres on the pixel
            "x": round(float(sum_x) / cnt + 0.5, 2),
            "y": round(float(sum_y) / cnt + 0.5, 2),
            "n_colors": int(n),
            "colors": cols,
        })
    out.sort(key=lambda d: (-d["n_colors"], -len(d["colors"]), d["y"], d["x"]))
    # final agglomerative merge of near-duplicate junctions (~1px apart).
    return _merge_close_points(out, merge_dist)
