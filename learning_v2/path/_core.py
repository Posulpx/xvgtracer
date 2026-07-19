"""Path assembly: to_path and SVG rendering."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from .._image import Point, components_merged, mask_for_color, quantize
from ..classify import classify_shape
from ..fit import _fit_bezier_lsq, _fit_circle_lsq, _fit_ellipse_lsq, _spline_path


_MIDPOINT_SPLIT_THRESH = 20
_BLOB_SUBSAMPLE = 24
_ELLIPSE_ASPECT_MIN = 1.05
_MIN_COMPONENT_AREA = 50


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
            if max(rx, ry) / min(rx, ry) > _ELLIPSE_ASPECT_MIN:
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
        step = max(1, n // _BLOB_SUBSAMPLE)
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
            N = len(ring)
            i0c, i1c = i0, i1
            if i1c < i0c:
                i1c += N
            cnt_len = i1c - i0c + 1
            if cnt_len > _MIDPOINT_SPLIT_THRESH:
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
        if mask.sum() < _MIN_COMPONENT_AREA:
            continue
        lum = sum(col) / 3.0
        if lum < 12 or lum > 243:
            continue
        res = classify_shape(mask)
        res["color"] = tuple(int(v) for v in col)
        out.append(res)
    return out


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
