"""Shape generators for building ground-truth test images."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple


def polygon(cx: float, cy: float, r: float, n: int,
            rot: float = 0) -> List[Tuple[float, float]]:
    return [(cx + r * math.cos(rot + 2 * math.pi * i / n),
             cy - r * math.sin(rot + 2 * math.pi * i / n))
            for i in range(n)]


def star(cx: float, cy: float, r: float,
         n_points: int) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for i in range(2 * n_points):
        a = math.pi / 2 + math.pi * i / n_points
        rad = r if i % 2 == 0 else r * 0.4
        pts.append((cx + rad * math.cos(a), cy - rad * math.sin(a)))
    return pts


def ellipse_pts(cx: float, cy: float, w: float, h: float,
                n: int = 60) -> List[Tuple[float, float]]:
    return [(cx + w * h * math.cos(t) / math.hypot(h * math.cos(t), w * math.sin(t)),
             cy - w * h * math.sin(t) / math.hypot(h * math.cos(t), w * math.sin(t)))
            for t in [2 * math.pi * i / n for i in range(n)]]


def blob(cx: float, cy: float, r: float, n: int = 60,
         amp: float = 0.15, freq: int = 4) -> List[Tuple[float, float]]:
    return [(cx + r * (1 + amp * math.sin(freq * t + 0.5 * math.sin(2 * t))) * math.cos(t),
             cy - r * (1 + amp * math.sin(freq * t + 0.5 * math.sin(2 * t))) * math.sin(t))
            for t in [2 * math.pi * i / n for i in range(n)]]


def lens_lune_pts(cx: float, cy: float, r: float, offset: float,
                  n: int = 60, lens: bool = True) -> List[Tuple[float, float]]:
    d2 = offset / 2.0
    h = math.sqrt(max(0, r * r - d2 * d2))
    cx2 = cx + offset
    a_up = math.atan2(-h, d2)
    a_dn = math.atan2(h, d2)
    b_dn = math.atan2(h, -d2)
    b_up = math.atan2(-h, -d2)
    if b_up < 0:
        b_up += 2 * math.pi
    half = n // 2
    pts: List[Tuple[float, float]] = []
    if lens:
        for i in range(half + 1):
            t = a_up + (a_dn - a_up) * i / half
            pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))
    else:
        for i in range(half + 1):
            t = a_up + (a_dn - 2 * math.pi - a_up) * i / half
            pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))
    for i in range(1, half + 1):
        t = b_dn + (b_up - b_dn) * i / half
        pts.append((cx2 + r * math.cos(t), cy + r * math.sin(t)))
    return pts


# ==============================================================================
# Standard 12-shape test layout
# ==============================================================================

SHAPE_DEFS = [
    ("circle",     100, 130, 60, None),
    ("triangle",   300, 130, 65, None),
    ("rect",       500, 130, 55, "square"),
    ("pentagon",   700, 130, 60, None),
    ("hexagon",    100, 400, 60, None),
    ("star",       300, 400, 60, 5),
    ("irregular",  500, 400, 0, None),
    ("concave",    700, 400, 0, None),
    ("blob",       100, 670, 55, None),
    ("ellipse",    300, 670, 0, (50, 30)),
    ("lune",       500, 670, 55, 35),
    ("lens",       700, 670, 55, 35),
]

IRREGULAR_PTS = [(430, 320), (490, 360), (550, 450), (520, 530), (400, 490)]
CONCAVE_PTS = [(620, 330), (680, 360), (710, 430), (760, 480), (700, 530), (650, 420)]


def generate_shapes() -> List[dict]:
    import colorsys
    colors = [tuple(round(255 * c) for c in colorsys.hsv_to_rgb(i / 12, 1.0, 200 / 255))
              for i in range(12)]
    shapes = []
    for (typ, cx, cy, r, ex), col in zip(SHAPE_DEFS, colors):
        if typ == "circle":
            pts = polygon(cx, cy, r, 60)
        elif typ == "triangle":
            pts = polygon(cx, cy, r, 3)
        elif typ == "rect":
            pts = polygon(cx, cy, r, 4, rot=math.pi / 4)
        elif typ == "pentagon":
            pts = polygon(cx, cy, r, 5)
        elif typ == "hexagon":
            pts = polygon(cx, cy, r, 6)
        elif typ == "star":
            pts = star(cx, cy, r, ex)
        elif typ == "irregular":
            pts = IRREGULAR_PTS
        elif typ == "concave":
            pts = CONCAVE_PTS
        elif typ == "blob":
            pts = blob(cx, cy, r)
        elif typ == "ellipse":
            pts = ellipse_pts(cx, cy, *ex)
        elif typ == "lune":
            pts = lens_lune_pts(cx, cy, r, ex, lens=False)
        elif typ == "lens":
            pts = lens_lune_pts(cx, cy, r, ex, lens=True)
        else:
            pts = []
        shapes.append({
            "type": typ,
            "pts": [[round(p[0], 1), round(p[1], 1)] for p in pts],
            "color": list(col),
        })
    return shapes


def render_ground_truth(width: int = 800, height: int = 800,
                        shapes: Optional[List[dict]] = None):
    from PIL import Image, ImageDraw
    if shapes is None:
        shapes = generate_shapes()
    img = Image.new("RGB", (width, height), "black")
    dr = ImageDraw.Draw(img)
    for s in shapes:
        pts = [(float(p[0]), float(p[1])) for p in s["pts"]]
        dr.polygon(pts, fill=tuple(s["color"]))
    return img
