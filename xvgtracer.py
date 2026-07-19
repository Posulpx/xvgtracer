"""XVGTracer core: color quantization + per-layer contour tracing -> layered SVG.

Pure-Python pipeline. Uses Pillow for perceptual quantization and scikit-image's
marching squares to trace each color layer into smooth SVG paths.
"""
from __future__ import annotations

import re

import numpy as np
from PIL import Image, ImageDraw
from skimage import measure


def quantize(image_path: str, num_colors: int = 5):
    """Quantize an image to `num_colors` perceptual colors.

    Returns (quantized_rgb_image, list_of_rgb_tuples).
    """
    img = Image.open(image_path).convert("RGB")
    # Perceptual quantization via median-cut + kmeans refinement.
    q = img.quantize(colors=num_colors, method=Image.FASTOCTREE, kmeans=num_colors)
    palette = np.array(q.getpalette()[: num_colors * 3]).reshape(-1, 3)
    colors = [tuple(int(c) for c in row) for row in palette]
    rgb = q.convert("RGB")
    return rgb, colors


def _binary_mask_for_color(arr: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    """Boolean mask where the quantized image equals `color` (touched by any channel)."""
    diff = np.abs(arr.astype(int) - np.array(color)).sum(axis=2)
    return diff < 8


def _rdp_open(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    """Standard (open) Ramer-Douglas-Peucker polyline simplification."""
    if len(points) < 3:
        return points
    arr = np.array(points, dtype=float)
    start, end = arr[0], arr[-1]
    vec = end - start
    norm = np.hypot(*vec)
    if norm == 0:
        return [points[0], points[-1]]
    rel = arr - start
    d = np.abs(rel[:, 0] * vec[1] - rel[:, 1] * vec[0]) / norm
    i = int(np.argmax(d))
    if d[i] > epsilon:
        left = _rdp_open(points[: i + 1], epsilon)
        right = _rdp_open(points[i:], epsilon)
        return left[:-1] + right
    return [points[0], points[-1]]


def _rdp(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    """Closed-contour RDP: simplify the ring as an open polyline by dropping the
    duplicate closing point, then re-append it so the path stays closed."""
    if len(points) < 3:
        return points
    # find_contours returns a closed loop (first≈last); drop the trailing duplicate
    ring = points[:]
    while len(ring) > 1 and abs(ring[0][0] - ring[-1][0]) < 1e-6 and abs(ring[0][1] - ring[-1][1]) < 1e-6:
        ring = ring[:-1]
    if len(ring) < 3:
        return points
    simplified = _rdp_open(ring, epsilon)
    return simplified + [simplified[0]]


def trace_layer(mask: np.ndarray, simplify: float = 1.0,
                 angle_threshold: float = 30.0, max_segment: float = 8.0,
                 smooth_sigma: float = 1.0) -> list[str]:
    """Trace a binary mask into SVG path `d` strings via marching squares.

    `simplify` controls contour smoothing (higher = fewer points / smoother).
    `angle_threshold` (degrees): vertices turning sharper than this stay hard
    corners; gentler runs are smoothed into cubic Bézier curves.
    `smooth_sigma` (px): mild low-pass applied to the contour point ring before
    tracing. This removes pixel-grid quantisation jitter (the cause of "polygonal
    hints") so smooth curves read as true curves, while preserving real corners
    and tiny details far better than blurring the image itself.
    """
    paths: list[str] = []
    contours = measure.find_contours(mask.astype(float), 0.5)
    for contour in contours:  # contour is (row=y, col=x)
        if len(contour) < 4:
            continue
        pts = [(float(x), float(y)) for y, x in contour]
        # dedupe consecutive points
        dedup = []
        for p in pts:
            if not dedup or abs(dedup[-1][0] - p[0]) > 1e-6 or abs(dedup[-1][1] - p[1]) > 1e-6:
                dedup.append(p)
        if len(dedup) < 3:
            continue
        # low-pass the point ring to kill grid jitter without shifting the boundary
        if smooth_sigma > 0:
            n = len(dedup)
            arr = np.array(dedup, dtype=float)
            k = max(1, int(round(smooth_sigma * 2)))
            for _ in range(k):
                s = np.zeros_like(arr)
                s[0] = (arr[-1] + 2 * arr[0] + arr[1]) / 4
                s[1:-1] = (arr[:-2] + 2 * arr[1:-1] + arr[2:]) / 4
                s[-1] = (arr[-2] + 2 * arr[-1] + arr[0]) / 4
                arr = s
            dedup = [(float(x), float(y)) for x, y in arr]
        simplified = _rdp(dedup, epsilon=simplify)
        d = _to_smooth_path(simplified, angle_threshold=angle_threshold,
                            max_segment=max_segment)
        paths.append(d)
    return paths


def _to_smooth_path(points: list[tuple[float, float]], angle_threshold: float = 30.0,
                    max_segment: float = 8.0) -> str:
    """Build an SVG path string from a closed polygon.

    Vertices whose turning angle exceeds `angle_threshold` (degrees) are kept as
    hard corners (line segments). Gentler runs are smoothed with a Catmull-Rom ->
    cubic Bézier spline, subdivided so curves stay dense enough to read as true
    (not facets). This is robust for any closed shape (circles, ellipses, stars)
    and avoids SVG-arc failure modes on near-full curves.
    """
    n = len(points)
    if n < 3:
        return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in points) + " Z"

    # closed ring with wrap-around neighbours for tangent estimation
    pts = points + [points[0], points[1], points[2]]

    def turn(i: int) -> float:
        """Exterior bend angle (deg) at ring vertex i (0..n-1). 180 = straight."""
        p0 = np.array(pts[i - 1])
        p1 = np.array(pts[i])
        p2 = np.array(pts[i + 1])
        v1 = p0 - p1
        v2 = p2 - p1
        n1 = np.hypot(*v1)
        n2 = np.hypot(*v2)
        if n1 == 0 or n2 == 0:
            return 0.0  # degenerate -> treat as sharp corner
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_a)))

    # A vertex is a hard corner when the path bends more than `angle_threshold`
    # away from straight (180 - angle_threshold).
    corner = [turn(i) < (180.0 - angle_threshold) or np.isnan(turn(i)) for i in range(n)]

    def subdivide(a, b):
        """Insert evenly spaced points between a and b so no segment exceeds max_segment."""
        dist = np.hypot(b[0] - a[0], b[1] - a[1])
        steps = max(1, int(np.ceil(dist / max_segment)))
        return [(a[0] + (b[0] - a[0]) * t / steps, a[1] + (b[1] - a[1]) * t / steps)
                for t in range(1, steps)]

    d = f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"
    i = 0
    while i < n:
        if corner[i]:
            d += f" L{pts[i][0]:.1f},{pts[i][1]:.1f}"
            i += 1
            continue
        # collect a smooth run of gentle vertices [i .. j]
        run = [pts[i]]
        j = i + 1
        while j < n and not corner[j]:
            run.append(pts[j])
            j += 1
        run.append(pts[j])  # next anchor (corner or wrap start)
        # densify each leg of the run so curves don't look faceted
        dense = [run[0]]
        for k in range(len(run) - 1):
            dense.extend(subdivide(run[k], run[k + 1]))
            dense.append(run[k + 1])
        # Smooth runs are rendered as a Catmull-Rom -> cubic Bézier spline. This is
        # robust for any closed curve (circles, ellipses, stars) and avoids the
        # failure modes of SVG arc commands (large-arc/sweep flags on near-full
        # circles). The path is closed and dense enough to read as a true curve.
        for k in range(1, len(dense) - 1):
            p_prev = np.array(dense[k - 1])
            p_curr = np.array(dense[k])
            p_next = np.array(dense[k + 1])
            c1 = p_curr + (p_next - p_prev) / 6.0
            c2 = (p_next - (np.array(dense[k + 2]) - p_curr) / 6.0
                  if k + 2 < len(dense) else p_next - (p_next - p_curr) / 6.0)
            d += (f" C{c1[0]:.1f},{c1[1]:.1f}"
                  f" {c2[0]:.1f},{c2[1]:.1f}"
                  f" {p_next[0]:.1f},{p_next[1]:.1f}")
        i = j
        if j >= n:
            break
    d += " Z"
    # drop consecutive duplicate points (e.g. echoed start vertex).
    # A command: A rx ry rot la,sweep x y ; others: M/L/C x,y (+ C has 3 pairs).
    import re
    last_pt = None
    out_parts = []
    # match a command letter, then its numeric args
    for m in re.finditer(r"([MLCZA])([^MLCZA]*)", d):
        cmd = m.group(1)
        if cmd == "Z":
            out_parts.append("Z")
            continue
        nums = re.findall(r"-?\d+\.?\d*", m.group(2))
        if cmd == "A":
            # rx ry rot la,sweep x y
            rx, ry, rot, la, sw, x, y = (float(v) for v in nums)
            pt = (round(x, 1), round(y, 1))
            if pt != last_pt:
                out_parts.append(f"A{rx:.1f},{ry:.1f} {int(rot)} {int(la)},{int(sw)} {x:.1f},{y:.1f}")
                last_pt = pt
        else:
            # group into coordinate pairs
            pts = [(round(float(nums[i]), 1), round(float(nums[i + 1]), 1))
                   for i in range(0, len(nums) - 1, 2)]
            kept = []
            for pt in pts:
                if pt != last_pt:
                    kept.append(f"{pt[0]:.1f},{pt[1]:.1f}")
                    last_pt = pt
            if kept:
                out_parts.append(cmd + " ".join(kept))
    return " ".join(out_parts)
    return " ".join(clean)


def generate_svg(
    rgb_image: Image.Image,
    colors: list[tuple[int, int, int]],
    output: str = "output_colored.svg",
    simplify: float = 1.0,
    angle_threshold: float = 30.0,
    smooth_sigma: float = 1.0,
    background: bool = True,
) -> str:
    """Generate a layered, grouped SVG with one <g> per color.

    Tracing uses marching-squares (skimage `find_contours`) + RDP simplification
    + spline smoothing (`_to_smooth_path`), which produces smooth Bézier/arc
    curves for organic shapes and crisp corners for rectilinear ones.
    """
    arr = np.asarray(rgb_image)
    w, h = rgb_image.size
    svg = [
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        f'xmlns="http://www.w3.org/2000/svg">'
    ]
    svg.append('<metadata>XVGTracer: per-color contour traced layers</metadata>')
    if background:
        # bottom-most layer = most frequent color as solid backdrop
        svg.append('<rect width="100%" height="100%" '
                   f'fill="rgb{tuple(colors[0])}" />')
    for idx, color in enumerate(colors):
        mask = _binary_mask_for_color(arr, color)
        if not mask.any():
            continue
        paths = trace_layer(mask, simplify=simplify, angle_threshold=angle_threshold,
                            smooth_sigma=smooth_sigma)
        if not paths:
            continue
        fill = f"rgb({color[0]},{color[1]},{color[2]})"
        svg.append(f'<g id="layer_{idx}" data-color="{fill}" '
                   f'class="xvg-layer" fill="{fill}" fill-rule="evenodd">')
        for d in paths:
            svg.append(f'  <path d="{d}" />')
        svg.append("</g>")
    svg.append("</svg>")
    out = "\n".join(svg)
    with open(output, "w", encoding="utf-8") as f:
        f.write(out)
    return out


def process(image_path: str, num_colors: int = 5, output: str = "output_colored.svg",
             simplify: float = 1.0, angle_threshold: float = 30.0,
             smooth_sigma: float = 1.0) -> str:
    rgb, colors = quantize(image_path, num_colors)
    rgb.save("quantized.png")
    svg = generate_svg(rgb, colors, output=output, simplify=simplify,
                       angle_threshold=angle_threshold, smooth_sigma=smooth_sigma)
    return svg


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python xvgtracer.py input.png [num_colors] [output.svg] "
              "[angle_threshold] [smooth_sigma]")
        sys.exit(1)
    nc = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    out = sys.argv[3] if len(sys.argv) > 3 else "output_colored.svg"
    ang = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
    sig = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    process(sys.argv[1], nc, out, angle_threshold=ang, smooth_sigma=sig)
    print(f"Colored layered SVG saved: {out}")
