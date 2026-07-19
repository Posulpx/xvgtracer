from __future__ import annotations

from typing import List, Tuple

import numpy as np
from PIL import Image

__all__ = [
    "Point",
    "components",
    "components_merged",
    "contour_vertices",
    "mask_for_color",
    "quantize",
    "repair_mask",
]

Point = Tuple[float, float]


def quantize(image_path: str, num_colors: int = 5):
    img = Image.open(image_path).convert("RGB")
    q = img.quantize(colors=num_colors, method=Image.FASTOCTREE, kmeans=num_colors)
    palette = np.array(q.getpalette()[: num_colors * 3]).reshape(-1, 3)
    colors = [tuple(int(c) for c in row) for row in palette]
    rgb = q.convert("RGB")
    return rgb, colors


def mask_for_color(arr: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    diff = np.abs(arr.astype(int) - np.array(color)).sum(axis=2)
    return diff < 8


def components(mask: np.ndarray, min_size: int = 16) -> List[np.ndarray]:
    from scipy.ndimage import label
    lab, n = label(mask)
    return [lab == k for k in range(1, n + 1) if (lab == k).sum() >= min_size]


def components_merged(mask: np.ndarray, merge_gap: int = 0,
                      min_size: int = 16) -> List[np.ndarray]:
    if merge_gap <= 0:
        return components(mask, min_size=min_size)
    from scipy.ndimage import label, binary_dilation
    conn = binary_dilation(mask, iterations=merge_gap)
    lab, n = label(conn)
    return [mask & (lab == k) for k in range(1, n + 1) if (mask & (lab == k)).sum() >= min_size]


def repair_mask(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_closing, binary_fill_holes
    m = binary_closing(mask, iterations=2)
    m = binary_fill_holes(m)
    return m


def contour_vertices(mask: np.ndarray, fill_holes: bool = True) -> List[Point]:
    from skimage import measure
    if not mask.any():
        return []
    m = mask
    if fill_holes:
        from scipy.ndimage import binary_fill_holes
        m = binary_fill_holes(m)
    contours = measure.find_contours(m.astype(float), 0.5)
    if not contours:
        return []
    c = max(contours, key=len)
    pts = [(float(y), float(x)) for x, y in c]
    if pts and pts[0] != pts[-1]:
        pts.append(pts[0])
    return pts
