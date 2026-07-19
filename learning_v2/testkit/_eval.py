"""SVG evaluation pipeline — rasterize with resvg and compute per-shape IoU."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


def render_svg(svg_path: str, out_png: str,
               resvg: str = "") -> None:
    if not resvg:
        resvg = r"C:\Users\pipos\AppData\Local\Temp\opencode\resvg\resvg.exe"
    subprocess.run([resvg, svg_path, out_png], check=True, capture_output=True)


def label_predicted_pixels(pred_rgb: np.ndarray,
                           gt_colors: List[Tuple[int, int, int]],
                           max_color_dist: float = 30) -> np.ndarray:
    gt_arr = np.array(gt_colors, dtype=np.float32)
    flat = pred_rgb.reshape(-1, 3).astype(np.float32)
    dists = np.sqrt(((flat[:, None] - gt_arr[None, :]) ** 2).sum(axis=2))
    nearest = dists.argmin(axis=1)
    label = nearest.reshape(pred_rgb.shape[:2]).copy()
    label[dists.min(axis=1).reshape(pred_rgb.shape[:2]) > max_color_dist] = -1
    return label


def compute_iou(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    inter = np.sum(gt_mask & pred_mask)
    union = np.sum(gt_mask | pred_mask)
    return inter / union if union else 0.0


def evaluate_svg(svg_path: str,
                 gt_image_path: str,
                 gt_json_path: str,
                 resvg: str = "") -> dict:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    render_svg(svg_path, tmp_path, resvg)
    pred = Image.open(tmp_path).convert("RGBA")
    bg = Image.new("RGBA", pred.size, (255, 255, 255, 255))
    bg.paste(pred, (0, 0), pred)
    pred_rgb = np.array(bg.convert("RGB"))
    os.unlink(tmp_path)
    gt = np.array(Image.open(gt_image_path))
    d = json.load(open(gt_json_path))
    gt_colors = [tuple(s["color"]) for s in d]
    label = label_predicted_pixels(pred_rgb, gt_colors)
    results = {}
    for i, s in enumerate(d):
        col = tuple(s["color"])
        gt_m = np.all(gt == col, axis=2)
        pred_m = label == i
        results[s["type"]] = compute_iou(gt_m, pred_m)
    total_inter = 0
    total_union = 0
    for i, s in enumerate(d):
        col = tuple(s["color"])
        gt_m = np.all(gt == col, axis=2)
        pred_m = label == i
        total_inter += np.sum(gt_m & pred_m)
        total_union += np.sum(gt_m | pred_m)
    results["overall"] = total_inter / total_union if total_union else 0.0
    return results
