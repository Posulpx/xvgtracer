"""Boundary evaluation: straight (incl. staircase) vs curve.

This is the low-level geometric evaluator that sits *below* primitive shape
detection. Before a contour can be called a circle/rect/line we must know the
local character of its boundary: is a run a STRAIGHT segment or a CURVE?

The key subtlety is the **staircase**: a quantized straight edge is rendered as
stepped pixels. A naive corner-counting check would misread those steps as
corners and conclude "not straight". This evaluator explicitly tolerates
stair-steps: it measures deviation from the run's chord, and a stair-stepped
straight edge has near-zero chord deviation (the steps stay close to the ideal
line), so it correctly evaluates as STRAIGHT. A true arc has large chord
deviation AND fits a circle well, so it evaluates as CURVE.

The evaluator works on a contour as a list of (x, y) points and returns, per
segmented run, a classification plus an overall summary of the boundary.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

Point = Tuple[float, float]


def _np_array(p):
    import numpy as np
    if isinstance(p, tuple):
        return np.array(p, dtype=float)
    return np.asarray(p, dtype=float)


def _chord_deviation(pts: List[Point]) -> float:
    """Max perpendicular distance of points from the chord (first->last line).

    Near-zero for a straight OR stair-stepped-straight run; grows with curvature.
    """
    if len(pts) < 3:
        return 0.0
    a = _np_array(pts[0])
    b = _np_array(pts[-1])
    v = b - a
    L = math.hypot(*v)
    if L < 1e-9:
        return 0.0
    nx, ny = -v[1] / L, v[0] / L
    return float(max(abs((p[0] - a[0]) * nx + (p[1] - a[1]) * ny) for p in pts))


def _run_window(pts: List[Point], target_runs: int = 4,
                min_pts: int = 8, max_pts: int = 200) -> int:
    """Choose a run length (in points) scaled to the contour's sampling.

    A fixed point-count window fails on densely-sampled contours: each window
    then covers only a tiny, nearly-flat arc of a real curve and reads as
    straight. Sizing the window so the contour breaks into ~``target_runs``
    runs makes each run span a meaningful fraction of the boundary: a genuine
    curve then shows real sagitta while a stair-stepped straight edge's steps
    average out, keeping its chord deviation near zero.
    """
    n = len(pts)
    if n <= min_pts:
        return n
    return max(min_pts, min(max_pts, n // target_runs))


def _segment_runs(pts: List[Point], max_run: int = None) -> List[List[Point]]:
    """Break a contour into sliding windows (runs) for local evaluation."""
    win = max_run if max_run is not None else _run_window(pts)
    if len(pts) <= win:
        return [list(pts)]
    runs = []
    step = max(1, win // 2)
    for i in range(0, len(pts) - win + 1, step):
        runs.append(list(pts[i:i + win]))
    return runs or [list(pts)]


def _rdp_vertex_count(pts: List[Point], eps: float) -> int:
    """Number of vertices after RDP simplification at tolerance `eps`.

    A straight (or stair-stepped) run collapses to 2 vertices; a curved run
    needs several — this is what separates a staircase edge from a true curve.
    """
    if len(pts) < 3:
        return len(pts)
    import numpy as np
    a = np.asarray(pts, dtype=float)
    ax, ay = a[0]
    bx, by = a[-1]
    vx, vy = bx - ax, by - ay
    L = math.hypot(vx, vy)
    if L < 1e-9:
        return 2
    d = [abs((p[0] - ax) * vy - (p[1] - ay) * vx) / L for p in a[1:-1]]
    i = int(np.argmax(d))
    if d[i] > eps:
        return (_rdp_vertex_count(pts[:i + 2], eps)
                + _rdp_vertex_count(pts[i + 1:], eps) - 1)
    return 2


def _run_type(run: List[Point], curve_vertices: int = 4,
              eps: float = 2.0) -> str:
    """Classify a single run as 'straight' (incl. staircase) or 'curve'.

    RDP-simplifies the run at a small *fixed* tolerance and counts vertices:

      * a stair-stepped straight edge collapses to ~2 vertices (steps are
        within tolerance),
      * a run crossing a single polygon corner keeps ~3 vertices (two straight
        edges meeting at one sharp bend),
      * a smooth curve needs >= 4 vertices (many small consistent bends).

    Using a fixed epsilon (rather than one scaled to the run) is what lets us
    tell a sharp polygon corner apart from smooth curvature: both deviate from
    the chord, but only the curve requires many vertices to represent. Runs
    with >= ``curve_vertices`` vertices are CURVE.
    """
    if len(run) < 3:
        return "straight"
    n = _rdp_vertex_count(run, eps)
    return "curve" if n >= curve_vertices else "straight"


def evaluate_boundary(pts: List[Point], curve_vertices: int = 4) -> Dict:
    """Evaluate a whole boundary into a straight/curve summary.

    Returns a dict:
      ``runs``      -> list of per-run classifications
      ``counts``    -> tallies per type
      ``dominant``  -> the boundary's overall character ('straight'/'curve')
      ``is_straight``-> True if the boundary is predominantly straight
                       (incl. stair-stepped), False if predominantly curved
    """
    runs = _segment_runs(pts)
    types = [_run_type(r, curve_vertices) for r in runs]
    counts = {t: types.count(t) for t in ("straight", "curve")}
    dominant = max(counts, key=counts.get) if counts else "straight"
    return {
        "runs": types,
        "counts": counts,
        "dominant": dominant,
        "is_straight": counts.get("straight", 0) >= counts.get("curve", 0),
    }
