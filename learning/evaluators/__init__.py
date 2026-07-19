"""Evaluators package: low-level geometric evaluation of boundaries.

These sit *below* primitive shape detection: before a contour can be called a
circle/rect/line we must know the local character of its boundary (straight,
possibly stair-stepped, vs curve). See `boundary_evaluator`.
"""

from .boundary_evaluator import (
    evaluate_boundary,
    _run_type,
    _chord_deviation,
    _rdp_vertex_count,
)

__all__ = [
    "evaluate_boundary",
    "_run_type",
    "_chord_deviation",
    "_rdp_vertex_count",
]
