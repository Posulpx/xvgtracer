"""metrics package: lightweight geometric comparison of shapes/contours."""

from .iou import iou_masks, iou_polygons, coverage
from .contour_error import (
    mean_contour_error,
    rms_contour_error,
    max_contour_error,
)
from .hausdorff import directed_hausdorff, hausdorff

__all__ = [
    "iou_masks",
    "iou_polygons",
    "coverage",
    "mean_contour_error",
    "rms_contour_error",
    "max_contour_error",
    "directed_hausdorff",
    "hausdorff",
]
