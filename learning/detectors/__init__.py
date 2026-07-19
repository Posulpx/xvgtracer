"""detectors package: convert a binary mask into contours, edges, and components."""

from .contour_detector import contour_vertices, all_contours
from .edge_detector import (
    repair_mask,
    smooth_mask,
    sobel_edges,
    edge_density,
)
from .component_detector import (
    mask_extent,
    components,
    components_merged,
    is_background,
    centroid_of_mask,
)
from .collision_detector import collision_points

__all__ = [
    "contour_vertices",
    "all_contours",
    "repair_mask",
    "smooth_mask",
    "sobel_edges",
    "edge_density",
    "mask_extent",
    "components",
    "components_merged",
    "is_background",
    "centroid_of_mask",
    "collision_points",
]
