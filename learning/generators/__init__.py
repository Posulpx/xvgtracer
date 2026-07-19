"""generators package: build candidate point lists for primitive shapes."""

from .circle import circle_points, circle_from_bbox
from .rectangle import rectangle_points, rounded_rectangle_points
from .ellipse import ellipse_points, ellipse_from_bbox
from .polygon import (
    polygon_points,
    triangle_points,
    regular_polygon,
    star_points,
)

__all__ = [
    "circle_points",
    "circle_from_bbox",
    "rectangle_points",
    "rounded_rectangle_points",
    "ellipse_points",
    "ellipse_from_bbox",
    "polygon_points",
    "triangle_points",
    "regular_polygon",
    "star_points",
]
