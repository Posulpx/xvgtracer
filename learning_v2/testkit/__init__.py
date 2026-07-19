import json
from ._gen import (
    polygon,
    star,
    ellipse_pts,
    blob,
    lens_lune_pts,
    generate_shapes,
    render_ground_truth,
    SHAPE_DEFS,
    IRREGULAR_PTS,
    CONCAVE_PTS,
)
from ._eval import (
    render_svg,
    label_predicted_pixels,
    compute_iou,
    evaluate_svg,
)


def regenerate_standard_test(path: str = "tests/primitives.png",
                             json_path: str = "tests/primitives.json") -> None:
    shapes = generate_shapes()
    img = render_ground_truth(800, 800, shapes)
    img.save(path)
    with open(json_path, "w") as f:
        json.dump(shapes, f)


def run_standard_eval(svg_path: str = "tests/primitives.svg",
                      gt_image: str = "tests/primitives.png",
                      gt_json: str = "tests/primitives.json") -> dict:
    return evaluate_svg(svg_path, gt_image, gt_json)

