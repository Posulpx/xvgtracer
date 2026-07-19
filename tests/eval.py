import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from learning_v2 import render_svg
render_svg("tests/primitives.png", num_colors=13, out="tests/primitives.svg")
from learning_v2.testkit import run_standard_eval

results = run_standard_eval()
for name in ["circle", "triangle", "rect", "pentagon", "hexagon", "star",
             "irregular", "concave", "blob", "ellipse", "lune", "lens"]:
    print(f"{name:12s} IoU={results.get(name, 0):.3f}")
print(f"\nOverall IoU={results.get('overall', 0):.3f}")
