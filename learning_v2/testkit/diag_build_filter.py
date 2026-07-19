"""Diagnostic: dump CP angle thresholds and final filter results."""

import sys, json, math
sys.path.insert(0, '.')
from learning_v2 import quantize, mask_for_color, components_merged, contour_vertices, repair_mask
from learning_v2 import _closed_ring, find_convergence_points, build_lc_path
import numpy as np

img, cols = quantize('tests/primitives.png', 13)
rgb = np.asarray(img)

d = json.load(open('tests/primitives.json'))
for s in d:
    name = s['type']
    if name not in ('ellipse', 'lune', 'lens', 'irregular', 'concave', 'blob'):
        continue
    col = s['color']
    c = mask_for_color(rgb, tuple(col))
    comps = components_merged(c, merge_gap=0)
    mask = max(comps, key=lambda m: m.sum())
    contour_pts = contour_vertices(repair_mask(mask), fill_holes=True)
    raw = _closed_ring(contour_pts)
    cps = find_convergence_points(raw)
    n = len(raw)
    print(f'\n{name}: {len(cps)} CPs (raw)')
    for k, cp in enumerate(cps):
        prev = (k - 1) % len(cps)
        nxt = (k + 1) % len(cps)
        i0 = (cps[prev] + n - 5) % n
        i1 = cps[k]
        i2 = (cps[nxt] + 5) % n
        v1x = raw[i1][0] - raw[i0][0]
        v1y = raw[i1][1] - raw[i0][1]
        v2x = raw[i2][0] - raw[i1][0]
        v2y = raw[i2][1] - raw[i1][1]
        dot = v1x * v2x + v1y * v2y
        cross = v1x * v2y - v1y * v2x
        angle = math.atan2(abs(cross), dot)
        print(f'  CP[{k}] idx={cp} angle={angle:.4f} keep={angle>=0.3}')
    lc = build_lc_path(raw)
    print(f'  after filter: {len(lc["convergence"])} CPs, segs={"".join(lc["seg_types"])}')
