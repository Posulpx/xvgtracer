import sys, json, math
sys.path.insert(0, '.')
from learning_v2 import quantize, mask_for_color, components_merged, contour_vertices, repair_mask
from learning_v2.__init__ import _closed_ring, find_convergence_points, build_lc_path
import numpy as np

img, cols = quantize('tests/primitives.png', 13)
rgb = np.asarray(img)

d = json.load(open('tests/primitives.json'))
for s in d:
    name = s['type']
    if name not in ('ellipse', 'lune', 'lens', 'irregular', 'concave'):
        continue
    col = s['color']
    c = mask_for_color(rgb, tuple(col))
    comps = components_merged(c, merge_gap=0)
    mask = max(comps, key=lambda m: m.sum())
    contour_pts = contour_vertices(repair_mask(mask), fill_holes=True)
    raw = _closed_ring(contour_pts)
    cps = find_convergence_points(raw)
    print(f'\n{name}: {len(cps)} CPs (raw)')
    lc = build_lc_path(raw)
    print(f'  after filter: {len(lc["convergence"])} CPs, segs={"".join(lc["seg_types"])}')
    if len(cps) >= 4:
        for k in range(1, len(cps) - 1):
            p0 = raw[cps[k-1]]
            p1 = raw[cps[k]]
            p2 = raw[cps[k+1]]
            chord = math.hypot(p2[0] - p0[0], p2[1] - p0[1])
            if chord > 1:
                d = abs((p2[0]-p0[0])*(p0[1]-p1[1]) - (p0[0]-p1[0])*(p2[1]-p0[1])) / chord
                print(f'  CP[{k}] idx={cps[k]} d/chord={d/chord:.4f}')
