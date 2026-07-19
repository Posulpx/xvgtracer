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
    if name not in ('lune', 'lens', 'ellipse'):
        continue
    col = s['color']
    c = mask_for_color(rgb, tuple(col))
    comps = components_merged(c, merge_gap=0)
    mask = max(comps, key=lambda m: m.sum())
    contour_pts = contour_vertices(repair_mask(mask), fill_holes=True)
    raw = _closed_ring(contour_pts)
    cps = find_convergence_points(raw)
    n = len(raw)
    print(f'\n{name}: {len(cps)} CPs from find_conv: {cps[:15]}')
    lc = build_lc_path(raw)
    final = lc['convergence']
    print(f'  after build_lc_path: {len(final)} CPs: {final}')
    print(f'  segs: {"".join(lc["seg_types"])}')
