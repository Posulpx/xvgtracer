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
    if name not in ('blob', 'lens'):
        continue
    col = s['color']
    c = mask_for_color(rgb, tuple(col))
    comps = components_merged(c, merge_gap=0)
    mask = max(comps, key=lambda m: m.sum())
    contour_pts = contour_vertices(repair_mask(mask), fill_holes=True)
    raw = _closed_ring(contour_pts)
    cps = find_convergence_points(raw)
    n = len(raw)
    print(f'\n{name}: {len(cps)} CPs from find_conv')
    for k, cp in enumerate(cps):
        prev = (k - 1) % len(cps)
        nxt = (k + 1) % len(cps)
        p0 = raw[cps[prev]]
        p1 = raw[cp]
        p2 = raw[cps[nxt]]
        chord = math.hypot(p2[0] - p0[0], p2[1] - p0[1])
        if chord < 1e-6:
            d_ratio = 0.0
        else:
            d = abs((p2[0]-p0[0])*(p0[1]-p1[1]) - (p0[0]-p1[0])*(p2[1]-p0[1])) / chord
            d_ratio = d / chord
        print(f'  CP[{k}] idx={cp} d/chord={d_ratio:.4f} corner={d_ratio>0.25}')
    lc = build_lc_path(raw)
    print(f'  final: {len(lc["convergence"])} CPs segs={"".join(lc["seg_types"])}')
