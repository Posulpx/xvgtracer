import sys, json
sys.path.insert(0, '.')
from learning_v2 import quantize, mask_for_color, components_merged, contour_vertices, repair_mask
from learning_v2.__init__ import _closed_ring, _simplified_signature, find_convergence_points
import numpy as np

img, cols = quantize('tests/primitives.png', 13)
rgb = np.asarray(img)

d = json.load(open('tests/primitives.json'))
for s in d:
    name = s['type']
    col = s['color']
    c = mask_for_color(rgb, tuple(col))
    comps = components_merged(c, merge_gap=0)
    mask = max(comps, key=lambda m: m.sum())
    contour_pts = contour_vertices(repair_mask(mask), fill_holes=True)
    raw = _closed_ring(contour_pts)
    xs = [p[0] for p in raw]
    ys = [p[1] for p in raw]
    diag = np.hypot(max(xs)-min(xs), max(ys)-min(ys))
    eps = max(1.0, 0.012 * diag)
    simp, sig = _simplified_signature(raw, eps)
    mag = np.abs(sig)
    n_high = int(np.sum(mag > 0.55))
    n_035 = int(np.sum(mag > 0.35))
    n_max = float(np.max(mag)) if len(mag) else 0
    cps = find_convergence_points(raw)
    print(f'{name:12s} diag={diag:.0f} rdp={len(simp)} n_035={n_035} n_high={n_high} max={n_max:.2f} cps={len(cps)}')
