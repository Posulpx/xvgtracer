import sys, math
sys.path.insert(0, '.')
import numpy as np
from PIL import Image
from learning_v2._image import quantize, mask_for_color, components_merged, contour_vertices, repair_mask
from learning_v2 import _closed_ring, _smooth_ring, find_convergence_points, _simplified_signature, build_lc_path

img, cols = quantize('tests/primitives.png', 13)
rgb = np.asarray(img)
c = mask_for_color(rgb, (0,0,200))
comps = components_merged(c, merge_gap=0)
mask = max(comps, key=lambda m: m.sum()) if comps else c
contour = contour_vertices(repair_mask(mask), fill_holes=True)
raw = _closed_ring(contour)
xs = [p[0] for p in raw]; ys = [p[1] for p in raw]
diag = math.hypot(max(xs)-min(xs), max(ys)-min(ys))

eps = max(1.0, 0.012 * diag)
simp, sig = _simplified_signature(raw, eps)
asig = np.abs(sig)

print(f'blob: n_raw={len(raw)} n_simp={len(simp)} diag={diag:.0f} eps={eps:.1f}')
print(f'RDP sig: {np.round(sig, 3)}')
print(f'max_turn={np.max(asig):.3f} var={np.var(asig):.5f} n_peaks={int(np.sum(asig>0.35))}')

cps = find_convergence_points(raw)
print(f'convergence points ({len(cps)}):')
for ci in cps:
    print(f'  cp[{ci}]: ({raw[ci][0]:.1f}, {raw[ci][1]:.1f})')

lc = build_lc_path(raw)
print(f'seg_types: {lc["seg_types"]}')

ring = _smooth_ring(raw, win=3)
print(f'tangents at CPs:')
for ci in cps:
    i_before = (ci - 1) % len(ring)
    i_after = (ci + 1) % len(ring)
    dx = ring[i_after][0] - ring[i_before][0]
    dy = ring[i_after][1] - ring[i_before][1]
    angle = math.degrees(math.atan2(dy, dx))
    print(f'  cp[{ci}]: ({ring[ci][0]:.1f},{ring[ci][1]:.1f})  tangent={angle:.0f}deg')
