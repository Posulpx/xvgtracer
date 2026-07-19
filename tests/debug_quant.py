import json
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

img = Image.open('tests/primitives.png')
arr = np.array(img)

# Manual quantize (same method as learning_v2)
h, w = arr.shape[:2]
px = arr.reshape(-1, 3).astype(np.float32)
kmeans = KMeans(n_clusters=12, random_state=0, n_init=3).fit(px)
labels = kmeans.labels_.reshape(h, w)
palette = np.round(kmeans.cluster_centers_).astype(np.uint8)
q = palette[labels]

print('Palette:')
for i, c in enumerate(palette):
    m = labels == i
    print(f'  {i}: rgb({c[0]},{c[1]},{c[2]})  {np.sum(m)} px')
print()

# Map each ground-truth color to palette colors
d = json.load(open('tests/primitives.json'))
gt_a = np.array(img)
for s in d:
    col = tuple(s['color'])
    gt_m = np.all(gt_a == col, axis=2)
    if np.sum(gt_m) == 0:
        print(f'{s["type"]:12s} NOT FOUND in image')
        continue
    overlaps = []
    for i in range(12):
        inter = np.sum(gt_m & (labels == i))
        if inter > 0:
            overlaps.append((inter, i))
    overlaps.sort(reverse=True)
    total = np.sum(gt_m)
    cov = overlaps[0][0] / total if overlaps else 0
    best_idx = overlaps[0][1] if overlaps else -1
    print(f'{s["type"]:12s} col={col}  -> pal#{best_idx}={tuple(palette[best_idx])}  cov={cov:.3f} ({overlaps[0][0]}/{total})')
    if len(overlaps) > 1:
        for ov, i in overlaps[1:]:
            print(f'                        also pal#{i}={tuple(palette[i])}  px={ov}')
