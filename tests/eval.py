import subprocess, json
import numpy as np
from PIL import Image

subprocess.run([
    r'C:\Users\pipos\AppData\Local\Temp\opencode\resvg\resvg.exe',
    'tests/primitives.svg','tests/primitives_render.png'
], check=True, capture_output=True)

pred = Image.open('tests/primitives_render.png').convert('RGBA')
bg = Image.new('RGBA', pred.size, (255,255,255,255))
bg.paste(pred, (0,0), pred)
pred_rgb = np.array(bg.convert('RGB'))

gt = Image.open('tests/primitives.png')
gt_a = np.array(gt)

d = json.load(open('tests/primitives.json'))
gt_colors = [tuple(s['color']) for s in d]
gt_colors_arr = np.array(gt_colors, dtype=np.float32)

pred_flat = pred_rgb.reshape(-1, 3).astype(np.float32)
dists = np.sqrt(((pred_flat[:, None] - gt_colors_arr[None, :]) ** 2).sum(axis=2))
nearest_idx = dists.argmin(axis=1)
nearest_dist = dists.min(axis=1)
pred_label = nearest_idx.reshape(800, 800).copy()
pred_label[nearest_dist.reshape(800, 800) > 30] = -1

for i, s in enumerate(d):
    col = tuple(s['color'])
    gt_m = np.all(gt_a == col, axis=2)
    pred_m = pred_label == i
    inter = np.sum(gt_m & pred_m)
    union = np.sum(gt_m | pred_m)
    iou = inter / union if union else 0
    print(f'{s["type"]:12s} IoU={iou:.3f}  gt={np.sum(gt_m):5d} pred={np.sum(pred_m):5d} inter={inter:5d}')

print()
total_inter = 0; total_union = 0
for i, s in enumerate(d):
    col = tuple(s['color'])
    gt_m = np.all(gt_a == col, axis=2)
    pred_m = pred_label == i
    total_inter += np.sum(gt_m & pred_m)
    total_union += np.sum(gt_m | pred_m)
print(f'Overall IoU={total_inter/total_union:.3f}')
