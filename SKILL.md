---
name: xvgtracer
description: Build and extend XVGTracer — a color image to layered/editable SVG tracer using Pillow quantization + per-color contour tracing. Use when tracing raster images into vector SVG layers, adding BiRefNet matting, or wiring the FastAPI + SvelteKit app.
---

# XVGTracer Skill

XVGTracer converts a raster image into a **layered, editable SVG** — one `<g>` group per
dominant color, each traced into vector paths (marching-squares + spline-smoothed curves).

## When to use
- User wants to vectorize / trace an image into SVG with separate color layers.
- User mentions XVGTracer, color-trace, potrace, image-to-SVG, or editable vector layers.
- Extending the POC: BiRefNet matting, real Potrace Béziers, SVGO optimization.

## Project layout
- `xvgtracer.py` — core. `quantize()`, `_binary_mask_for_color()`, `trace_layer()` (marching-squares + spline smoothing), `generate_svg()`, `process()`.
- `learning/` — learning **package** (`from learning import learner`).
  - `learner.py` — orchestrator. `learn_image()` (full pipeline), `model_to_svg()`, `identify_shape()` / `rebuild_shape()` (delegates to `svg_renderer._node_d`) / `detect_overlaps()` public wrappers, `save_model()`/`load_model()`.
  - `classifiers/` — **mask→node logic**. `mask_classifier.classify_mask()` (repair → extent → best-fit IoU → star → rotated → polygon fallback), `_fit_rotated_primitive()`, `estimate_orientation()`, `build_transform()` (single owner of rotate-Transform attachment).
  - `shape_registry.py` — vocabulary (`PRIMITIVES/COMPOSITES/TRANSFORMS`), primitive→reconstructor map, shared `fit_candidates()` IoU-ranking helper, `register()` bookkeeping.
  - `composites.py` — `build_composites()`: overlap records → `difference`/`union` composite nodes.
  - `metrics/` — `iou` (mask/polygon IoU, coverage), `contour_error` (mean/rms/max), `hausdorff`. Pure geometry.
  - `detectors/` — `contour_detector` (marching-squares → (x,y) ring + gap/hole repair), `edge_detector` (mask repair, Gaussian smoothing, Sobel edges), `component_detector` (mask extent, components, `is_background`, centroid).
  - `generators/` — `circle`, `rectangle` (+rounded), `ellipse`, `polygon` (+triangle/star/regular): each builds candidate point lists.
  - `reconstructors/` — `primitive_reconstructor` (point/circle/ellipse/arc/line/rect/rounded_rect → `d`), `polygon_reconstructor` (triangle/polygon/star → `d`), `bezier_reconstructor` (explicit cubic or Catmull-Rom smooth).
  - `renderers/` — `raster_renderer` (`rasterize_polygon`, `rasterize_node` — shared mask backend), `svg_renderer` (`model_to_svg`, `_node_d`, `transform_to_svg`).
- `tests/` — synthetic test images (`clean_shapes.png`, `transform_shapes.png`) + fixtures.
- `api.py` — FastAPI: `POST /trace` (multipart: `image`, `num_colors`, `simplify`, `angle_threshold`, `smooth_sigma`) → SVG; `POST /learn` (multipart: `image`, `num_colors`) → `{model, svg, summary}`; `GET /health`.
- `frontend/` — SvelteKit. `src/routes/+page.svelte` handles upload, inline `{@html svg}`, layer checkboxes, **Learn** button (calls `/api/learn`), SVGO, download. Vite proxies `/api` → `:8000`.

## Tracing engine (single)
- `trace_layer()` (skimage `find_contours` marching-squares) → `_rdp` → `_to_smooth_path`.
  **Potrace was evaluated and dropped**: the `potracer` pure-Python port is buggy
  (rects→diamonds, triangles→under-filled kites, IoU ~0.5) and `pypotrace` can't build
  here (needs libagg/pkg-config). The marching-squares path with spline smoothing
  reliably traces all shapes (per-layer IoU 0.95–0.97 on test images).

## Learning system (`learner.py` + package)
Goal: turn each color mask into a **node hierarchy** that is editable/structured:
```
Primitive : point | line | circle | ellipse | rect | rounded_rect |
            triangle | polygon(N) | bezier | arc
Composite : union(A,B) | difference(A,B) | intersection(A,B) | xor(A,B)
Transform : translate | rotate | scale | skew | mirror   (attached to a node)
```
`learner.learn_image()` orchestrates three stages across the package:
1. **identify** — `classifiers.mask_classifier.classify_mask()` repairs the mask (`detectors.edge_detector.repair_mask`,
   gap-close + hole-fill, robust to anti-aliased quantization) → `detectors.component_detector.mask_extent`
   (true pixel extent, not the largest contour, which can fragment). Detection order: **triangle** (clean 3-corner fit,
   confirmed by re-rasterised IoU so stars are rejected) → **star** (strictly-alternating radial tips/valleys on the
   raw contour, via angle-binned radii) → best-fit IoU via `shape_registry.fit_candidates` over candidates
    (circle/ellipse/rect/rounded_rect/triangle); `rounded_rect` wins when its IoU beats `rect`. When a star is
    detected, `classify_mask` runs BOTH `_star_vertices` (angle-binned tip maxima, de-duplicated by angular
    separation, valleys via windowed-min at the mid-angle) and `_star_vertices_angle` (pure angle-space local
    maxima with 30° spacing + min-radius gap valleys) and keeps whichever re-rasterises with higher IoU. The
    `_detect_star` guard requires ≥5 significant radial maxima (rmin + 0.25·range) so a clipped circle (one
    overlap-notch) is NOT mis-classified as a star. Rotated rects/ellipses are
   caught by `_fit_rotated_primitive()` (min-area-rectangle orientation search → de-rotate → re-fit → `rotate`
   Transform node); everything else → `polygon`. `build_transform()` is the single owner of rotate-Transform attachment.
   Gotcha: the `rounded_rectangle_points` generator must use **radians** for its corner arcs (a degree/radian bug
   previously made rounded-rect rasterise to a thin ring).
 2. **rebuild** — `learner.rebuild_shape()` delegates to `renderers.svg_renderer._node_d`, which dispatches to
    `reconstructors/` (`primitive_reconstructor` for circle/ellipse/rect/rounded_rect/arc/line/point,
    `polygon_reconstructor` for polygon/star/triangle, `bezier_reconstructor` for cubic/Bezier).
 3. **postprocess** — `learner.detect_overlaps()` rasterises each shape via `renderers.raster_renderer`
    and records pairwise IoU. `composites.build_composites()` builds **composite** nodes: `keep_topmost` → `difference`,
    `contained` → `union`. Background layers (>60% of frame, via `is_background`) excluded.
    `renderers.svg_renderer.transform_to_svg()` renders attached Transform nodes.
    **Composite SVG rendering**: the `model_to_svg` renderer does NOT paint a composite as one flat group colour.
    Instead it emits each child as its own z-ordered, coloured `<path>` (bottom first, top last) and skips the
    standalone draw of composite members. This keeps every shape's colour and makes the overlap show the top
    shape's colour (natural overpainting) — so `difference`/`union` both look correct rather than flattening to
    the top colour. (Earlier versions used a single evenodd path that punched holes to the background.)

Key gotchas (learned the hard way):
- `find_contours` returns **(row, col)**; `detectors.contour_detector` swaps to **(x, y)** for drawing.
- Anti-aliased masks fragment into multiple contours → never trust `max(contours, key=len)` for bbox;
  use `mask_extent` on raw mask pixels. Repair with `binary_closing` before fitting so edge gaps don't
  penalise primitive IoU.
- Rotated-rectangle PCA orientation is unreliable; `shape_registry.estimate_orientation` uses a
  min-area-rectangle angle search instead.
- Keep the package dependency direction acyclic: `metrics` ← `detectors`/`generators` ←
  `reconstructors`/`renderers` ← `shape_registry` ← `learner`. `learner`/`shape_registry` import from
  the sub-packages; sub-packages never import `learner`.

## Key implementation notes
- Quantization: `Image.quantize(colors=N, method=Image.FASTOCTREE, kmeans=N)` then read palette
  from the **P-mode** image BEFORE `.convert("RGB")` (converting drops the palette).
- Contour tracing uses skimage `measure.find_contours(mask, 0.5)` (marching squares).
- Simplification: `_rdp()` is a **closed-contour-aware** Ramer–Douglas–Peucker. For closed
  loops, first split at the farthest point pair (else a circle collapses to 2 points). Use the
  2D perpendicular-distance formula `|rel_x*vec_y - rel_y*vec_x| / norm` — numpy `cross` is 3D only.
- `simplify` param = RDP epsilon in pixels (default 2.0). Higher = fewer path points.
- `smooth_sigma` (px, default 1.0): low-pass on the **contour point ring** before tracing
  (NOT image blur — blur drifts the boundary and worsens shape error). Kills grid jitter
  that causes "polygonal hints". Higher = smoother but can round micro-details.
- `angle_threshold` (deg, default 30) = spline corner sensitivity. `_to_smooth_path()`
  classifies each vertex by its *exterior bend angle* (180° = straight, small = sharp).
  Vertices bending more than `angle_threshold` from straight become hard corners (`L`);
  gentler runs become cubic Béziers via Catmull-Rom→Bézier. So circles/arcs curve,
  rectangles/stars stay crisp. NOTE: turn is measured as `180 - angle`, not the raw angle.
- Output: each layer `<g id="layer_i" data-color="rgb(...)" fill-rule="evenodd">`.

## Run
```bash
pip install pillow numpy scikit-image scipy fastapi uvicorn
python -m learning.learner tests/clean_shapes.png 8                    # CLI -> _learned.json + .svg
uvicorn api:app --reload --port 8000                        # API (/trace + /learn)
cd frontend && npm install && npm run dev                   # UI
```

## Upgrade paths
- **Better curves**: improve `_to_smooth_path` tight Béziers / arc fitting (no Potrace — see above).
- **Composite inference**: add `intersection`/`xor` detection from overlap geometry (node types exist).
- **Clean foreground**: run BiRefNet matting first, trace only foreground layers.
- **Better colors**: OKLab / pngquant instead of median-cut.
- **Optimization**: server-side SVGO (svgwrite) instead of the client regex pass.
