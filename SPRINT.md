# SPRINT — XVGTracer

One-shot build of a color-image → layered SVG tracer (Potrace-style curves, no system potrace),
plus a learning system that identifies shapes as editable primitives, composites, and transforms.

## Status: ✅ Working (verified end-to-end)

- Python core: quantize → per-color contour tracing → grouped SVG ✅
- **Learning system** (`learner.py`): identify → rebuild → postprocess into a node hierarchy ✅
- FastAPI endpoints `/trace` and `/learn` return SVG / JSON model ✅
- SvelteKit frontend: upload, inline SVG, layer toggle, **Learn**, SVGO, download ✅

## Architecture

```
XVGTracer/
├── xvgtracer.py        # core: quantize() + trace_layer() + generate_svg()
├── api.py              # FastAPI wrapper (POST /trace, POST /learn, GET /health)
├── learning/           # learning system (package)
│   ├── learner.py      # orchestrator (wires the package together)
│   ├── shape_registry.py  # vocabulary + primitive->reconstructor map + fit_candidates
│   ├── classifiers/    # mask_classifier (classify_mask, rotated-fit, build_transform)
│   ├── composites.py   # overlap records -> composite nodes
│   ├── metrics/        # iou, contour_error, hausdorff
│   ├── detectors/      # contour_detector, edge_detector, component_detector
│   ├── generators/     # circle, rectangle, ellipse, polygon
│   ├── reconstructors/ # primitive, polygon, bezier
│   └── renderers/      # raster_renderer, svg_renderer
├── tests/              # test images + fixtures
│   ├── clean_shapes.png    # rect/circle/ellipse/triangle/star/poly
│   └── transform_shapes.png# rotated ellipse + overlapping circles
└── frontend/           # SvelteKit app
    └── src/routes/+page.svelte
```

### Learning-system package layout (`learning/`)
The learner is split into focused modules (orchestrated by `learning/learner.py`,
imported as `from learning import learner`):
- **metrics/** — `iou` (mask/polygon IoU, coverage), `contour_error` (mean/rms/max
  nearest-neighbour), `hausdorff` (directed + undirected). Pure geometry.
- **detectors/** — `contour_detector` (marching-squares → (x,y) ring, gap/hole fix),
  `edge_detector` (mask repair, Gaussian smoothing, Sobel edges), `component_detector`
  (mask extent, connected components, background test, centroid).
- **generators/** — one module per primitive producing candidate point lists:
  `circle`, `rectangle` (+rounded), `ellipse`, `polygon` (+triangle/star/regular).
- **reconstructors/** — `primitive_reconstructor` (point/circle/ellipse/arc/line/
  rect/rounded_rect → `d`), `polygon_reconstructor` (triangle/polygon/star → `d`),
  `bezier_reconstructor` (explicit cubic control points or Catmull-Rom smooth).
- **renderers/** — `raster_renderer` (polygon/node → boolean mask, shared backend),
  `svg_renderer` (model → layered SVG, transform → `transform=`).
- **classifiers/** — `mask_classifier`: owns all mask→node logic — `classify_mask()`
  (repair → extent → **triangle** (3-corner fit, IoU-confirmed) → **star**
  (strict alternation on angle-binned radii + re-rasterised-IoU guard) →
  best-fit IoU via `fit_candidates` (circle/ellipse/rect/rounded_rect/triangle)
  → rotated → **custom polygon** fallback), `_fit_rotated_primitive()`,
  `estimate_orientation()`, `build_transform()`.
  - *Discipline:* not everything is a primitive. If a mask qualifies as none of
    the primitives (and no rotated primitive fits), it MUST be a `polygon`. A
    star whose arms are clipped by an overlap (e.g. a yellow circle cutting
    through an arm) fails `_detect_star`'s symmetry/IoU guard and is NOT
    mis-identified as a star — it falls through to **custom shape
    reconstruction** (`_custom_polygon_node`): the smoothed, adaptively
    simplified contour emitted as a `polygon` node flagged `custom: True`,
    faithfully tracing the genuine (clipped) silhouette.
- **shape_registry.py** — vocabulary (`PRIMITIVES/COMPOSITES/TRANSFORMS`), the
  primitive→reconstructor map, the shared `fit_candidates()` IoU-ranking helper,
  and `register()` for bookkeeping.
- **composites.py** — `build_composites()`: turns overlap records into
  `difference`/`union` composite nodes (moved out of the orchestrator).

### Tracing pipeline (`xvgtracer.py`)
1. `quantize()` — Pillow `quantize(colors=N, kmeans=N)` → palette + RGB image.
2. Per color: `_binary_mask_for_color()` builds a boolean mask.
3. `generate_svg()` — one `<g id="layer_i" data-color=...>` per color, evenodd fill.
   Tracing engine: `trace_layer()` — skimage `find_contours` (marching squares) →
   `_rdp()` polyline simplification → `_to_smooth_path()` (angle-gated Catmull-Rom
   Béziers). This is the single, reliable tracer (no potrace — see limitations).

### Learning pipeline (`learner.py` + package)
`learner.learn_image(image_path, num_colors, merge_gap=0)` orchestrates three
stages over the per-color masks:
- **Component split + gap merge.** Each color mask is split into connected
  components (`detectors.components_merged`). `merge_gap` (px) is a tolerance for
  rejoining *nearby* fragments that visually belong to one silhouette — e.g. a
  drawn star whose sharp tip/arm was split off by a hairline anti-aliasing gap.
  With `merge_gap=0`, every disjoint component is its own silhouette (so you can
  *see* the split); raising it (e.g. 15) re-fuses the pointy sides into one star.
  Only connectivity is merged — the returned masks keep the original pixels, so
  shapes are not artificially fattened.
1. **identify** (`shape_registry.classify_mask`) — classify each mask into a
   primitive node by best-fit IoU against candidate primitives (built by
   `generators/`). Repairs anti-aliased masks (`detectors.edge_detector`) and uses
   pixel extent (`detectors.component_detector`) before fitting. Detects stars via
   radial alternation; attaches a **Transform** node (rotate) when a rect/ellipse is
   rotated (min-area-rectangle search, de-rotate, re-fit).
2. **rebuild** (`reconstructors/`) — emit an ideal SVG `d` for each primitive
   (arc for circle/ellipse, `Q` for rounded_rect, polyline for polygon/star/triangle,
   cubic `C` for bezier).
3. **postprocess** (`learner.detect_overlaps` + `renderers.raster_renderer`) — detect
   overlaps; build **Composite** nodes:
   - overlapping & topmost → `difference` (top punches a hole, evenodd fill-rule)
   - one contained in another → `union`
   Background layers (cover >60% of frame, via `is_background`) are excluded.

The model is JSON-serialisable and carries the full hierarchy:
`primitives | composites | transforms` (see `model["hierarchy"]`). It also carries
a `collisions` list of color-region junctions (see below) and a `summary` block
(`n_silhouettes` incl. canvas, `n_collisions`, etc.).

**Every island is drawn (per-shape composite ownership).** One color can split
into several disjoint silhouettes that share a `layer_index` (e.g. a star clipped
into a big body + a small fragment). The SVG renderer decides "is this shape
already rendered by a composite?" **per shape** — by identity plus an
`in_composite` tag set in `build_composites` — never by `layer_index`. A previous
layer-based skip dropped the small island because its larger sibling was a
composite member. The fix is universal: only the exact island inside a composite
is skipped; all other islands still get their standalone draw.
Metrics in `metrics/` (IoU, contour error, Hausdorff) are available for evaluating
reconstruction fidelity.

### Collision points (`detectors.collision_detector`)
`collision_points(rgb, min_colors=2, ...)` scans the *quantized* image pixel-by-
pixel: any pixel whose local window contains ≥ `min_colors` distinct colors (the
canvas/background is included by default) is a convergence point. 2-color and
3-color junctions use the *same* procedure; 3-color convergences are simply
prioritised by sorting on descending `n_colors`. Hits are clustered (radius 3px)
so one junction yields one dot, and results are sorted by descending `n_colors`
so **3-color convergences come first** (the strongest structural corners). With
`boundary_only=True` (default) a point is kept only if its centre pixel sits on a
real region boundary (differs from a 4-neighbour), dropping internal points inside
a single shape. Each entry is
`{x, y, n_colors, colors}` where `colors` are the actual RGB triples meeting there.
The renderer draws them as dots on top of the SVG (`<g id="collisions">`): red for
3+ color junctions, blue for 2-color edges. Ensure `min_colors=2` to also include
plain edges.

After the in-detection clustering, a final **agglomerative merge** fuses any two
*emitted* junctions whose centres lie within `merge_dist` px (default **1.0**,
adjustable via `learn_image(..., collision_merge_dist=)`). This handles the
quantized-scale case where two genuinely-distinct junctions land ~1px apart and
should read as one convergence: the merged point takes the area-weighted centroid
of its members, the **union** of their converging colors, and the max `n_colors`
(so a 2-color + 2-color pair that share one color can promote to a 3-color
junction). `merge_dist=0` disables the pass. It repeatedly merges the closest
in-range pair until none remain.

### Boundary evaluation: straight (incl. staircase) vs curve (`learning/evaluators/`)
This is the low-level ingredient that sits *below* primitive shape detection.
Before a contour can be called a circle/rect/line, `evaluate_boundary(pts)`
segments the contour into runs and classifies each run as **straight** or
**curve**, returning per-run types, tallies, a `dominant` character, and
`is_straight`. The single hard problem it solves is the **staircase**: a
quantized straight edge is rendered as stepped pixels, and those steps must NOT
be mistaken for curvature.

How it decides (two robust ingredients):
  * **Perimeter-scaled runs** (`_run_window`, `target_runs≈4`) — the run length
    scales to the contour so each run spans a meaningful fraction of the
    boundary. A fixed point-count window fails on densely-sampled contours
    (each window covers a tiny near-flat arc and reads straight); scaling makes
    a genuine curve reveal real sagitta while staircase steps average out.
  * **Fixed-epsilon RDP vertex count** (`_run_type`, `eps=2.0`) — RDP-simplify
    each run at a small *fixed* tolerance and count vertices: a stair-stepped
    edge collapses to ~2, a single polygon **corner** keeps ~3 (two straight
    legs meeting sharply), and a smooth **curve** needs ≥4. The fixed epsilon is
    what tells a sharp corner apart from smooth curvature — both deviate from
    the chord, but only a curve needs many vertices. Runs with ≥`curve_vertices`
    (default 4) vertices are CURVE.

`boundary_eval` is attached to **every** node by the `classify_mask` wrapper
(covering early-return unknown/background cases too), never left null.

Verified — synthetic: line→straight, stair(s=1/s=2)→straight, triangle→straight
(corners, not curves), circle/small-circle→curve. Real images: rects & triangles
& polygonal stars → straight; circles/ellipses/rounded_rect → curve. No `ladder`
type exists — it was removed from `PRIMITIVES` and the reconstructor map (the
"comb/ladder primitive" was a misroute; the real requirement was staircase
straight-vs-curve, now handled above).

### Corner-driven geometry refinement (`learning/corners.py`)
Polygonal shapes are rebuilt from their **validated contour corners** (the
turning-angle curvature maxima = the human-confirmed ideal line-to-line
endpoints) rather than the radius/IoU primitive fit, which can be a few px off
and misses the true tip. `refine_shapes_from_corners(shapes, masks)` runs BEFORE
the reconstructor and overwrites `params.points`, stamping `corner_refined:true`.

  * **Fixed-vertex primitives** (`_FIXED_VERTEX_SHAPES`, e.g. `triangle`→3):
    rebuilt from `strongest_corners(ring, k)` — the k sharpest corners in
    boundary order, discarding false edge corners from staircase jitter. Fixed a
    previously broken triangle fit (two near-coincident corners) and lands the
    triangle exactly on its 3 real vertices.
  * **Custom polygons** (traced silhouettes, no fixed vertex count):
    `corners_by_threshold` keeps **every** contour corner above a turn threshold,
    in boundary order. An equal tip/valley (star) split is deliberately NOT
    enforced — real stars can be asymmetric, so balancing tips against valleys
    would discard a genuine corner (it dropped the red star's (292.5,272)). The
    red star rebuilds from all 7 of its corners. `star_corners` remains as an
    independent diagnostic corner source but no longer drives the rebuild.
  * **IoU-guarded span selection** (`_CUSTOM_REFINE_IOU_TOL=0.01`): the corner
    detector's non-maximum-suppression `span` (default 6) can swallow genuine
    corners on fine-featured silhouettes. On `ladder_shape` (a 10-tooth comb with
    5px-wide slots) span=6 suppressed the wall-top corners at each notch,
    collapsing the rectangular U-slots into V-points and dropping the reconstruction
    to IoU 0.78. The refinement now tries spans `(6,4,3,2)`, rasterizes each
    candidate against the mask, and keeps the corner set with the best IoU (within
    tol of the traced baseline). Results: ladder 0.78→0.89 (U-slots recovered,
    77 verts), drawn_shapes custom poly 0.73→0.90, clipped_star/transform_shapes
    unchanged (their default span was already optimal). This also disproved the
    initial "convergence points biased toward white" hypothesis — the notch tips
    were perfectly even (12px); the loss was span NMS, not collision-point bias
    or merge tolerance.

### Scale validation (`1024-test.png`, 1025×1025) + two robustness fixes
Ran the full pipeline on a 1025px, 3-colour image (blue/green shapes + orange
ellipse on white) to confirm the foundation scales. It does (~78s): blue polygon
IoU 0.981, green polygon 0.979. Two defects surfaced and were fixed:

  * **IoU-guarded rotation transform** (`build_transform`, `mask_classifier.py`):
    `estimate_orientation`'s min-area-rect brute search reported a 42° angle for
    the orange ellipse, which is actually **upright** (PCA major axis = 90°).
    Blindly applying it rotated a correct ellipse and dropped it to IoU 0.666.
    `build_transform` now rasterizes the rotated vs untransformed node against the
    mask and keeps the rotation only if it *improves* fit (`> base + 1e-3`).
    Orange 0.666→0.907; transform_shapes' genuinely-rotated teal ellipse still
    rotates (48°, IoU 0.957) — the guard rejects only harmful rotations.
  * **Fringe-island area floor** (`learn_image`, `min_component_area`): a 29px
    anti-alias sliver in the seam between two shapes became a spurious 0.5px
    "ellipse". Components below `max(32, w*h*5e-5)` are now dropped before
    classification. Small real shapes are far above the floor (30k–66k px here).

### Two-circle boolean recovery: lens + lunes (`learning/circle_boolean.py`)
`1024-test.png` is two overlapping circles (r≈197, centres (424,490)/(600,490)).
Colour quantization splits it into three regions the per-region classifier fits
in isolation and gets **wrong** in two ways the user flagged: (a) the shared
centre was fit as an **ellipse (IoU 0.907)** though it tapers to sharp points —
it is a **lens** (circle∩circle); (b) each outer crescent was a **faceted
polygon** (23/22 verts) though its boundary is two circular **arcs** — a **lune**
(circle − lens). `recover_two_circle_booleans` fixes both:

  * **Detect** an `ellipse` node whose mask actually tapers to near-points at both
    ends of its long axis (`_tapers_to_points`, < 12% of cross-span) — the lens
    signature that separates it from a real rounded ellipse.
  * **Find** its two neighbouring regions (dilated-halo overlap ≥ 15%). Each
    lune ∪ lens should reconstitute a full disk.
  * **Fit** a circle to each `lune ∪ lens` via an IoU-maximizing coarse→fine grid
    search seeded from the bbox (Kasa/algebraic edge fit was biased by the
    straight lens seam → r 172 vs true 197; grid fit nails r=197, IoU ≥ 0.994).
  * **Confirm** the two circles' intersection reproduces the lens (IoU ≥ 0.95),
    then rewrite all three nodes: lens → `type='lens'` (`params.circles`=[c0,c1]),
    each crescent → `type='lune'` (`params.circle`, `params.cut`).
  * **Arc reconstruction** (`lens_d`, `lune_d`): crossings ordered deterministically
    by the sign of `(cB-cA)×(P-cA)`; lens = two MINOR arcs (large=0) meeting at
    the tips; lune = own-circle MAJOR arc (large=1,sweep=1) + inner cut MINOR arc
    (large=0,sweep=0). Flags verified by brute-rasterizing all 16 combos vs mask.
  * **Rasterization** (`raster_renderer`): `lens`→`c0 & c1`, `lune`→`c & ~cut`
    (true boolean, not the old naive union), so IoU/overlap metrics are correct.

  Results on 1024-test: lens 0.907→**0.989** (SVG-`d` render), lunes facets→
  **0.990/0.988** smooth arcs. Fires ONLY here — the genuine (non-tapering)
  ellipses in clean_shapes/drawn_shapes/transform_shapes are untouched.

### 12-color stress test + simplicity-preference guards (`tests/color12.png`)
A 360×260 image with 12 mutually-intersecting shapes + background (13 colours)
validates the pipeline on a busier scene: **9.8 s, mean foreground IoU 0.950**,
all types correct. It surfaced two "fits a curve/rounded primitive that a simpler
one explains better" flaws, both fixed by margin-gated simplicity preferences in
`mask_classifier.py`:

  * **Polygon-over-curve** (`_fit_corner_polygon`): a regular N-gon can clear the
    circle/ellipse IoU threshold, so after a circle/ellipse is accepted we extract
    corners (`corners_by_threshold(span=6)`) and, if there is a small stable count
    (3–10) whose rasterised polygon beats the curve by ≥ `_POLY_OVER_CURVE_GAIN`
    (0.03), return a polygon instead. A genuine curve yields 0–2 real corners or
    many spurious ones that rasterise near-0, so the guard never fires on real
    circles/ellipses. Lime hexagon **ellipse 0.804 → polygon 0.979**.
  * **Rect-over-rounded_rect** (`_ROUNDED_OVER_RECT_GAIN`=0.02): a sharp rect whose
    edges are nibbled by AA/occlusion can edge out a plain rect by <0.004, spuriously
    reading as rounded. rounded_rect is now kept only if it beats a re-fit plain rect
    by ≥ 0.02. Navy thin rect **rounded_rect 0.830 → rect 0.827** (correct type; IoU
    capped by occlusion). Genuine rounded_rect in drawn_shapes is preserved.

Both guards verified regression-free across clean_shapes / drawn_shapes /
transform_shapes / 1024-test / 1024-test-02 (all primitives, stars, unions, lens,
lunes untouched).

### Occluded circles = smooth arcs, not facets (`learning/occluded_circle.py`)
A circle partly hidden by other shapes reads as an irregular region: its visible
boundary is a circular **arc**, the hidden side a straight **chord** where the
occluder cut across. Its bbox-circle IoU is too low to pass as a primitive, so it
was falling back to a **faceted polygon** — turning a smooth arc into a staircase.
`fit_occluded_circle` recovers the underlying circle and reconstructs the region
with true arcs:

  * **Robust fit** — a bbox-constrained **RANSAC** circle fit (least-squares/Kasa
    is wrecked by the straight chords: green went 9% on-arc). RANSAC picks the
    circle with the most boundary inliers, radius/centre pinned near the bbox to
    reject the giant near-collinear circles a locally straight staircase edge would
    otherwise produce, then refines on the inlier set.
  * **Gate (does NOT fire on ellipses/blobs)** — inlier fraction ≥ 0.45, inlier
    **angular coverage** ≥ 0.55 (an ellipse matches a circle only near two
    vertices → low coverage), mask ⊆ fitted disk (containment ≥ 0.90) and disk IoU
    ≥ 0.45 (rejects an ellipse inscribed in a large circle).
  * **Arc reconstruction** (`occluded_circle_d`) — walk the ring; runs of on-circle
    vertices coalesce into a single SVG `A` arc (large-arc split at a semicircle),
    off-circle chord runs are RDP-simplified to a few straight `L` segments. Red
    circle went 348 arclets+138 lines → **3 arcs + 7 lines**, IoU 0.974.
  * Runs in the classifier fallback AFTER rotated-primitive and BEFORE the polygon
    fallback; dispatched by `reconstruct` / `rasterize_node` (polygon-fill footprint
    for metrics). On `color12`: red **0.975**, green **0.950**, cyan **0.928** — all
    smooth. Magenta ellipse stays `ellipse` (0.972), yellow stays `polygon` (0.935,
    not a clean circle), lime hexagon stays `polygon` (0.979). No regression on
    clean_shapes/drawn_shapes circles or 1024-test/-02 lens+lunes.

### Collision swarms → single convergence points (`collision_detector.py`)
Near a near-tangent seam (e.g. a lens tip) many pixels along the boundary each see
all converging colours, producing a **swarm** of magenta collision crosses that is
really one junction; on a large image the swarm spans dozens of px, so a fixed 1px
merge left a smear. Two size-aware fixes:

  * **`min_region_frac`** (default 0.15) — a colour only *converges* if it fills a
    real share of the sampling window, not a 1px anti-alias transition sliver.
    Along a smooth curved edge quantization leaves a thin intermediate band that
    would otherwise inflate the colour count all along the arc (spurious corners);
    the fraction gate drops it.
  * **Size-scaled merge** — `learn_image` sets `collision_merge_dist` to ~3% of the
    image diagonal when the caller leaves it default (still a 1px floor), so a
    tip-swarm collapses to a single dot on large images while small images stay
    tight. 1024-test tips went from a ~8-point swarm to a single **n=7** dot each;
    small scenes keep 4–7 sensible points.

### Boundary-frame alignment (parametric vs traced) & flawless-primitive snap gate
Two independent causes of a ~1px gap between a rendered parametric primitive and
a neighbouring traced shape (e.g. the yellow circle vs the small clipped red
fragment on the tiny `drawn_shapes` source):

  1. **Coordinate-frame mismatch.** `mask_extent` returns a cell box `(min,
     max+1)` whose centre sits **+0.5px** off the boundary the contour tracer
     (`find_contours` at the 0.5 iso-level) produces. Traced polygons/corners/
     clip edges live in that boundary frame, so a parametric circle built from
     the cell box renders half a pixel shifted. Fix (`mask_classifier._BOUNDARY_ADJ
     = 0.5`): keep classification/detection in pixel-index space (star/triangle
     radial math is sensitive to the centre — shifting it globally flipped a real
     star to a polygon), and subtract 0.5 **only from the emitted parametric
     params** (circle/ellipse centre, rect/rounded_rect origin; width/height/
     radius unchanged). The classifier now emits the yellow circle at exactly
     (280.0,250.0) = the true boundary centre.
  2. **Snap re-shifting.** `snap_shapes_to_collisions` was then dragging the
     circle centre to a 3-colour junction, reopening the gap. Gate added
     (`snap._HIGH_FIT_IOU = 0.95`): a *centre-moving* parametric (ellipse/rect/
     rounded_rect) whose `fit_iou` is near-perfect is **exempt** from collision
     snapping — keep its aligned centre and let neighbours meet it. Circles are
     NOT exempt because they now snap locally (see below).
  * Not yet aligned: the rotated-primitive path emits its centre in a de-rotated
    frame, so `_BOUNDARY_ADJ` isn't applied there (rare; left as a follow-up).

### Circles as closed 4-node cubic splines (locally snappable)
Circles are emitted as a **closed 4-node cubic Bézier** (`primitive_reconstructor
.circle_spline_d`, default `as_spline=True`) instead of the SVG `a` arc: 4 anchors
at E/S/W/N, each quarter one cubic with tangential handles of length `kappa*r`
(`_KAPPA=0.5523`). With zero offsets it renders **identically** to the arc (raster
IoU 1.0). The point is editability: `params['anchor_offsets']=[dE,dS,dW,dN]` moves
each anchor **radially** and independently, its handles rescaling to stay smooth.

### Standalone circle = primitive; overlapping circle = node-spline on junctions
A **standalone** circle stays this clean parametric primitive. But once a circle
**overlaps** another shape it can no longer be a primitive: its rim must pass
*exactly* through every shared convergence point, and those can sit **anywhere**
on the rim — not just on the four E/S/W/N axes. So `snap.py._snap_circle_nodes`
rebuilds an overlapping circle as an **explicit node-spline** (`primitive_
reconstructor.circle_nodes_spline_d`, `params['spline_nodes']`): the node ring =
the four cardinal anchors **plus one node placed exactly on each converging
target**, sorted by angle. Each segment uses circular tangents with `kappa`
scaled by its actual arc span (`4/3·tan(Δθ/4)·r`), so the curve stays round
between nodes and only **bulges locally** to touch each junction (centre & radius
otherwise preserved). A cardinal that nearly coincides with a target (≤8°) is
dropped so there are no degenerate segments. `circle_d` routes to the node-spline
whenever `spline_nodes` is present, else the 4-anchor spline, else the arc.

This is the general rule: **curves involved in an overlap must be node-splines
with a node on every tandem convergence point, never bare primitives.** Verified
on `drawn_shapes`: the standalone green circle stays parametric; the yellow
circle overlapping the small red triangle island gains an extra SE node at
(293.75, 272.33) and its path now visibly touches the island, while its other
four nodes remain exactly on radius (dev 0.0px).

### Snap reconstructed nodes onto collision points (`learning/snap.py`)
The ideal reconstruction (circle/ellipse/rect/polygon …) often misses the
quantized junctions by several pixels, so the markers would float off the
vectors. After collisions are found, `snap_shapes_to_collisions` deforms every
converging shape so its boundary passes through the collision point:
- polygon / star / triangle / bezier: each target is snapped onto the ring —
  a near vertex is moved onto it, otherwise the nearest edge is split by
  inserting the target itself (boundary is pulled onto the point). A `claimed`
  set stops later targets from stealing an already-snapped vertex.
- circle / ellipse / rect / rounded_rect: only snapped when a target is already
  within ~8px of the ideal boundary, so clean primitives are never dragged
  across the image to a distant quantized corner (which would wreck their form).
Collision points are grouped per shape and snapped in one stable angular-ordered
pass. **Only 3+ color convergences snap shapes** (true structural corners). A
2-color collision is just where two shapes touch — snapping both to every point
along such an edge drags a straight edge onto the neighbour's curve (e.g. the red
star's straight side got bent to follow the yellow circle), so 2-color points are
left as contact markers, not deformation targets. Gating also uses the shape's
**actual boundary distance** (tolerance ~8px) rather than its bbox, so a small
island sharing a color/footprint with a bigger neighbour is NOT dragged onto
collisions belonging to that neighbour (which would decimate it). Result: 3-color
junctions are hit exactly (dist 0.0); the small red island keeps its shape; ideal
primitives stay clean; shared straight edges are preserved.
merely touching the backdrop is not mistaken for a junction.

Verified across all three test images (nc=8):
- `clean_shapes.png` (7 shapes): rect, circle, ellipse + 4 stars; the (230,130,30)
  star is a 5-point star at IoU 0.56, the others 0.91–0.99.
- `drawn_shapes.png`: rounded_rect, circle, ellipse, triangle (IoU 0.75),
  circle + 2 yellow-fill `difference` composites; the red drawn "star" is correctly
  traced as a 10-vertex `polygon` (IoU 0.82) rather than force-fit to a 5-point star.
- `transform_shapes.png`: rotated ellipse as `ellipse` + `rotate`, overlapping
  clipped circles correctly fall through to `polygon` (IoU 0.91) — NOT mis-detected
  as stars.

Star robustness: `_detect_star` requires ≥5 significant radial maxima, near-even
tip/valley balance, and strict alternation so a clipped circle (single
overlap-notch) is rejected; it also re-rasterises the candidate star against the
*actual* mask and rejects it unless IoU ≥ 0.80, so an arm-clipped star (cut by an
overlapping yellow circle) is no longer mis-detected — it falls through to custom
polygon reconstruction. `classify_mask` keeps the better of two vertex extractors by
re-rasterised IoU.

- `clipped_star.png`: a 5-point star whose arm is cut through by a yellow circle is
  correctly classified as `polygon` (`custom: True`, 11 vertices, re-con IoU 0.78)
  rather than a mis-fit `star`; a separate clean star stays `star`.

## How to run

### CLI (learner)
```bash
pip install pillow numpy scikit-image scipy
python -m learning.learner tests/clean_shapes.png 8      # -> clean_shapes_learned.json + .svg
```

### API + Frontend
```bash
pip install fastapi uvicorn
uvicorn api:app --reload --port 8000
# in another shell:
cd frontend && npm install && npm run dev
# frontend proxies /api -> :8000
```
`POST /learn` (multipart: image, num_colors) → JSON `{ model, svg, summary }`.

## Known limitations (POC)
- Tracing uses marching-squares (staircase edges) with spline smoothing — not true
  Potrace Bézier curves. (Potrace was evaluated and abandoned: the `potracer`
  pure-Python port is buggy and `pypotrace` can't build on this Windows/Py3.13 box.)
- Learning is fit-based (IoU threshold 0.85): very rotated *rectangles* with jagged
  quantized edges may fall back to `polygon` rather than `rect + rotate`. Ellipses and
  circles rotate cleanly. No `intersection`/`xor` composites yet (only `union`/`difference`
  are inferred; the node types exist for manual/forward use).
- No BiRefNet foreground matting yet (see next steps).

## Next steps
- [ ] BiRefNet matting before quantization (foreground-only tracing).
- [ ] Intersection / XOR composite inference from overlap geometry.
- [ ] Real SVGO server-side pass (svgwrite / svgo node).
- [ ] OKLab-based quantization for better perceptual colors.
- [ ] Export per-layer as separate editable groups / Figma import.

