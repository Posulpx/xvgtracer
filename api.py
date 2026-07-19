"""FastAPI wrapper around XVGTracer core.

Run:  uvicorn api:app --reload
POST /trace  multipart form: image, num_colors, simplify
Returns the generated SVG as text/xml and saves output_colored.svg.
POST /learn  multipart form: image, num_colors
Returns a structured JSON model (primitive/composite/transform hierarchy) plus
an SVG rebuilt from the learned primitives.
"""
from __future__ import annotations

import json
import os
import tempfile

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from xvgtracer import process

from learning import learner

app = FastAPI(title="XVGTracer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/trace", response_class=PlainTextResponse)
async def trace(
    image: UploadFile = File(...),
    num_colors: int = Form(5),
    simplify: float = Form(1.0),
    angle_threshold: float = Form(30.0),
    smooth_sigma: float = Form(1.0),
):
    suffix = os.path.splitext(image.filename or "img.png")[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name
    try:
        svg = process(tmp_path, num_colors=num_colors, simplify=simplify,
                      angle_threshold=angle_threshold, smooth_sigma=smooth_sigma,
                      output="output_colored.svg")
    finally:
        os.remove(tmp_path)
    return PlainTextResponse(svg, media_type="image/svg+xml")


@app.post("/learn")
async def learn(
    image: UploadFile = File(...),
    num_colors: int = Form(8),
    merge_gap: int = Form(0),
    min_colors: int = Form(2),
):
    """Run the learning pipeline and return the structured model + SVG.

    The model is a hierarchy of nodes:
      Primitive : point | line | circle | ellipse | rect | rounded_rect |
                  triangle | polygon | bezier | arc
      Composite : union | difference | intersection | xor
      Transform : translate | rotate | scale | skew | mirror
    """
    suffix = os.path.splitext(image.filename or "img.png")[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name
    try:
        model = learner.learn_image(tmp_path, num_colors=num_colors,
                                     merge_gap=merge_gap,
                                     min_colors=min_colors)
    finally:
        os.remove(tmp_path)
    svg = learner.model_to_svg(model)
    payload = {
        "model": model,
        "svg": svg,
        "summary": {
            "shapes": len(model["shapes"]),
            "primitives": [s["type"] for s in model["shapes"]],
            "composites": [c["type"] for c in model.get("composites", [])],
            "collisions": len(model.get("collisions", [])),
        },
    }
    return JSONResponse(payload)
