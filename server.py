"""FreshCheck API + static host.

FastAPI backend that runs the trained TinyFreshNet model and serves the static
single-page app. One origin, no build step required for the frontend.

    python server.py            # serves http://127.0.0.1:8000
"""

import io
import json
import os
import time

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from inference import FreshnessClassifier, CKPT_PATH

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "web")
METRICS_PATH = os.path.join(HERE, "metrics.json")

app = FastAPI(title="FreshCheck API", version="1.0.0")

clf = FreshnessClassifier() if os.path.exists(CKPT_PATH) else None

# Warm up so the first real request reports a representative latency.
if clf is not None:
    with torch.no_grad():
        _dummy = torch.zeros(1, 3, clf.img_size, clf.img_size, device=clf.device)
        clf.model(_dummy)


def _load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}


# Copy a couple of held-out images into the static dir so the SPA can offer
# one-click example tests without exposing the whole dataset directory.
EXAMPLES_DIR = os.path.join(WEB_DIR, "examples")
EXAMPLE_URLS = []
if os.path.isdir(EXAMPLES_DIR):
    EXAMPLE_URLS = [f"/examples/{n}" for n in sorted(os.listdir(EXAMPLES_DIR))
                    if n.lower().endswith((".png", ".jpg", ".jpeg"))]


@app.get("/api/examples")
def examples():
    return EXAMPLE_URLS


@app.get("/api/meta")
def meta():
    m = _load_metrics()
    m["model_loaded"] = clf is not None
    m["inference_device"] = clf.device if clf else None
    return m


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if clf is None:
        raise HTTPException(503, "Model not loaded. Run train.py then evaluate.py.")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image.")

    width, height = image.size

    t0 = time.perf_counter()
    _verdict, scores = clf.predict(image)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    probabilities = sorted(
        ({"label": k, "prob": v} for k, v in scores.items()),
        key=lambda d: d["prob"], reverse=True,
    )
    top = probabilities[0]

    return JSONResponse({
        "predicted_class": top["label"],
        "confidence": top["prob"],
        "probabilities": probabilities,
        "latency_ms": round(latency_ms, 1),
        "device": clf.device,
        "image": {
            "filename": file.filename,
            "width": width,
            "height": height,
            "bytes": len(raw),
            "format": (file.content_type or "image").split("/")[-1],
        },
    })


# Static SPA last, so /api/* routes take precedence.
if os.path.isdir(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
