"""Export the trained TinyFreshNet to ONNX for in-browser inference.

Produces web/model.onnx (run by onnxruntime-web in the SPA) and web/meta.json
(static model facts the SPA reads instead of the /api/meta endpoint).

    python export_onnx.py
"""

import glob
import json
import os

import onnx
import torch

from model import TinyFreshNet, count_params
from inference import CKPT_PATH

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
ONNX_PATH = os.path.join(WEB, "model.onnx")
META_PATH = os.path.join(WEB, "meta.json")
METRICS_PATH = "metrics.json"


def main():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    classes = ckpt["classes"]
    img_size = ckpt.get("img_size", 64)

    model = TinyFreshNet(num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dummy = torch.zeros(1, 3, img_size, img_size)
    torch.onnx.export(
        model, dummy, ONNX_PATH,
        input_names=["input"], output_names=["logits"],
        opset_version=18,
    )
    # The exporter may write weights as an external .data file; onnxruntime-web
    # needs a single self-contained model, so inline all tensors and clean up.
    model_proto = onnx.load(ONNX_PATH)  # pulls external data into memory
    onnx.save_model(model_proto, ONNX_PATH, save_as_external_data=False)
    for extra in glob.glob(os.path.join(WEB, "*.onnx.data")) + glob.glob(os.path.join(WEB, "*.onnx_data")):
        os.remove(extra)
    print(f"Wrote {ONNX_PATH} ({os.path.getsize(ONNX_PATH)} bytes, self-contained)")

    # Build the static meta the SPA reads (mirrors /api/meta).
    meta = json.load(open(METRICS_PATH)) if os.path.exists(METRICS_PATH) else {}
    meta["params"] = count_params(model)
    meta["classes"] = classes
    meta["img_size"] = img_size
    meta["inference_device"] = "browser (wasm)"
    meta["normalize"] = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

    examples = []
    ex_dir = os.path.join(WEB, "examples")
    if os.path.isdir(ex_dir):
        examples = [f"examples/{n}" for n in sorted(os.listdir(ex_dir))
                    if n.lower().endswith((".png", ".jpg", ".jpeg"))]
    meta["examples"] = examples

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {META_PATH}")


if __name__ == "__main__":
    main()
