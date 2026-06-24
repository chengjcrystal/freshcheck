"""Compute honest train/val accuracy + model/dataset facts and write metrics.json.

These numbers are surfaced in the web UI's Model Insights dashboard, so they are
measured directly from the trained checkpoint — nothing is hard-coded.

    python evaluate.py
"""

import json
import os

import torch
from torchvision import datasets

from model import TinyFreshNet, count_params
from data_utils import eval_transform
from inference import CKPT_PATH

METRICS_PATH = "metrics.json"


@torch.no_grad()
def _accuracy(model, ds, device):
    correct = 0
    loader = torch.utils.data.DataLoader(ds, batch_size=64)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
    return correct / len(ds) if len(ds) else 0.0


def main():
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(CKPT_PATH, map_location=device)
    classes, img_size = ckpt["classes"], ckpt.get("img_size", 64)

    model = TinyFreshNet(num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tf = eval_transform(img_size)
    train_ds = datasets.ImageFolder("data/train", transform=tf)
    val_ds = datasets.ImageFolder("data/val", transform=tf)

    metrics = {
        "architecture": "TinyFreshNet (3 conv blocks + global average pool + linear head)",
        "params": count_params(model),
        "img_size": img_size,
        "classes": classes,
        "device": device,
        "dataset": {
            "train_images": len(train_ds),
            "val_images": len(val_ds),
            "source": "nikibout/fresh-and-rotten-fruit (Hugging Face)",
        },
        "accuracy": {
            "train": round(_accuracy(model, train_ds, device), 4),
            "val": round(_accuracy(model, val_ds, device), 4),
        },
        "layers": [
            "Conv(3→16) + BN + ReLU + MaxPool",
            "Conv(16→32) + BN + ReLU + MaxPool",
            "Conv(32→64) + BN + ReLU + MaxPool",
            "AdaptiveAvgPool → Dropout(0.3) → Linear(64→2)",
        ],
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"\nWrote {METRICS_PATH}")


if __name__ == "__main__":
    main()
