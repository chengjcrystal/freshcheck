"""Compute honest train/val/test accuracy + model/dataset facts, write metrics.json.

These numbers are surfaced in the web UI's Model Insights dashboard, so they are
measured directly from the trained checkpoint, nothing is hard-coded.

The headline number is TEST accuracy: the test split is held out by
prepare_hf_banana.py and is never seen during training or checkpoint selection,
so it reflects generalization rather than memorization. Train and val are also
reported for context (val is the checkpoint-selection set, so it reads high).

    python evaluate.py
"""

import json
import math

import torch
from torchvision import datasets

from model import TinyFreshNet, count_params
from data_utils import eval_transform
from inference import CKPT_PATH

METRICS_PATH = "metrics.json"


@torch.no_grad()
def _correct_total(model, ds, device):
    correct = 0
    loader = torch.utils.data.DataLoader(ds, batch_size=64)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
    return correct, len(ds)


def _accuracy(model, ds, device):
    correct, total = _correct_total(model, ds, device)
    return correct / total if total else 0.0


def _wilson_ci(correct, total, z=1.96):
    """Wilson score 95% confidence interval for a binomial proportion.

    Better than the normal approximation for small n like our 44-image test set.
    Returns (low, high) as fractions.
    """
    if total == 0:
        return 0.0, 0.0
    p = correct / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return max(0.0, center - margin), min(1.0, center + margin)


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
    test_ds = datasets.ImageFolder("data/test", transform=tf)

    test_correct, test_total = _correct_total(model, test_ds, device)
    ci_low, ci_high = _wilson_ci(test_correct, test_total)

    metrics = {
        "architecture": "TinyFreshNet (3 conv blocks + global average pool + linear head)",
        "params": count_params(model),
        "img_size": img_size,
        "classes": classes,
        "device": device,
        "dataset": {
            "train_images": len(train_ds),
            "val_images": len(val_ds),
            "test_images": len(test_ds),
            "source": "nikibout/fresh-and-rotten-fruit (Hugging Face)",
            "note": "300 unique banana images after de-duplication; "
                    "leak-free group-aware 70/15/15 train/val/test split.",
        },
        "accuracy": {
            "test": round(test_correct / test_total, 4) if test_total else 0.0,
            "test_correct": test_correct,
            "test_total": test_total,
            "test_ci95": [round(ci_low, 4), round(ci_high, 4)],
            "val": round(_accuracy(model, val_ds, device), 4),
            "train": round(_accuracy(model, train_ds, device), 4),
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
