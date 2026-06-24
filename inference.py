"""Shared inference: load the trained model and classify a single PIL image."""

import torch
from PIL import Image

from model import TinyFreshNet
from data_utils import eval_transform

CKPT_PATH = "checkpoints/freshnet.pt"


class FreshnessClassifier:
    def __init__(self, ckpt_path=CKPT_PATH, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.classes = ckpt["classes"]
        self.img_size = ckpt.get("img_size", 64)

        self.model = TinyFreshNet(num_classes=len(self.classes)).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.transform = eval_transform(self.img_size)

    @torch.no_grad()
    def predict(self, image: Image.Image):
        """Return (verdict_str, {class_name: probability}) for a PIL image."""
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1).squeeze(0)
        scores = {cls: float(probs[i]) for i, cls in enumerate(self.classes)}

        top = max(scores, key=scores.get)
        verdict = self._verdict(top, scores[top])
        return verdict, scores

    @staticmethod
    def _verdict(label, confidence):
        pct = f"{confidence * 100:.1f}%"
        if "rotten" in label.lower() or "bad" in label.lower():
            return f"🤢 Gone bad — looks ROTTEN ({pct} confident)"
        return f"✅ Looks FRESH ({pct} confident)"
