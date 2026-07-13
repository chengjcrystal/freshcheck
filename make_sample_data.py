"""Generate a small SYNTHETIC fresh/rotten dataset so the full pipeline runs
end-to-end without any download. Fresh fruit = smooth bright color; rotten =
same fruit with dark-brown decay blotches.

This is only for verifying the pipeline and demoing the app. For real accuracy,
replace data/ with actual fruit photos (see README -> "Use real data").

    python make_sample_data.py            # 300 train + 60 val + 60 test per class
"""

import argparse
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

IMG = 96
FRUIT_COLORS = [(210, 50, 40), (240, 200, 40), (90, 170, 60), (245, 150, 40)]  # apple/banana/lime/orange
DECAY = (70, 45, 25)


def _draw_fruit(rotten, rng):
    img = Image.new("RGB", (IMG, IMG), (235, 235, 235))
    d = ImageDraw.Draw(img)
    color = rng.choice(FRUIT_COLORS)
    # slight per-image color variation
    color = tuple(int(np.clip(c + rng.randint(-20, 20), 0, 255)) for c in color)
    cx, cy = IMG // 2 + rng.randint(-6, 6), IMG // 2 + rng.randint(-6, 6)
    r = rng.randint(30, 38)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

    if rotten:
        for _ in range(rng.randint(4, 8)):
            bx = cx + rng.randint(-r + 5, r - 5)
            by = cy + rng.randint(-r + 5, r - 5)
            br = rng.randint(4, 11)
            blot = tuple(int(np.clip(c + rng.randint(-15, 15), 0, 255)) for c in DECAY)
            d.ellipse([bx - br, by - br, bx + br, by + br], fill=blot)

    return img.filter(ImageFilter.GaussianBlur(0.6))


def _make_split(root, split, n, rng):
    for label, rotten in [("fresh", False), ("rotten", True)]:
        out = os.path.join(root, split, label)
        os.makedirs(out, exist_ok=True)
        for i in range(n):
            _draw_fruit(rotten, rng).save(os.path.join(out, f"{label}_{i:04d}.jpg"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--train", type=int, default=300)
    ap.add_argument("--val", type=int, default=60)
    ap.add_argument("--test", type=int, default=60)
    args = ap.parse_args()

    rng = random.Random(0)
    np.random.seed(0)
    _make_split(args.root, "train", args.train, rng)
    _make_split(args.root, "val", args.val, rng)
    _make_split(args.root, "test", args.test, rng)
    print(f"Wrote synthetic data to {args.root}/ "
          f"({args.train} train + {args.val} val + {args.test} test per class)")


if __name__ == "__main__":
    main()
