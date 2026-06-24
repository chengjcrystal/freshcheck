"""Download the real banana freshness dataset from Hugging Face
(nikibout/fresh-and-rotten-fruit, ~120 MB, no auth) and lay it out as
data/{train,val}/{fresh,rotten}.

The source uses classes `freshbanana`/`rottenbanana` split across Train/Test.
We pool ALL images, map fresh*->fresh and rotten*->rotten, then do a seeded,
stratified 85/15 train/val split so each split is class-balanced.

    python prepare_hf_banana.py
"""

import argparse
import os
import random
import shutil

from huggingface_hub import snapshot_download

REPO_ID = "nikibout/fresh-and-rotten-fruit"


def _bucket(path):
    p = path.lower()
    if "freshbanana" in p:
        return "fresh"
    if "rottenbanana" in p:
        return "rotten"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default="data")
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    print(f"Downloading {REPO_ID} from Hugging Face ...")
    src = snapshot_download(repo_id=REPO_ID, repo_type="dataset",
                            allow_patterns=["*.png", "*.jpg", "*.jpeg"])
    print(f"Downloaded to {src}")

    # Collect images per bucket from the whole repo (Train + Test pooled).
    buckets = {"fresh": [], "rotten": []}
    for root, _dirs, files in os.walk(src):
        for fname in files:
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            b = _bucket(os.path.join(root, fname))
            if b:
                buckets[b].append(os.path.join(root, fname))

    # Wipe any previous data/ contents (e.g. synthetic) so splits stay clean.
    for split in ("train", "val"):
        for cls in ("fresh", "rotten"):
            d = os.path.join(args.dst, split, cls)
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

    rng = random.Random(0)
    totals = {}
    for cls, paths in buckets.items():
        rng.shuffle(paths)
        n_val = int(len(paths) * args.val_frac)
        for i, src_path in enumerate(paths):
            split = "val" if i < n_val else "train"
            ext = os.path.splitext(src_path)[1].lower()
            dst_path = os.path.join(args.dst, split, cls, f"{cls}_{i:04d}{ext}")
            shutil.copy(src_path, dst_path)
        totals[cls] = (len(paths) - n_val, n_val)

    print("\nDone. Real banana data laid out:")
    for cls, (tr, va) in totals.items():
        print(f"  {cls:7s} train={tr:4d}  val={va:3d}")


if __name__ == "__main__":
    main()
