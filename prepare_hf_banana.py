"""Download the real banana freshness dataset from Hugging Face
(nikibout/fresh-and-rotten-fruit, ~120 MB, no auth) and lay it out as
data/{train,val,test}/{fresh,rotten} with a leak-free split.

Why this is not a naive split:

The source ships `dataset2/{Train,Test}/{freshbanana,rottenbanana}`, but the
Train and Test folders overlap heavily. Every Train image is a byte-for-byte
duplicate of a Test image (verified via md5), so the raw folders hold only
~300 UNIQUE images (150 fresh + 150 rotten), not the 510 file entries. On top
of that, several images are near-duplicate frames of the same banana.

Pooling everything and splitting at the file level (the old behaviour) leaked
exact duplicates into both train and val, which is what produced the
suspicious "100% validation accuracy". Here we instead:

  1. Drop exact duplicates by content hash.
  2. Group near-duplicates (perceptual average-hash, Hamming <= 4) so frames of
     the same banana never straddle a split boundary.
  3. Do a seeded, stratified, group-aware 70/15/15 train/val/test split. The
     largest near-dup clusters are pinned into train, leaving val and test as
     diverse singletons for an honest generalization estimate.

The test split is written once and must never be used for training or
checkpoint selection.

    python prepare_hf_banana.py
"""

import argparse
import hashlib
import os
import shutil

from PIL import Image
from huggingface_hub import snapshot_download

REPO_ID = "nikibout/fresh-and-rotten-fruit"
IMG_EXTS = (".png", ".jpg", ".jpeg")


def _bucket(path):
    p = path.lower()
    if "freshbanana" in p:
        return "fresh"
    if "rottenbanana" in p:
        return "rotten"
    return None


def _content_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _ahash(path, size=8):
    """Perceptual average-hash: 64-bit fingerprint robust to small changes."""
    im = Image.open(path).convert("L").resize((size, size))
    px = list(im.getdata())
    avg = sum(px) / len(px)
    bits = 0
    for i, p in enumerate(px):
        if p > avg:
            bits |= (1 << i)
    return bits


def _hamming(a, b):
    return bin(a ^ b).count("1")


def _cluster_near_dups(files, threshold=4):
    """Union-find clustering of files whose average-hashes are within
    `threshold` bits. Returns a list of clusters (each a list of paths)."""
    hashes = [(f, _ahash(f)) for f in files]
    parent = list(range(len(hashes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if _hamming(hashes[i][1], hashes[j][1]) <= threshold:
                parent[find(i)] = find(j)

    groups = {}
    for i, (f, _) in enumerate(hashes):
        groups.setdefault(find(i), []).append(f)
    return list(groups.values())


def _assign_splits(clusters, rng, val_frac, test_frac):
    """Assign whole clusters to train/val/test for one class.

    Largest clusters go to train first (so val/test stay diverse singletons),
    then remaining clusters fill the val and test image quotas.
    """
    total = sum(len(c) for c in clusters)
    n_val = round(total * val_frac)
    n_test = round(total * test_frac)

    clusters = sorted(clusters, key=len, reverse=True)
    # Pin the biggest cluster(s) to train up front.
    big = [c for c in clusters if len(c) > 1]
    small = [c for c in clusters if len(c) == 1]
    rng.shuffle(small)

    splits = {"train": [], "val": [], "test": []}
    for c in big:
        splits["train"].extend(c)

    # Fill test then val from singletons, remainder to train.
    for c in small:
        if len(splits["test"]) < n_test:
            splits["test"].extend(c)
        elif len(splits["val"]) < n_val:
            splits["val"].extend(c)
        else:
            splits["train"].extend(c)
    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default="data")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Downloading {REPO_ID} from Hugging Face ...")
    src = snapshot_download(repo_id=REPO_ID, repo_type="dataset",
                            allow_patterns=[f"*{e}" for e in IMG_EXTS])
    print(f"Downloaded to {src}")

    # Collect images per class, dropping exact duplicates by content hash.
    buckets = {"fresh": [], "rotten": []}
    seen = set()
    dup_count = 0
    for root, _dirs, files in os.walk(src):
        for fname in sorted(files):
            if not fname.lower().endswith(IMG_EXTS):
                continue
            path = os.path.join(root, fname)
            b = _bucket(path)
            if not b:
                continue
            h = _content_hash(path)
            if h in seen:
                dup_count += 1
                continue
            seen.add(h)
            buckets[b].append(path)
    print(f"Dropped {dup_count} exact-duplicate files. Unique images: "
          f"fresh={len(buckets['fresh'])}, rotten={len(buckets['rotten'])}")

    # Wipe any previous data/ contents so splits stay clean.
    for split in ("train", "val", "test"):
        for cls in ("fresh", "rotten"):
            d = os.path.join(args.dst, split, cls)
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

    import random
    rng = random.Random(args.seed)
    totals = {}
    for cls, paths in buckets.items():
        clusters = _cluster_near_dups(paths)
        splits = _assign_splits(clusters, rng, args.val_frac, args.test_frac)
        for split, items in splits.items():
            for i, src_path in enumerate(sorted(items)):
                ext = os.path.splitext(src_path)[1].lower()
                dst_path = os.path.join(args.dst, split, cls, f"{cls}_{i:04d}{ext}")
                shutil.copy(src_path, dst_path)
        totals[cls] = {s: len(v) for s, v in splits.items()}

    print("\nDone. Leak-free banana data laid out:")
    print(f"  {'class':8s} {'train':>6s} {'val':>5s} {'test':>5s}")
    for cls, t in totals.items():
        print(f"  {cls:8s} {t['train']:6d} {t['val']:5d} {t['test']:5d}")
    print("\nTest split is held out: never used for training or checkpoint selection.")


if __name__ == "__main__":
    main()
