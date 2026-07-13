"""Convert a downloaded Kaggle 'Fruits fresh and rotten for classification'
dataset into the binary fresh/rotten layout this project expects.

1. Download + unzip from:
     https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification
   You'll get folders like: dataset/train/freshapples, dataset/train/rottenbanana, ...

2. Run:
     python prepare_real_data.py --src /path/to/dataset --dst data

It merges every fresh* class -> fresh, every rotten* class -> rotten, and
splits the source images into train/val/test (default 70/15/15). The test split
is held out and read only by evaluate.py, never during training.
"""

import argparse
import hashlib
import os
import random
import shutil


def _bucket(class_name):
    name = class_name.lower()
    if name.startswith("fresh"):
        return "fresh"
    if name.startswith("rotten"):
        return "rotten"
    return None


def _content_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="unzipped Kaggle dataset root")
    ap.add_argument("--dst", default="data")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    args = ap.parse_args()

    rng = random.Random(0)
    src_train = os.path.join(args.src, "train")
    if not os.path.isdir(src_train):
        src_train = args.src  # allow pointing directly at the class folders

    count = 0
    for class_name in os.listdir(src_train):
        class_dir = os.path.join(src_train, class_name)
        if not os.path.isdir(class_dir):
            continue
        bucket = _bucket(class_name)
        if bucket is None:
            print(f"  skipping unrecognized class: {class_name}")
            continue

        # De-duplicate by content hash so identical images can't leak across splits.
        seen, files = set(), []
        for f in os.listdir(class_dir):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            h = _content_hash(os.path.join(class_dir, f))
            if h not in seen:
                seen.add(h)
                files.append(f)
        rng.shuffle(files)
        n_test = int(len(files) * args.test_frac)
        n_val = int(len(files) * args.val_frac)

        for i, fname in enumerate(files):
            if i < n_test:
                split = "test"
            elif i < n_test + n_val:
                split = "val"
            else:
                split = "train"
            out_dir = os.path.join(args.dst, split, bucket)
            os.makedirs(out_dir, exist_ok=True)
            shutil.copy(os.path.join(class_dir, fname),
                        os.path.join(out_dir, f"{class_name}_{fname}"))
            count += 1

    print(f"Copied {count} images into {args.dst}/(train|val|test)/(fresh|rotten)")


if __name__ == "__main__":
    main()
