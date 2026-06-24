"""Train TinyFreshNet on a fresh/rotten fruit dataset.

Expected layout:
    data/train/fresh/*.jpg
    data/train/rotten/*.jpg
    data/val/fresh/*.jpg
    data/val/rotten/*.jpg

Run:
    python train.py --data-dir data --epochs 15
"""

import argparse
import collections
import os

import torch
import torch.nn as nn

from model import TinyFreshNet, count_params
from data_utils import make_loaders, IMG_SIZE

CKPT_PATH = "checkpoints/freshnet.pt"


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    total, correct, loss_sum = 0, 0, 0.0

    torch.set_grad_enabled(train)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_sum += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)

    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, classes = make_loaders(
        args.data_dir, args.batch_size, args.img_size)
    print(f"Classes: {classes}")

    model = TinyFreshNet(num_classes=len(classes)).to(device)
    print(f"Trainable parameters: {count_params(model):,}")

    # Real datasets are often class-imbalanced; weight the loss inversely to
    # each class's frequency so the rarer class isn't ignored.
    counts = collections.Counter(train_loader.dataset.targets)
    weights = torch.tensor(
        [len(train_loader.dataset) / (len(classes) * counts[i]) for i in range(len(classes))],
        dtype=torch.float, device=device)
    print(f"Class counts: { {classes[i]: counts[i] for i in range(len(classes))} } "
          f"-> weights {[round(w, 2) for w in weights.tolist()]}")
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs("checkpoints", exist_ok=True)
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f}")

        if va_acc >= best_acc:
            best_acc = va_acc
            torch.save({
                "state_dict": model.state_dict(),
                "classes": classes,
                "img_size": args.img_size,
            }, CKPT_PATH)
            print(f"  ✓ saved checkpoint (val acc {best_acc:.3f}) -> {CKPT_PATH}")

    print(f"\nDone. Best val accuracy: {best_acc:.3f}")


if __name__ == "__main__":
    main()
