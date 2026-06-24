"""Image transforms and dataset loading shared by training and inference."""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMG_SIZE = 64

# ImageNet-ish normalization keeps values well-scaled for the BatchNorm layers.
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def train_transform(img_size=IMG_SIZE):
    """Augmented transform used during training (makes the tiny net generalize)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def eval_transform(img_size=IMG_SIZE):
    """Deterministic transform used for validation and single-image prediction."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def make_loaders(data_dir, batch_size=32, img_size=IMG_SIZE):
    """Build train/val DataLoaders from `data_dir/{train,val}/<class>/*.jpg`."""
    train_ds = datasets.ImageFolder(f"{data_dir}/train", transform=train_transform(img_size))
    val_ds = datasets.ImageFolder(f"{data_dir}/val", transform=eval_transform(img_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, train_ds.classes
