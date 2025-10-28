import os
from typing import Tuple, List, Dict

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DEFAULT_IMG_SIZE: Tuple[int, int] = (32, 32)


def build_transforms(img_size: Tuple[int, int] = DEFAULT_IMG_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def load_datasets(train_dir: str, test_dir: str, img_size: Tuple[int, int] = DEFAULT_IMG_SIZE):
    """
    Uses torchvision ImageFolder to automatically encode labels from subfolder names.
    """
    transform = build_transforms(img_size)

    train_ds = datasets.ImageFolder(root=train_dir, transform=transform)
    test_ds = datasets.ImageFolder(root=test_dir, transform=transform)

    # class_to_idx is the label encoding mapping
    class_to_idx: Dict[str, int] = train_ds.class_to_idx
    idx_to_class: Dict[int, str] = {v: k for k, v in class_to_idx.items()}

    return train_ds, test_ds, class_to_idx, idx_to_class


def get_dataloaders(
    train_dir: str = "./data/devanagari_dataset/Train",
    test_dir: str = "./data/devanagari_dataset/Test",
    batch_size: int = 128,
    num_workers: int = 2,
    img_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
):
    train_ds, test_ds, class_to_idx, idx_to_class = load_datasets(train_dir, test_dir, img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader, class_to_idx, idx_to_class


def count_classes(train_dir: str = "./data/devanagari_dataset/Train", img_size: Tuple[int, int] = DEFAULT_IMG_SIZE) -> int:
    ds = datasets.ImageFolder(root=train_dir, transform=build_transforms(img_size))
    return len(ds.classes)


if __name__ == "__main__":
    train_loader, test_loader, class_to_idx, idx_to_class = get_dataloaders()
    print(f"Classes ({len(class_to_idx)}):")
    print(sorted(class_to_idx.items(), key=lambda x: x[1])[:5], "...")
    xb, yb = next(iter(train_loader))
    print("Batch:", xb.shape, yb.shape)
