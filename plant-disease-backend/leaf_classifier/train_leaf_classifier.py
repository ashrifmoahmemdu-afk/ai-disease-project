"""
train_leaf_classifier.py

Trains a binary classifier to distinguish plant leaves from non-leaf images.
Uses the 69-class PlantVillage dataset as "leaf" examples
and user-provided images in leaf_classifier/non_leaf/ as "non-leaf" examples.

Usage:
  1. Place non-leaf images in  backend/leaf_classifier/non_leaf/
  2. Run:   python leaf_classifier/train_leaf_classifier.py
  3. Trained model saved to: leaf_classifier/leaf_classifier.pth
"""

import os, sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import numpy as np

# ── Paths ──
LEAF_DATA_DIR = Path(r"D:\ai data\Final\Merge-Project\resized_merged")
NON_LEAF_DIR = Path(__file__).parent / "non_leaf"
OUTPUT_PATH  = Path(__file__).parent / "leaf_classifier.pth"

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_image_paths():
    """Return (leaf_paths, non_leaf_paths)."""
    leaf_paths = []
    if LEAF_DATA_DIR.is_dir():
        for cls_dir in sorted(LEAF_DATA_DIR.iterdir()):
            if cls_dir.is_dir():
                leaf_paths.extend(cls_dir.glob("*.*"))
        # Cap at 5000 per class to avoid class imbalance
    else:
        print(f"WARNING: Leaf dataset not found at {LEAF_DATA_DIR}")
        print("Place leaf images in leaf_classifier/leaf/ or set LEAF_DATA_DIR")

    non_leaf_paths = list(NON_LEAF_DIR.glob("*.*")) if NON_LEAF_DIR.is_dir() else []
    non_leaf_paths = [p for p in non_leaf_paths if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]

    # Filter leaf paths similarly
    leaf_paths = [p for p in leaf_paths if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]

    # Limit to avoid extreme imbalance (sample if needed)
    max_per_class = 50000
    if len(leaf_paths) > len(non_leaf_paths) * 10 and len(non_leaf_paths) > 0:
        # Keep roughly balanced
        target = min(len(non_leaf_paths) * 3, max_per_class)
        if len(leaf_paths) > target:
            import random
            random.seed(42)
            leaf_paths = random.sample(leaf_paths, target)

    return leaf_paths, non_leaf_paths


class LeafDataset(Dataset):
    """Binary dataset: label=1 for leaf, 0 for non-leaf."""
    def __init__(self, leaf_paths, non_leaf_paths, transform):
        self.paths = [(p, 1) for p in leaf_paths] + [(p, 0) for p in non_leaf_paths]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path, label = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Return a blank image on error
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        img = self.transform(img)
        return img, label


def train():
    print(f"Using device: {DEVICE}")
    print(f"Leaf data dir: {LEAF_DATA_DIR}")
    print(f"Non-leaf dir:  {NON_LEAF_DIR}")

    # Load paths
    leaf_paths, non_leaf_paths = load_image_paths()
    print(f"Leaf images:    {len(leaf_paths)}")
    print(f"Non-leaf images: {len(non_leaf_paths)}")

    if len(leaf_paths) == 0 or len(non_leaf_paths) == 0:
        print("ERROR: Both leaf and non-leaf images required.")
        print(f"  Leaf count: {len(leaf_paths)}  (check LEAF_DATA_DIR)")
        print(f"  Non-leaf count: {len(non_leaf_paths)}  (place images in {NON_LEAF_DIR})")
        sys.exit(1)

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.15, 0.15, 0.15, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Split into train/val
    dataset = LeafDataset(leaf_paths, non_leaf_paths, train_transform)
    val_dataset = LeafDataset(leaf_paths, non_leaf_paths, val_transform)

    # Use same split indices for both
    total = len(dataset)
    val_size = max(1, int(total * 0.15))
    train_size = total - val_size
    generator = torch.Generator().manual_seed(42)
    train_idx, val_idx = random_split(range(total), [train_size, val_size], generator=generator)

    # Wrapped datasets
    class SubsetWithTransform:
        def __init__(self, base_dataset, indices, transform):
            self.base = base_dataset
            self.indices = indices
            self.transform = transform
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            path, label = self.base.paths[self.indices[i]]
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
            return self.transform(img), label

    train_ds = SubsetWithTransform(dataset, train_idx.indices, train_transform)
    val_ds   = SubsetWithTransform(val_dataset, val_idx.indices, val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Model: ResNet18 with binary head
    model = models.resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        train_acc = train_correct / train_total * 100
        val_acc = val_correct / val_total * 100
        scheduler.step()

        print(f"Epoch {epoch+1:2d}/{EPOCHS}  "
              f"Train Loss: {train_loss/train_total:.4f}  "
              f"Train Acc: {train_acc:.1f}%  "
              f"Val Acc: {val_acc:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), str(OUTPUT_PATH))
            print(f"  -> Saved model (val_acc={val_acc:.1f}%)")

    print(f"\nDone! Best val acc: {best_acc:.1f}%")
    print(f"Model saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    train()
