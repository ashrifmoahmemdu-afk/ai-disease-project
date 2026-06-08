"""
download_non_leaf.py — Downloads ~200 diverse non-leaf images for binary classifier training.
Categories cover common non-plant objects, animals, people, scenes, etc.
"""
import os, sys, time
from pathlib import Path

NON_LEAF_DIR = Path(__file__).parent / "leaf_classifier" / "non_leaf"
NON_LEAF_DIR.mkdir(parents=True, exist_ok=True)

# Diverse non-leaf categories — 20 categories, ~10 each = 200 images
CATEGORIES = [
    "human face portrait",
    "car vehicle",
    "dog animal",
    "cat animal",
    "building house",
    "shoe footwear",
    "food plate meal",
    "chair furniture",
    "smartphone device",
    "book magazine",
    "bicycle bicycle",
    "watch clock",
    "bottle container",
    "ball sports",
    "hat cap clothing",
    "tree trunk bark",
    "rock stone",
    "sky clouds",
    "water ocean sea",
    "hand finger",
]

def download():
    from bing_image_downloader import downloader

    total = 0
    for i, cat in enumerate(CATEGORIES, 1):
        out_dir = NON_LEAF_DIR / f"_{i:02d}_{cat.replace(' ', '_')}"
        if out_dir.exists():
            # already downloaded
            count = len(list(out_dir.glob("*.*")))
            total += count
            print(f"[{i}/{len(CATEGORIES)}] {cat}: already have {count} images")
            continue

        try:
            downloader.download(
                cat,
                limit=12,
                output_dir=str(NON_LEAF_DIR),
                adult_filter_off=False,
                force_replace=False,
                timeout=30,
            )
            # Rename folder to include index
            raw_dir = NON_LEAF_DIR / cat
            if raw_dir.exists() and raw_dir != out_dir:
                raw_dir.rename(out_dir)
            count = len(list(out_dir.glob("*.*"))) if out_dir.exists() else 0
            total += count
            print(f"[{i}/{len(CATEGORIES)}] {cat}: downloaded {count} images")
        except Exception as e:
            print(f"[{i}/{len(CATEGORIES)}] {cat}: FAILED — {e}")

        time.sleep(1)  # be polite

    print(f"\nTotal non-leaf images: {total}")
    return total


def flatten():
    """Move all images from subdirs into the flat non_leaf/ directory."""
    count = 0
    for subdir in sorted(NON_LEAF_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        for img in subdir.glob("*.*"):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"):
                continue
            # Rename with prefix to avoid collisions
            new_name = f"{subdir.name}_{img.name}"
            dest = NON_LEAF_DIR / new_name
            if not dest.exists():
                img.rename(dest)
                count += 1
        # Remove empty subdir
        try:
            subdir.rmdir()
        except OSError:
            pass

    # Remove any non-image files
    for f in NON_LEAF_DIR.iterdir():
        if f.is_dir():
            continue
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"):
            try:
                f.unlink()
            except OSError:
                pass

    print(f"Flattened: {count} images in {NON_LEAF_DIR}")


if __name__ == "__main__":
    n = download()
    if n > 0:
        flatten()
        # Final count
        final = len(list(NON_LEAF_DIR.glob("*.*")))
        print(f"Final non-leaf image count: {final}")
