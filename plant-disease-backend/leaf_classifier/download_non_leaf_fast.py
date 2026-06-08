"""
download_non_leaf_fast.py — Fast parallel download of ~200 non-leaf images.
Uses icrawler (Bing image search) which is much faster.
"""
import sys
from pathlib import Path
from icrawler.builtin import BingImageCrawler

NON_LEAF_DIR = Path(__file__).parent / "non_leaf"

CATEGORIES = [
    ("human_face", "human face portrait"),
    ("vehicle_car", "car vehicle front view"),
    ("animal_dog", "dog animal sitting"),
    ("animal_cat", "cat animal closeup"),
    ("building_house", "building house exterior"),
    ("shoe_footwear", "shoe footwear sneaker"),
    ("food_plate", "food plate meal top view"),
    ("chair_furniture", "chair furniture wooden"),
    ("smartphone", "smartphone mobile phone"),
    ("book_magazine", "book magazine stack"),
    ("bicycle", "bicycle bike outdoor"),
    ("watch_clock", "watch clock wrist"),
    ("bottle", "bottle water plastic"),
    ("ball_sports", "ball football soccer"),
    ("hat_cap", "hat cap clothing accessory"),
    ("tree_bark", "tree trunk bark closeup"),
    ("rock_stone", "rock stone nature"),
    ("sky_clouds", "sky clouds blue"),
    ("water_ocean", "water ocean sea wave"),
    ("hand_fingers", "hand fingers human"),
]

def download():
    total = 0
    for short_name, query in CATEGORIES:
        out_dir = NON_LEAF_DIR / short_name
        if out_dir.exists() and len(list(out_dir.glob("*.*"))) > 5:
            count = len(list(out_dir.glob("*.*")))
            total += count
            print(f"[{short_name}]: already {count}")
            continue

        crawler = BingImageCrawler(storage={"root_dir": str(out_dir)})
        try:
            crawler.crawl(keyword=query, max_num=12, file_idx_offset=0)
        except Exception as e:
            print(f"[{short_name}]: error — {e}")
            continue

        count = len(list(out_dir.glob("*.*")))
        total += count
        print(f"[{short_name}]: downloaded {count}")

    print(f"\nTotal: {total}")


def flatten():
    """Move all images into flat non_leaf/ dir with unique names."""
    count = 0
    for subdir in sorted(NON_LEAF_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        for img in list(subdir.glob("*.*")):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                try:
                    img.unlink()
                except:
                    pass
                continue
            new_name = f"{subdir.name}_{img.name}"
            dest = NON_LEAF_DIR / new_name
            if not dest.exists():
                try:
                    img.rename(dest)
                    count += 1
                except:
                    pass
        # remove subdir
        try:
            import shutil
            shutil.rmtree(subdir)
        except:
            pass

    # Filter non-image files
    for f in list(NON_LEAF_DIR.iterdir()):
        if f.is_dir():
            continue
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            try:
                f.unlink()
            except:
                pass

    final = len(list(NON_LEAF_DIR.glob("*.*")))
    print(f"Flattened: {count} images. Final count: {final}")


if __name__ == "__main__":
    download()
    flatten()
