import sys; sys.path.insert(0, r"D:\ai data\New folder (4)\plant-disease-backend")
import cv2, numpy as np
from leaf_analysis import detect_crop_family, is_likely_a_leaf
from pathlib import Path

paths = list(Path(r"C:\Users\MIM.ASHRIF\Desktop\leaf\New folder (2)").glob("*.jpg"))
if not paths:
    print("No images found in folder")
else:
    for p in paths[:5]:
        img = cv2.imread(str(p))
        if img is None:
            print(f"{p.name}: FAILED to load")
            continue
        crop, conf, metrics = detect_crop_family(img)
        is_leaf, reason = is_likely_a_leaf(metrics)
        coverage = metrics.get("area", 0) / 65536.0
        print(f"{p.name}: leaf={is_leaf}  reason=\"{reason}\"  coverage={coverage:.2%}  solidity={metrics.get('solidity',0):.2f}  regions={metrics.get('num_regions',0)}  border={metrics.get('border_touching',False)}")
