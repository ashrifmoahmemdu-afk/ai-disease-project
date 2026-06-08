#!/usr/bin/env python3
"""Quick debug: predict on the latest uploaded image and show all predictions."""
from pathlib import Path
from predict import predict_disease, _load_models, _predict_disease_ml_only
import json
import cv2
import numpy as np

upload_dir = Path(__file__).resolve().parent / "uploads"
uploads = sorted(upload_dir.glob("*.jpg"))
if not uploads:
    print("No uploads found")
    exit(1)
npm run build
latest = uploads[-1]
print(f"Testing: {latest.name}\n")
image_bytes = latest.read_bytes()

# First, get raw ML-only prediction
print("=" * 60)
print("RAW ML PREDICTION (before heuristics)")
print("=" * 60)
ml_disease, ml_conf, ml_info, ml_top, ml_plant, _ = _predict_disease_ml_only(image_bytes)
print(f"Top prediction: {ml_disease} ({ml_conf}%)\n")
print("Top 5 ML predictions:")
for i, pred in enumerate(ml_top[:5], 1):
    print(f"  {i}. {pred['disease']} — {pred['confidence']}%")

# Now get full prediction with heuristics
print("\n" + "=" * 60)
print("FINAL PREDICTION (after heuristics + adjustments)")
print("=" * 60)
disease, confidence, info, top_predictions, plant_type, metrics = predict_disease(image_bytes)

print(f"\nTop prediction:")
print(f"  Disease: {disease}")
print(f"  Display name: {info.get('display_name')}")
print(f"  Confidence: {confidence}%")
print(f"  Plant type: {plant_type}")

print(f"\nAll top 5 predictions:")
for i, pred in enumerate(top_predictions, 1):
    print(f"  {i}. {pred['disease']} ({pred['display_name']}) — {pred['confidence']}%")

print(f"\nDetected plant: {info.get('detected_plant', 'N/A')}")
print(f"Unsupported: {info.get('unsupported_plant', False)}")

if metrics:
    print(f"\nKey metrics (crop detection):")
    for k, v in sorted(metrics.items())[:15]:
        print(f"  {k}: {v}")

