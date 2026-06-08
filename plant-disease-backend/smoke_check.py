"""Quick smoke check — run: python smoke_check.py"""
from __future__ import annotations

import io
import sys

import cv2
import numpy as np

sys.path.insert(0, ".")


def main() -> int:
    errors: list[str] = []

    # Models
    try:
        import joblib
        from pathlib import Path

        for name in ("model.pkl", "scaler.pkl", "label_encoder.pkl"):
            p = Path("models") / name
            if not p.exists():
                errors.append(f"Missing {p}")
        if not errors:
            le = joblib.load("models/label_encoder.pkl")
            print(f"OK models: {len(le.classes_)} classes")
    except Exception as exc:
        errors.append(f"Models: {exc}")

    # Predict pipeline
    try:
        from predict import extract_features, predict_disease

        img = np.zeros((128, 128, 3), dtype=np.uint8)
        img[:, :, 1] = 140
        feat = extract_features(img)
        if feat.shape[0] != 103:
            errors.append(f"Feature length {feat.shape[0]} != 103")
        else:
            print("OK extract_features shape 103")

        ok, buf = cv2.imencode(".jpg", img)
        assert ok
        disease, conf, info, top, plant, _ = predict_disease(buf.tobytes())
        if disease is None:
            errors.append("predict_disease returned None")
        else:
            print(f"OK predict: {disease} ({conf}%)")
    except Exception as exc:
        errors.append(f"Predict: {exc}")

    # Flask health
    try:
        from app import app

        client = app.test_client()
        r = client.get("/health")
        if r.status_code != 200:
            errors.append(f"/health status {r.status_code}")
        else:
            print("OK /health")
    except Exception as exc:
        errors.append(f"App: {exc}")

    # QA names aligned with model
    try:
        from qa_engine import get_clarification_questions

        q = get_clarification_questions("Tomato_Early_blight", "Tomato_Late_blight")
        if not q.get("questions"):
            errors.append("QA tomato pair empty")
        else:
            print(f"OK QA tomato ({len(q['questions'])} questions)")
    except Exception as exc:
        errors.append(f"QA: {exc}")

    if errors:
        print("\nFAILED:")
        for e in errors:
            print(" -", e)
        return 1
    print("\nAll smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
