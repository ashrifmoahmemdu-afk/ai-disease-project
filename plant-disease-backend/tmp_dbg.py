from pathlib import Path
import cv2
import numpy as np
from predict import _load_models, extract_features
from leaf_analysis import (
    detect_crop_family, refine_probabilities, apply_sl_crop_correction,
    finalize_sl_prediction, detect_shaped_sl_crop, _metrics_mango_oval,
    _metrics_mango_spotted_blade, is_mango_leaf, is_tea_leaf, crop_max_proba,
)
from predict import _class_crop, _predict_disease_with_heuristics

p = Path(r"C:\Users\MIM.ASHRIF\.cursor\projects\d-ai-data-Final\assets\c__Users_MIM.ASHRIF_AppData_Roaming_Cursor_User_workspaceStorage_49c62e1b5ef46494cd6acfbd321f5bfb_images_image-0de3bff6-9abf-436f-97e3-4fbdb98d8230.png")
b = p.read_bytes()
img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
model, scaler, le = _load_models()
_, _, metrics = detect_crop_family(img)
print("metrics", metrics)
print("shaped", detect_shaped_sl_crop(metrics))
print("mango_oval", _metrics_mango_oval(metrics))
print("mango_spotted", _metrics_mango_spotted_blade(metrics))
print("is_mango", is_mango_leaf(metrics))
print("is_tea", is_tea_leaf(metrics))
features = extract_features(img)
proba = model.predict_proba(scaler.transform([features]))[0]
best_idx, adjusted, crop_family, metrics2 = refine_probabilities(img, le.classes_, proba)
disease = str(le.classes_[best_idx])
conf = round(float(adjusted[best_idx])*100,1)
raw_top = int(np.argmax(proba))
raw_crop = _class_crop(str(le.classes_[raw_top]))
print("raw", le.classes_[raw_top], round(proba[raw_top]*100,1), "raw_crop", raw_crop)
print("after refine", disease, conf, crop_family)
d,c,bi,ff,fc,adj = apply_sl_crop_correction(metrics, le.classes_, proba, adjusted, disease, conf, best_idx, crop_family, fc, raw_crop)
print("after sl_corr", d, c, ff, fc)
d,c,bi,ff,fc,adj = finalize_sl_prediction(metrics, le.classes_, proba, adj, d, c, bi, ff, fc)
print("final", d, c, ff, fc)
print("mango_ml", crop_max_proba(le.classes_, proba, "Mango"), "tea_ml", crop_max_proba(le.classes_, proba, "Tea"))
