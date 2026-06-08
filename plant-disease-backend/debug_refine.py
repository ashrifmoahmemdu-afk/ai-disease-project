import sys
sys.path.insert(0, '.')
import joblib, cv2, numpy as np
from config import MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(LABEL_ENCODER_PATH)

f = open('test_images/frogeye_botryosphaeria_obtusa_001.jpg', 'rb')
img_bytes = f.read()
f.close()
arr = np.frombuffer(img_bytes, dtype=np.uint8)
img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

from leaf_analysis import detect_crop_family

crop, crop_conf, metrics = detect_crop_family(img)
print('=== detect_crop_family ===')
print(f'crop={crop}, conf={crop_conf}')
for k, v in sorted(metrics.items()):
    print(f'  {k}={v}')

# Check refine_probabilities with real probs
from leaf_analysis import refine_probabilities

img128 = cv2.resize(img, (128, 128))
hsv = cv2.cvtColor(img128, cv2.COLOR_BGR2HSV)
features = []
for i in range(3):
    hist = cv2.calcHist([hsv], [i], None, [32], [0, 256])
    features.extend(hist.flatten())
gray = cv2.cvtColor(img128, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)
features.append(edges.mean())
for i in range(3):
    features.append(img128[:, :, i].mean())
    features.append(img128[:, :, i].std())
feat = np.array(features)
scaled = scaler.transform([feat])
proba = model.predict_proba(scaled)[0]

print('\n=== refine_probabilities ===')
idx, adj, cf, m2 = refine_probabilities(img, encoder.classes_, proba)
print(f'best={encoder.classes_[idx]}, cf={cf}')
top5 = np.argsort(adj)[::-1][:5]
for i in top5:
    print(f'  {encoder.classes_[i]}: {adj[i]*100:.1f}%')
