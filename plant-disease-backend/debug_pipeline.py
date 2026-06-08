import sys
sys.path.insert(0, '.')
import logging
logging.basicConfig(level=logging.DEBUG)

from predict import predict_disease, _predict_with_pytorch, _predict_with_rf
from leaf_analysis import detect_crop_family, is_likely_a_leaf
import cv2
import numpy as np

f = open('test_images/frogeye_botryosphaeria_obtusa_001.jpg', 'rb')
img_bytes = f.read()
f.close()

arr = np.frombuffer(img_bytes, dtype=np.uint8)
img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

print('=== STAGE 1: Leaf Gate ===')
crop, crop_conf, metrics = detect_crop_family(img)
is_leaf, reason = is_likely_a_leaf(metrics)
print(f'is_leaf={is_leaf}, reason={reason}')
print(f'crop={crop}, crop_conf={crop_conf}')
print(f'lobes={metrics.get("lobes")}, aspect={metrics.get("aspect")}')
print(f'serration={metrics.get("serration")}, solidity={metrics.get("solidity")}')
print(f'lesion_ratio={metrics.get("lesion_ratio")}')
print(f'green_ratio={metrics.get("green_ratio")}')

print('\n=== STAGE 3: ResNet50 ===')
res = _predict_with_pytorch(img_bytes)
if res:
    print(f'Result: {res[0]} @ {res[1]}% plant={res[2]}')
else:
    print('SKIPPED (no model file)')

print('\n=== STAGE 4: Random Forest ===')
rf = _predict_with_rf(img_bytes)
if rf:
    print(f'Result: {rf[0]} @ {rf[1]}% plant={rf[2]}')
    for t in rf[3][:5]:
        print(f'  {t["disease"]}: {t["confidence"]}%')
else:
    print('FAILED')

print('\n=== STAGE 5: Vision API ===')
# Won't actually call API, just check if import works
try:
    from groq_predict import predict_with_groq
    print('groq_predict import OK')
except Exception as e:
    print(f'groq_predict import FAILED: {e}')

print('\n=== Full predict_disease ===')
result = predict_disease(img_bytes)
print(f'Final: disease={result[0]}, conf={result[1]}, plant={result[4]}, source={result[5].get("source")}')
