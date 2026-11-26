from pathlib import Path
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

BASE_DIR = Path(__file__).resolve().parent.parent

# Create challenging image
img = np.full((400, 800, 3), 240, dtype=np.uint8)
for i in range(400):
    img[i, :] = np.clip(img[i, :].astype(int) - int(i * 0.15), 0, 255).astype(np.uint8)

cv2.putText(img, "SMART ASSISTIVE GLASSES FOR VISUALLY IMPAIRED", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2)
cv2.putText(img, "This paragraph contains small fine printed characters.", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1)
cv2.putText(img, "Previously, faint letters like i, l, t, and commas were missed.", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
cv2.putText(img, "High contrast enhancement restores every missing letter.", (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
cv2.putText(img, "Date: 25 November 2026 | Verified Accurate Document", (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)

engine = RapidOCR()
result, _ = engine(img)

print("=" * 60)
print("RapidOCR on Challenging Degraded Document:")
if result:
    for item in result:
        print(f"  [{item[2]:.2f}] {item[1]}")
    full = " ".join([it[1] for it in result])
    print(f"\nExtracted:\n'{full}'")
print("=" * 60)

