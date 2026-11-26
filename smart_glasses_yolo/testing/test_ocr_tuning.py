"""Test OCR improvements with preprocessing and tuned hyperparameters."""

from pathlib import Path
import cv2
import numpy as np
import easyocr

BASE_DIR = Path(__file__).resolve().parent.parent
EASYOCR_DIR = BASE_DIR / ".easyocr" / "model"

# Create a challenging document: smaller text, light gray, with slight shadow
img = np.full((400, 800, 3), 240, dtype=np.uint8)
# Add a shadow gradient
for i in range(400):
    img[i, :] = np.clip(img[i, :].astype(int) - int(i * 0.15), 0, 255).astype(np.uint8)

cv2.putText(img, "SMART ASSISTIVE GLASSES FOR VISUALLY IMPAIRED", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2)
cv2.putText(img, "This paragraph contains small fine printed characters.", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1)
cv2.putText(img, "Previously, faint letters like i, l, t, and commas were missed.", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
cv2.putText(img, "High contrast enhancement restores every missing letter.", (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
cv2.putText(img, "Date: 25 November 2026 | Verified Accurate Document", (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)

reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=str(EASYOCR_DIR), user_network_directory=str(EASYOCR_DIR))

print("=" * 60)
print("1. RAW DEFAULT EASYOCR RUN:")
raw_res = reader.readtext(img)
raw_text = " ".join([t[1] for t in raw_res])
print(f"Detected {len(raw_res)} fragments:")
print(f"'{raw_text}'\n")

print("2. TUNED & PREPROCESSED EASYOCR RUN:")
# Preprocessing
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)
gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
h, w = sharpened.shape[:2]
upscaled = cv2.resize(sharpened, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

tuned_res = reader.readtext(
    upscaled,
    contrast_ths=0.1,
    adjust_contrast=0.5,
    text_threshold=0.35,
    low_text=0.25,
    link_threshold=0.35,
    mag_ratio=1.5,
    slope_ths=0.2,
    width_ths=0.6,
)
tuned_text = " ".join([t[1] for t in tuned_res])
print(f"Detected {len(tuned_res)} fragments:")
print(f"'{tuned_text}'")
print("=" * 60)

