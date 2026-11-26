from pathlib import Path
import time
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

BASE_DIR = Path(__file__).resolve().parent.parent
test_doc = BASE_DIR / "outputs" / "ocr_scans" / "sample_test_doc.jpg"

engine = RapidOCR()

t0 = time.time()
result, elapse = engine(str(test_doc))
elapsed_total = time.time() - t0

print("=" * 60)
print(f"RapidOCR Test on {test_doc.name} (took {elapsed_total:.2f}s):")
if result:
    for item in result:
        bbox, text, conf = item
        print(f"  [{conf:.2f}] {text}")
    full_text = " ".join([item[1] for item in result])
    print(f"\nFull Extracted Text:\n'{full_text}'")
else:
    print("No text detected.")
print("=" * 60)

