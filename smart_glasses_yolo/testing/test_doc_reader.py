"""Test document reader OCR and TTS offline functionality."""

from pathlib import Path
import cv2
import numpy as np
import easyocr
import pyttsx3

BASE_DIR = Path(__file__).resolve().parent.parent
EASYOCR_DIR = BASE_DIR / ".easyocr" / "model"
EASYOCR_DIR.mkdir(parents=True, exist_ok=True)

# 1. Create a synthetic document image with clear text
img = np.full((350, 700, 3), 255, dtype=np.uint8)
cv2.putText(img, "SMART ASSISTIVE GLASSES", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
cv2.putText(img, "Chapter 1: Final Year Project", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2)
cv2.putText(img, "Real-time Assistive Vision System", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 0), 2)
cv2.putText(img, "Extracted text will be read aloud.", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 0, 0), 2)

test_path = BASE_DIR / "outputs" / "ocr_scans" / "sample_test_doc.jpg"
test_path.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(test_path), img)

print("=" * 60)
print("Testing EasyOCR on Synthetic Document Image...")
reader = easyocr.Reader(
    ["en"],
    gpu=False,
    model_storage_directory=str(EASYOCR_DIR),
    user_network_directory=str(EASYOCR_DIR),
)
results = reader.readtext(str(test_path))

extracted_text = []
for bbox, text, conf in results:
    if conf > 0.2:
        extracted_text.append(text)
        print(f"  Detected: '{text}' (Confidence: {conf:.2f})")

full_sentence = " ".join(extracted_text)
print("\nFull Extracted Text:")
print(f"'{full_sentence}'")

print("\nTesting pyttsx3 offline text-to-speech engine...")
engine = pyttsx3.init()
engine.setProperty("rate", 160)
print("Speaking text aloud...")
engine.say("Smart Assistive Glasses. Document reader test successful.")
engine.runAndWait()
engine.stop()
print("Speech synthesis test passed successfully!")
print("=" * 60)
