"""Verifies dedicated YOLO26n currency model across denominations."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
os.environ["YOLO_CONFIG_DIR"] = str(BASE_DIR / ".ultralytics")
(BASE_DIR / ".ultralytics").mkdir(parents=True, exist_ok=True)

from ultralytics import YOLO

def main():
    pt_path = BASE_DIR / "models" / "yolo26n_currency.pt"
    onnx_path = BASE_DIR / "models" / "yolo26n_currency.onnx"
    val_dir = BASE_DIR / "datasets" / "indian_currency" / "images" / "val"

    print("=" * 60)
    print("Testing Dedicated YOLO26n Indian Currency Detector")
    print(f"PT Model   : {pt_path}")
    print(f"ONNX Model : {onnx_path}")
    print("=" * 60)

    model = YOLO(str(pt_path))
    print(f"Classes: {model.names}")

    # Test one image from each denomination
    denominations = ["10", "20", "50", "100", "200", "500", "2000"]
    sample_images = []
    for d in denominations:
        imgs = list(val_dir.glob(f"{d}_*.jpg"))
        if imgs:
            sample_images.append((d, imgs[0]))

    print(f"\nRunning PyTorch inference across {len(sample_images)} denominations...")
    for denom, img_path in sample_images:
        results = model.predict(source=str(img_path), conf=0.25, verbose=False)
        boxes = results[0].boxes
        dets = [f"{model.names[int(b.cls[0].item())]} ({float(b.conf[0].item()):.2%})" for b in boxes]
        print(f"  Target: INR {denom:<4} -> Detected: {', '.join(dets) if dets else 'None'}")

    print("\nRunning ONNX Runtime inference on sample note...")
    onnx_model = YOLO(str(onnx_path))
    res = onnx_model.predict(source=str(sample_images[0][1]), imgsz=416, conf=0.25, verbose=False)
    onnx_dets = [f"{onnx_model.names[int(b.cls[0].item())]} ({float(b.conf[0].item()):.2%})" for b in res[0].boxes]
    print(f"  ONNX Output: {', '.join(onnx_dets)}")
    print("Dedicated Currency Model verification completed successfully!")

if __name__ == "__main__":
    main()

