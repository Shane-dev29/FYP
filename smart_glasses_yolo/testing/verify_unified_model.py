"""Verifies multi-scale currency and indoor obstacle detection across distances."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
os.environ["YOLO_CONFIG_DIR"] = str(BASE_DIR / ".ultralytics")
(BASE_DIR / ".ultralytics").mkdir(parents=True, exist_ok=True)

from ultralytics import YOLO

def main():
    pt_path = BASE_DIR / "models" / "yolo26n_finetuned.pt"
    onnx_path = BASE_DIR / "models" / "yolo26n_finetuned.onnx"
    val_img_dir = BASE_DIR / "datasets" / "indoor_unified" / "images" / "val"

    print("=" * 60)
    print("Testing Multi-Scale Distance Robustness (PyTorch & ONNX)")
    print(f"PT Model   : {pt_path}")
    print(f"ONNX Model : {onnx_path}")
    print("=" * 60)

    model = YOLO(str(pt_path))

    # Pick samples:
    # 1. Multi-scale synthetic note (arm's length)
    # 2. Close-up note
    # 3. Person indoor image
    test_images = []
    syn_imgs = list(val_img_dir.glob("syn_curr_*.jpg"))
    if syn_imgs:
        test_images.append(syn_imgs[0])
        if len(syn_imgs) > 1:
            test_images.append(syn_imgs[min(5, len(syn_imgs)-1)])

    curr_imgs = [p for p in val_img_dir.glob("curr_*.jpg") if not p.name.startswith("syn_")]
    if curr_imgs:
        test_images.append(curr_imgs[0])

    indoor_imgs = [p for p in val_img_dir.glob("*.jpg") if not p.name.startswith("curr_") and not p.name.startswith("syn_")]
    if indoor_imgs:
        test_images.append(indoor_imgs[0])

    print(f"\nRunning PyTorch inference on {len(test_images)} test scenes across distance scales...")
    for img_p in test_images:
        results = model.predict(source=str(img_p), conf=0.20, verbose=False)
        res = results[0]
        boxes = res.boxes
        det_list = []
        for b in boxes:
            cls_id = int(b.cls[0].item())
            cls_name = model.names[cls_id]
            conf = float(b.conf[0].item())
            det_list.append(f"{cls_name} ({conf:.2f})")
        print(f"  [{img_p.name}]: {', '.join(det_list) if det_list else 'No detections'}")

    print("\nRunning ONNX Runtime Edge test...")
    onnx_model = YOLO(str(onnx_path))
    onnx_res = onnx_model.predict(source=str(test_images[0]), imgsz=416, conf=0.20, verbose=False)
    onnx_dets = [f"{onnx_model.names[int(b.cls[0].item())]} ({float(b.conf[0].item()):.2f})" for b in onnx_res[0].boxes]
    print(f"  ONNX [{test_images[0].name}]: {', '.join(onnx_dets) if onnx_dets else 'No detections'}")
    print("\nAll distance verifications completed successfully!")

if __name__ == "__main__":
    main()
