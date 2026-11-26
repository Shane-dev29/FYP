"""Model testing suite evaluating indoor robustness: occlusions, lighting, and scales."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger("SmartGlasses.TestModel")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs" / "test_evaluations"


def apply_lighting_simulation(img: np.ndarray, condition: str) -> np.ndarray:
    """Simulates realistic indoor lighting conditions for robustness stress-testing."""
    if condition == "low_light":
        # Scale brightness down and add slight sensor noise
        low = np.clip(img.astype(np.float32) * 0.35, 0, 255).astype(np.uint8)
        noise = np.random.normal(0, 8, img.shape).astype(np.int16)
        return np.clip(low.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif condition == "bright_glare":
        # Simulate high exposure glare near windows/lamps
        bright = np.clip(img.astype(np.float32) * 1.5 + 40, 0, 255).astype(np.uint8)
        return bright
    elif condition == "partial_occlusion":
        # Simulate an obstacle partially blocking the field of view
        occ = img.copy()
        h, w = occ.shape[:2]
        cv2.rectangle(occ, (int(w * 0.25), int(h * 0.35)), (int(w * 0.55), int(h * 0.75)), (30, 30, 30), -1)
        return occ
    return img


def run_robustness_test(
    model_path: str,
    image_paths: List[Path],
    device: str = "cuda:0",
    conf: float = 0.30,
) -> Dict[str, Any]:
    """Tests model detection stability across normal, low light, bright glare, and partial occlusion."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_path)
    conditions = ["normal", "low_light", "bright_glare", "partial_occlusion"]

    results_by_condition = {c: [] for c in conditions}

    logger.info(f"Running robustness test on {len(image_paths)} images across {len(conditions)} conditions...")

    for img_p in image_paths:
        orig = cv2.imread(str(img_p))
        if orig is None:
            continue

        for cond in conditions:
            test_img = apply_lighting_simulation(orig, cond)
            preds = model.predict(test_img, conf=conf, device=device, verbose=False)[0]

            det_count = len(preds.boxes) if preds.boxes is not None else 0
            classes = [model.names[int(c)] for c in preds.boxes.cls.cpu().numpy()] if det_count > 0 else []
            confs = [float(s) for s in preds.boxes.conf.cpu().numpy()] if det_count > 0 else []

            results_by_condition[cond].append({
                "image": img_p.name,
                "detection_count": det_count,
                "classes": classes,
                "mean_confidence": round(float(np.mean(confs)), 3) if confs else 0.0,
            })

            # Save annotated example for inspection
            out_img_path = OUTPUTS_DIR / f"{img_p.stem}_{cond}_annotated.jpg"
            cv2.imwrite(str(out_img_path), preds.plot())

    summary = {}
    print("\n" + "=" * 65)
    print("  INDOOR ROBUSTNESS STRESS TEST RESULTS")
    print("=" * 65)
    print(f"{'Condition':<20} {'Total Detections':<18} {'Mean Conf':<12}")
    print("-" * 65)
    for cond, entries in results_by_condition.items():
        total_dets = sum(e["detection_count"] for e in entries)
        all_confs = [e["mean_confidence"] for e in entries if e["mean_confidence"] > 0]
        mean_c = round(float(np.mean(all_confs)), 3) if all_confs else 0.0
        summary[cond] = {"total_detections": total_dets, "mean_confidence": mean_c}
        print(f"{cond:<20} {total_dets:<18} {mean_c:<12}")
    print("=" * 65 + "\n")

    out_json = OUTPUTS_DIR / f"robustness_{Path(model_path).stem}.json"
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "details": results_by_condition}, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Test model under indoor environmental conditions.")
    parser.add_argument("--model", type=str, default="smart_glasses_yolo/models/yolo26n.pt", help="Model path")
    parser.add_argument("--images-dir", type=str, default=None, help="Directory containing test images")
    parser.add_argument("--device", type=str, default="cuda:0", help="Execution device")
    args = parser.parse_args()

    # Find test images from validation split or testing dir
    if args.images_dir:
        img_paths = list(Path(args.images_dir).glob("*.jpg"))
    else:
        val_dir = BASE_DIR / "datasets" / "indoor_unified" / "images" / "val"
        img_paths = list(val_dir.glob("*.jpg"))[:10]  # Sample 10 validation images

    run_robustness_test(
        model_path=args.model,
        image_paths=img_paths,
        device=args.device,
    )


if __name__ == "__main__":
    main()

