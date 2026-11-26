"""Evaluation script to measure precision, recall, mAP50, mAP50-95 on indoor validation set."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict
import torch
from ultralytics import YOLO

logger = logging.getLogger("SmartGlasses.Evaluate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_YAML = BASE_DIR / "training" / "data.yaml"
OUTPUTS_DIR = BASE_DIR / "outputs"


def evaluate_model(
    model_path: str,
    data_yaml: str = str(DATA_YAML),
    imgsz: int = 640,
    device: str = "0" if torch.cuda.is_available() else "cpu",
    split: str = "val",
) -> Dict[str, Any]:
    """Runs strict evaluation of a model against the unified dataset validation split."""
    model_p = Path(model_path)
    if not model_p.exists():
        raise FileNotFoundError(f"Model file not found: {model_p}")

    logger.info(f"Evaluating {model_p.name} on {split} set (device={device}, imgsz={imgsz})...")
    model = YOLO(str(model_p))

    metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        device=device,
        split=split,
        verbose=False,
    )

    # Extract overall metrics
    mp = float(metrics.box.mp)  # Mean precision
    mr = float(metrics.box.mr)  # Mean recall
    map50 = float(metrics.box.map50)  # mAP@0.50
    map50_95 = float(metrics.box.map)  # mAP@0.50:0.95
    f1 = 2 * (mp * mr) / (mp + mr + 1e-7)

    # Latency breakdown
    speed = metrics.speed
    preprocess_ms = speed.get("preprocess", 0.0)
    inference_ms = speed.get("inference", 0.0)
    loss_ms = speed.get("loss", 0.0)
    postprocess_ms = speed.get("postprocess", 0.0)
    total_ms = preprocess_ms + inference_ms + postprocess_ms
    fps = 1000.0 / total_ms if total_ms > 0 else 0.0

    # Per-class AP
    class_names = metrics.names
    per_class_ap50 = {}
    per_class_ap = {}
    if hasattr(metrics.box, "ap50") and metrics.box.ap50 is not None:
        for idx, ap in enumerate(metrics.box.ap50):
            c_name = class_names.get(idx, f"class_{idx}")
            per_class_ap50[c_name] = round(float(ap), 4)

    if hasattr(metrics.box, "ap") and metrics.box.ap is not None:
        for idx, ap in enumerate(metrics.box.ap):
            c_name = class_names.get(idx, f"class_{idx}")
            per_class_ap[c_name] = round(float(ap), 4)

    results = {
        "model_name": model_p.name,
        "device": device,
        "imgsz": imgsz,
        "precision": round(mp, 4),
        "recall": round(mr, 4),
        "f1_score": round(f1, 4),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "latency_ms": {
            "preprocess": round(preprocess_ms, 2),
            "inference": round(inference_ms, 2),
            "postprocess": round(postprocess_ms, 2),
            "total": round(total_ms, 2),
        },
        "fps": round(fps, 2),
        "per_class_mAP50": per_class_ap50,
        "per_class_mAP50_95": per_class_ap,
    }

    out_file = OUTPUTS_DIR / f"eval_{model_p.stem}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation finished for {model_p.name}:")
    logger.info(f"mAP50: {map50:.4f} | mAP50-95: {map50_95:.4f} | Latency: {total_ms:.2f}ms | FPS: {fps:.1f}")
    logger.info(f"Saved results to: {out_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO models on indoor dataset.")
    parser.add_argument("--model", type=str, required=True, help="Path to model .pt file")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    evaluate_model(model_path=args.model, imgsz=args.imgsz, device=args.device)


if __name__ == "__main__":
    main()

