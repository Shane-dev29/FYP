"""Fine-tuning pipeline for YOLO indoor object detection models."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
os.environ["YOLO_CONFIG_DIR"] = str(BASE_DIR / ".ultralytics")
(BASE_DIR / ".ultralytics").mkdir(parents=True, exist_ok=True)

import torch
from ultralytics import YOLO

logger = logging.getLogger("SmartGlasses.Train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_YAML = BASE_DIR / "training" / "data.yaml"
MODELS_DIR = BASE_DIR / "models"
RUNS_DIR = BASE_DIR / "training" / "runs"


def train_model(
    model_name: str,
    epochs: int = 10,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "0" if torch.cuda.is_available() else "cpu",
    patience: int = 10,
    workers: int = 2,
    data_yaml: Optional[Path] = None,
    output_name: Optional[str] = None,
) -> Path:
    """Fine-tunes a candidate YOLO model using transfer learning on the specified dataset."""
    data_path = Path(data_yaml) if data_yaml else DATA_YAML
    model_weight = MODELS_DIR / f"{model_name}.pt"
    if not model_weight.exists():
        model_weight = MODELS_DIR / f"{model_name}"
        if not model_weight.exists():
            raise FileNotFoundError(f"Weights for {model_name} not found in {MODELS_DIR}")

    exp_tag = output_name or f"{Path(model_name).stem}_indoor"
    exp_name = f"{exp_tag}_run"
    save_dir = RUNS_DIR / exp_name

    logger.info("=" * 60)
    logger.info(f"Starting Fine-Tuning: {model_name}")
    logger.info(f"Base Weights : {model_weight}")
    logger.info(f"Dataset      : {data_path}")
    logger.info(f"Epochs       : {epochs} | Batch: {batch_size} | ImgSz: {imgsz}")
    logger.info(f"Device       : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    logger.info(f"Experiment   : {save_dir}")
    logger.info("=" * 60)

    # Initialize model
    model = YOLO(str(model_weight))

    t0 = time.time()
    # Train
    train_args = {
        "data": str(data_path),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": imgsz,
        "device": device,
        "workers": workers,
        "patience": patience,
        "project": str(RUNS_DIR),
        "name": exp_name,
        "exist_ok": True,
        "save": True,
        "plots": True,
        "verbose": True,
    }

    # MuSGD optimizer specifically for YOLO26 if supported, else auto
    if "26" in model_name:
        train_args["optimizer"] = "auto"

    results = model.train(**train_args)
    elapsed_min = (time.time() - t0) / 60.0
    logger.info(f"Training completed in {elapsed_min:.2f} minutes.")

    # Locate best.pt
    best_pt = save_dir / "weights" / "best.pt"
    target_pt = MODELS_DIR / f"{exp_tag}.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, target_pt)
        logger.info(f"Saved best fine-tuned model checkpoint to: {target_pt}")
    else:
        logger.warning(f"best.pt not found at {best_pt}, checking last.pt...")
        last_pt = save_dir / "weights" / "last.pt"
        if last_pt.exists():
            shutil.copy2(last_pt, target_pt)
            logger.info(f"Saved last checkpoint to: {target_pt}")

    return target_pt


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO models on assistive datasets.")
    parser.add_argument("--model", type=str, default="yolo26n", help="Model variant (e.g. yolo26n, yolo11n, yolov8n)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--data", type=str, default=None, help="Dataset YAML path")
    parser.add_argument("--name", type=str, default=None, help="Output model name tag")
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        data_yaml=args.data,
        output_name=args.name,
    )


if __name__ == "__main__":
    main()

