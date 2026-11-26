"""Dedicated training script for YOLO26n Indian Currency Detector."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
os.environ["YOLO_CONFIG_DIR"] = str(BASE_DIR / ".ultralytics")
(BASE_DIR / ".ultralytics").mkdir(parents=True, exist_ok=True)

import torch
from ultralytics import YOLO

logger = logging.getLogger("SmartGlasses.CurrencyTrain")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

DATA_YAML = BASE_DIR / "training" / "currency_data.yaml"
MODELS_DIR = BASE_DIR / "models"
RUNS_DIR = BASE_DIR / "training" / "runs"


def train_currency_model(epochs: int = 6, batch_size: int = 16, imgsz: int = 640):
    base_weight = MODELS_DIR / "yolo26n.pt"
    if not base_weight.exists():
        raise FileNotFoundError(f"Base YOLO26n weights not found at {base_weight}")

    exp_name = "yolo26n_currency_run"
    save_dir = RUNS_DIR / exp_name

    logger.info("=" * 60)
    logger.info("Starting Dedicated Indian Currency YOLO26n Training")
    logger.info(f"Base Weights : {base_weight}")
    logger.info(f"Dataset      : {DATA_YAML}")
    logger.info(f"Epochs       : {epochs} | Batch: {batch_size} | ImgSz: {imgsz}")
    logger.info(f"Device       : {'CUDA (RTX 3050)' if torch.cuda.is_available() else 'CPU'}")
    logger.info("=" * 60)

    model = YOLO(str(base_weight))

    train_args = {
        "data": str(DATA_YAML),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": imgsz,
        "device": "0" if torch.cuda.is_available() else "cpu",
        "workers": 2,
        "project": str(RUNS_DIR),
        "name": exp_name,
        "exist_ok": True,
        "save": True,
        "plots": True,
        "verbose": True,
        "optimizer": "auto",
        "scale": 0.7,        # High scale variation for distance robustness
        "mosaic": 1.0,       # Multi-image context mosaic
        "degrees": 15.0,     # Rotation variation
        "perspective": 0.0005,
    }

    t0 = time.time()
    results = model.train(**train_args)
    elapsed_min = (time.time() - t0) / 60.0
    logger.info(f"Training completed in {elapsed_min:.2f} minutes.")

    best_pt = save_dir / "weights" / "best.pt"
    target_pt = MODELS_DIR / "yolo26n_currency.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, target_pt)
        logger.info(f"Saved dedicated currency model to: {target_pt}")
    else:
        last_pt = save_dir / "weights" / "last.pt"
        if last_pt.exists():
            shutil.copy2(last_pt, target_pt)
            logger.info(f"Saved last checkpoint to: {target_pt}")

    return target_pt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    train_currency_model(epochs=args.epochs, batch_size=args.batch, imgsz=args.imgsz)

