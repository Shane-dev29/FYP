"""Export engine for deploying YOLO models to Edge targets (Raspberry Pi / ARM).
Supports ONNX, NCNN, OpenVINO, and TFLite formats with optional FP16/INT8 quantization.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
os.environ["YOLO_CONFIG_DIR"] = str(BASE_DIR / ".ultralytics")
(BASE_DIR / ".ultralytics").mkdir(parents=True, exist_ok=True)

from ultralytics import YOLO

logger = logging.getLogger("SmartGlasses.Export")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"


def export_edge_model(
    model_path: Path,
    export_format: str = "onnx",
    imgsz: int = 640,
    half: bool = False,
    int8: bool = False,
    data_yaml: Optional[str] = None,
) -> Optional[Path]:
    """Exports a YOLO model to the designated target edge deployment format."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info("=" * 60)
    logger.info(f"Exporting: {model_path.name}")
    logger.info(f"Target Format : {export_format.upper()}")
    logger.info(f"Resolution    : {imgsz}x{imgsz}")
    logger.info(f"Precision     : {'INT8' if int8 else ('FP16' if half else 'FP32')}")
    logger.info("=" * 60)

    model = YOLO(str(model_path))

    export_args = {
        "format": export_format,
        "imgsz": imgsz,
        "half": half,
        "int8": int8,
        "verbose": True,
    }
    if data_yaml and int8:
        export_args["data"] = data_yaml

    t0 = time.time()
    try:
        exported_path = model.export(**export_args)
        elapsed = time.time() - t0
        logger.info(f"Export successful in {elapsed:.2f} seconds!")
        logger.info(f"Exported artifact location: {exported_path}")

        # Check file or directory size
        p = Path(exported_path)
        if p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            logger.info(f"Artifact file size: {size_mb:.2f} MB")
        elif p.is_dir():
            size_mb = sum(f.stat().st_size for f in p.glob("**/*") if f.is_file()) / (1024 * 1024)
            logger.info(f"Artifact folder size: {size_mb:.2f} MB")

        return Path(exported_path)
    except Exception as e:
        logger.error(f"Failed to export {model_path.name} to {export_format}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Export YOLO models to edge formats.")
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="Model checkpoint path or name")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "ncnn", "openvino", "tflite", "engine"], help="Export format")
    parser.add_argument("--imgsz", type=int, default=640, help="Input resolution")
    parser.add_argument("--half", action="store_true", help="FP16 half-precision")
    parser.add_argument("--int8", action="store_true", help="INT8 quantization")
    parser.add_argument("--data", type=str, default=None, help="Dataset yaml for INT8 calibration")
    args = parser.parse_args()

    m_path = Path(args.model)
    if not m_path.is_absolute() and not m_path.exists():
        candidate = MODELS_DIR / m_path.name
        if candidate.exists():
            m_path = candidate

    export_edge_model(
        model_path=m_path,
        export_format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        int8=args.int8,
        data_yaml=args.data,
    )


if __name__ == "__main__":
    main()

