"""Dedicated Indian Currency Recognition Reader for Smart Assistive Glasses."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

os.environ["YOLO_CONFIG_DIR"] = str(BASE_DIR / ".ultralytics")
(BASE_DIR / ".ultralytics").mkdir(parents=True, exist_ok=True)

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger("SmartGlasses.CurrencyReader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

MODELS_DIR = BASE_DIR / "models"
CURRENCY_PT = MODELS_DIR / "yolo26n_currency.pt"
CURRENCY_ONNX = MODELS_DIR / "yolo26n_currency.onnx"

DENOMINATION_MAP = {
    "currency_10": "₹10 (Ten Rupees)",
    "currency_20": "₹20 (Twenty Rupees)",
    "currency_50": "₹50 (Fifty Rupees)",
    "currency_100": "₹100 (One Hundred Rupees)",
    "currency_200": "₹200 (Two Hundred Rupees)",
    "currency_500": "₹500 (Five Hundred Rupees)",
    "currency_2000": "₹2000 (Two Thousand Rupees)",
}


def run_currency_reader(model_path: Path, source: str = "0", conf_thresh: float = 0.30):
    logger.info("=" * 65)
    logger.info("Smart Assistive Glasses - Dedicated Currency Recognition Mode")
    logger.info(f"Active Model  : {model_path.name}")
    logger.info(f"Target Device : {'CUDA (RTX 3050)' if 'pt' in model_path.suffix else 'CPU (ONNX Edge)'}")
    logger.info(f"Confidence    : {conf_thresh:.2f}")
    logger.info("=" * 65)

    model = YOLO(str(model_path))

    # Open Camera
    cam_idx = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        logger.warning(f"Camera index {source} unavailable. Generating synthetic test pattern...")
        cap = None

    window_name = "Smart Glasses - Dedicated Currency Recognition (YOLO26n)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 640)

    last_spoken = ""
    last_spoken_time = 0.0

    fps = 0.0
    frame_count = 0
    t_start = time.time()

    try:
        while True:
            t0 = time.time()

            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                # Synthetic fallback canvas
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Present an Indian Banknote to Camera", (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            h, w = frame.shape[:2]

            # Run inference
            results = model.predict(source=frame, conf=conf_thresh, imgsz=640 if "pt" in model_path.suffix else 416, verbose=False)
            res = results[0]
            boxes = res.boxes

            # Top HUD banner
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 65), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            cv2.line(frame, (0, 65), (w, 65), (0, 200, 255), 2)

            cv2.putText(frame, f"CURRENCY MODE: {model_path.name}", (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.1f} | Conf: {conf_thresh:.2f} | Notes: {len(boxes)}", (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 200), 1)

            # Bottom help bar
            cv2.rectangle(frame, (0, h - 25), (w, h), (15, 15, 15), -1)
            cv2.putText(frame, "[Q] Quit  |  [+] Conf +0.05  |  [-] Conf -0.05", (10, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1)

            # Draw detected notes
            for b in boxes:
                cls_id = int(b.cls[0].item())
                raw_name = model.names[cls_id]
                friendly_name = DENOMINATION_MAP.get(raw_name, raw_name)
                conf = float(b.conf[0].item())
                coords = b.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords

                # Gold border for currency
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 215, 255), 3)

                label_str = f"{friendly_name} [{conf:.1%}]"
                (lw, lh), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, max(0, y1 - lh - 8)), (x1 + lw + 8, max(lh + 8, y1)), (0, 215, 255), -1)
                cv2.putText(frame, label_str, (x1 + 4, max(lh + 4, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                # Audio notification logging (throttled)
                if friendly_name != last_spoken or (time.time() - last_spoken_time > 3.0):
                    print(f"\n>>> [ASSISTIVE AUDIO ANNOUNCEMENT]: \"{friendly_name} identified ({conf:.1%} confidence)\" <<<")
                    last_spoken = friendly_name
                    last_spoken_time = time.time()

            cv2.imshow(window_name, frame)

            # Calculate FPS
            frame_count += 1
            dt = time.time() - t0
            fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 0.001))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key in (ord("+"), ord("=")):
                conf_thresh = min(0.95, conf_thresh + 0.05)
            elif key in (ord("-"), ord("_")):
                conf_thresh = max(0.05, conf_thresh - 0.05)

    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dedicated Indian Currency Reader for Smart Glasses")
    parser.add_argument("--model", type=str, default="yolo26n_currency.pt")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    m_path = Path(args.model)
    if not m_path.exists():
        m_path = MODELS_DIR / args.model

    run_currency_reader(m_path, source=args.source, conf_thresh=args.conf)

