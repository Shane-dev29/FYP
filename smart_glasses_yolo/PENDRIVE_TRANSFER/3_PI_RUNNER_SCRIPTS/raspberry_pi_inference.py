"""Edge Inference Engine for Raspberry Pi 5 (8GB RAM) Assistive Smart Glasses.
Optimized for low-latency ARM CPU inference with telemetry (FPS, CPU load, CPU temp, thermal throttling).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import psutil
from ultralytics import YOLO

logger = logging.getLogger("SmartGlasses.EdgePi")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def read_rpi_cpu_temperature() -> Optional[float]:
    """Reads SoC temperature on Raspberry Pi OS via sysfs."""
    thermal_path = Path("/sys/class/thermal/thermal_zone0/temp")
    if thermal_path.exists():
        try:
            raw = thermal_path.read_text().strip()
            return round(float(raw) / 1000.0, 1)
        except Exception:
            pass
    return None


def read_rpi_throttling_status() -> Optional[str]:
    """Queries vcgencmd for under-voltage or thermal throttling flags on Raspberry Pi."""
    try:
        out = subprocess.check_output(["vcgencmd", "get_throttled"], text=True).strip()
        # Format is throttled=0x0 (normal)
        return out
    except Exception:
        return None


class EdgeDetectionRecord:
    """Detection item matching the smart glasses interface."""
    def __init__(self, class_name: str, class_id: int, confidence: float, bbox: List[int]):
        self.class_name = class_name
        self.class_id = int(class_id)
        self.confidence = round(float(confidence), 4)
        self.bbox = [int(x) for x in bbox]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_name": self.class_name,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "bbox": self.bbox,
        }


class EdgeFrameDetections:
    """Frame detections structured for downstream depth sensing and haptic integration."""
    def __init__(
        self,
        frame_id: int,
        timestamp: int,
        detections: List[EdgeDetectionRecord],
        latency_ms: Dict[str, float],
        cpu_temp: Optional[float] = None,
        cpu_util: Optional[float] = None,
        ram_mb: Optional[float] = None,
    ):
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.detections = detections
        self.latency_ms = latency_ms
        self.cpu_temp = cpu_temp
        self.cpu_util = cpu_util
        self.ram_mb = ram_mb

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "detections": [d.to_dict() for d in self.detections],
            "telemetry": {
                "latency_breakdown_ms": self.latency_ms,
                "cpu_temperature_c": self.cpu_temp,
                "cpu_utilization_pct": self.cpu_util,
                "process_ram_mb": self.ram_mb,
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class RaspberryPiDetector:
    """Edge detector wrapper running on Raspberry Pi."""

    def __init__(
        self,
        model_path: str,
        imgsz: int = 416,
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        self.imgsz = imgsz
        self.conf = conf_thresh
        self.iou = iou_thresh

        logger.info(f"Loading edge model for Raspberry Pi: {self.model_path.name} (imgsz={imgsz})")
        self.model = YOLO(str(self.model_path))
        self.names = self.model.names if hasattr(self.model, "names") else {}
        self.process = psutil.Process(os.getpid())

    def run_inference(self, frame: np.ndarray, frame_id: int) -> Tuple[np.ndarray, EdgeFrameDetections]:
        """Runs single-frame edge inference and telemetry gathering."""
        t_start = time.perf_counter()
        timestamp_ms = int(time.time() * 1000)

        # Run model prediction on CPU
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device="cpu",
            verbose=False,
        )
        t_end = time.perf_counter()

        res = results[0]
        records: List[EdgeDetectionRecord] = []

        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)

            for box, score, cid in zip(boxes, confs, clss):
                x1, y1, x2, y2 = box.astype(int)
                cname = self.names.get(cid, f"class_{cid}")
                records.append(EdgeDetectionRecord(cname, cid, float(score), [int(x1), int(y1), int(x2), int(y2)]))

        speed = res.speed if hasattr(res, "speed") else {}
        pre_ms = round(speed.get("preprocess", 0.0), 2)
        inf_ms = round(speed.get("inference", (t_end - t_start) * 1000.0), 2)
        post_ms = round(speed.get("postprocess", 0.0), 2)
        total_ms = round(pre_ms + inf_ms + post_ms, 2)

        ram_mb = round(self.process.memory_info().rss / (1024 * 1024), 1)
        cpu_pct = round(psutil.cpu_percent(interval=None), 1)
        cpu_temp = read_rpi_cpu_temperature()

        detections = EdgeFrameDetections(
            frame_id=frame_id,
            timestamp=timestamp_ms,
            detections=records,
            latency_ms={"preprocess": pre_ms, "inference": inf_ms, "postprocess": post_ms, "total": total_ms},
            cpu_temp=cpu_temp,
            cpu_util=cpu_pct,
            ram_mb=ram_mb,
        )

        annotated_frame = res.plot()
        return annotated_frame, detections


def start_edge_pipeline(
    model_path: str,
    camera_idx: int = 0,
    imgsz: int = 416,
    conf: float = 0.35,
    max_frames: Optional[int] = None,
    headless: bool = True,
    output_log: Optional[str] = None,
):
    """Starts the edge detection loop on Raspberry Pi."""
    detector = RaspberryPiDetector(model_path=model_path, imgsz=imgsz, conf_thresh=conf)
    cap = cv2.VideoCapture(camera_idx)

    if not cap.isOpened():
        logger.warning(f"Hardware camera {camera_idx} unavailable. Using test pattern generator.")
        use_synthetic = True
    else:
        use_synthetic = False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    log_file = open(output_log, "a", encoding="utf-8") if output_log else None
    frame_id = 0
    fps_start = time.time()
    current_fps = 0.0

    print("\n" + "=" * 65)
    print("  RASPBERRY PI 5 - SMART ASSISTIVE GLASSES EDGE RUNTIME")
    print("=" * 65)
    print(f" Model       : {Path(model_path).name}")
    print(f" Resolution  : {imgsz}x{imgsz}")
    print(f" Platform    : {sys.platform} ({os.uname().machine if hasattr(os, 'uname') else 'x86/win'})")
    print(f" Throttling  : {read_rpi_throttling_status() or 'N/A'}")
    print("=" * 65 + "\n")

    try:
        while True:
            if use_synthetic:
                frame = np.full((480, 640, 3), 50, dtype=np.uint8)
                cv2.putText(frame, "RPi Test Stream", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                time.sleep(0.03)
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            frame_id += 1
            if frame_id % 10 == 0:
                elapsed = time.time() - fps_start
                if elapsed > 0:
                    current_fps = 10.0 / elapsed
                fps_start = time.time()

            annotated_frame, detections = detector.run_inference(frame, frame_id=frame_id)

            if log_file:
                log_file.write(detections.to_json() + "\n")
                log_file.flush()

            if frame_id % 10 == 0:
                temp_str = f" | Temp: {detections.cpu_temp}°C" if detections.cpu_temp else ""
                logger.info(
                    f"Frame {frame_id:4d} | FPS: {current_fps:4.1f} | "
                    f"Latency: {detections.latency_ms['total']:5.1f}ms | "
                    f"RAM: {detections.ram_mb}MB{temp_str} | "
                    f"Detections: {len(detections.detections)}"
                )

            if not headless:
                cv2.imshow("Raspberry Pi 5 Edge Detector", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if max_frames and frame_id >= max_frames:
                logger.info(f"Reached frame limit ({max_frames}). Stopping.")
                break

    finally:
        if not use_synthetic:
            cap.release()
        if not headless:
            cv2.destroyAllWindows()
        if log_file:
            log_file.close()


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi Edge Inference Runtime")
    parser.add_argument("--model", type=str, required=True, help="Path to exported model (.pt, .onnx, _ncnn_model, _openvino_model)")
    parser.add_argument("--imgsz", type=int, default=416, help="Input resolution (416 recommended for edge real-time)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--headless", action="store_true", default=True, help="Headless execution")
    parser.add_argument("--output-log", type=str, default=None, help="Output JSONL log path")
    args = parser.parse_args()

    start_edge_pipeline(
        model_path=args.model,
        camera_idx=args.camera,
        imgsz=args.imgsz,
        conf=args.conf,
        max_frames=args.max_frames,
        headless=args.headless,
        output_log=args.output_log,
    )


if __name__ == "__main__":
    main()

