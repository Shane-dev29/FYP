"""Object Detection engine with modular structured output for downstream depth and haptic subsystems."""

from __future__ import annotations

import json
import logging
import time
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

BASE_DIR = Path(__file__).resolve().parent.parent
os.environ["YOLO_CONFIG_DIR"] = str(BASE_DIR / ".ultralytics")
(BASE_DIR / ".ultralytics").mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
from ultralytics import YOLO

from .config import DetectorConfig

logger = logging.getLogger("SmartGlasses.Detector")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@dataclass
class DetectionRecord:
    """Individual object detection item in standard bounding box representation."""
    class_name: str
    class_id: int
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2] pixel coordinates (integers)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_name": self.class_name,
            "class_id": int(self.class_id),
            "confidence": round(float(self.confidence), 4),
            "bbox": [int(c) for c in self.bbox],
        }


@dataclass
class FrameDetections:
    """Modular structured frame detection output for future depth sensing and haptic integration.

    Conforms to the required interface:
    {
      "frame_id": 120,
      "timestamp": 123456789,
      "detections": [
        {
          "class_name": "chair",
          "class_id": 56,
          "confidence": 0.94,
          "bbox": [100, 120, 300, 400]
        }
      ]
    }
    """
    frame_id: int
    timestamp: int  # epoch milliseconds
    detections: List[DetectionRecord] = field(default_factory=list)
    model_name: str = ""
    inference_time_ms: float = 0.0
    preprocess_time_ms: float = 0.0
    postprocess_time_ms: float = 0.0

    @property
    def total_latency_ms(self) -> float:
        return self.preprocess_time_ms + self.inference_time_ms + self.postprocess_time_ms

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "latency_breakdown_ms": {
                "preprocess": round(self.preprocess_time_ms, 2),
                "inference": round(self.inference_time_ms, 2),
                "postprocess": round(self.postprocess_time_ms, 2),
                "total": round(self.total_latency_ms, 2),
            },
            "total_detections": len(self.detections),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": int(self.frame_id),
            "timestamp": int(self.timestamp),
            "detections": [d.to_dict() for d in self.detections],
            "metadata": self.metadata,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class SmartGlassesDetector:
    """Core object detector wrapper managing YOLO models and telemetry."""

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self.model: Optional[YOLO] = None
        self.model_path: Path = Path(self.config.model_path)
        self.class_names: Dict[int, str] = {}
        self.frame_counter: int = 0
        self._load_model(self.model_path)

    def _load_model(self, model_path: Union[str, Path]) -> None:
        """Loads or hot-swaps a YOLO model checkpoint."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        logger.info(f"Loading YOLO model from: {model_path} onto device: {self.config.device}")
        try:
            self.model = YOLO(str(model_path))
            self.model_path = model_path
            self.config.model_name = model_path.name

            # Cache class names
            if hasattr(self.model, "names") and self.model.names:
                self.class_names = self.model.names
            elif hasattr(self.model.model, "names") and self.model.model.names:
                self.class_names = self.model.model.names
            else:
                self.class_names = {}

            logger.info(f"Successfully loaded '{model_path.name}' with {len(self.class_names)} classes.")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    def switch_model(self, model_path: Union[str, Path]) -> bool:
        """Switch active model checkpoint at runtime."""
        try:
            self._load_model(model_path)
            return True
        except Exception as e:
            logger.error(f"Failed switching to model {model_path}: {e}")
            return False

    def set_confidence(self, conf: float) -> None:
        """Dynamically update confidence threshold."""
        self.config.confidence_threshold = max(0.05, min(0.95, conf))
        logger.info(f"Updated confidence threshold to: {self.config.confidence_threshold:.2f}")

    def detect(
        self,
        frame: np.ndarray,
        frame_id: Optional[int] = None,
        conf_threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, FrameDetections]:
        """Runs inference on a single image frame and produces structured outputs.

        Args:
            frame: BGR numpy image from OpenCV.
            frame_id: Optional sequence number. If None, internal counter is incremented.
            conf_threshold: Optional override for confidence threshold.

        Returns:
            Tuple of (annotated_frame, FrameDetections object).
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or invalid.")

        if frame_id is None:
            self.frame_counter += 1
            fid = self.frame_counter
        else:
            fid = frame_id

        conf = conf_threshold if conf_threshold is not None else self.config.confidence_threshold
        timestamp_ms = int(time.time() * 1000)

        # Run inference
        results = self.model.predict(
            source=frame,
            conf=conf,
            iou=self.config.iou_threshold,
            imgsz=self.config.image_size,
            device=self.config.device,
            verbose=self.config.verbose,
            max_det=self.config.max_detections,
        )

        res = results[0]
        records: List[DetectionRecord] = []

        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)

            for box, score, cls_id in zip(boxes, confs, clss):
                x1, y1, x2, y2 = box.astype(int)
                c_name = self.class_names.get(cls_id, f"class_{cls_id}")
                record = DetectionRecord(
                    class_name=c_name,
                    class_id=int(cls_id),
                    confidence=float(score),
                    bbox=[int(x1), int(y1), int(x2), int(y2)],
                )
                records.append(record)

        # Extract timing profiles from Ultralytics result
        preprocess_ms = res.speed.get("preprocess", 0.0) if hasattr(res, "speed") else 0.0
        inference_ms = res.speed.get("inference", 0.0) if hasattr(res, "speed") else 0.0
        postprocess_ms = res.speed.get("postprocess", 0.0) if hasattr(res, "speed") else 0.0

        frame_detections = FrameDetections(
            frame_id=fid,
            timestamp=timestamp_ms,
            detections=records,
            model_name=self.config.model_name,
            inference_time_ms=inference_ms,
            preprocess_time_ms=preprocess_ms,
            postprocess_time_ms=postprocess_ms,
        )

        # Annotated frame
        annotated_frame = res.plot()

        return annotated_frame, frame_detections

