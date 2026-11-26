"""Configuration module for Smart Assistive Glasses Object Detection System."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
import torch


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATASETS_DIR = BASE_DIR / "datasets"
OUTPUTS_DIR = BASE_DIR / "outputs"
RESEARCH_DIR = BASE_DIR / "research"
BENCHMARKING_DIR = BASE_DIR / "benchmarking"


@dataclass
class DetectorConfig:
    """Core detector and hardware execution configuration."""
    model_name: str = "yolo26n_finetuned.pt"
    model_path: Path = field(default_factory=lambda: MODELS_DIR / "yolo26n_finetuned.pt")
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    image_size: int = 640
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    half_precision: bool = torch.cuda.is_available()  # FP16 on GPU
    max_detections: int = 50
    verbose: bool = False


@dataclass
class CameraConfig:
    """Camera acquisition settings."""
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 30
    buffer_size: int = 1


@dataclass
class AssistiveOutputConfig:
    """Structured output settings for downstream depth sensing and haptic integration."""
    save_json_log: bool = False
    json_log_path: Path = field(default_factory=lambda: OUTPUTS_DIR / "detections_stream.jsonl")
    # Categories of paramount interest for visually impaired navigation
    obstacle_priority_classes: list[str] = field(default_factory=lambda: [
        "person", "chair", "table", "door", "stairs", "sofa", "bed", "tv", "laptop"
    ])
    currency_classes: list[str] = field(default_factory=lambda: [
        "currency_10", "currency_20", "currency_50", "currency_100", "currency_200", "currency_500", "currency_2000"
    ])


@dataclass
class SystemConfig:
    """Global system configuration container."""
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    assistive: AssistiveOutputConfig = field(default_factory=AssistiveOutputConfig)


# Default singleton instance
DEFAULT_CONFIG = SystemConfig()

