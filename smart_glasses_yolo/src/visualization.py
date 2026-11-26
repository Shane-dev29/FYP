"""Visualization and assistive Head-Up Display (HUD) rendering for smart glasses feed."""

from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np

from .detector import FrameDetections


class Visualizer:
    """Renders informative HUD overlays and detections for testing and assistive display."""

    # High-contrast color palette (BGR) for crisp visibility
    COLORS = {
        "hud_bg": (25, 25, 25),
        "hud_border": (80, 80, 80),
        "text_white": (245, 245, 245),
        "accent_cyan": (255, 200, 0),
        "accent_green": (80, 220, 100),
        "accent_orange": (0, 140, 255),
        "accent_red": (60, 60, 240),
        "box_color": (0, 215, 255),
    }

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_hud(
        self,
        frame: np.ndarray,
        detections: FrameDetections,
        current_fps: float,
        conf_threshold: float,
        model_name: str,
        device: str,
    ) -> np.ndarray:
        """Overlays a clean diagnostic and assistive status panel on top of the frame."""
        h, w = frame.shape[:2]
        canvas = frame.copy()

        # HUD Top Banner Dimensions
        hud_h = 75
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), self.COLORS["hud_bg"], -1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
        cv2.line(canvas, (0, hud_h), (w, hud_h), self.COLORS["hud_border"], 1)

        # Left Section: Model & Execution Target
        model_str = f"Model: {model_name} ({device.upper()})"
        cv2.putText(canvas, model_str, (14, 24), self.font, 0.60, self.COLORS["text_white"], 1, cv2.LINE_AA)

        # Center/Left: Performance Telemetry
        lat = detections.metadata["latency_breakdown_ms"]
        telemetry_str = f"FPS: {current_fps:4.1f} | Latency: {lat['total']:4.1f}ms (Pre:{lat['preprocess']:.1f} Inf:{lat['inference']:.1f} Post:{lat['postprocess']:.1f})"
        cv2.putText(canvas, telemetry_str, (14, 48), self.font, 0.52, self.COLORS["accent_cyan"], 1, cv2.LINE_AA)

        # Right Section: Conf threshold & count
        right_info = f"Conf: {conf_threshold:.2f} | Detections: {len(detections.detections)}"
        cv2.putText(canvas, right_info, (max(14, w - 270), 24), self.font, 0.55, self.COLORS["accent_green"], 1, cv2.LINE_AA)

        # Bottom Controls Banner
        help_bar_h = 24
        cv2.rectangle(canvas, (0, h - help_bar_h), (w, h), (15, 15, 15), -1)
        controls_text = "[Q] Exit | [+] Conf +0.05 | [-] Conf -0.05 | [M] Switch Model | [S] Save Log"
        cv2.putText(canvas, controls_text, (10, h - 7), self.font, 0.44, (190, 190, 190), 1, cv2.LINE_AA)

        return canvas

    def draw_custom_boxes(
        self,
        frame: np.ndarray,
        detections: FrameDetections,
    ) -> np.ndarray:
        """Custom box rendering if custom styling beyond ultralytics plot() is required."""
        canvas = frame.copy()
        for det in detections.detections:
            x1, y1, x2, y2 = det.bbox
            label = f"{det.class_name} {det.confidence:.2f}"

            # Box outline
            cv2.rectangle(canvas, (x1, y1), (x2, y2), self.COLORS["box_color"], 2)

            # Label banner
            (lw, lh), _ = cv2.getTextSize(label, self.font, 0.5, 1)
            cv2.rectangle(canvas, (x1, max(0, y1 - lh - 6)), (x1 + lw + 6, max(lh + 6, y1)), self.COLORS["box_color"], -1)
            cv2.putText(canvas, label, (x1 + 3, max(lh + 1, y1 - 4)), self.font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return canvas

