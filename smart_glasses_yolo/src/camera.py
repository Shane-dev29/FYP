"""Camera acquisition module handling webcam streams, video files, and synthetic test frames."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Generator, Optional, Tuple, Union
import cv2
import numpy as np

from .config import CameraConfig

logger = logging.getLogger("SmartGlasses.Camera")


class CameraStream:
    """Manages video input from USB cameras, CSI modules, files, or synthetic sources."""

    def __init__(self, source: Union[int, str, Path] = 0, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self.source = source
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_synthetic = False
        self.synthetic_frame_count = 0
        self._init_source()

    def _init_source(self) -> None:
        """Initializes the capture device or determines fallback."""
        if isinstance(self.source, str) and Path(self.source).is_file():
            logger.info(f"Opening video file source: {self.source}")
            self.cap = cv2.VideoCapture(str(self.source))
        elif isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
            idx = int(self.source)
            logger.info(f"Opening camera index: {idx}")
            # Try DSHOW on Windows first for fast startup, then default
            self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                logger.warning(f"CAP_DSHOW failed for camera {idx}, trying default backend...")
                self.cap = cv2.VideoCapture(idx)

            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
                self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
            else:
                logger.warning(f"Could not open physical camera {idx}. Falling back to synthetic test generator.")
                self.is_synthetic = True
        else:
            logger.warning(f"Unrecognized source: '{self.source}'. Falling back to synthetic generator.")
            self.is_synthetic = True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads the next available frame."""
        if self.is_synthetic:
            return True, self._generate_synthetic_frame()

        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            # If video file reached EOF, loop or stop
            if isinstance(self.source, str) and Path(self.source).is_file():
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to retrieve frame from video source.")
                return False, None

        return True, frame

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Generates dynamic synthetic indoor test frames for testing when hardware camera is offline."""
        self.synthetic_frame_count += 1
        w, h = self.config.frame_width, self.config.frame_height
        canvas = np.full((h, w, 3), 40, dtype=np.uint8)

        # Draw a simulated indoor room background: floor, wall, doorway
        cv2.rectangle(canvas, (0, 0), (w, int(h * 0.65)), (70, 60, 60), -1)  # Wall
        cv2.rectangle(canvas, (0, int(h * 0.65)), (w, h), (40, 50, 45), -1)  # Floor
        cv2.line(canvas, (0, int(h * 0.65)), (w, int(h * 0.65)), (100, 100, 100), 2)  # Baseboard

        # Doorway outline
        cv2.rectangle(canvas, (int(w * 0.75), int(h * 0.2)), (int(w * 0.95), int(h * 0.75)), (110, 80, 50), -1)

        # Table & Chair
        cv2.rectangle(canvas, (80, int(h * 0.55)), (260, int(h * 0.75)), (120, 80, 40), -1)
        cv2.rectangle(canvas, (100, int(h * 0.45)), (170, int(h * 0.65)), (80, 80, 120), -1)

        # Moving object to simulate dynamic assistive scenario
        t = self.synthetic_frame_count * 0.05
        cx = int(w * 0.45 + np.sin(t) * 80)
        cy = int(h * 0.55 + np.cos(t) * 20)
        # Person silhouette (head & body)
        cv2.circle(canvas, (cx, cy - 40), 22, (200, 180, 160), -1)
        cv2.rectangle(canvas, (cx - 24, cy - 15), (cx + 24, cy + 60), (180, 60, 50), -1)

        cv2.putText(
            canvas,
            f"SYNTHETIC CAMERA TEST PATTERN (Frame {self.synthetic_frame_count})",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def release(self) -> None:
        """Releases underlying resources."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            logger.info("Released camera capture device.")

