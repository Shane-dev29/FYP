"""High-Accuracy Live Document & Text Reader for Smart Assistive Glasses.
Supports PP-OCRv4 (RapidOCR ONNX) and EasyOCR with automatic motion-blur reduction,
document contrast enhancement, and offline text-to-speech narration.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import List, Tuple

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

os.environ["YOLO_CONFIG_DIR"] = str(BASE_DIR / ".ultralytics")
(BASE_DIR / ".ultralytics").mkdir(parents=True, exist_ok=True)

import cv2
import numpy as np

logger = logging.getLogger("SmartGlasses.DocReader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

OUTPUTS_DIR = BASE_DIR / "outputs" / "ocr_scans"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
EASYOCR_DIR = BASE_DIR / ".easyocr" / "model"
EASYOCR_DIR.mkdir(parents=True, exist_ok=True)


def calculate_image_sharpness(img: np.ndarray) -> float:
    """Calculates focus and sharpness variance using Laplacian operator."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def enhance_document_for_ocr(img: np.ndarray) -> np.ndarray:
    """Applies adaptive contrast equalization to bring out fine print and eliminate shadows."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Convert back to 3-channel for OCR engines expecting BGR
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


class TextToSpeechWorker:
    """Thread-safe offline speech engine using pyttsx3."""

    def __init__(self, speech_rate: int = 165):
        self.speech_rate = speech_rate
        self.lock = threading.Lock()
        self.current_thread: threading.Thread | None = None
        self._stop_flag = False

    def speak(self, text: str):
        if not text or not text.strip():
            return
        self.stop()
        self._stop_flag = False
        self.current_thread = threading.Thread(target=self._run_tts, args=(text,), daemon=True)
        self.current_thread.start()

    def _run_tts(self, text: str):
        with self.lock:
            try:
                import pyttsx3

                engine = pyttsx3.init()
                engine.setProperty("rate", self.speech_rate)
                voices = engine.getProperty("voices")
                if voices:
                    engine.setProperty("voice", voices[0].id)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                logger.error(f"TTS Error: {e}")

    def stop(self):
        self._stop_flag = True


class DocumentReader:
    """High-accuracy live document reader using RapidOCR (PP-OCRv4) or EasyOCR."""

    def __init__(self, camera_index: int = 0, engine_type: str = "rapidocr", languages: List[str] = None):
        self.camera_index = camera_index
        self.engine_type = engine_type.lower()
        self.languages = languages or ["en"]

        logger.info("=" * 65)
        logger.info("Initializing Smart Assistive Glasses Document Reader")
        logger.info(f"Primary OCR Engine : {self.engine_type.upper()}")
        logger.info("=" * 65)

        self.rapid_engine = None
        self.easy_engine = None

        if self.engine_type == "rapidocr":
            try:
                from rapidocr_onnxruntime import RapidOCR

                self.rapid_engine = RapidOCR()
                logger.info("RapidOCR (PP-OCRv4 ONNX) engine initialized successfully.")
            except Exception as e:
                logger.warning(f"Could not initialize RapidOCR: {e}. Falling back to EasyOCR.")
                self.engine_type = "easyocr"

        if self.engine_type == "easyocr":
            import easyocr
            import torch

            use_gpu = torch.cuda.is_available()
            self.easy_engine = easyocr.Reader(
                self.languages,
                gpu=use_gpu,
                model_storage_directory=str(EASYOCR_DIR),
                user_network_directory=str(EASYOCR_DIR),
            )
            logger.info("EasyOCR initialized successfully.")

        self.tts = TextToSpeechWorker()

        # State tracking
        self.is_processing = False
        self.status_message = "Ready. Place document in box and press [SPACE] to read."
        self.last_snap: np.ndarray | None = None
        self.detected_boxes: List[Tuple] = []  # (bbox, text, conf)
        self.full_text = ""

    def process_frame_async(self, frame_to_process: np.ndarray):
        """Runs OCR in a separate thread so camera feed remains smooth."""
        if self.is_processing:
            return

        self.is_processing = True
        self.status_message = "Analyzing document and extracting all text..."
        self.last_snap = frame_to_process.copy()
        self.detected_boxes = []
        self.full_text = ""

        thread = threading.Thread(target=self._do_ocr, args=(self.last_snap,), daemon=True)
        thread.start()

    def _do_ocr(self, raw_img: np.ndarray):
        t0 = time.time()
        try:
            enhanced_img = enhance_document_for_ocr(raw_img)
            boxes_data = []
            extracted_lines = []

            if self.engine_type == "rapidocr" and self.rapid_engine is not None:
                # RapidOCR (PP-OCRv4)
                results, _ = self.rapid_engine(enhanced_img)
                if results:
                    for bbox, text, conf in results:
                        clean_t = text.strip()
                        if float(conf) > 0.40 and clean_t:
                            extracted_lines.append(clean_t)
                            # Convert RapidOCR polygon points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] to int list
                            pts = [[int(pt[0]), int(pt[1])] for pt in bbox]
                            boxes_data.append((pts, clean_t, float(conf)))

            elif self.easy_engine is not None:
                # EasyOCR
                results = self.easy_engine.readtext(
                    enhanced_img,
                    detail=1,
                    paragraph=False,
                    contrast_ths=0.10,
                    adjust_contrast=0.60,
                    text_threshold=0.35,
                    low_text=0.25,
                    link_threshold=0.35,
                    mag_ratio=1.5,
                    slope_ths=0.25,
                    width_ths=0.60,
                )
                if results:
                    # Sort top-to-bottom, left-to-right
                    def sort_key(item):
                        top_y = min(pt[1] for pt in item[0])
                        left_x = min(pt[0] for pt in item[0])
                        return (int(top_y // 25), left_x)

                    sorted_res = sorted(results, key=sort_key)
                    for bbox, text, conf in sorted_res:
                        clean_t = text.strip()
                        if float(conf) > 0.20 and clean_t:
                            extracted_lines.append(clean_t)
                            pts = [[int(pt[0]), int(pt[1])] for pt in bbox]
                            boxes_data.append((pts, clean_t, float(conf)))

            elapsed = time.time() - t0
            self.detected_boxes = boxes_data
            self.full_text = " ".join(extracted_lines)

            if self.full_text:
                self.status_message = f"Extracted {len(extracted_lines)} lines/phrases in {elapsed:.1f}s. Reading aloud..."
                print("\n" + "=" * 65)
                print(f">>> [DOCUMENT TEXT EXTRACTED ({self.engine_type.upper()})]:")
                for line in extracted_lines:
                    print(f"  • {line}")
                print("=" * 65 + "\n")
                self.tts.speak(self.full_text)
            else:
                self.status_message = "No readable text detected. Please hold closer or steady."
                print("\n>>> [ASSISTIVE AUDIO]: No readable text detected. <<<\n")
                self.tts.speak("No readable text detected. Please bring the document closer.")

        except Exception as e:
            logger.error(f"OCR Exception: {e}")
            self.status_message = f"Error during OCR: {e}"
        finally:
            self.is_processing = False

    def save_current_scan(self):
        """Saves current snapped frame and text to outputs folder."""
        if self.last_snap is None or not self.full_text:
            logger.warning("No document text to save.")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        img_out = OUTPUTS_DIR / f"doc_scan_{timestamp}.jpg"
        txt_out = OUTPUTS_DIR / f"doc_scan_{timestamp}.txt"

        cv2.imwrite(str(img_out), self.last_snap)
        with open(txt_out, "w", encoding="utf-8") as f:
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Engine: {self.engine_type}\n")
            f.write(f"Extracted Text:\n{self.full_text}\n")

        self.status_message = f"Saved scan to {txt_out.name}"
        print(f"\n[SAVED] Document text saved to: {txt_out}")

    def clear(self):
        """Clears detections and stops speech."""
        self.tts.stop()
        self.detected_boxes = []
        self.full_text = ""
        self.status_message = "Cleared. Press [SPACE] to scan a new document."

    def run(self):
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_index)

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"Camera stream active: {actual_w}x{actual_h}")
        else:
            logger.warning(f"Camera index {self.camera_index} unavailable. Using fallback canvas.")
            cap = None

        window_name = f"Smart Glasses - Document Reader ({self.engine_type.upper()})"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1024, 680)

        fps = 0.0

        try:
            while True:
                t0 = time.time()
                if cap is not None and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                else:
                    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.putText(
                        frame,
                        "Hold Document in Front of Camera",
                        (100, 360),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        2,
                    )

                h, w = frame.shape[:2]
                display_frame = frame.copy()

                # Document Alignment Guide Box (helps user center the document for max focus)
                box_margin_x = int(w * 0.12)
                box_margin_y = int(h * 0.15)
                guide_color = (0, 255, 200) if not self.is_processing else (0, 165, 255)
                cv2.rectangle(
                    display_frame,
                    (box_margin_x, box_margin_y),
                    (w - box_margin_x, h - box_margin_y),
                    guide_color,
                    2,
                )
                cv2.putText(
                    display_frame,
                    "DOCUMENT TARGET ZONE",
                    (box_margin_x + 8, box_margin_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    guide_color,
                    1,
                    cv2.LINE_AA,
                )

                # Draw detected text bounding boxes
                for bbox, text, conf in self.detected_boxes:
                    pts = np.array(bbox, dtype=np.int32)
                    cv2.polylines(display_frame, [pts], isClosed=True, color=(0, 255, 100), thickness=2)
                    top_left = tuple(pts[0])
                    cv2.putText(
                        display_frame,
                        text,
                        (top_left[0], max(18, top_left[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

                # Top Status Banner
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 75), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.80, display_frame, 0.20, 0, display_frame)
                cv2.line(display_frame, (0, 75), (w, 75), (0, 200, 255), 2)

                cv2.putText(
                    display_frame,
                    f"SMART GLASSES - DOCUMENT READER ({self.engine_type.upper()})",
                    (15, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                status_color = (0, 200, 255) if self.is_processing else (0, 255, 150)
                cv2.putText(
                    display_frame,
                    f"Status: {self.status_message}",
                    (15, 54),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    status_color,
                    1,
                )

                # Bottom Controls Banner
                cv2.rectangle(display_frame, (0, h - 30), (w, h), (15, 15, 15), -1)
                help_text = "[SPACE] or [T] Snap & Read Aloud  |  [S] Save  |  [C] Clear  |  [M] Switch Engine  |  [Q] Quit"
                cv2.putText(
                    display_frame,
                    help_text,
                    (10, h - 9),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (220, 220, 220),
                    1,
                )

                # Bottom overlay for extracted speech text preview
                if self.full_text and not self.is_processing:
                    preview_box_h = 70
                    p_overlay = display_frame.copy()
                    cv2.rectangle(
                        p_overlay,
                        (0, h - 30 - preview_box_h),
                        (w, h - 30),
                        (30, 30, 30),
                        -1,
                    )
                    cv2.addWeighted(p_overlay, 0.85, display_frame, 0.15, 0, display_frame)
                    cv2.line(
                        display_frame,
                        (0, h - 30 - preview_box_h),
                        (w, h - 30 - preview_box_h),
                        (100, 100, 100),
                        1,
                    )

                    preview_snippet = self.full_text[:140] + ("..." if len(self.full_text) > 140 else "")
                    cv2.putText(
                        display_frame,
                        f"Speech: \"{preview_snippet}\"",
                        (12, h - 30 - preview_box_h + 38),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (255, 230, 100),
                        1,
                    )

                cv2.imshow(window_name, display_frame)

                dt = time.time() - t0
                fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 0.001))

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key in (ord(" "), ord("t"), ord("T")):
                    # Capture 3 quick frames and pick the sharpest one to eliminate motion blur
                    best_frame = frame
                    if cap is not None and cap.isOpened():
                        best_sharpness = calculate_image_sharpness(frame)
                        for _ in range(2):
                            r, f = cap.read()
                            if r:
                                s = calculate_image_sharpness(f)
                                if s > best_sharpness:
                                    best_sharpness = s
                                    best_frame = f
                    self.process_frame_async(best_frame)
                elif key in (ord("s"), ord("S")):
                    self.save_current_scan()
                elif key in (ord("c"), ord("C")):
                    self.clear()
                elif key in (ord("m"), ord("M")):
                    # Switch between RapidOCR and EasyOCR
                    self.engine_type = "easyocr" if self.engine_type == "rapidocr" else "rapidocr"
                    if self.engine_type == "easyocr" and self.easy_engine is None:
                        import easyocr
                        import torch
                        self.easy_engine = easyocr.Reader(
                            self.languages,
                            gpu=torch.cuda.is_available(),
                            model_storage_directory=str(EASYOCR_DIR),
                            user_network_directory=str(EASYOCR_DIR),
                        )
                    self.status_message = f"Switched OCR engine to {self.engine_type.upper()}"
                    print(f"\n[ENGINE SWITCH]: Now using {self.engine_type.upper()}")

        finally:
            self.tts.stop()
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Assistive Glasses Document & Text Reader")
    parser.add_argument("--source", type=int, default=0, help="Camera index")
    parser.add_argument("--engine", type=str, default="rapidocr", choices=["rapidocr", "easyocr"], help="OCR engine")
    parser.add_argument("--langs", nargs="+", default=["en"], help="Languages (e.g. en hi)")
    args = parser.parse_args()

    reader = DocumentReader(camera_index=args.source, engine_type=args.engine, languages=args.langs)
    reader.run()
