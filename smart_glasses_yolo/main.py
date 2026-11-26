"""Main entrypoint for Smart Assistive Glasses Object Detection System."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2

# Ensure local package import
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.camera import CameraStream
from src.config import CameraConfig, DetectorConfig
from src.detector import SmartGlassesDetector
from src.visualization import Visualizer

logger = logging.getLogger("SmartGlasses.Main")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def discover_models(models_dir: Path) -> List[Path]:
    """Finds available YOLO weights for interactive switching."""
    if not models_dir.exists():
        return []
    pt_files = sorted(list(models_dir.glob("*.pt")))
    return pt_files


def run_pipeline(
    model_path: str,
    source: str = "0",
    conf: float = 0.35,
    device: Optional[str] = None,
    imgsz: int = 640,
    headless: bool = False,
    max_frames: Optional[int] = None,
    output_log: Optional[str] = None,
) -> None:
    """Executes the real-time detection loop with interactive controls."""
    models_dir = Path(__file__).resolve().parent / "models"
    available_models = discover_models(models_dir)

    active_model_path = Path(model_path)
    if not active_model_path.is_absolute() and not active_model_path.exists():
        candidate = models_dir / active_model_path.name
        if candidate.exists():
            active_model_path = candidate

    det_cfg = DetectorConfig(
        model_path=active_model_path,
        confidence_threshold=conf,
        image_size=imgsz,
    )
    if device:
        det_cfg.device = device

    logger.info(f"Initializing detector with model: {active_model_path} on device: {det_cfg.device}")
    detector = SmartGlassesDetector(det_cfg)
    visualizer = Visualizer()

    # Source handling (integer camera index vs file path)
    cam_source = int(source) if source.isdigit() else source
    cam = CameraStream(source=cam_source, config=CameraConfig())

    json_log_file = None
    if output_log:
        out_p = Path(output_log)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        json_log_file = open(out_p, "a", encoding="utf-8")
        logger.info(f"Logging structured detections to: {out_p}")

    model_idx = 0
    if available_models and active_model_path in available_models:
        model_idx = available_models.index(active_model_path)

    frame_count = 0
    fps_start_time = time.time()
    current_fps = 0.0

    print("\n" + "=" * 60)
    print("  SMART ASSISTIVE GLASSES - OBJECT DETECTION SUBSYSTEM")
    print("=" * 60)
    print(f" Active Model   : {active_model_path.name}")
    print(f" Device         : {det_cfg.device}")
    print(f" Confidence     : {detector.config.confidence_threshold:.2f}")
    print(f" Input Source   : {source}")
    print(f" Mode           : {'Headless (CLI/Testing)' if headless else 'Interactive Display'}")
    if not headless:
        print(" Hotkeys        : [Q] Quit | [+] Conf + | [-] Conf - | [M] Cycle Model | [S] Save Snapshot")
    print("=" * 60 + "\n")

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                logger.info("End of video stream or cannot capture frame.")
                break

            frame_count += 1
            if frame_count % 15 == 0:
                elapsed = time.time() - fps_start_time
                if elapsed > 0:
                    current_fps = 15.0 / elapsed
                fps_start_time = time.time()

            # Run detection & obtain structured output
            annotated_frame, detections = detector.detect(
                frame=frame,
                frame_id=frame_count,
            )

            # Write structured JSON log if requested
            if json_log_file:
                json_log_file.write(detections.to_json() + "\n")
                json_log_file.flush()

            if headless:
                # In headless mode, log periodically
                if frame_count % 10 == 0:
                    logger.info(
                        f"Frame {frame_count:4d} | FPS: {current_fps:4.1f} | "
                        f"Detections: {len(detections.detections):2d} | "
                        f"Latency: {detections.metadata['latency_breakdown_ms']['total']:.1f}ms"
                    )
            else:
                # Render HUD
                hud_frame = visualizer.draw_hud(
                    frame=annotated_frame,
                    detections=detections,
                    current_fps=current_fps,
                    conf_threshold=detector.config.confidence_threshold,
                    model_name=detector.config.model_name,
                    device=detector.config.device,
                )

                cv2.imshow("Smart Assistive Glasses - Object Detection HUD", hud_frame)
                key = cv2.waitKey(1) & 0xFF

                if key in [ord("q"), ord("Q"), 27]:  # 27 is ESC
                    logger.info("Exit command received.")
                    break
                elif key in [ord("+"), ord("=")]:
                    detector.set_confidence(detector.config.confidence_threshold + 0.05)
                elif key in [ord("-"), ord("_")]:
                    detector.set_confidence(detector.config.confidence_threshold - 0.05)
                elif key in [ord("m"), ord("M")] and available_models:
                    model_idx = (model_idx + 1) % len(available_models)
                    next_model = available_models[model_idx]
                    logger.info(f"Switching model to: {next_model.name}")
                    detector.switch_model(next_model)
                elif key in [ord("s"), ord("S")]:
                    snap_path = Path(__file__).parent / "outputs" / f"snapshot_frame_{frame_count}.jpg"
                    snap_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(snap_path), hud_frame)
                    logger.info(f"Saved snapshot to: {snap_path}")
                    # Also log the JSON
                    print(f"\n[FRAME {frame_count} STRUCTURED OUTPUT]:")
                    print(detections.to_json(indent=2))

            if max_frames and frame_count >= max_frames:
                logger.info(f"Reached requested limit of {max_frames} frames. Halting.")
                break

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        cam.release()
        if not headless:
            cv2.destroyAllWindows()
        if json_log_file:
            json_log_file.close()
        logger.info(f"Pipeline stopped. Processed {frame_count} frames total.")


def main():
    parser = argparse.ArgumentParser(
        description="Smart Assistive Glasses Object Detection Subsystem (YOLO Prototype)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_model = Path(__file__).resolve().parent / "models" / "yolo26n.pt"
    parser.add_argument("--model", type=str, default=str(default_model), help="Path to YOLO .pt model")
    parser.add_argument("--source", type=str, default="0", help="Webcam index ('0') or video/image path")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--device", type=str, default=None, help="Execution device ('cuda:0' or 'cpu')")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--headless", action="store_true", help="Run without graphical display window")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after processing N frames")
    parser.add_argument("--output-log", type=str, default=None, help="Filepath for structured JSON detections stream")

    args = parser.parse_args()
    run_pipeline(
        model_path=args.model,
        source=args.source,
        conf=args.conf,
        device=args.device,
        imgsz=args.imgsz,
        headless=args.headless,
        max_frames=args.max_frames,
        output_log=args.output_log,
    )


if __name__ == "__main__":
    main()

