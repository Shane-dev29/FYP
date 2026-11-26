"""Comprehensive benchmarking pipeline measuring accuracy, latency, memory, VRAM, and GFLOPs."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import psutil
import torch
from ultralytics import YOLO

logger = logging.getLogger("SmartGlasses.Benchmark")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_YAML = BASE_DIR / "training" / "data.yaml"


def get_vram_usage_mb() -> float:
    """Returns current allocated GPU memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def benchmark_single_model(
    model_path: Path,
    device: str = "cuda:0",
    imgsz: int = 640,
    warmup_iters: int = 15,
    test_iters: int = 100,
    eval_accuracy: bool = True,
) -> Dict[str, Any]:
    """Measures precise empirical performance of a model under controlled conditions."""
    logger.info(f"--- Benchmarking {model_path.name} on {device.upper()} ---")

    # Clean memory before benchmark
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = YOLO(str(model_path))
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    # Count parameters & GFLOPs
    params = sum(p.numel() for p in model.model.parameters())

    # Create dummy frame for latency profiling
    dummy_input = np.random.randint(0, 256, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    for _ in range(warmup_iters):
        _ = model.predict(dummy_input, imgsz=imgsz, device=device, verbose=False)

    latencies = []
    process = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss / (1024 * 1024)
    cpu_percentages = []

    # Measurement loop
    t_start = time.perf_counter()
    for _ in range(test_iters):
        cpu_percentages.append(psutil.cpu_percent(interval=None))
        t0 = time.perf_counter()
        _ = model.predict(dummy_input, imgsz=imgsz, device=device, verbose=False)
        if "cuda" in device and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    ram_after = process.memory_info().rss / (1024 * 1024)
    peak_vram_mb = get_vram_usage_mb() if "cuda" in device else 0.0

    lat_avg = float(np.mean(latencies))
    lat_median = float(np.median(latencies))
    lat_p95 = float(np.percentile(latencies, 95))
    fps = 1000.0 / lat_avg if lat_avg > 0 else 0.0
    avg_cpu = float(np.mean(cpu_percentages)) if cpu_percentages else 0.0

    # Accuracy validation if requested
    acc_metrics = {"mAP50": None, "mAP50_95": None, "precision": None, "recall": None, "f1": None}
    if eval_accuracy and DATA_YAML.exists():
        try:
            val_res = model.val(data=str(DATA_YAML), imgsz=imgsz, device=device, split="val", verbose=False)
            mp = float(val_res.box.mp)
            mr = float(val_res.box.mr)
            f1 = 2 * (mp * mr) / (mp + mr + 1e-7)
            acc_metrics = {
                "mAP50": round(float(val_res.box.map50), 4),
                "mAP50_95": round(float(val_res.box.map), 4),
                "precision": round(mp, 4),
                "recall": round(mr, 4),
                "f1": round(f1, 4),
            }
        except Exception as e:
            logger.warning(f"Could not compute validation metrics for {model_path.name}: {e}")

    result = {
        "model": model_path.name,
        "device": device,
        "parameters": params,
        "file_size_mb": round(file_size_mb, 2),
        "latency_avg_ms": round(lat_avg, 2),
        "latency_median_ms": round(lat_median, 2),
        "latency_p95_ms": round(lat_p95, 2),
        "fps": round(fps, 1),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "ram_usage_mb": round(ram_after, 1),
        "cpu_util_pct": round(avg_cpu, 1),
        **acc_metrics,
    }

    logger.info(
        f"Result: {model_path.name} ({device}) -> Latency: {lat_avg:.2f}ms | FPS: {fps:.1f} | "
        f"mAP50: {acc_metrics['mAP50']} | mAP50-95: {acc_metrics['mAP50_95']}"
    )
    return result


def run_benchmarks(
    models: List[str],
    devices: List[str] = ["cuda:0", "cpu"],
    imgsz: int = 640,
    iterations: int = 60,
    eval_acc: bool = True,
) -> pd.DataFrame:
    """Runs systematic benchmarks over all specified models and devices."""
    results = []
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    for m_name in models:
        m_path = MODELS_DIR / m_name
        if not m_path.exists():
            logger.warning(f"Skipping {m_name}, file not found.")
            continue

        for dev in devices:
            if "cuda" in dev and not torch.cuda.is_available():
                continue
            res = benchmark_single_model(
                model_path=m_path,
                device=dev,
                imgsz=imgsz,
                warmup_iters=10,
                test_iters=iterations,
                eval_accuracy=eval_acc and ("cuda" in dev or len(devices) == 1),
            )
            results.append(res)

    df = pd.DataFrame(results)
    json_path = OUTPUTS_DIR / "benchmark_results.json"
    csv_path = OUTPUTS_DIR / "benchmark_summary.csv"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    df.to_csv(csv_path, index=False)

    logger.info(f"Saved benchmark results to {json_path} and {csv_path}")
    print("\n" + "=" * 90)
    print("  EMPIRICAL BENCHMARK SUMMARY TABLE")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90 + "\n")

    return df


def main():
    parser = argparse.ArgumentParser(description="Systematic YOLO Benchmarking Suite.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--iterations", type=int, default=50, help="Number of test iterations")
    parser.add_argument("--cpu-only", action="store_true", help="Benchmark only CPU")
    parser.add_argument("--gpu-only", action="store_true", help="Benchmark only GPU")
    parser.add_argument("--skip-acc", action="store_true", help="Skip accuracy validation")
    args = parser.parse_args()

    candidate_models = [
        "yolo26n.pt",
        "yolo11n.pt",
        "yolov10n.pt",
        "yolov8n.pt",
        "yolov9t.pt",
        "yolov5nu.pt",
        "yolo12n.pt",
    ]

    # Include any fine-tuned models if present
    for pt in MODELS_DIR.glob("*_finetuned.pt"):
        if pt.name not in candidate_models:
            candidate_models.append(pt.name)

    devices = []
    if not args.cpu_only and torch.cuda.is_available():
        devices.append("cuda:0")
    if not args.gpu_only:
        devices.append("cpu")

    run_benchmarks(
        models=candidate_models,
        devices=devices,
        imgsz=args.imgsz,
        iterations=args.iterations,
        eval_acc=not args.skip_acc,
    )


if __name__ == "__main__":
    main()

