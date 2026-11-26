"""Raspberry Pi 5 Edge Benchmarking Suite.
Executes repeated, reproducible latency and resource benchmarking directly on ARM edge devices.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import psutil
from ultralytics import YOLO

logger = logging.getLogger("SmartGlasses.BenchmarkPi")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"


def get_cpu_temp() -> Optional[float]:
    """Reads Linux sysfs thermal sensor."""
    p = Path("/sys/class/thermal/thermal_zone0/temp")
    if p.exists():
        try:
            return round(float(p.read_text().strip()) / 1000.0, 1)
        except Exception:
            pass
    return None


def get_throttled() -> str:
    """Checks Raspberry Pi vcgencmd throttled state."""
    try:
        res = subprocess.check_output(["vcgencmd", "get_throttled"], text=True).strip()
        return res
    except Exception:
        return "N/A (Non-RPi or vcgencmd unavailable)"


def run_pi_benchmark(
    model_path: str,
    imgsz: int = 416,
    iterations: int = 50,
    warmup: int = 10,
) -> Dict[str, Any]:
    """Runs controlled inference benchmarking on CPU / ARM edge processor."""
    m_path = Path(model_path)
    if not m_path.exists():
        raise FileNotFoundError(f"Model path not found: {m_path}")

    logger.info(f"Starting Pi Benchmark: {m_path.name} @ {imgsz}x{imgsz} ({iterations} iterations)")
    model = YOLO(str(m_path))

    dummy_input = np.random.randint(0, 256, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    for _ in range(warmup):
        _ = model.predict(dummy_input, imgsz=imgsz, device="cpu", verbose=False)

    latencies = []
    cpu_usages = []
    temps = []
    process = psutil.Process(os.getpid())

    # Measurement loop
    for _ in range(iterations):
        cpu_usages.append(psutil.cpu_percent(interval=None))
        t0 = time.perf_counter()
        _ = model.predict(dummy_input, imgsz=imgsz, device="cpu", verbose=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

        t_val = get_cpu_temp()
        if t_val:
            temps.append(t_val)

    avg_lat = float(np.mean(latencies))
    med_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    fps = 1000.0 / avg_lat if avg_lat > 0 else 0.0
    ram_mb = float(process.memory_info().rss / (1024 * 1024))
    avg_cpu = float(np.mean(cpu_usages)) if cpu_usages else 0.0
    avg_temp = float(np.mean(temps)) if temps else None

    # Model file or folder size
    if m_path.is_file():
        size_mb = m_path.stat().st_size / (1024 * 1024)
    else:
        size_mb = sum(f.stat().st_size for f in m_path.glob("**/*") if f.is_file()) / (1024 * 1024)

    res = {
        "model": m_path.name,
        "format": m_path.suffix.replace(".", "") if m_path.is_file() else "dir",
        "imgsz": imgsz,
        "avg_latency_ms": round(avg_lat, 2),
        "median_latency_ms": round(med_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "fps": round(fps, 1),
        "ram_usage_mb": round(ram_mb, 1),
        "cpu_util_pct": round(avg_cpu, 1),
        "avg_cpu_temp_c": avg_temp,
        "throttling": get_throttled(),
        "model_size_mb": round(size_mb, 2),
        "platform": f"{platform.system()} {platform.machine()}",
    }

    logger.info(
        f"Pi Benchmark Complete: {m_path.name} ({imgsz}x{imgsz}) -> "
        f"Latency: {avg_lat:.1f}ms | FPS: {fps:.1f} | RAM: {ram_mb:.1f}MB"
    )
    return res


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi Edge Model Benchmarking")
    parser.add_argument("--models", nargs="+", required=True, help="List of model paths to benchmark")
    parser.add_argument("--resolutions", nargs="+", type=int, default=[416, 512, 640], help="Input resolutions")
    parser.add_argument("--iterations", type=int, default=40, help="Number of benchmark iterations")
    parser.add_argument("--output", type=str, default="outputs/rpi_benchmark_results.json", help="Output JSON path")
    args = parser.parse_args()

    results = []
    for m in args.models:
        for r in args.resolutions:
            try:
                r_res = run_pi_benchmark(model_path=m, imgsz=r, iterations=args.iterations)
                results.append(r_res)
            except Exception as e:
                logger.error(f"Failed benchmark for {m} @ {r}: {e}")

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved Raspberry Pi benchmark summary to {out_p}")


if __name__ == "__main__":
    main()

