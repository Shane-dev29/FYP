"""Academic visualization generator for comparative benchmark reports."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger("SmartGlasses.Reports")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"


def generate_all_plots(json_path: Path = OUTPUTS_DIR / "benchmark_results.json"):
    """Generates all academic comparison plots from empirical benchmark data."""
    if not json_path.exists():
        logger.error(f"Benchmark file not found: {json_path}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Clean model names for display
    df["display_name"] = df["model"].str.replace(".pt", "", regex=False)

    sns.set_theme(style="whitegrid", font="sans-serif")

    # 1. ACCURACY COMPARISON (mAP50 & mAP50-95)
    acc_df = df[df["mAP50"].notnull()].drop_duplicates(subset=["display_name"])
    if not acc_df.empty:
        plt.figure(figsize=(11, 5.5))
        x = np.arange(len(acc_df))
        width = 0.35

        plt.bar(x - width/2, acc_df["mAP50"], width=width, label="mAP@50", color="#1f77b4", edgecolor="black", alpha=0.9)
        plt.bar(x + width/2, acc_df["mAP50_95"], width=width, label="mAP@50-95", color="#2ca02c", edgecolor="black", alpha=0.9)

        plt.xticks(x, acc_df["display_name"], rotation=25, ha="right", fontsize=11, fontweight="bold")
        plt.ylabel("Detection Accuracy (mAP)", fontsize=12)
        plt.title("Object Detection Accuracy on Indoor Dataset (Validation Split)", fontsize=13, fontweight="bold", pad=12)
        plt.ylim(0, max(acc_df["mAP50"].max() * 1.15, 0.5))
        plt.legend(frameon=True, facecolor="white", loc="upper left")
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        for i in range(len(acc_df)):
            plt.text(i - width/2, acc_df["mAP50"].iloc[i] + 0.01, f"{acc_df['mAP50'].iloc[i]:.3f}", ha="center", fontsize=8.5)
            plt.text(i + width/2, acc_df["mAP50_95"].iloc[i] + 0.01, f"{acc_df['mAP50_95'].iloc[i]:.3f}", ha="center", fontsize=8.5)

        plt.tight_layout()
        out1 = OUTPUTS_DIR / "accuracy_comparison.png"
        plt.savefig(out1, dpi=300)
        plt.close()
        logger.info(f"Generated: {out1}")

    # 2. LATENCY COMPARISON (GPU vs CPU)
    plt.figure(figsize=(11, 5.5))
    gpu_df = df[df["device"].str.contains("cuda")].set_index("display_name")
    cpu_df = df[df["device"] == "cpu"].set_index("display_name")

    models = sorted(list(set(df["display_name"])))
    x = np.arange(len(models))
    width = 0.35

    gpu_lats = [gpu_df.loc[m]["latency_avg_ms"] if m in gpu_df.index else 0 for m in models]
    cpu_lats = [cpu_df.loc[m]["latency_avg_ms"] if m in cpu_df.index else 0 for m in models]

    plt.bar(x - width/2, gpu_lats, width=width, label="NVIDIA RTX 3050 Laptop GPU", color="#76b900", edgecolor="black", alpha=0.9)
    plt.bar(x + width/2, cpu_lats, width=width, label="Host CPU", color="#0071c5", edgecolor="black", alpha=0.9)

    plt.xticks(x, models, rotation=25, ha="right", fontsize=11, fontweight="bold")
    plt.ylabel("Inference Latency (ms) - Lower is Better", fontsize=12)
    plt.title("Inference Latency: RTX 3050 vs CPU Across YOLO Architectures", fontsize=13, fontweight="bold", pad=12)
    plt.legend(frameon=True, facecolor="white")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for i in range(len(models)):
        if gpu_lats[i] > 0:
            plt.text(i - width/2, gpu_lats[i] + 1, f"{gpu_lats[i]:.1f}", ha="center", fontsize=8.5)
        if cpu_lats[i] > 0:
            plt.text(i + width/2, cpu_lats[i] + 1, f"{cpu_lats[i]:.1f}", ha="center", fontsize=8.5)

    plt.tight_layout()
    out2 = OUTPUTS_DIR / "latency_comparison.png"
    plt.savefig(out2, dpi=300)
    plt.close()
    logger.info(f"Generated: {out2}")

    # 3. FPS COMPARISON (Frames Per Second)
    plt.figure(figsize=(11, 5.5))
    gpu_fps = [gpu_df.loc[m]["fps"] if m in gpu_df.index else 0 for m in models]
    cpu_fps = [cpu_df.loc[m]["fps"] if m in cpu_df.index else 0 for m in models]

    plt.bar(x - width/2, gpu_fps, width=width, label="RTX 3050 GPU", color="#2ca02c", edgecolor="black", alpha=0.9)
    plt.bar(x + width/2, cpu_fps, width=width, label="CPU", color="#ff7f0e", edgecolor="black", alpha=0.9)

    # Reference Real-Time Line at 30 FPS
    plt.axhline(y=30, color="red", linestyle=":", linewidth=2, label="30 FPS Real-Time Threshold")

    plt.xticks(x, models, rotation=25, ha="right", fontsize=11, fontweight="bold")
    plt.ylabel("Frames Per Second (FPS) - Higher is Better", fontsize=12)
    plt.title("Throughput Comparison: RTX 3050 GPU vs CPU", fontsize=13, fontweight="bold", pad=12)
    plt.legend(frameon=True, facecolor="white")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for i in range(len(models)):
        if gpu_fps[i] > 0:
            plt.text(i - width/2, gpu_fps[i] + 1, f"{gpu_fps[i]:.0f}", ha="center", fontsize=8.5)
        if cpu_fps[i] > 0:
            plt.text(i + width/2, cpu_fps[i] + 0.5, f"{cpu_fps[i]:.1f}", ha="center", fontsize=8.5)

    plt.tight_layout()
    out3 = OUTPUTS_DIR / "fps_comparison.png"
    plt.savefig(out3, dpi=300)
    plt.close()
    logger.info(f"Generated: {out3}")

    # 4. MODEL SIZE & PARAMETER FOOTPRINT
    uniq_df = df.drop_duplicates(subset=["display_name"]).sort_values("parameters")
    fig, ax1 = plt.subplots(figsize=(11, 5.5))

    x_pos = np.arange(len(uniq_df))
    w = 0.35

    p1 = ax1.bar(x_pos - w/2, uniq_df["parameters"] / 1e6, width=w, color="#8c564b", edgecolor="black", label="Parameters (Millions)")
    ax1.set_ylabel("Parameters (Millions)", color="#8c564b", fontsize=12, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#8c564b")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(uniq_df["display_name"], rotation=25, ha="right", fontsize=11, fontweight="bold")

    ax2 = ax1.twinx()
    p2 = ax2.bar(x_pos + w/2, uniq_df["file_size_mb"], width=w, color="#9467bd", edgecolor="black", label="Model Size (MB)")
    ax2.set_ylabel("Disk Footprint (MB)", color="#9467bd", fontsize=12, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#9467bd")

    plt.title("Model Computational Footprint: Parameters vs Storage", fontsize=13, fontweight="bold", pad=12)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Combined legend
    lines = [p1, p2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=True, facecolor="white")

    plt.tight_layout()
    out4 = OUTPUTS_DIR / "model_size_params.png"
    plt.savefig(out4, dpi=300)
    plt.close()
    logger.info(f"Generated: {out4}")

    # 5. PARETO OPTIMALITY: SPEED VS ACCURACY FRONTIER
    pareto_df = df[(df["device"] == "cpu") & (df["mAP50_95"].notnull())]
    if not pareto_df.empty and len(pareto_df) > 1:
        plt.figure(figsize=(9, 6))
        for _, row in pareto_df.iterrows():
            plt.scatter(row["latency_avg_ms"], row["mAP50_95"], s=180, alpha=0.85, edgecolors="black", linewidth=1.5)
            plt.annotate(
                row["display_name"],
                (row["latency_avg_ms"], row["mAP50_95"]),
                textcoords="offset points",
                xytext=(8, 6),
                ha="left",
                fontsize=10,
                fontweight="bold",
            )

        plt.xlabel("CPU Latency (ms) - Lower is Faster", fontsize=12)
        plt.ylabel("mAP@50-95 - Higher is More Accurate", fontsize=12)
        plt.title("Speed vs Accuracy Trade-off (CPU Inference for Edge Deployment)", fontsize=13, fontweight="bold", pad=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        out5 = OUTPUTS_DIR / "pareto_tradeoff.png"
        plt.savefig(out5, dpi=300)
        plt.close()
        logger.info(f"Generated: {out5}")

    print("\nAll academic comparison plots successfully generated in:", OUTPUTS_DIR)


if __name__ == "__main__":
    generate_all_plots()

