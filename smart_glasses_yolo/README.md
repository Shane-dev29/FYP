# Smart Assistive Glasses – YOLO Indoor Object Detection & Comparative Research

Final-Year Computer Engineering Project: **Smart Assistive Glasses for Visually Impaired Individuals**

This repository contains the software prototype, comparative research study, dataset pipelines, benchmarking suite, and edge deployment runtimes for the visual object detection subsystem.

---

## Repository Structure

```
smart_glasses_yolo/
│
├── models/                         # Checkpoints & exported edge models
│   ├── yolo26n.pt                  # Pretrained YOLO26 Nano baseline
│   ├── yolo26n_finetuned.pt        # Fine-tuned 13-class indoor model (RTX 3050)
│   ├── yolo26n_finetuned.onnx      # Exported ONNX Runtime model for Raspberry Pi
│   ├── yolo11n.pt                  # Pretrained YOLO11 Nano baseline
│   ├── yolo11n_finetuned.pt        # Fine-tuned comparative model
│   └── ...                         # YOLOv10n, YOLOv8n, YOLOv9t, YOLOv5nu, YOLO12n
│
├── datasets/
│   ├── indoor_unified/             # 13-class unified indoor dataset
│   │   ├── images/ (train, val)    # 2,333 train, 417 val images
│   │   └── labels/ (train, val)    # Harmonized YOLO format annotations
│   └── homeobjects-3K/             # Base HomeObjects-3K dataset
│
├── src/                            # Modular detector package
│   ├── __init__.py
│   ├── config.py                   # Configuration parameters
│   ├── detector.py                 # Core detector & structured telemetry
│   ├── camera.py                   # Camera stream & synthetic test generator
│   └── visualization.py            # HUD overlays & bounding box rendering
│
├── training/
│   ├── data.yaml                   # Unified dataset configuration
│   ├── prepare_dataset.py          # Data curation, class mapping & person integration
│   ├── train.py                    # Multi-model fine-tuning pipeline
│   └── evaluate.py                 # Strict mAP & latency validation script
│
├── benchmarking/
│   ├── benchmark_models.py         # Multi-model GPU/CPU systematic benchmark suite
│   ├── benchmark_pi.py             # Dedicated ARM / Raspberry Pi benchmark suite
│   └── generate_reports.py         # Academic comparison plot generator
│
├── testing/
│   └── test_model.py               # Robustness stress tester (glare, low light, occlusion)
│
├── deployment/
│   ├── export_model.py             # Export to ONNX, NCNN, OpenVINO, TFLite
│   └── raspberry_pi_inference.py   # Low-latency Raspberry Pi 5 edge runtime
│
├── research/
│   └── YOLO_Comparative_Study.md   # Comprehensive academic research report
│
├── outputs/                        # Benchmark plots, CSVs, and evaluation JSONs
│   ├── accuracy_comparison.png
│   ├── latency_comparison.png
│   ├── fps_comparison.png
│   ├── model_size_params.png
│   ├── class_distribution.png
│   └── benchmark_summary.csv
│
├── requirements.txt
├── README.md
└── main.py                         # Interactive & headless CLI entrypoint
```

---

## Quickstart Guide

### 1. Environment Setup
```bash
# Install core dependencies
pip install -r requirements.txt
```

### 2. Run Real-Time Object Detection Prototype
```bash
# Run with webcam and interactive HUD (or synthetic test stream if camera unavailable)
python smart_glasses_yolo/main.py --model smart_glasses_yolo/models/yolo26n_finetuned.pt --conf 0.35

# Interactive Hotkeys:
# [Q] Quit
# [+] Increase Confidence (+0.05)
# [-] Decrease Confidence (-0.05)
# [M] Cycle through available models in models/
# [S] Save Snapshot & print structured telemetry
```

### 3. Run Headless Integration Test
```bash
python smart_glasses_yolo/main.py \
  --model smart_glasses_yolo/models/yolo26n_finetuned.pt \
  --headless \
  --max-frames 50 \
  --output-log smart_glasses_yolo/outputs/detections_stream.jsonl
```

### 4. Fine-Tune Models on Unified Indoor Dataset
```bash
# Fine-tune YOLO26n on RTX 3050 Laptop GPU
python smart_glasses_yolo/training/train.py --model yolo26n --epochs 10 --batch 16 --device 0

# Fine-tune YOLO11n for comparison
python smart_glasses_yolo/training/train.py --model yolo11n --epochs 10 --batch 16 --device 0
```

### 5. Run Controlled Model Benchmark Suite
```bash
python smart_glasses_yolo/benchmarking/benchmark_models.py --iterations 50
python smart_glasses_yolo/benchmarking/generate_reports.py
```

### 6. Export for Raspberry Pi Edge Deployment
```bash
# Export fine-tuned YOLO26n to ONNX at 416x416 resolution
python smart_glasses_yolo/deployment/export_model.py \
  --model smart_glasses_yolo/models/yolo26n_finetuned.pt \
  --format onnx \
  --imgsz 416
```

### 7. Run on Raspberry Pi 5 (8GB RAM)
```bash
python smart_glasses_yolo/deployment/raspberry_pi_inference.py \
  --model smart_glasses_yolo/models/yolo26n_finetuned.onnx \
  --imgsz 416 \
  --camera 0 \
  --output-log detections.jsonl
```

---

## Modular Structured Output Specification

To support future integration with **Depth Sensing** and **Haptic Feedback** modules, the detector exposes a clean JSON stream:

```json
{
  "frame_id": 120,
  "timestamp": 1793846757802,
  "detections": [
    {
      "class_name": "chair",
      "class_id": 2,
      "confidence": 0.9412,
      "bbox": [100, 120, 300, 400]
    },
    {
      "class_name": "person",
      "class_id": 12,
      "confidence": 0.9715,
      "bbox": [320, 80, 510, 460]
    }
  ],
  "metadata": {
    "model_name": "yolo26n_finetuned.onnx",
    "latency_breakdown_ms": {
      "preprocess": 0.85,
      "inference": 24.10,
      "postprocess": 1.25,
      "total": 26.20
    },
    "total_detections": 2
  }
}
```

---

## Research Summary & Key Findings

| Architecture | Parameters | GFLOPs | Precision | mAP@50 | mAP@50-95 | GPU Latency | ONNX CPU Latency (416) | FPS (Pi 5) |
|---|---|---|---|---|---|---|---|---|
| **YOLO26n (Fine-Tuned)** | **2.39M** | **5.4** | **68.5%** | **66.8%** | **46.5%** | 31.2 ms | **24.1 ms** | **41.5 FPS** |
| **YOLO11n (Fine-Tuned)** | 2.60M | 6.5 | 58.3% | 64.3% | 44.0% | 26.6 ms | 31.8 ms | 31.4 FPS |
| **YOLOv10n (COCO)** | 2.78M | 8.2 | -- | -- | -- | 25.8 ms | 38.5 ms | 26.0 FPS |
| **YOLOv8n (COCO)** | 3.16M | 8.7 | -- | -- | -- | 26.5 ms | 35.2 ms | 28.4 FPS |

**Definitive Conclusion:** YOLO26n provides the optimal balance for assistive smart glasses: it achieves the highest precision (+10.2% over YOLO11n), smallest parameter footprint (2.39M), lowest GFLOPs (5.4), and achieves **41.5 FPS** on CPU edge hardware via ONNX Runtime at $416\times416$.

