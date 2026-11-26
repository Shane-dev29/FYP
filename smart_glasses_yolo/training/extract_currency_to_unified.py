"""Extracts a balanced subset of Indian Currency Dataset from archive (2).zip
and integrates directly into the unified 20-class dataset (indoor obstacles + currency).
"""

from __future__ import annotations

import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ZIP_PATH = r"C:\Users\remed\Downloads\archive (2).zip"
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "indoor_unified"
IMG_TRAIN_DIR = DATASET_DIR / "images" / "train"
LBL_TRAIN_DIR = DATASET_DIR / "labels" / "train"
IMG_VAL_DIR = DATASET_DIR / "images" / "val"
LBL_VAL_DIR = DATASET_DIR / "labels" / "val"

CURRENCY_MAP = {
    "10": 13,
    "20": 14,
    "50": 15,
    "100": 16,
    "200": 17,
    "500": 18,
    "2000": 19,
}

MAX_TRAIN_PER_CLASS = 60
MAX_VAL_PER_CLASS = 15


def extract_and_integrate():
    print("=" * 60)
    print("Integrating Indian Currency Dataset into Unified Detector")
    print(f"Source Archive : {ZIP_PATH}")
    print(f"Target Dataset : {DATASET_DIR}")
    print("=" * 60)

    if not os.path.exists(ZIP_PATH):
        print(f"ERROR: Archive not found at {ZIP_PATH}", file=sys.stderr)
        sys.exit(1)

    for d in [IMG_TRAIN_DIR, LBL_TRAIN_DIR, IMG_VAL_DIR, LBL_VAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    train_counts = defaultdict(int)
    val_counts = defaultdict(int)

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        namelist = zf.namelist()
        print(f"Total entries in zip archive: {len(namelist)}")

        for entry in namelist:
            if not entry.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            # Standard path: Indian_Currency_Dataset/Train/<class>/<file> or Indian_Currency_Dataset/Test/<class>/<file>
            parts = [p for p in entry.split("/") if p]
            if len(parts) < 4:
                continue

            split_type = parts[1]  # Train or Test
            cls_name = parts[2]    # 10, 20, 50, etc.
            filename = parts[3]

            if cls_name not in CURRENCY_MAP:
                continue

            class_id = CURRENCY_MAP[cls_name]

            if split_type.lower() == "train" and train_counts[cls_name] < MAX_TRAIN_PER_CLASS:
                idx = train_counts[cls_name]
                out_img_name = f"curr_{cls_name}_tr_{idx:03d}.jpg"
                out_lbl_name = f"curr_{cls_name}_tr_{idx:03d}.txt"

                # Write image
                img_data = zf.read(entry)
                with open(IMG_TRAIN_DIR / out_img_name, "wb") as f:
                    f.write(img_data)

                # Write bounding box label (center: 0.5, 0.5, width: 0.90, height: 0.85)
                with open(LBL_TRAIN_DIR / out_lbl_name, "w", encoding="utf-8") as f:
                    f.write(f"{class_id} 0.500000 0.500000 0.900000 0.850000\n")

                train_counts[cls_name] += 1

            elif split_type.lower() in ("test", "val") and val_counts[cls_name] < MAX_VAL_PER_CLASS:
                idx = val_counts[cls_name]
                out_img_name = f"curr_{cls_name}_val_{idx:03d}.jpg"
                out_lbl_name = f"curr_{cls_name}_val_{idx:03d}.txt"

                # Write image
                img_data = zf.read(entry)
                with open(IMG_VAL_DIR / out_img_name, "wb") as f:
                    f.write(img_data)

                # Write bounding box label
                with open(LBL_VAL_DIR / out_lbl_name, "w", encoding="utf-8") as f:
                    f.write(f"{class_id} 0.500000 0.500000 0.900000 0.850000\n")

                val_counts[cls_name] += 1

    print("\nIntegration Summary:")
    print("-" * 50)
    print(f"{'Denomination':<15} {'Class ID':<10} {'Train Samples':<15} {'Val Samples':<15}")
    print("-" * 50)
    for c, cid in CURRENCY_MAP.items():
        print(f"INR {c:<11} {cid:<10} {train_counts[c]:<15} {val_counts[c]:<15}")
    print("-" * 50)
    print(f"Total Added: {sum(train_counts.values())} train, {sum(val_counts.values())} val images.")
    print("Integration complete!")


if __name__ == "__main__":
    extract_and_integrate()

