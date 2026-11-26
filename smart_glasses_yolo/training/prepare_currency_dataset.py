"""Extracts a balanced, lightweight subset of Indian Currency Dataset for YOLO classification training."""

from __future__ import annotations

import os
import zipfile
from collections import defaultdict
from pathlib import Path

ZIP_PATH = r"C:\Users\remed\Downloads\archive (2).zip"
BASE_DIR = Path(__file__).resolve().parent.parent
CURRENCY_DIR = BASE_DIR / "datasets" / "currency"
TRAIN_DIR = CURRENCY_DIR / "train"
VAL_DIR = CURRENCY_DIR / "val"

TARGET_CLASSES = ["10", "20", "50", "100", "200", "500", "2000", "Invalid"]
MAX_TRAIN_PER_CLASS = 150  # Balanced subset to save disk space and train in ~2 minutes


def prepare_currency():
    print("=" * 60)
    print("Preparing Indian Currency Dataset (Balanced Subset)")
    print(f"Source: {ZIP_PATH}")
    print(f"Target: {CURRENCY_DIR}")
    print("=" * 60)

    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(f"Zip file not found at: {ZIP_PATH}")

    for c in TARGET_CLASSES:
        (TRAIN_DIR / c).mkdir(parents=True, exist_ok=True)
        (VAL_DIR / c).mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        namelist = zf.namelist()
        print(f"Total files in zip: {len(namelist)}")

        train_counts = defaultdict(int)
        val_counts = defaultdict(int)

        for entry in namelist:
            if not entry.endswith(".jpg"):
                continue

            parts = entry.split("/")
            if len(parts) >= 4:
                # Format: Indian_Currency_Dataset/Train/<class>/<file>.jpg
                split_type = parts[1]
                cls_name = parts[2]
                filename = parts[3]

                if cls_name not in TARGET_CLASSES:
                    continue

                if split_type == "Train":
                    if train_counts[cls_name] < MAX_TRAIN_PER_CLASS:
                        data = zf.read(entry)
                        out_path = TRAIN_DIR / cls_name / filename
                        with open(out_path, "wb") as f:
                            f.write(data)
                        train_counts[cls_name] += 1
                elif split_type == "Test":
                    data = zf.read(entry)
                    out_path = VAL_DIR / cls_name / filename
                    with open(out_path, "wb") as f:
                        f.write(data)
                    val_counts[cls_name] += 1

    print("\nExtraction Summary:")
    print("-" * 45)
    print(f"{'Class (Denomination)':<25} {'Train':<10} {'Val':<10}")
    print("-" * 45)
    for c in TARGET_CLASSES:
        print(f"{c:<25} {train_counts[c]:<10} {val_counts[c]:<10}")
    print("-" * 45)
    print(f"Total Train Images: {sum(train_counts.values())}")
    print(f"Total Val Images  : {sum(val_counts.values())}")
    print("Dataset preparation complete!")


if __name__ == "__main__":
    prepare_currency()

