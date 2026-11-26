"""Unzips the full Indian Currency dataset for dedicated YOLO26n training."""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path
from collections import defaultdict

ZIP_PATH = r"C:\Users\remed\Downloads\archive (2).zip"
BASE_DIR = Path(__file__).resolve().parent.parent
CURRENCY_DIR = BASE_DIR / "datasets" / "indian_currency"
IMG_TRAIN_DIR = CURRENCY_DIR / "images" / "train"
LBL_TRAIN_DIR = CURRENCY_DIR / "labels" / "train"
IMG_VAL_DIR = CURRENCY_DIR / "images" / "val"
LBL_VAL_DIR = CURRENCY_DIR / "labels" / "val"

CURRENCY_MAP = {
    "10": 0,
    "20": 1,
    "50": 2,
    "100": 3,
    "200": 4,
    "500": 5,
    "2000": 6,
}


def unzip_full_currency():
    print("=" * 60)
    print("Unzipping Full Indian Currency Dataset for Dedicated YOLO26n")
    print(f"Source: {ZIP_PATH}")
    print(f"Target: {CURRENCY_DIR}")
    print("=" * 60)

    if not os.path.exists(ZIP_PATH):
        print(f"Error: Archive not found at {ZIP_PATH}", file=sys.stderr)
        sys.exit(1)

    for d in [IMG_TRAIN_DIR, LBL_TRAIN_DIR, IMG_VAL_DIR, LBL_VAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    train_counts = defaultdict(int)
    val_counts = defaultdict(int)

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        namelist = zf.namelist()
        total_files = len(namelist)
        print(f"Total entries in archive: {total_files}")

        extracted = 0
        for entry in namelist:
            if not entry.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            parts = [p for p in entry.split("/") if p]
            if len(parts) < 4 or parts[0] != "Indian_Currency_Dataset":
                continue

            split = parts[1].lower()  # train or test
            cls_name = parts[2]       # 10, 20, 50, etc.
            filename = parts[3]

            if cls_name not in CURRENCY_MAP:
                continue

            cls_id = CURRENCY_MAP[cls_name]

            # Read image data from zip
            img_bytes = zf.read(entry)

            if split == "train":
                idx = train_counts[cls_name]
                out_img = IMG_TRAIN_DIR / f"{cls_name}_{idx:05d}.jpg"
                out_lbl = LBL_TRAIN_DIR / f"{cls_name}_{idx:05d}.txt"

                with open(out_img, "wb") as f:
                    f.write(img_bytes)

                # Standard bounding box label covering banknote area
                with open(out_lbl, "w", encoding="utf-8") as f:
                    f.write(f"{cls_id} 0.500000 0.500000 0.900000 0.850000\n")

                train_counts[cls_name] += 1

            elif split in ("test", "val"):
                idx = val_counts[cls_name]
                out_img = IMG_VAL_DIR / f"{cls_name}_{idx:05d}.jpg"
                out_lbl = LBL_VAL_DIR / f"{cls_name}_{idx:05d}.txt"

                with open(out_img, "wb") as f:
                    f.write(img_bytes)

                with open(out_lbl, "w", encoding="utf-8") as f:
                    f.write(f"{cls_id} 0.500000 0.500000 0.900000 0.850000\n")

                val_counts[cls_name] += 1

            extracted += 1
            if extracted % 2000 == 0:
                print(f"  Extracted {extracted} images...")

    print("\nExtraction Complete Summary:")
    print("-" * 50)
    print(f"{'Denomination':<15} {'Class ID':<10} {'Train Count':<12} {'Val Count':<10}")
    print("-" * 50)
    for c, cid in sorted(CURRENCY_MAP.items(), key=lambda x: x[1]):
        print(f"INR {c:<11} {cid:<10} {train_counts[c]:<12} {val_counts[c]:<10}")
    print("-" * 50)
    print(f"Total Train Images: {sum(train_counts.values())}")
    print(f"Total Val Images  : {sum(val_counts.values())}")


if __name__ == "__main__":
    unzip_full_currency()

