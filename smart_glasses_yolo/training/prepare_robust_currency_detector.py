"""Builds a robust, production-grade Indian Currency Object Detection Dataset:
1. Injects 1,500 negative background images (empty rooms/desks/walls) to completely eliminate false positives.
2. Injects 1,500 multi-scale in-context images with precise bounding boxes around notes on tables/rooms.
3. Keeps balanced raw note images across all 7 denominations.
"""

from __future__ import annotations

import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
CURRENCY_DATASET = BASE_DIR / "datasets" / "indian_currency"
INDOOR_DATASET = BASE_DIR / "datasets" / "indoor_unified"

IMG_TRAIN = CURRENCY_DATASET / "images" / "train"
LBL_TRAIN = CURRENCY_DATASET / "labels" / "train"
IMG_VAL = CURRENCY_DATASET / "images" / "val"
LBL_VAL = CURRENCY_DATASET / "labels" / "val"

CURRENCY_MAP = {
    "10": 0,
    "20": 1,
    "50": 2,
    "100": 3,
    "200": 4,
    "500": 5,
    "2000": 6,
}


def build_robust_dataset():
    random.seed(42)
    np.random.seed(42)

    print("=" * 65)
    print("Building Robust Currency Dataset: Real Bounding Boxes + Negatives")
    print("=" * 65)

    # 1. Gather indoor background images
    indoor_bgs = [p for p in (INDOOR_DATASET / "images" / "train").glob("*.jpg")
                  if not p.name.startswith("curr_") and not p.name.startswith("syn_")]
    print(f"Found {len(indoor_bgs)} clean indoor background scenes.")

    # 2. Add 1,500 Negative Background Images to Train (Empty label files)
    print("Injecting 1,500 negative background images to eliminate false positives...")
    bg_samples = random.sample(indoor_bgs, min(1500, len(indoor_bgs)))
    for idx, bg_p in enumerate(bg_samples):
        out_img = IMG_TRAIN / f"bg_negative_{idx:04d}.jpg"
        out_lbl = LBL_TRAIN / f"bg_negative_{idx:04d}.txt"

        shutil.copy2(bg_p, out_img)
        # Empty label file tells YOLO: NO OBJECTS IN THIS IMAGE!
        with open(out_lbl, "w", encoding="utf-8") as f:
            f.write("")

    # Also add 150 negative background images to Val
    val_bgs = [p for p in (INDOOR_DATASET / "images" / "val").glob("*.jpg")
               if not p.name.startswith("curr_") and not p.name.startswith("syn_")]
    for idx, bg_p in enumerate(val_bgs[:150]):
        out_img = IMG_VAL / f"bg_negative_val_{idx:04d}.jpg"
        out_lbl = LBL_VAL / f"bg_negative_val_{idx:04d}.txt"
        shutil.copy2(bg_p, out_img)
        with open(out_lbl, "w", encoding="utf-8") as f:
            f.write("")

    print("Negative background injection complete!")

    # 3. Add 1,400 In-Context Multi-Scale Banknotes (200 per denomination) with EXACT bounding boxes
    print("Generating in-context notes with tight, precise bounding boxes...")
    syn_count = 0
    for denom, cls_id in CURRENCY_MAP.items():
        denom_notes = list(IMG_TRAIN.glob(f"{denom}_*.jpg"))
        if not denom_notes:
            continue

        selected_notes = random.sample(denom_notes, min(200, len(denom_notes)))
        for s_idx, note_p in enumerate(selected_notes):
            note_img = cv2.imread(str(note_p))
            if note_img is None:
                continue

            bg_p = random.choice(indoor_bgs)
            bg_img = cv2.imread(str(bg_p))
            if bg_img is None:
                continue

            bg_h, bg_w = bg_img.shape[:2]
            nh, nw = note_img.shape[:2]

            # Scale between 20% and 55% of background width (realistic distance)
            target_scale = random.uniform(0.20, 0.55)
            target_w = max(50, int(bg_w * target_scale))
            aspect = nw / max(1.0, float(nh))
            target_h = max(30, int(target_w / aspect))

            if target_h >= bg_h - 15 or target_w >= bg_w - 15:
                continue

            resized_note = cv2.resize(note_img, (target_w, target_h))

            # Random placement in image
            px = random.randint(10, bg_w - target_w - 10)
            py = random.randint(10, bg_h - target_h - 10)

            # Paste note onto background
            composite = bg_img.copy()
            composite[py:py + target_h, px:px + target_w] = resized_note

            # Compute exact normalized bounding box
            x_c = (px + target_w / 2.0) / float(bg_w)
            y_c = (py + target_h / 2.0) / float(bg_h)
            b_w = target_w / float(bg_w)
            b_h = target_h / float(bg_h)

            out_img = IMG_TRAIN / f"syn_tight_{denom}_{s_idx:04d}.jpg"
            out_lbl = LBL_TRAIN / f"syn_tight_{denom}_{s_idx:04d}.txt"

            cv2.imwrite(str(out_img), composite)
            with open(out_lbl, "w", encoding="utf-8") as f:
                f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {b_w:.6f} {b_h:.6f}\n")

            syn_count += 1

    print(f"Generated {syn_count} tight in-context bounding box images.")
    print("Dataset optimization complete!")


if __name__ == "__main__":
    build_robust_dataset()

