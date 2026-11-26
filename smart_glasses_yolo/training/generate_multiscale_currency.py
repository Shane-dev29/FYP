"""Generates realistic multi-scale currency training data by compositing banknotes
at varied scales (small tabletop, medium arm's length, close-up) onto diverse indoor scenes.
"""

from __future__ import annotations

import random
from pathlib import Path
import cv2
import numpy as np

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

# Distance scale categories for assistive wearable vision
SCALES = [
    ("small", 0.16, 0.28),    # ~1.0m away on desk/table
    ("medium", 0.30, 0.48),   # ~40-60cm away at arm's length (handheld)
    ("close", 0.52, 0.75),    # ~20-30cm close inspection
]


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotates an image by an angle in degrees, expanding bounds if needed."""
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(mat[0, 0])
    sin = np.abs(mat[0, 1])
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))
    mat[0, 2] += (nw / 2) - center[0]
    mat[1, 2] += (nh / 2) - center[1]
    return cv2.warpAffine(image, mat, (nw, nh), borderMode=cv2.BORDER_REFLECT_101)


def generate_multiscale_dataset():
    random.seed(42)
    np.random.seed(42)

    indoor_train_bgs = [p for p in IMG_TRAIN_DIR.glob("*.jpg") if not p.name.startswith("curr_") and not p.name.startswith("syn_")]
    indoor_val_bgs = [p for p in IMG_VAL_DIR.glob("*.jpg") if not p.name.startswith("curr_") and not p.name.startswith("syn_")]

    curr_train_imgs = [p for p in IMG_TRAIN_DIR.glob("curr_*.jpg")]
    curr_val_imgs = [p for p in IMG_VAL_DIR.glob("curr_*.jpg")]

    print("=" * 60)
    print("Generating Multi-Scale Currency Samples for Wearable Distance Robustness")
    print(f"Indoor Backgrounds: Train={len(indoor_train_bgs)}, Val={len(indoor_val_bgs)}")
    print(f"Source Currency Notes: Train={len(curr_train_imgs)}, Val={len(curr_val_imgs)}")
    print("=" * 60)

    # Process training set
    syn_train_count = 0
    for idx, curr_path in enumerate(curr_train_imgs):
        # Extract class denomination from name: curr_<cls>_tr_...
        parts = curr_path.stem.split("_")
        cls_str = parts[1]
        cls_id = CURRENCY_MAP.get(cls_str)
        if cls_id is None:
            continue

        note_img = cv2.imread(str(curr_path))
        if note_img is None:
            continue

        # Generate 2 multi-scale synthetic variations for each note
        for s_idx in range(2):
            scale_name, s_min, s_max = random.choice(SCALES)
            bg_path = random.choice(indoor_train_bgs)
            bg_img = cv2.imread(str(bg_path))
            if bg_img is None:
                continue

            bg_h, bg_w = bg_img.shape[:2]

            # Slight rotation (-18 to +18 deg)
            angle = random.uniform(-18.0, 18.0)
            rot_note = rotate_image(note_img, angle)
            rn_h, rn_w = rot_note.shape[:2]

            # Random target scale relative to background width
            target_scale = random.uniform(s_min, s_max)
            new_w = max(40, int(bg_w * target_scale))
            aspect = rn_w / max(1, rn_h)
            new_h = max(25, int(new_w / aspect))

            if new_h >= bg_h - 10 or new_w >= bg_w - 10:
                continue

            resized_note = cv2.resize(rot_note, (new_w, new_h))

            # Random placement in visible frame
            x_pos = random.randint(5, bg_w - new_w - 5)
            y_pos = random.randint(5, bg_h - new_h - 5)

            # Composite onto background
            composite = bg_img.copy()
            composite[y_pos:y_pos + new_h, x_pos:x_pos + new_w] = resized_note

            # Calculate normalized YOLO bbox
            x_center = (x_pos + new_w / 2.0) / float(bg_w)
            y_center = (y_pos + new_h / 2.0) / float(bg_h)
            norm_w = new_w / float(bg_w)
            norm_h = new_h / float(bg_h)

            out_img_name = f"syn_curr_{cls_str}_tr_{idx:03d}_{s_idx}.jpg"
            out_lbl_name = f"syn_curr_{cls_str}_tr_{idx:03d}_{s_idx}.txt"

            cv2.imwrite(str(IMG_TRAIN_DIR / out_img_name), composite)

            # Read background's existing labels if present and append
            bg_lbl_path = LBL_TRAIN_DIR / f"{bg_path.stem}.txt"
            existing_lines = ""
            if bg_lbl_path.exists():
                with open(bg_lbl_path, "r", encoding="utf-8") as f:
                    existing_lines = f.read().strip()

            with open(LBL_TRAIN_DIR / out_lbl_name, "w", encoding="utf-8") as f:
                if existing_lines:
                    f.write(existing_lines + "\n")
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

            syn_train_count += 1

    # Process validation set
    syn_val_count = 0
    for idx, curr_path in enumerate(curr_val_imgs):
        parts = curr_path.stem.split("_")
        cls_str = parts[1]
        cls_id = CURRENCY_MAP.get(cls_str)
        if cls_id is None:
            continue

        note_img = cv2.imread(str(curr_path))
        if note_img is None:
            continue

        scale_name, s_min, s_max = random.choice(SCALES)
        bg_path = random.choice(indoor_val_bgs)
        bg_img = cv2.imread(str(bg_path))
        if bg_img is None:
            continue

        bg_h, bg_w = bg_img.shape[:2]
        target_scale = random.uniform(s_min, s_max)
        new_w = max(40, int(bg_w * target_scale))
        aspect = note_img.shape[1] / max(1, note_img.shape[0])
        new_h = max(25, int(new_w / aspect))

        if new_h >= bg_h - 10 or new_w >= bg_w - 10:
            continue

        resized_note = cv2.resize(note_img, (new_w, new_h))
        x_pos = random.randint(5, bg_w - new_w - 5)
        y_pos = random.randint(5, bg_h - new_h - 5)

        composite = bg_img.copy()
        composite[y_pos:y_pos + new_h, x_pos:x_pos + new_w] = resized_note

        x_center = (x_pos + new_w / 2.0) / float(bg_w)
        y_center = (y_pos + new_h / 2.0) / float(bg_h)
        norm_w = new_w / float(bg_w)
        norm_h = new_h / float(bg_h)

        out_img_name = f"syn_curr_{cls_str}_val_{idx:03d}.jpg"
        out_lbl_name = f"syn_curr_{cls_str}_val_{idx:03d}.txt"

        cv2.imwrite(str(IMG_VAL_DIR / out_img_name), composite)

        bg_lbl_path = LBL_VAL_DIR / f"{bg_path.stem}.txt"
        existing_lines = ""
        if bg_lbl_path.exists():
            with open(bg_lbl_path, "r", encoding="utf-8") as f:
                existing_lines = f.read().strip()

        with open(LBL_VAL_DIR / out_lbl_name, "w", encoding="utf-8") as f:
            if existing_lines:
                f.write(existing_lines + "\n")
            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        syn_val_count += 1

    print(f"Generated {syn_train_count} multi-scale training images and {syn_val_count} validation images.")
    print("Multi-scale dataset synthesis complete!")


if __name__ == "__main__":
    generate_multiscale_dataset()

