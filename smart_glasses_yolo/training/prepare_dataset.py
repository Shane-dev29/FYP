"""Dataset preparation and unification script.
Combines HomeObjects-3K with indoor person instances, producing a unified 13-class indoor dataset.
"""

from __future__ import annotations

import os
import shutil
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
OUTPUTS_DIR = BASE_DIR / "outputs"
TRAINING_DIR = BASE_DIR / "training"

UNIFIED_DIR = DATASETS_DIR / "indoor_unified"
UNIFIED_TRAIN_IMG = UNIFIED_DIR / "images" / "train"
UNIFIED_VAL_IMG = UNIFIED_DIR / "images" / "val"
UNIFIED_TRAIN_LBL = UNIFIED_DIR / "labels" / "train"
UNIFIED_VAL_LBL = UNIFIED_DIR / "labels" / "val"

CLASSES = {
    0: "bed",
    1: "sofa",
    2: "chair",
    3: "table",
    4: "lamp",
    5: "tv",
    6: "laptop",
    7: "wardrobe",
    8: "window",
    9: "door",
    10: "potted plant",
    11: "photo frame",
    12: "person",
}


def build_unified_dataset():
    """Constructs the unified dataset avoiding duplicate leakage and standardizing class IDs."""
    print("=" * 60)
    print("Preparing Unified Indoor Object Detection Dataset")
    print("=" * 60)

    for p in [UNIFIED_TRAIN_IMG, UNIFIED_VAL_IMG, UNIFIED_TRAIN_LBL, UNIFIED_VAL_LBL]:
        p.mkdir(parents=True, exist_ok=True)

    # 1. Copy HomeObjects-3K
    ho_img_train = DATASETS_DIR / "images" / "train"
    ho_img_val = DATASETS_DIR / "images" / "val"
    ho_lbl_train = DATASETS_DIR / "labels" / "train"
    ho_lbl_val = DATASETS_DIR / "labels" / "val"

    print("Copying HomeObjects-3K train samples...")
    train_count = 0
    for img_p in ho_img_train.glob("*.jpg"):
        lbl_p = ho_lbl_train / f"{img_p.stem}.txt"
        if lbl_p.exists():
            shutil.copy2(img_p, UNIFIED_TRAIN_IMG / img_p.name)
            shutil.copy2(lbl_p, UNIFIED_TRAIN_LBL / lbl_p.name)
            train_count += 1

    print(f"Copied {train_count} HomeObjects-3K training samples.")

    print("Copying HomeObjects-3K validation samples...")
    val_count = 0
    for img_p in ho_img_val.glob("*.jpg"):
        lbl_p = ho_lbl_val / f"{img_p.stem}.txt"
        if lbl_p.exists():
            shutil.copy2(img_p, UNIFIED_VAL_IMG / img_p.name)
            shutil.copy2(lbl_p, UNIFIED_VAL_LBL / lbl_p.name)
            val_count += 1

    print(f"Copied {val_count} HomeObjects-3K validation samples.")

    # 2. Extract person annotations from coco128 and remap class 0 -> 12
    coco128_dir = Path("datasets/coco128")
    if not coco128_dir.exists():
        coco128_dir = Path(__file__).resolve().parent.parent.parent / "datasets" / "coco128"

    coco_imgs = sorted(list((coco128_dir / "images" / "train2017").glob("*.jpg")))
    coco_lbls_dir = coco128_dir / "labels" / "train2017"

    person_samples = []
    for c_img in coco_imgs:
        c_lbl = coco_lbls_dir / f"{c_img.stem}.txt"
        if not c_lbl.exists():
            continue
        # Read lines, filter and remap COCO person (id 0) to 12
        person_lines = []
        for line in open(c_lbl):
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "0":  # COCO person
                person_lines.append(f"12 {' '.join(parts[1:])}\n")

        if person_lines:
            person_samples.append((c_img, person_lines))

    print(f"Found {len(person_samples)} COCO samples containing 'person'.")

    # Split: 80% train, 20% val
    num_train_person = int(len(person_samples) * 0.8)
    train_person = person_samples[:num_train_person]
    val_person = person_samples[num_train_person:]

    print(f"Adding {len(train_person)} person samples to train split...")
    for img_p, lines in train_person:
        dest_img_name = f"coco_person_{img_p.name}"
        dest_lbl_name = f"coco_person_{img_p.stem}.txt"
        shutil.copy2(img_p, UNIFIED_TRAIN_IMG / dest_img_name)
        with open(UNIFIED_TRAIN_LBL / dest_lbl_name, "w") as f:
            f.writelines(lines)

    print(f"Adding {len(val_person)} person samples to validation split...")
    for img_p, lines in val_person:
        dest_img_name = f"coco_person_{img_p.name}"
        dest_lbl_name = f"coco_person_{img_p.stem}.txt"
        shutil.copy2(img_p, UNIFIED_VAL_IMG / dest_img_name)
        with open(UNIFIED_VAL_LBL / dest_lbl_name, "w") as f:
            f.writelines(lines)

    # 3. Create data.yaml
    data_yaml = {
        "path": str(UNIFIED_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": CLASSES,
        "nc": len(CLASSES),
    }

    yaml_path = TRAINING_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    print(f"Written unified dataset config to: {yaml_path}")

    # 4. Compute statistics
    train_counts = Counter()
    for lf in UNIFIED_TRAIN_LBL.glob("*.txt"):
        for line in open(lf):
            parts = line.strip().split()
            if parts:
                train_counts[int(parts[0])] += 1

    val_counts = Counter()
    for lf in UNIFIED_VAL_LBL.glob("*.txt"):
        for line in open(lf):
            parts = line.strip().split()
            if parts:
                val_counts[int(parts[0])] += 1

    print("\nUNIFIED DATASET CLASS BREAKDOWN:")
    print("-" * 55)
    print(f"{'Class ID':<10} {'Name':<15} {'Train':<10} {'Val':<10} {'Total':<10}")
    print("-" * 55)
    class_names_list = []
    train_vals = []
    val_vals = []
    for cid in range(len(CLASSES)):
        name = CLASSES[cid]
        tc = train_counts[cid]
        vc = val_counts[cid]
        print(f"{cid:<10} {name:<15} {tc:<10} {vc:<10} {tc+vc:<10}")
        class_names_list.append(name)
        train_vals.append(tc)
        val_vals.append(vc)
    print("-" * 55)

    # 5. Plot distribution chart
    plt.figure(figsize=(12, 6))
    x = range(len(CLASSES))
    width = 0.4
    plt.bar([p - width/2 for p in x], train_vals, width=width, label="Train Instances", color="#2b5c8f")
    plt.bar([p + width/2 for p in x], val_vals, width=width, label="Val Instances", color="#48a999")
    plt.xticks(x, class_names_list, rotation=45, ha="right", fontsize=10)
    plt.ylabel("Number of Bounding Box Annotations", fontsize=11)
    plt.title("Unified Indoor Dataset - Class Distribution (HomeObjects-3K + Person)", fontsize=13, fontweight="bold")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_chart = OUTPUTS_DIR / "class_distribution.png"
    plt.savefig(out_chart, dpi=200)
    plt.close()
    print(f"Class distribution chart saved to: {out_chart}")


if __name__ == "__main__":
    build_unified_dataset()

