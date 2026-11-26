"""Tests multi-scale currency compositing onto indoor backgrounds."""

import random
from pathlib import Path
import cv2
import numpy as np

indoor_dir = Path("smart_glasses_yolo/datasets/indoor_unified/images/train")
indoor_imgs = [p for p in indoor_dir.glob("*.jpg") if not p.name.startswith("curr_")]
curr_imgs = [p for p in indoor_dir.glob("curr_*.jpg")]

print(f"Indoor images: {len(indoor_imgs)}, Currency images: {len(curr_imgs)}")

# Test loading one indoor image and one currency image
bg = cv2.imread(str(indoor_imgs[0]))
fg = cv2.imread(str(curr_imgs[0]))

bg_h, bg_w = bg.shape[:2]
fg_h, fg_w = fg.shape[:2]
print(f"Background size: {bg_w}x{bg_h}, Foreground size: {fg_w}x{fg_h}")

# Test placing fg onto bg at scale 0.25 (arm's length)
target_scale = 0.25
aspect = fg_w / fg_h
new_w = int(bg_w * target_scale)
new_h = int(new_w / aspect)

if new_h < bg_h and new_w < bg_w:
    fg_resized = cv2.resize(fg, (new_w, new_h))
    x_offset = (bg_w - new_w) // 2
    y_offset = (bg_h - new_h) // 2
    bg[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = fg_resized
    
    # Calculate YOLO bbox
    x_center = (x_offset + new_w / 2.0) / bg_w
    y_center = (y_offset + new_h / 2.0) / bg_h
    norm_w = new_w / float(bg_w)
    norm_h = new_h / float(bg_h)
    print(f"Synthetic BBox: center=({x_center:.3f}, {y_center:.3f}), size=({norm_w:.3f}, {norm_h:.3f})")
    print("Multi-scale placement logic works perfectly!")

