"""
Football Player Role Classification
prepare_dataset.py

This script extracts individual object crops from the raw YOLO-format dataset.
Each image in the raw dataset contains multiple annotated objects (ball, goalkeeper,
player, referee). This script reads each image alongside its corresponding YOLO
label file, crops each annotated bounding box, and saves the crops into separate
class folders — transforming a detection dataset into a classification dataset.

The original Roboflow train/valid/test split is preserved so that crops from
different match images never leak between splits during training or evaluation.

Input structure:
    data/
    ├── train/
    │   ├── images/   <- original match images
    │   └── labels/   <- YOLO annotation files (.txt)
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/

Output structure:
    dataset/
    ├── train/
    │   ├── ball/
    │   ├── goalkeeper/
    │   ├── player/
    │   └── referee/
    ├── valid/
    │   └── ...
    └── test/
        └── ...
"""

import cv2
from pathlib import Path

# ============================================================
# CONFIG — paths are relative to this script's directory
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / 'data'      # raw YOLO dataset
OUTPUT_DIR  = PROJECT_DIR / 'dataset'  # output classification dataset

# Class names must match the order defined in data/data.yaml
CLASSES = ['ball', 'goalkeeper', 'player', 'referee']

# Dataset splits to process — mirrors the Roboflow dataset structure
SPLITS = ['train', 'valid', 'test']

# Minimum bounding box size in pixels — smaller crops are likely noise
MIN_CROP_SIZE = 10

# Create one output folder per split and class
for split in SPLITS:
    for cls in CLASSES:
        (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

print(f"Project directory : {PROJECT_DIR}")
print(f"Data directory    : {DATA_DIR}")
print(f"Output directory  : {OUTPUT_DIR}")

# ============================================================
# CROP EXTRACTION
# ============================================================
counts = {split: {cls: 0 for cls in CLASSES} for split in SPLITS}

for split in SPLITS:
    images_dir = DATA_DIR / split / 'images'
    labels_dir = DATA_DIR / split / 'labels'

    if not images_dir.exists():
        print(f"Skipping '{split}' — directory not found.")
        continue

    # Collect all image files in this split
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    print(f"\nProcessing '{split}': {len(image_files)} images found.")

    for img_path in image_files:
        # Each image has a corresponding .txt label file with the same stem
        label_path = labels_dir / (img_path.stem + '.txt')

        if not label_path.exists():
            continue

        # Load the image using OpenCV
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        # Read all annotations for this image
        with open(label_path) as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            parts = line.strip().split()

            # Each YOLO annotation line: class_id cx cy bw bh (all normalized 0-1)
            if len(parts) != 5:
                continue

            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])

            # Convert YOLO normalized coordinates to absolute pixel coordinates
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            # Clamp coordinates to stay within image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Skip bounding boxes that are too small to be meaningful
            if (x2 - x1) < MIN_CROP_SIZE or (y2 - y1) < MIN_CROP_SIZE:
                continue

            # Crop the object from the image
            crop = img[y1:y2, x1:x2]
            cls_name = CLASSES[cls_id]

            # Save inside the split folder to preserve the original dataset partition
            out_path = OUTPUT_DIR / split / cls_name / f"{img_path.stem}_{i}.jpg"
            cv2.imwrite(str(out_path), crop)
            counts[split][cls_name] += 1

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*40)
print("Dataset Extraction Summary")
print("="*40)
for split in SPLITS:
    split_total = sum(counts[split].values())
    if split_total == 0:
        continue
    print(f"\n  [{split}]")
    for cls, count in counts[split].items():
        print(f"    {cls:12s}: {count:>6} crops")
    print(f"    {'TOTAL':12s}: {split_total:>6} crops")

grand_total = sum(sum(c.values()) for c in counts.values())
print(f"\n  {'GRAND TOTAL':12s}: {grand_total:>6} crops")
print(f"\nDataset saved to: {OUTPUT_DIR}")
