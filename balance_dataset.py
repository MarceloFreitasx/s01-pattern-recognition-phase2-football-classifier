"""
Football Player Role Classification
balance_dataset.py

This script addresses the severe class imbalance present in the training split.
After running prepare_dataset.py, the class distribution in the train split is
highly skewed (player class dominates). Training directly on this imbalanced
data would cause the model to heavily favour the 'player' class.

Strategy:
    - train  — randomly undersample to MAX_PER_CLASS images per class
    - valid  — copied as-is (preserves Roboflow validation split)
    - test   — copied as-is (preserves Roboflow test split for final evaluation)
"""

import shutil
import random
from pathlib import Path

# ============================================================
# CONFIG — paths are relative to this script's directory
# ============================================================
PROJECT_DIR   = Path(__file__).resolve().parent
SRC_DIR       = PROJECT_DIR / 'dataset'           # extracted crops by split
DST_DIR       = PROJECT_DIR / 'dataset_balanced'  # balanced output

CLASSES       = ['ball', 'goalkeeper', 'player', 'referee']
SPLITS        = ['train', 'valid', 'test']
MAX_PER_CLASS = 1500  # maximum images per class in the training split
SEED          = 42    # random seed for reproducibility

random.seed(SEED)

print(f"Project directory : {PROJECT_DIR}")
print(f"Source dataset    : {SRC_DIR}")
print(f"Balanced dataset  : {DST_DIR}")
print(f"Max per class     : {MAX_PER_CLASS} (train split only)")

# ============================================================
# BUILD BALANCED DATASET
# ============================================================
print("\n" + "="*40)
print("Building Dataset (train balanced, val/test preserved)")
print("="*40)

for split in SPLITS:
    print(f"\n  [{split}]")
    for cls in CLASSES:
        src_cls = SRC_DIR / split / cls
        dst_cls = DST_DIR / split / cls
        dst_cls.mkdir(parents=True, exist_ok=True)

        images = list(src_cls.glob('*.jpg'))

        if split == 'train':
            # Undersample the training split to balance classes
            selected = random.sample(images, min(MAX_PER_CLASS, len(images)))
        else:
            # Keep validation and test splits unchanged
            selected = images

        for img in selected:
            shutil.copy(img, dst_cls / img.name)

        print(f"    {cls:12s}: {len(selected):>6} images")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*40)
print("Summary")
print("="*40)
for split in SPLITS:
    total = sum(len(list((DST_DIR / split / cls).glob('*.jpg'))) for cls in CLASSES)
    if total > 0:
        print(f"  {split:8s}: {total:>6} images")

grand_total = sum(
    len(list((DST_DIR / split / cls).glob('*.jpg')))
    for split in SPLITS for cls in CLASSES
)
print(f"\n  {'TOTAL':8s}: {grand_total:>6} images")
print(f"\nDataset saved to: {DST_DIR}")
