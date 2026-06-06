"""
Football Player Role Classification
balance_dataset.py

This script addresses the severe class imbalance present in the extracted dataset.
After running prepare_dataset.py, the class distribution is highly skewed:
    ball:       ~405 images
    goalkeeper: ~473 images
    player:    ~13239 images
    referee:   ~1519 images

Training directly on this imbalanced data would cause the model to heavily favour
the 'player' class, leading to poor performance on minority classes.

This script randomly samples up to MAX_PER_CLASS images from each class folder
and copies them into a new balanced dataset folder, ensuring fair class representation
during model training.
"""

import shutil
import random
from pathlib import Path

# ============================================================
# CONFIG — paths are relative to this script's directory
# ============================================================
PROJECT_DIR   = Path(__file__).resolve().parent
SRC_DIR       = PROJECT_DIR / 'dataset'           # extracted crops
DST_DIR       = PROJECT_DIR / 'dataset_balanced'  # balanced output

CLASSES       = ['ball', 'goalkeeper', 'player', 'referee']
MAX_PER_CLASS = 1500  # maximum images per class after balancing
SEED          = 42    # random seed for reproducibility

random.seed(SEED)

print(f"Project directory : {PROJECT_DIR}")
print(f"Source dataset    : {SRC_DIR}")
print(f"Balanced dataset  : {DST_DIR}")
print(f"Max per class     : {MAX_PER_CLASS}")

# ============================================================
# BALANCE DATASET
# ============================================================
print("\n" + "="*40)
print("Balancing Dataset")
print("="*40)

for cls in CLASSES:
    src_cls = SRC_DIR / cls
    dst_cls = DST_DIR / cls
    dst_cls.mkdir(parents=True, exist_ok=True)

    # Gather all available images for this class
    images = list(src_cls.glob('*.jpg'))

    # Randomly sample up to MAX_PER_CLASS images
    # If the class has fewer images than the limit, use all of them
    selected = random.sample(images, min(MAX_PER_CLASS, len(images)))

    # Copy selected images to the balanced dataset folder
    for img in selected:
        shutil.copy(img, dst_cls / img.name)

    print(f"  {cls:12s}: {len(selected):>6} images copied")

# ============================================================
# SUMMARY
# ============================================================
total = sum(
    len(list((DST_DIR / cls).glob('*.jpg')))
    for cls in CLASSES
)
print(f"\n  {'TOTAL':12s}: {total:>6} images")
print(f"\nBalanced dataset saved to: {DST_DIR}")