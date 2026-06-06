"""
Football Player Role Classification
main.py

This is the main entry point for the full project pipeline.
Running this script will execute all steps in order:

    Step 1 — prepare_dataset.py
        Extracts individual object crops from the YOLO-format raw dataset,
        creating a classification-ready dataset with one folder per class.

    Step 2 — balance_dataset.py
        Balances the dataset by randomly sampling up to MAX_PER_CLASS images
        per class, reducing the impact of class imbalance during training.

    Step 3 — train.py
        Trains three CNN models (Custom CNN, MobileNetV2, ResNet50),
        evaluates them, and saves all figures and model weights to outputs/.

    Step 4 — gradcam.py
        Loads the trained models and generates Grad-CAM visualizations
        to explain which image regions influenced each model's predictions.

    Step 5 — inference_demo.py
        Loads a real match image from the dataset, draws ground truth
        bounding boxes, and overlays MobileNetV2 predictions on each object.

Usage:
    python main.py

Requirements:
    - Python 3.10+
    - See requirements.txt for all dependencies
    - GPU recommended (NVIDIA CUDA-compatible)
"""

import subprocess
import sys
import time
from pathlib import Path

# All scripts are in the same directory as this file
PROJECT_DIR = Path(__file__).resolve().parent

STEPS = [
    ("Extracting crops from dataset",              "prepare_dataset.py"),
    ("Balancing dataset",                          "balance_dataset.py"),
    ("Training models",                            "train.py"),
    ("Generating Grad-CAM visualizations",         "gradcam.py"),
    ("Generating inference demo on match image",   "inference_demo.py"),
]


def run_step(description, script):
    """Runs a single pipeline step and exits if it fails."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run([sys.executable, PROJECT_DIR / script])
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\nError in {script}. Aborting pipeline.")
        sys.exit(1)
    print(f"  Completed in {elapsed:.1f}s")


def main():
    print("\nFootball Player Role Classification — Pipeline")
    print("=" * 60)
    print(f"Project directory: {PROJECT_DIR}")
    total_start = time.time()

    for description, script in STEPS:
        run_step(description, script)

    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete! Total time: {total:.1f}s")
    print(f"  Outputs saved to: {PROJECT_DIR / 'outputs'}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()