"""
Football Player Role Classification
inference_demo.py

Visualizes ground truth annotations from a real match image,
using the original YOLO labels for bounding boxes and colors.
For larger objects, the MobileNetV2 model prediction is also shown.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from pathlib import Path
import random

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / 'data'
OUTPUT_DIR  = PROJECT_DIR / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

CLASSES = ['ball', 'goalkeeper', 'player', 'referee']
IMG_SIZE = 224
MIN_SIZE = 20        # ignore very small objects
CONF_THRESHOLD = 0.6 # only show model label if confidence >= 60%

COLORS = {
    'ball':       '#2196F3',
    'goalkeeper': '#FF9800',
    'player':     '#4CAF50',
    'referee':    '#E91E63',
}

# Load MobileNetV2
print("Loading MobileNetV2...")
inputs = keras.Input(shape=(224, 224, 3))
base = MobileNetV2(weights='imagenet', include_top=False, input_tensor=inputs)
x = base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(4, activation='softmax')(x)
mobilenet = keras.Model(inputs, out, name='MobileNetV2')
mobilenet.load_weights(str(OUTPUT_DIR / 'best_mobilenet_ft.keras'))
print("Model loaded!")

# Pick image with most annotations
test_images = list((DATA_DIR / 'train' / 'images').glob('*.jpg'))
random.seed(42)

best_img = None
best_count = 0
for img_path in random.sample(test_images, min(50, len(test_images))):
    label_path = DATA_DIR / 'train' / 'labels' / (img_path.stem + '.txt')
    if label_path.exists():
        with open(label_path) as f:
            count = len(f.readlines())
        if count > best_count:
            best_count = count
            best_img = img_path

print(f"Selected: {best_img.name} ({best_count} objects)")

img_bgr = cv2.imread(str(best_img))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w = img_rgb.shape[:2]

label_path = DATA_DIR / 'train' / 'labels' / (best_img.stem + '.txt')
with open(label_path) as f:
    lines = f.readlines()

fig, ax = plt.subplots(1, 1, figsize=(18, 11))
ax.imshow(img_rgb)

class_counts = {cls: 0 for cls in CLASSES}

for line in lines:
    parts = line.strip().split()
    if len(parts) != 5:
        continue

    true_cls_id = int(parts[0])
    cx, cy, bw, bh = map(float, parts[1:])

    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    box_w = x2 - x1
    box_h = y2 - y1

    if box_w < 5 or box_h < 5:
        continue

    # Use ground truth class for color and base label
    true_cls = CLASSES[true_cls_id]
    color = COLORS[true_cls]
    class_counts[true_cls] += 1

    # For larger objects, also run model prediction
    label = true_cls
    if box_w >= MIN_SIZE and box_h >= MIN_SIZE:
        crop = img_rgb[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        crop_norm = np.expand_dims(crop_resized / 255.0, axis=0)
        preds = mobilenet.predict(crop_norm, verbose=0)[0]
        pred_cls = CLASSES[np.argmax(preds)]
        confidence = np.max(preds)

        if confidence >= CONF_THRESHOLD:
            # Show model prediction alongside ground truth
            label = f"{true_cls} → {pred_cls} {confidence:.0%}"
        else:
            label = f"{true_cls}"

    # Draw box
    rect = patches.Rectangle(
        (x1, y1), box_w, box_h,
        linewidth=2, edgecolor=color, facecolor='none'
    )
    ax.add_patch(rect)

    # Label
    ax.text(
        x1, y1 - 3, label,
        fontsize=6.5, color='white', fontweight='bold',
        bbox=dict(facecolor=color, edgecolor='none', pad=1.5, alpha=0.9)
    )

# Legend
handles = []
for cls, color in COLORS.items():
    count = class_counts[cls]
    patch = patches.Patch(color=color, label=f'{cls} ({count})')
    handles.append(patch)

ax.legend(handles=handles, loc='upper right', fontsize=10, framealpha=0.9)
ax.set_title(
    'Football Role Classification — Ground Truth Annotations with MobileNetV2 Predictions',
    fontsize=13, fontweight='bold', pad=10
)
ax.axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'inference_demo.pdf', format='pdf', bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'inference_demo.png', dpi=200, bbox_inches='tight')
plt.close()

print(f"Objects: {class_counts}")
print("Saved: outputs/inference_demo.pdf")
