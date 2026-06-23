"""
Football Player Role Classification
robustness.py

Evaluates how the three trained models degrade when the held-out test images
are corrupted, simulating realistic broadcast conditions: motion/defocus blur,
sensor noise, illumination changes, partial occlusion, and JPEG compression.

For every corruption type and severity level (0 = clean), all test crops are
transformed and re-classified. Accuracy and macro F1-score are recorded per
model. Results are written to a CSV file and a multi-panel line plot.
"""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

from tensorflow import keras
from sklearn.metrics import f1_score

from shap_analysis import (
    build_custom_cnn, build_mobilenetv2, build_resnet50,
    OUTPUT_DIR, DATASET_PATH, CLASSES, IMG_SIZE,
)

SEVERITIES = [0, 1, 2, 3, 4]  # 0 = clean reference


# ============================================================
# CORRUPTIONS (operate on a [0,1] float image of shape HxWx3)
# ============================================================
def corrupt_blur(img, sev):
    if sev == 0:
        return img
    k = [0, 3, 5, 7, 9][sev]
    return cv2.GaussianBlur(img, (k, k), 0)


def corrupt_noise(img, sev):
    if sev == 0:
        return img
    std = [0, 0.04, 0.08, 0.14, 0.22][sev]
    noisy = img + np.random.normal(0, std, img.shape).astype(np.float32)
    return np.clip(noisy, 0, 1)


def corrupt_brightness(img, sev):
    if sev == 0:
        return img
    factor = [1.0, 0.75, 0.55, 0.40, 0.28][sev]  # progressively darker
    return np.clip(img * factor, 0, 1)


def corrupt_occlusion(img, sev):
    if sev == 0:
        return img
    frac = [0, 0.15, 0.30, 0.45, 0.60][sev]
    h, w = img.shape[:2]
    bh, bw = int(h * frac), int(w * frac)
    out = img.copy()
    y = (h - bh) // 2
    x = (w - bw) // 2
    out[y:y + bh, x:x + bw] = 0.0  # central black patch
    return out


def corrupt_jpeg(img, sev):
    if sev == 0:
        return img
    quality = [100, 30, 20, 12, 6][sev]
    bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


CORRUPTIONS = {
    'Blur':       corrupt_blur,
    'Noise':      corrupt_noise,
    'Brightness': corrupt_brightness,
    'Occlusion':  corrupt_occlusion,
    'JPEG':       corrupt_jpeg,
}


# ============================================================
# DATA + MODELS
# ============================================================
def load_test_set():
    images, y_true = [], []
    for idx, cls in enumerate(CLASSES):
        for f in sorted((DATASET_PATH / 'test' / cls).glob('*.jpg')):
            img = keras.utils.load_img(str(f), target_size=(IMG_SIZE, IMG_SIZE))
            images.append(keras.utils.img_to_array(img) / 255.0)
            y_true.append(idx)
    return np.array(images, dtype=np.float32), np.array(y_true)


def load_models():
    print("\nBuilding and loading models...")
    cnn = build_custom_cnn(); cnn.load_weights(str(OUTPUT_DIR / 'best_custom_cnn.keras'))
    mob = build_mobilenetv2(); mob.load_weights(str(OUTPUT_DIR / 'best_mobilenet_ft.keras'))
    res = build_resnet50();   res.load_weights(str(OUTPUT_DIR / 'best_resnet_ft.keras'))
    print("All models loaded.")
    return {'Custom_CNN': cnn, 'MobileNetV2': mob, 'ResNet50': res}


def apply_corruption(images, fn, sev):
    return np.array([fn(img, sev) for img in images], dtype=np.float32)


# ============================================================
# MAIN
# ============================================================
def main():
    np.random.seed(42)
    images, y_true = load_test_set()
    print(f"Loaded {len(images)} test images.")
    models = load_models()

    rows = []  # (model, corruption, severity, accuracy, macro_f1)

    for cname, fn in CORRUPTIONS.items():
        print(f"\nCorruption: {cname}")
        for sev in SEVERITIES:
            corrupted = apply_corruption(images, fn, sev)
            for mname, model in models.items():
                preds = model.predict(corrupted, batch_size=64, verbose=0)
                y_pred = np.argmax(preds, axis=1)
                acc = float(np.mean(y_pred == y_true))
                f1 = float(f1_score(y_true, y_pred, average='macro'))
                rows.append((mname, cname, sev, acc, f1))
            print(f"  severity {sev}: done")

    # Save CSV
    csv_path = OUTPUT_DIR / 'robustness.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'corruption', 'severity', 'accuracy', 'macro_f1'])
        w.writerows(rows)
    print(f"\nSaved: {csv_path}")

    # Plot accuracy vs severity, one panel per corruption
    model_names = list(models.keys())
    colors = {'Custom_CNN': '#2196F3', 'MobileNetV2': '#4CAF50', 'ResNet50': '#FF5722'}
    fig, axes = plt.subplots(1, len(CORRUPTIONS), figsize=(4 * len(CORRUPTIONS), 4), sharey=True)
    for ax, cname in zip(axes, CORRUPTIONS):
        for mname in model_names:
            ys = [acc for (m, c, s, acc, f1) in rows if m == mname and c == cname]
            ax.plot(SEVERITIES, ys, marker='o', label=mname, color=colors[mname])
        ax.set_title(cname, fontweight='bold')
        ax.set_xlabel('Severity')
        ax.set_xticks(SEVERITIES)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('Accuracy')
    axes[-1].legend(fontsize=8, loc='lower left')
    plt.suptitle('Model Robustness to Image Corruptions (Test Set)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'robustness_accuracy.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("Saved: robustness_accuracy.pdf")

    # Console summary (clean vs most severe)
    print("\n" + "=" * 60)
    print("ROBUSTNESS SUMMARY — accuracy at severity 0 -> 4")
    print("=" * 60)
    for cname in CORRUPTIONS:
        print(f"\n{cname}")
        for mname in model_names:
            seq = [acc for (m, c, s, acc, f1) in rows if m == mname and c == cname]
            print(f"  {mname:12s}: " + " -> ".join(f"{a*100:5.1f}%" for a in seq))


if __name__ == '__main__':
    main()
