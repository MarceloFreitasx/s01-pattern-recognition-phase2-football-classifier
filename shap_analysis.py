"""
Football Player Role Classification
shap_analysis.py

Generates SHAP (SHapley Additive exPlanations) visualizations for all three
trained models. While Grad-CAM highlights coarse class-discriminative regions,
SHAP attributes the prediction to individual image regions with both positive
(supporting) and negative (opposing) evidence.

A model-agnostic Partition explainer with an inpainting image masker is used.
This only relies on the model's forward pass (predict), which keeps it robust
across Keras 3 / TensorFlow versions.

For each model a figure is produced with one row per sample image, showing the
SHAP attribution map for the predicted class. Output is saved as PDF.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from pathlib import Path

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, ResNet50

# ============================================================
# CONFIG
# ============================================================
PROJECT_DIR  = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / 'dataset_balanced'
OUTPUT_DIR   = PROJECT_DIR / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

CLASSES           = ['ball', 'goalkeeper', 'player', 'referee']
SAMPLES_PER_CLASS = 2     # number of test images explained per class
MAX_EVALS         = 600   # SHAP masking evaluations per image (higher = sharper)
IMG_SIZE          = 224

print(f"Dataset path     : {DATASET_PATH}")
print(f"Output directory : {OUTPUT_DIR}")


# ============================================================
# MODEL BUILDERS (must match train.py exactly)
# ============================================================
def build_custom_cnn():
    inp = keras.Input(shape=(224, 224, 3))
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='last_conv')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(4, activation='softmax')(x)
    return keras.Model(inp, out, name='Custom_CNN')


def build_mobilenetv2():
    inputs = keras.Input(shape=(224, 224, 3))
    base = MobileNetV2(weights='imagenet', include_top=False, input_tensor=inputs)
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(4, activation='softmax')(x)
    return keras.Model(inputs, out, name='MobileNetV2')


def build_resnet50():
    inputs = keras.Input(shape=(224, 224, 3))
    base = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(4, activation='softmax')(x)
    return keras.Model(inputs, out, name='ResNet50')


def load_models():
    print("\nBuilding and loading models...")
    cnn = build_custom_cnn()
    cnn.load_weights(str(OUTPUT_DIR / 'best_custom_cnn.keras'))
    mob = build_mobilenetv2()
    mob.load_weights(str(OUTPUT_DIR / 'best_mobilenet_ft.keras'))
    res = build_resnet50()
    res.load_weights(str(OUTPUT_DIR / 'best_resnet_ft.keras'))
    print("All models loaded.")
    return cnn, mob, res


# ============================================================
# SAMPLE SELECTION
# ============================================================
def load_sample_images():
    """Loads SAMPLES_PER_CLASS test images per class, normalized to [0, 1]."""
    images, labels = [], []
    for cls in CLASSES:
        files = sorted((DATASET_PATH / 'test' / cls).glob('*.jpg'))[:SAMPLES_PER_CLASS]
        for f in files:
            img = keras.utils.load_img(str(f), target_size=(IMG_SIZE, IMG_SIZE))
            images.append(keras.utils.img_to_array(img) / 255.0)
            labels.append(cls)
    return np.array(images, dtype=np.float32), labels


# ============================================================
# SHAP FIGURE
# ============================================================
def generate_shap_figure(model, model_name, images, labels):
    """Computes SHAP attributions for the predicted class and saves a PDF."""
    print(f"\nGenerating SHAP for {model_name} ...")

    def predict_fn(x):
        return model.predict(x, verbose=0)

    masker = shap.maskers.Image("inpaint_telea", images[0].shape)
    explainer = shap.Explainer(predict_fn, masker, output_names=CLASSES)

    shap_values = explainer(
        images,
        max_evals=MAX_EVALS,
        batch_size=50,
        outputs=shap.Explanation.argsort.flip[:1],
    )

    shap.image_plot(shap_values, show=False)
    fig = plt.gcf()
    fig.suptitle(f'SHAP — {model_name}', fontsize=14, fontweight='bold')
    fig.savefig(OUTPUT_DIR / f'shap_{model_name}.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: shap_{model_name}.pdf")


def main():
    cnn, mob, res = load_models()
    images, labels = load_sample_images()
    print(f"Loaded {len(images)} test images ({SAMPLES_PER_CLASS} per class).")

    for model, name in [(cnn, 'Custom_CNN'), (mob, 'MobileNetV2'), (res, 'ResNet50')]:
        generate_shap_figure(model, name, images, labels)

    print("\nSHAP analysis complete!")


if __name__ == '__main__':
    main()
