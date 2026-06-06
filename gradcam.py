"""
Football Player Role Classification
gradcam.py

This script generates Grad-CAM (Gradient-weighted Class Activation Mapping)
visualizations for all three trained models.

Grad-CAM works by computing the gradient of the predicted class score with
respect to the feature maps of the last convolutional layer. These gradients
are then used to weight the activation maps, producing a heatmap that highlights
the image regions most important for the model's decision.

For each model, this script generates a figure with three rows:
    Row 1 — Original image with true and predicted label
    Row 2 — Raw Grad-CAM heatmap
    Row 3 — Heatmap overlaid on the original image

Figures are saved as PDF for publication-quality output.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from pathlib import Path

# ============================================================
# CONFIG — paths are relative to this script's directory
# ============================================================
PROJECT_DIR   = Path(__file__).resolve().parent
DATASET_PATH  = PROJECT_DIR / 'dataset_balanced'
OUTPUT_DIR    = PROJECT_DIR / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

CLASSES           = ['ball', 'goalkeeper', 'player', 'referee']
SAMPLES_PER_CLASS = 2  # number of validation images to visualize per class

print(f"Project directory : {PROJECT_DIR}")
print(f"Dataset path      : {DATASET_PATH}")
print(f"Output directory  : {OUTPUT_DIR}")

# ============================================================
# MODEL BUILDERS
# ============================================================
# These functions recreate the exact same architectures used in train.py.
# We need to rebuild the models before loading saved weights because Keras
# requires the architecture to be defined before weights can be restored.

def build_custom_cnn():
    """Builds the custom CNN architecture (must match train.py exactly)."""
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
    # 'last_conv' is the target layer for Grad-CAM
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
    """Builds the MobileNetV2 transfer learning architecture."""
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
    """Builds the ResNet50 transfer learning architecture."""
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


# ============================================================
# LOAD MODELS
# ============================================================
def load_models():
    """Rebuilds all three models and loads their saved weights."""
    print("\nBuilding and loading models...")
    custom_cnn = build_custom_cnn()
    custom_cnn.load_weights(str(OUTPUT_DIR / 'best_custom_cnn.keras'))

    mobilenet = build_mobilenetv2()
    mobilenet.load_weights(str(OUTPUT_DIR / 'best_mobilenet_ft.keras'))

    resnet = build_resnet50()
    resnet.load_weights(str(OUTPUT_DIR / 'best_resnet_ft.keras'))

    print("All models loaded.")
    return custom_cnn, mobilenet, resnet


# ============================================================
# GRAD-CAM
# ============================================================
def make_gradcam(img_exp, model, conv_layer_name):
    """
    Computes the Grad-CAM heatmap for the predicted class.

    Steps:
    1. Build a sub-model that outputs both the target conv layer and the predictions
    2. Record gradients of the predicted class score w.r.t. the conv layer output
    3. Pool gradients spatially to get importance weights per feature map channel
    4. Compute weighted sum of feature maps to get the heatmap
    5. Apply ReLU and normalize to [0, 1]

    Args:
        img_exp   : preprocessed image with batch dimension (1, 224, 224, 3)
        model     : trained Keras model
        conv_layer_name : name of the target convolutional layer

    Returns:
        heatmap   : 2D numpy array of shape (H, W) with values in [0, 1]
        preds     : raw prediction probabilities for all classes
    """
    # Create a model that returns the target conv layer output and final predictions
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_exp)
        # Identify the predicted class
        pred_index = tf.argmax(predictions[0])
        # Score for the predicted class
        class_channel = predictions[:, pred_index]

    # Compute gradients of the class score w.r.t. the conv feature maps
    grads = tape.gradient(class_channel, conv_outputs)

    # Average gradient across spatial dimensions -> one value per feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight each feature map by its corresponding gradient magnitude
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU (keep only positive activations) and normalize
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), predictions.numpy()[0]


def overlay_heatmap(img_array, heatmap):
    """
    Overlays the Grad-CAM heatmap on the original image using a JET colormap.

    Args:
        img_array : original image as float array [0, 1], shape (224, 224, 3)
        heatmap   : Grad-CAM heatmap, shape (H, W), values in [0, 1]

    Returns:
        Blended image as uint8 numpy array, shape (224, 224, 3)
    """
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    img_uint8 = np.uint8(img_array * 255)
    # Blend original image (60%) with heatmap (40%)
    return cv2.addWeighted(img_uint8, 0.6, heatmap_colored, 0.4, 0)


# ============================================================
# GENERATE GRAD-CAM FIGURES
# ============================================================
def generate_gradcam_figure(model, conv_layer_name, model_name, sample_files, class_names):
    """
    Generates and saves a Grad-CAM visualization figure for one model.
    The figure has 3 rows (Original, Heatmap, Overlay) and one column per sample image.
    Correct predictions are shown in green, incorrect predictions in red.
    """
    n = len(sample_files)
    fig, axes = plt.subplots(3, n, figsize=(n * 2.5, 9))

    for i, img_path in enumerate(sample_files):
        # Load and preprocess the image
        img = keras.utils.load_img(img_path, target_size=(224, 224))
        img_array = keras.utils.img_to_array(img) / 255.0
        img_exp = np.expand_dims(img_array, axis=0)
        true_class = Path(img_path).parent.name

        try:
            heatmap, preds = make_gradcam(img_exp, model, conv_layer_name)
            pred_class = class_names[np.argmax(preds)]
            confidence = np.max(preds)
            correct = true_class == pred_class

            # Row 0: original image with prediction label
            axes[0, i].imshow(img_array)
            axes[0, i].set_title(
                f'True: {true_class}\nPred: {pred_class}\n{confidence:.0%}',
                fontsize=7,
                color='green' if correct else 'red'
            )
            axes[0, i].axis('off')

            # Row 1: raw Grad-CAM heatmap
            axes[1, i].imshow(cv2.resize(heatmap, (224, 224)), cmap='jet', vmin=0, vmax=1)
            axes[1, i].set_title('Heatmap', fontsize=7)
            axes[1, i].axis('off')

            # Row 2: heatmap overlaid on original image
            axes[2, i].imshow(overlay_heatmap(img_array, heatmap))
            axes[2, i].set_title('Overlay', fontsize=7)
            axes[2, i].axis('off')

        except Exception as e:
            print(f"  Error on {Path(img_path).name}: {e}")
            for row in range(3):
                axes[row, i].axis('off')

    # Row labels on the left side
    for row, label in enumerate(['Original', 'Heatmap', 'Overlay']):
        axes[row, 0].set_ylabel(label, fontsize=9, fontweight='bold')

    plt.suptitle(f'Grad-CAM — {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'gradcam_{model_name}.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: gradcam_{model_name}.pdf")


# ============================================================
# MAIN
# ============================================================
def main():
    custom_cnn, mobilenet, resnet = load_models()

    # Set up validation generator to get file paths and class names
    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    val_gen = val_datagen.flow_from_directory(
        str(DATASET_PATH),
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        seed=42,
        shuffle=False
    )
    class_names = list(val_gen.class_indices.keys())

    # Select SAMPLES_PER_CLASS images from each class for visualization
    sample_files = []
    for cls in CLASSES:
        files = [f for f in val_gen.filepaths if f'/{cls}/' in f][:SAMPLES_PER_CLASS]
        sample_files.extend(files)
    print(f"Using {len(sample_files)} sample images ({SAMPLES_PER_CLASS} per class).")

    # Each model uses a different target conv layer for Grad-CAM
    models_config = [
        (custom_cnn, 'last_conv',           'Custom_CNN'),   # custom named layer
        (mobilenet,  'Conv_1',              'MobileNetV2'),  # last conv in MobileNetV2
        (resnet,     'conv5_block3_3_conv', 'ResNet50'),     # last conv in ResNet50
    ]

    for model, conv_layer, name in models_config:
        print(f"\nGenerating Grad-CAM for {name}...")
        generate_gradcam_figure(model, conv_layer, name, sample_files, class_names)

    print("\nGrad-CAM generation complete!")


if __name__ == '__main__':
    main()