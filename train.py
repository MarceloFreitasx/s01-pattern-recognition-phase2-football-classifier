"""
Football Player Role Classification
train.py

This script implements the full training pipeline for three CNN-based models:
    1. Custom CNN    — a lightweight convolutional network built from scratch
    2. MobileNetV2   — a pre-trained transfer learning model (fine-tuned)
    3. ResNet50      — a deeper pre-trained transfer learning model (fine-tuned)

The script covers the complete ML workflow:
    - Data loading and augmentation
    - Class imbalance handling via class weights
    - Model definition, compilation, and training
    - Evaluation using accuracy, precision, recall, F1-score, and confusion matrix
    - PDF export of all figures (training curves, confusion matrices, comparison chart)

All outputs are saved to the outputs/ directory.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # use non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

# ============================================================
# CONFIG — paths are relative to this script's directory
# ============================================================
PROJECT_DIR  = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / 'dataset_balanced'  # balanced classification dataset
OUTPUT_DIR   = PROJECT_DIR / 'outputs'           # directory for saved models and figures
OUTPUT_DIR.mkdir(exist_ok=True)

IMG_SIZE   = 224   # input image size expected by MobileNetV2 and ResNet50
BATCH_SIZE = 32    # number of samples per training step
SEED       = 42    # random seed for reproducibility
CLASSES    = ['ball', 'goalkeeper', 'player', 'referee']

print(f"Project directory : {PROJECT_DIR}")
print(f"Dataset path      : {DATASET_PATH}")
print(f"Output directory  : {OUTPUT_DIR}")

# ============================================================
# DATA GENERATORS
# ============================================================
# Training generator applies data augmentation to artificially increase
# dataset diversity and improve model generalization
train_datagen = ImageDataGenerator(
    rescale=1./255,            # normalize pixel values to [0, 1]
    rotation_range=20,         # randomly rotate images up to 20 degrees
    width_shift_range=0.15,    # randomly shift images horizontally
    height_shift_range=0.15,   # randomly shift images vertically
    horizontal_flip=True,      # randomly flip images left-right
    zoom_range=0.15,           # randomly zoom in/out
    brightness_range=[0.8, 1.2],  # randomly adjust brightness
    validation_split=0.2       # reserve 20% of data for validation
)

# Validation generator applies only rescaling — no augmentation
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    str(DATASET_PATH),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',   # one-hot encoded labels for multi-class
    subset='training',
    seed=SEED,
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    str(DATASET_PATH),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=SEED,
    shuffle=False  # keep order consistent for evaluation
)

print(f"\nClasses: {train_generator.class_indices}")
print(f"Train: {train_generator.samples} | Val: {val_generator.samples}")

# Compute class weights to handle residual imbalance after balancing
# This penalizes misclassifications of minority classes more heavily
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weight_dict}")

# ============================================================
# CLASS DISTRIBUTION PLOT
# ============================================================
counts = {cls: len(list(DATASET_PATH.glob(f'{cls}/*.jpg'))) for cls in CLASSES}

plt.figure(figsize=(8, 5))
plt.bar(counts.keys(), counts.values(),
        color=['#2196F3', '#FF9800', '#4CAF50', '#E91E63'])
plt.title('Class Distribution — Balanced Dataset', fontsize=14, fontweight='bold')
plt.ylabel('Number of Images')
for i, (k, v) in enumerate(counts.items()):
    plt.text(i, v + 5, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'class_distribution.pdf', format='pdf', bbox_inches='tight')
plt.close()
print("Saved: class_distribution.pdf")

# ============================================================
# CALLBACKS
# ============================================================
def get_callbacks(model_name):
    """
    Returns a list of training callbacks:
    - EarlyStopping: stops training when val_accuracy stops improving
    - ReduceLROnPlateau: reduces learning rate when val_loss plateaus
    - ModelCheckpoint: saves the best model weights during training
    """
    return [
        EarlyStopping(
            patience=8,
            restore_best_weights=True,
            monitor='val_accuracy'
        ),
        ReduceLROnPlateau(
            patience=4,
            factor=0.3,
            min_lr=1e-7,
            monitor='val_loss'
        ),
        ModelCheckpoint(
            str(OUTPUT_DIR / f'best_{model_name}.keras'),
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]

# ============================================================
# TRAINING CURVES PLOT
# ============================================================
def plot_history(history, model_name):
    """Plots and saves training/validation accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title(f'{model_name} — Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title(f'{model_name} — Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.suptitle(model_name, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'curves_{model_name}.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: curves_{model_name}.pdf")

# ============================================================
# EVALUATION
# ============================================================
def evaluate_model(model, generator, model_name):
    """
    Evaluates a trained model on the validation set.
    Prints a full classification report and saves the confusion matrix as PDF.
    Returns the overall accuracy.
    """
    generator.reset()
    preds = model.predict(generator, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = generator.classes
    class_names = list(generator.class_indices.keys())

    print(f"\n{'='*50}")
    print(f"Results: {model_name}")
    print('='*50)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix — {model_name}', fontweight='bold', fontsize=13)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'cm_{model_name}.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: cm_{model_name}.pdf")

    return np.mean(y_pred == y_true)

# ============================================================
# MODEL 1: CUSTOM CNN
# ============================================================
# A custom 4-block convolutional network built from scratch.
# Each block applies convolution, batch normalization, max pooling, and dropout.
# Increasing filter sizes (32 -> 64 -> 128 -> 256) allow the network to learn
# progressively more complex visual features.
print("\n" + "="*50)
print("Training Custom CNN...")
print("="*50)

inp = keras.Input(shape=(224, 224, 3))

# Block 1 — low-level features (edges, textures)
x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inp)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2,2)(x)
x = layers.Dropout(0.25)(x)

# Block 2 — mid-level features
x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2,2)(x)
x = layers.Dropout(0.25)(x)

# Block 3 — higher-level features
x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2,2)(x)
x = layers.Dropout(0.25)(x)

# Block 4 — high-level semantic features (used for Grad-CAM)
x = layers.Conv2D(256, (3,3), activation='relu', padding='same', name='last_conv')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2,2)(x)
x = layers.Dropout(0.25)(x)

# Classifier head
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
out = layers.Dense(4, activation='softmax')(x)  # 4 output classes

custom_cnn = keras.Model(inp, out, name='Custom_CNN')
custom_cnn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_cnn = custom_cnn.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=get_callbacks('custom_cnn'),
    class_weight=class_weight_dict,
    verbose=1
)
plot_history(history_cnn, 'Custom_CNN')

# ============================================================
# MODEL 2: MobileNetV2
# ============================================================
# MobileNetV2 is a lightweight architecture pre-trained on ImageNet.
# We use transfer learning in two phases:
#   Phase 1 — Feature extraction: freeze base weights, train only the new head
#   Phase 2 — Fine-tuning: unfreeze the last 30 layers and train with low LR
print("\n" + "="*50)
print("Training MobileNetV2...")
print("="*50)

inputs_m = keras.Input(shape=(224, 224, 3))
base_m = MobileNetV2(weights='imagenet', include_top=False, input_tensor=inputs_m)
base_m.trainable = False  # Phase 1: freeze all base layers

# Custom classification head
x = base_m.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
out_m = layers.Dense(4, activation='softmax')(x)
mobilenet = keras.Model(inputs_m, out_m, name='MobileNetV2')

mobilenet.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Phase 1: train classification head only
history_mobilenet = mobilenet.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=get_callbacks('mobilenet'),
    class_weight=class_weight_dict,
    verbose=1
)

# Phase 2: fine-tune — unfreeze last 30 layers of MobileNetV2
print("\nFine-tuning MobileNetV2...")
base_m.trainable = True
for layer in base_m.layers[:-30]:
    layer.trainable = False  # keep early layers frozen to preserve low-level features

# Use a much smaller learning rate to avoid destroying pre-trained weights
mobilenet.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_mobilenet_ft = mobilenet.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=get_callbacks('mobilenet_ft'),
    class_weight=class_weight_dict,
    verbose=1
)
plot_history(history_mobilenet_ft, 'MobileNetV2')

# ============================================================
# MODEL 3: ResNet50
# ============================================================
# ResNet50 uses residual (skip) connections to enable training of much deeper
# networks. Same two-phase transfer learning strategy as MobileNetV2.
print("\n" + "="*50)
print("Training ResNet50...")
print("="*50)

inputs_r = keras.Input(shape=(224, 224, 3))
base_r = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs_r)
base_r.trainable = False  # Phase 1: freeze base

x = base_r.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
out_r = layers.Dense(4, activation='softmax')(x)
resnet = keras.Model(inputs_r, out_r, name='ResNet50')

resnet.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_resnet = resnet.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=get_callbacks('resnet'),
    class_weight=class_weight_dict,
    verbose=1
)

# Phase 2: fine-tune last 15 layers of ResNet50
print("\nFine-tuning ResNet50...")
base_r.trainable = True
for layer in base_r.layers[:-15]:
    layer.trainable = False

resnet.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_resnet_ft = resnet.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=get_callbacks('resnet_ft'),
    class_weight=class_weight_dict,
    verbose=1
)
plot_history(history_resnet_ft, 'ResNet50')

# ============================================================
# EVALUATE ALL MODELS
# ============================================================
acc_cnn    = evaluate_model(custom_cnn, val_generator, 'Custom_CNN')
acc_mobile = evaluate_model(mobilenet,  val_generator, 'MobileNetV2')
acc_resnet = evaluate_model(resnet,     val_generator, 'ResNet50')

# ============================================================
# MODEL COMPARISON CHART
# ============================================================
models_names = ['Custom CNN', 'MobileNetV2', 'ResNet50']
accuracies   = [acc_cnn, acc_mobile, acc_resnet]

plt.figure(figsize=(8, 5))
bars = plt.bar(models_names, [a*100 for a in accuracies],
               color=['#2196F3', '#4CAF50', '#FF5722'])
plt.ylim(0, 100)
plt.title('Model Comparison — Validation Accuracy', fontweight='bold', fontsize=14)
plt.ylabel('Accuracy (%)')
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 1,
             f'{acc*100:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'model_comparison.pdf', format='pdf', bbox_inches='tight')
plt.close()
print("Saved: model_comparison.pdf")

# ============================================================
# SAMPLE IMAGES GRID
# ============================================================
# Visual overview of the dataset — 4 sample images per class
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, cls in enumerate(CLASSES):
    cls_files = list(DATASET_PATH.glob(f'{cls}/*.jpg'))[:4]
    for j, img_path in enumerate(cls_files):
        img = keras.utils.load_img(str(img_path), target_size=(224, 224))
        img_array = keras.utils.img_to_array(img) / 255.0
        axes[i, j].imshow(img_array)
        axes[i, j].set_title(cls if j == 0 else '', fontsize=11, fontweight='bold')
        axes[i, j].axis('off')
plt.suptitle('Dataset Sample Images', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'sample_images.pdf', format='pdf', bbox_inches='tight')
plt.close()
print("Saved: sample_images.pdf")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
for name, acc in zip(models_names, accuracies):
    print(f"{name}: {acc*100:.2f}%")
print(f"\nAll outputs saved to: {OUTPUT_DIR}")