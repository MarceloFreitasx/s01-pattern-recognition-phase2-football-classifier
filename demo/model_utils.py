"""Shared model loading for the Gradio demo."""

from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

from constants import CLASSES, IMG_SIZE

DEMO_DIR = Path(__file__).resolve().parent

MOBILENET_CANDIDATES = [
    DEMO_DIR / "model" / "best_mobilenet_ft.keras",
    DEMO_DIR.parent / "outputs" / "best_mobilenet_ft.keras",
]

CUSTOM_CNN_CANDIDATES = [
    DEMO_DIR / "model" / "best_custom_cnn.keras",
    DEMO_DIR.parent / "outputs" / "best_custom_cnn.keras",
]

GRADCAM_LAYER_MOBILENET = "Conv_1"
GRADCAM_LAYER_CUSTOM = "last_conv"


def build_custom_cnn() -> keras.Model:
    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="last_conv")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(len(CLASSES), activation="softmax")(x)
    return keras.Model(inp, out, name="Custom_CNN")


def build_mobilenetv2() -> keras.Model:
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    base = MobileNetV2(weights="imagenet", include_top=False, input_tensor=inputs)
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(len(CLASSES), activation="softmax")(x)
    return keras.Model(inputs, out, name="MobileNetV2")


def _resolve_path(candidates: list[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    searched = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(f"{label} weights not found. Expected one of:\n{searched}")


def load_custom_cnn() -> keras.Model:
    path = _resolve_path(CUSTOM_CNN_CANDIDATES, "Custom CNN")
    print(f"Loading Custom CNN from {path}...")
    model = build_custom_cnn()
    model.load_weights(str(path))
    print("Custom CNN loaded.")
    return model


def load_mobilenet() -> keras.Model:
    path = _resolve_path(MOBILENET_CANDIDATES, "MobileNetV2")
    print(f"Loading MobileNetV2 from {path}...")
    model = build_mobilenetv2()
    model.load_weights(str(path))
    print("MobileNetV2 loaded.")
    return model


def load_model() -> keras.Model:
    """Backward-compatible alias for MobileNetV2 (crop explorer tab)."""
    return load_mobilenet()


def preprocess_image(image) -> tf.Tensor:
    """RGB PIL or uint8 ndarray -> float tensor (224, 224, 3) in [0, 1]."""
    if hasattr(image, "convert"):
        rgb = image.convert("RGB")
        arr = tf.constant(rgb, dtype=tf.float32)
    else:
        arr = tf.constant(image, dtype=tf.float32)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
    resized = tf.image.resize(arr, (IMG_SIZE, IMG_SIZE))
    return resized / 255.0


def predict_probs(model: keras.Model, image) -> dict[str, float]:
    batch = tf.expand_dims(preprocess_image(image), axis=0)
    probs = model.predict(batch, verbose=0)[0]
    return {cls: float(probs[i]) for i, cls in enumerate(CLASSES)}
