"""Grad-CAM helpers for the crop explorer tab."""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from constants import CLASSES
from model_utils import GRADCAM_LAYER_MOBILENET as GRADCAM_LAYER, preprocess_image


def make_gradcam(
    model: keras.Model, image, conv_layer_name: str = GRADCAM_LAYER
) -> tuple[np.ndarray, np.ndarray]:
    """Return heatmap (H, W) in [0, 1] and class probabilities."""
    img_tensor = preprocess_image(image)
    img_exp = tf.expand_dims(img_tensor, axis=0)

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_exp)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), predictions.numpy()[0]


def overlay_heatmap(image, heatmap: np.ndarray) -> np.ndarray:
    """Blend JET heatmap on top of the input image."""
    if hasattr(image, "convert"):
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    else:
        rgb = np.array(image, dtype=np.uint8)
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]

    h, w = rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 0.6, heatmap_colored, 0.4, 0)


def explain_crop(model: keras.Model, image) -> tuple[dict[str, float], str, np.ndarray]:
    """Return probabilities, summary text, and Grad-CAM overlay image."""
    heatmap, probs = make_gradcam(model, image)
    prob_dict = {cls: float(probs[i]) for i, cls in enumerate(CLASSES)}
    pred_cls = max(prob_dict, key=prob_dict.get)
    confidence = prob_dict[pred_cls]

    # Show overlay on the same 224×224 view used by the classifier.
    if hasattr(image, "convert"):
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    else:
        rgb = np.array(image, dtype=np.uint8)
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
    rgb_224 = cv2.resize(rgb, (224, 224))
    overlay = overlay_heatmap(rgb_224, heatmap)

    summary = (
        f"**Predicted:** {pred_cls}  \n"
        f"**Confidence:** {confidence:.0%}"
    )
    return prob_dict, summary, overlay
