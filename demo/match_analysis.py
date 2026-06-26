"""Annotate full match frames — YOLO detection or dataset ground-truth boxes."""

from io import BytesIO
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from constants import CLASSES, IMG_SIZE
from detector import detect_objects

DEMO_DIR = Path(__file__).resolve().parent
MATCHES_DIR = DEMO_DIR / "examples" / "matches"

COLORS = {
    "ball": "#2196F3",
    "goalkeeper": "#FF9800",
    "player": "#4CAF50",
    "referee": "#E91E63",
}

MIN_SIZE = 20
CONF_THRESHOLD = 0.6


def list_match_examples() -> list[str]:
    if not MATCHES_DIR.exists():
        return []
    return sorted(p.stem for p in MATCHES_DIR.glob("*.jpg"))


def _match_paths(name: str) -> tuple[Path, Path]:
    image_path = MATCHES_DIR / f"{name}.jpg"
    label_path = MATCHES_DIR / f"{name}.txt"
    if not image_path.exists() or not label_path.exists():
        raise FileNotFoundError(f"Match example not found: {name}")
    return image_path, label_path


def _parse_yolo_labels(label_path: Path, width: int, height: int) -> list[dict]:
    objects = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            x1 = int((cx - bw / 2) * width)
            y1 = int((cy - bh / 2) * height)
            x2 = int((cx + bw / 2) * width)
            y2 = int((cy + bh / 2) * height)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            box_w, box_h = x2 - x1, y2 - y1
            if box_w < 5 or box_h < 5:
                continue
            true_cls = CLASSES[cls_id]
            objects.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "box_w": box_w,
                    "box_h": box_h,
                    "true_cls": true_cls,
                }
            )
    return objects


def _classify_crop(model, img_rgb: np.ndarray, x1: int, y1: int, box_w: int, box_h: int):
    import tensorflow as tf

    from model_utils import preprocess_image

    crop = img_rgb[y1 : y1 + box_h, x1 : x1 + box_w]
    tensor = preprocess_image(Image.fromarray(crop))
    probs = model.predict(tf.expand_dims(tensor, 0), verbose=0)[0]
    pred_cls = CLASSES[int(np.argmax(probs))]
    confidence = float(np.max(probs))
    return pred_cls, confidence


def _render_annotated_frame(
    img_rgb: np.ndarray,
    objects: list[dict],
    model,
    *,
    title: str,
    use_ground_truth: bool,
) -> tuple[Image.Image, dict[str, int]]:
    """Mirror outputs/inference_demo.png: box colour from GT or YOLO, label shows classifier."""
    class_counts = {cls: 0 for cls in CLASSES}

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.imshow(img_rgb)

    for obj in objects:
        x1, y1 = obj["x1"], obj["y1"]
        box_w, box_h = obj["box_w"], obj["box_h"]

        if use_ground_truth:
            base_cls = obj["true_cls"]
        else:
            base_cls = obj["det_cls"]

        color = COLORS[base_cls]
        class_counts[base_cls] += 1

        if box_w >= MIN_SIZE and box_h >= MIN_SIZE:
            pred_cls, confidence = _classify_crop(model, img_rgb, x1, y1, box_w, box_h)
        else:
            pred_cls, confidence = base_cls, 0.0

        if use_ground_truth:
            if confidence >= CONF_THRESHOLD:
                label = f"{base_cls} → {pred_cls} {confidence:.0%}"
            else:
                label = base_cls
        elif confidence >= CONF_THRESHOLD and pred_cls != base_cls:
            label = f"{base_cls} → {pred_cls} {confidence:.0%}"
        elif confidence >= CONF_THRESHOLD:
            label = f"{base_cls} {confidence:.0%}"
        else:
            label = base_cls

        rect = patches.Rectangle(
            (x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            max(y1 - 3, 8),
            label,
            fontsize=6,
            color="white",
            fontweight="bold",
            bbox=dict(facecolor=color, edgecolor="none", pad=1.2, alpha=0.9),
        )

    handles = [
        patches.Patch(color=COLORS[cls], label=f"{cls} ({class_counts[cls]})")
        for cls in CLASSES
        if class_counts[cls] > 0
    ]
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB"), class_counts


def annotate_uploaded_match(model, image) -> tuple[Image.Image, str]:
    """Detect with YOLO, classify crops with MobileNetV2 (same logic as inference_demo.py)."""
    if hasattr(image, "convert"):
        img_rgb = np.array(image.convert("RGB"))
    else:
        img_rgb = np.array(image)
        if img_rgb.shape[-1] == 4:
            img_rgb = img_rgb[..., :3]

    objects = detect_objects(img_rgb)
    if not objects:
        raise ValueError(
            "No objects detected. Try a clearer match frame with visible players, "
            "referees, goalkeepers, or the ball."
        )

    result_img, class_counts = _render_annotated_frame(
        img_rgb,
        objects,
        model,
        title="Football Role Classification — YOLO Detection + MobileNetV2",
        use_ground_truth=False,
    )

    summary_lines = [
        "**Uploaded match frame**",
        "",
        f"- **Objects detected:** {sum(class_counts.values())}",
    ]
    for cls in CLASSES:
        if class_counts[cls]:
            summary_lines.append(f"- **{cls}:** {class_counts[cls]}")
    summary_lines.append("")
    summary_lines.append(
        "Box colours follow **YOLO detections**; labels show "
        "`YOLO class → MobileNetV2 prediction`, like `inference_demo.png`."
    )
    return result_img, "\n".join(summary_lines)


def annotate_match(model, match_name: str) -> tuple[Image.Image, str]:
    """Use dataset ground-truth boxes — identical workflow to inference_demo.py."""
    image_path, label_path = _match_paths(match_name)
    img_bgr = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    height, width = img_rgb.shape[:2]
    objects = _parse_yolo_labels(label_path, width, height)

    result_img, class_counts = _render_annotated_frame(
        img_rgb,
        objects,
        model,
        title="Football Role Classification — Ground Truth + MobileNetV2",
        use_ground_truth=True,
    )

    summary_lines = [f"**Example frame:** {match_name}", ""]
    for cls in CLASSES:
        summary_lines.append(f"- **{cls}:** {class_counts[cls]} objects")
    summary_lines.append("")
    summary_lines.append(
        "Uses **dataset annotations** for boxes and colours — the same setup as "
        "`outputs/inference_demo.png`."
    )
    return result_img, "\n".join(summary_lines)
