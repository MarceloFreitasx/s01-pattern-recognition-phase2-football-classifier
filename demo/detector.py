"""YOLO object detection for user-uploaded match frames."""

from pathlib import Path

from constants import CLASSES

DEMO_DIR = Path(__file__).resolve().parent

YOLO_CANDIDATES = [
    DEMO_DIR / "model" / "yolo_detector.pt",
    DEMO_DIR / "model" / "detector" / "weights" / "best.pt",
    DEMO_DIR.parent / "demo" / "model" / "detector" / "weights" / "best.pt",
]

_yolo_model = None


def _resolve_yolo_path() -> Path:
    for path in YOLO_CANDIDATES:
        if path.exists():
            return path
    searched = "\n".join(f"  - {p}" for p in YOLO_CANDIDATES)
    raise FileNotFoundError(
        "YOLO detector weights not found. Expected one of:\n"
        f"{searched}\n\n"
        "Train the detector with:\n"
        "  yolo detect train model=yolov8n.pt data=demo/yolo_data.yaml "
        "epochs=25 imgsz=640 project=demo/model name=detector\n"
        "Then copy demo/model/detector/weights/best.pt to demo/model/yolo_detector.pt"
    )


def load_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO

        path = _resolve_yolo_path()
        print(f"Loading YOLO detector from {path}...")
        _yolo_model = YOLO(str(path))
        print("YOLO detector loaded.")
    return _yolo_model


def detect_objects(image_rgb, conf: float = 0.25, iou: float = 0.45) -> list[dict]:
    """
    Run YOLO on an RGB numpy array or PIL image.

    Returns a list of dicts with keys: x1, y1, box_w, box_h, det_cls.
    """
    import numpy as np

    detector = load_yolo()
    if hasattr(image_rgb, "convert"):
        frame = np.array(image_rgb.convert("RGB"))
    else:
        frame = np.array(image_rgb)
        if frame.shape[-1] == 4:
            frame = frame[..., :3]

    results = detector.predict(frame, conf=conf, iou=iou, verbose=False, device="cpu")[0]
    objects = []
    if results.boxes is None:
        return objects

    names = results.names
    for box in results.boxes:
        cls_id = int(box.cls[0])
        det_cls = names[cls_id] if isinstance(names, dict) else names[cls_id]
        if det_cls not in CLASSES:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        box_w, box_h = x2 - x1, y2 - y1
        if box_w < 5 or box_h < 5:
            continue
        objects.append(
            {
                "x1": max(0, x1),
                "y1": max(0, y1),
                "box_w": box_w,
                "box_h": box_h,
                "det_cls": det_cls,
                "det_conf": float(box.conf[0]),
            }
        )
    return objects
