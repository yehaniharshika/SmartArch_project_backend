"""
SmartArch — services/extraction/yolo_service.py

ONLY job: run the trained YOLOv8 model (best.pt) on an image and
return a list of raw detections (wall/door/window + pixel bbox).

Output format matches exactly what room_boundary_service.py expects:
  [{"label": "wall"|"door"|"window", "confidence": 0.0-1.0,
    "x1": .., "y1": .., "x2": .., "y2": ..}, ...]
"""
import os
import numpy as np
from config import Config

_yolo_model = None


def _get_yolo():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model

    from ultralytics import YOLO
    import torch

    # PyTorch 2.6+ blocks loading some checkpoints by default for
    # security reasons. Our best.pt is OUR OWN trained file — safe to allow.
    original_torch_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)
    torch.load = _patched_load

    weights = str(Config.WEIGHTS_PATH)
    if not os.path.exists(weights):
        raise FileNotFoundError(f"YOLO weights not found at: {weights}")

    print(f"[YOLO] Loading model from {weights} ...")
    _yolo_model = YOLO(weights)
    print("[YOLO] ✅ Model ready")
    return _yolo_model


def detect_structural_elements(img: np.ndarray) -> list:
    """
    Returns a list of dicts, one per detection:
      {"label": "wall"|"door"|"window", "confidence": 0.0-1.0,
       "x1": .., "y1": .., "x2": .., "y2": ..}
    """
    model = _get_yolo()
    results = model(
        img,
        conf=Config.YOLO_CONF,
        iou=Config.YOLO_IOU,
        imgsz=Config.YOLO_IMG_SIZE,
        verbose=False,
    )

    names = results[0].names
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "label": names[int(box.cls[0])].lower(),
            "confidence": float(box.conf[0]),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })

    counts = {}
    for d in detections:
        counts[d["label"]] = counts.get(d["label"], 0) + 1
    print("[YOLO] Detected: " +
          " | ".join(f"{k}:{v}" for k, v in counts.items()) +
          f"  (total={len(detections)})")

    return detections