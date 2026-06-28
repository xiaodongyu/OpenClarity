import sys
import time
from ultralytics import YOLO

IGNORED_CLASSES = {
    "sports ball", "kite", "frisbee", "snowboard", "skis",
    "surfboard", "tennis racket", "baseball bat", "baseball glove",
}

_model: YOLO | None = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
    return _model


def detect(frame, conf: float = 0.5) -> list[dict]:
    model = _get_model()
    t0 = time.perf_counter()
    results = model(frame, conf=conf, verbose=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"inference: {elapsed_ms:.1f} ms", file=sys.stderr)

    detections = []
    for result in results:
        names = result.names
        for box in result.boxes:
            label = names[int(box.cls)]
            if label in IGNORED_CLASSES:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            detections.append({
                "label": label,
                "confidence": float(box.conf),
                "bbox": (x1, y1, x2, y2),
                "centroid": (cx, cy),
            })
    return detections
