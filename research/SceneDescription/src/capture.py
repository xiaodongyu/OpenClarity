import base64
import os

import cv2
import numpy as np


CAPTURE_INTERVAL_SEC: float = float(os.environ.get("CAPTURE_INTERVAL_SEC", "2.0"))


def capture_frame(device: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera device {device}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to read frame from camera")
    return frame


def encode_jpeg(frame: np.ndarray, size: tuple[int, int] = (768, 768)) -> str:
    """Resize frame to `size` and return a base64-encoded JPEG string."""
    resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", resized)
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")
