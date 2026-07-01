import os
import cv2
import numpy as np


def capture_frame(device: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera device {device}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to read frame from camera")
    return frame


def select_zoom_region(frame: np.ndarray) -> tuple[int, int, int, int]:
    preset = os.environ.get("ZOOM_PRESET")
    if preset:
        parts = [int(v.strip()) for v in preset.split(",")]
        if len(parts) != 4:
            raise ValueError("ZOOM_PRESET must be 'x,y,w,h'")
        return tuple(parts)
    roi = cv2.selectROI("Select zoom region", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select zoom region")
    return roi  # (x, y, w, h)


def crop(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    return frame[y : y + h, x : x + w]
