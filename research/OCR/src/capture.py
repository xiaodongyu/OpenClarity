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


def select_roi(frame: np.ndarray) -> tuple[int, int, int, int]:
    preset = os.environ.get("ROI_PRESET")
    if preset:
        parts = [int(v.strip()) for v in preset.split(",")]
        if len(parts) != 4:
            raise ValueError("ROI_PRESET must be 'x,y,w,h'")
        return tuple(parts)
    roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    return roi  # (x, y, w, h)


def crop(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    return frame[y : y + h, x : x + w]
