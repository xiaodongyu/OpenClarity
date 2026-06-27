import cv2
import numpy as np

_cap: cv2.VideoCapture | None = None

_WIDTH = 640
_HEIGHT = 480


def _get_cap() -> cv2.VideoCapture:
    global _cap
    if _cap is None or not _cap.isOpened():
        _cap = cv2.VideoCapture(0)
        _cap.set(cv2.CAP_PROP_FRAME_WIDTH, _WIDTH)
        _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _HEIGHT)
        if not _cap.isOpened():
            raise RuntimeError("Cannot open webcam device 0")
    return _cap


def capture_frame() -> np.ndarray:
    cap = _get_cap()
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Failed to read frame from webcam")
    return frame


def get_frame_dims() -> tuple[int, int]:
    cap = _get_cap()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return (w, h)


def release() -> None:
    global _cap
    if _cap is not None:
        _cap.release()
        _cap = None
