import base64
import time

import numpy as np
import pytest

from src.capture import encode_jpeg


def _make_frame(h=480, w=640):
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def test_encode_jpeg_returns_string():
    frame = _make_frame()
    result = encode_jpeg(frame)
    assert isinstance(result, str)
    assert len(result) > 0


def test_encode_jpeg_is_decodable():
    frame = _make_frame()
    b64 = encode_jpeg(frame)
    raw = base64.b64decode(b64)
    # JPEG magic bytes: FF D8
    assert raw[:2] == b"\xff\xd8"


def test_encode_jpeg_default_size():
    frame = _make_frame(480, 640)
    b64 = encode_jpeg(frame)
    raw = base64.b64decode(b64)
    import cv2, numpy as np
    arr = np.frombuffer(raw, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert decoded.shape == (768, 768, 3)


def test_encode_jpeg_custom_size():
    frame = _make_frame()
    b64 = encode_jpeg(frame, size=(256, 256))
    raw = base64.b64decode(b64)
    import cv2, numpy as np
    arr = np.frombuffer(raw, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert decoded.shape == (256, 256, 3)


def test_capture_interval_env(monkeypatch):
    monkeypatch.setenv("CAPTURE_INTERVAL_SEC", "1.5")
    import importlib
    import src.capture as cap_mod
    importlib.reload(cap_mod)
    assert cap_mod.CAPTURE_INTERVAL_SEC == pytest.approx(1.5, abs=1e-9)
