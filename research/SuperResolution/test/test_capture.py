import numpy as np
import pytest
from src.capture import crop, select_zoom_region


def test_crop_dimensions():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    roi = (100, 80, 400, 300)
    result = crop(frame, roi)
    assert result.shape == (300, 400, 3)


def test_crop_exact_region():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[80:380, 100:500] = 255
    roi = (100, 80, 400, 300)
    result = crop(frame, roi)
    assert result.shape == (300, 400, 3)
    assert np.all(result == 255)


def test_zoom_preset(monkeypatch):
    monkeypatch.setenv("ZOOM_PRESET", "10,20,300,200")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    roi = select_zoom_region(frame)
    assert roi == (10, 20, 300, 200)
    result = crop(frame, roi)
    assert result.shape == (200, 300, 3)


def test_zoom_preset_invalid(monkeypatch):
    monkeypatch.setenv("ZOOM_PRESET", "10,20,300")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        select_zoom_region(frame)
