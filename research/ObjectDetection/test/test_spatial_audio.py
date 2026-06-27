import sys
import types
from unittest.mock import MagicMock, patch

# Stub sounddevice before spatial_audio is imported (PortAudio may not be installed)
_sd_stub = types.ModuleType("sounddevice")
_sd_stub.play = MagicMock()
sys.modules.setdefault("sounddevice", _sd_stub)

import numpy as np
import pytest
from src.spatial_audio import centroid_to_pan, bbox_to_proximity, _make_earcon, emit, SAMPLE_RATE


def test_centroid_to_pan_left():
    assert centroid_to_pan(0, 640) == pytest.approx(-1.0)


def test_centroid_to_pan_right():
    assert centroid_to_pan(640, 640) == pytest.approx(1.0)


def test_centroid_to_pan_centre():
    assert centroid_to_pan(320, 640) == pytest.approx(0.0)


def test_bbox_to_proximity_full_frame():
    assert bbox_to_proximity((0, 0, 640, 480), 640, 480) == pytest.approx(1.0)


def test_bbox_to_proximity_quarter_frame():
    assert bbox_to_proximity((0, 0, 320, 240), 640, 480) == pytest.approx(0.25)


def test_earcon_full_left_energy():
    """cx=0 (pan=-1) → all energy in left channel, none in right."""
    buf = _make_earcon(freq=440.0, pan=-1.0, proximity=1.0)
    assert buf.shape[1] == 2
    left_energy = np.sum(buf[:, 0] ** 2)
    right_energy = np.sum(buf[:, 1] ** 2)
    assert left_energy > 0
    assert right_energy == pytest.approx(0.0, abs=1e-6)


def test_earcon_full_right_energy():
    """cx=frame_width (pan=+1) → all energy in right channel, none in left."""
    buf = _make_earcon(freq=440.0, pan=1.0, proximity=1.0)
    left_energy = np.sum(buf[:, 0] ** 2)
    right_energy = np.sum(buf[:, 1] ** 2)
    assert right_energy > 0
    assert left_energy == pytest.approx(0.0, abs=1e-6)


def test_emit_calls_play():
    sd = sys.modules["sounddevice"]
    sd.play.reset_mock()
    detections = [{"label": "person", "confidence": 0.9,
                   "bbox": (0, 0, 320, 240), "centroid": (0, 120)}]
    emit(detections, frame_dims=(640, 480))
    sd.play.assert_called_once()
    args, kwargs = sd.play.call_args
    buf = args[0]
    assert buf.shape[1] == 2
    assert kwargs.get("blocking") is False


def test_emit_empty_detections_no_play():
    sd = sys.modules["sounddevice"]
    sd.play.reset_mock()
    emit([], frame_dims=(640, 480))
    sd.play.assert_not_called()
