import sys
import types
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

# Stub sounddevice before any spatial_audio import
_sd_stub = types.ModuleType("sounddevice")
_sd_stub.play = MagicMock()
sys.modules.setdefault("sounddevice", _sd_stub)


def _fake_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _fake_detections():
    return [
        {"label": "person", "confidence": 0.9, "bbox": (10, 10, 100, 200), "centroid": (55, 105)},
        {"label": "chair",  "confidence": 0.7, "bbox": (200, 50, 400, 300), "centroid": (300, 175)},
    ]


@patch("src.pipeline.emit")
@patch("src.pipeline.top_n", return_value=_fake_detections())
@patch("src.pipeline.detect", return_value=_fake_detections())
@patch("src.pipeline.get_frame_dims", return_value=(640, 480))
@patch("src.pipeline.capture_frame", side_effect=[_fake_frame(), _fake_frame(), KeyboardInterrupt])
@patch("src.pipeline.release_camera")
def test_pipeline_calls_each_stage(mock_release, mock_capture, mock_dims,
                                   mock_detect, mock_top_n, mock_emit):
    from src.pipeline import run, parse_args
    args = parse_args(["--conf", "0.4"])
    run(args)

    assert mock_capture.call_count == 3       # 2 frames + 1 that raises KeyboardInterrupt
    assert mock_detect.call_count == 2
    assert mock_top_n.call_count == 2
    assert mock_emit.call_count == 2
    mock_release.assert_called_once()


@patch("src.pipeline.emit")
@patch("src.pipeline.top_n", return_value=_fake_detections())
@patch("src.pipeline.detect", return_value=_fake_detections())
@patch("src.pipeline.get_frame_dims", return_value=(640, 480))
@patch("src.pipeline.capture_frame", side_effect=[_fake_frame(), KeyboardInterrupt])
@patch("src.pipeline.release_camera")
def test_no_audio_prints_detections(mock_release, mock_capture, mock_dims,
                                    mock_detect, mock_top_n, mock_emit, capsys):
    from src.pipeline import run, parse_args
    args = parse_args(["--no-audio"])
    run(args)

    mock_emit.assert_not_called()
    captured = capsys.readouterr()
    assert "person" in captured.out


@patch("src.pipeline.emit")
@patch("src.pipeline.top_n", return_value=_fake_detections())
@patch("src.pipeline.detect", return_value=_fake_detections())
@patch("src.pipeline.get_frame_dims", return_value=(640, 480))
@patch("src.pipeline.capture_frame", side_effect=[_fake_frame(), KeyboardInterrupt])
@patch("src.pipeline.release_camera")
def test_conf_flag_forwarded(mock_release, mock_capture, mock_dims,
                              mock_detect, mock_top_n, mock_emit):
    from src.pipeline import run, parse_args
    args = parse_args(["--conf", "0.8"])
    run(args)
    _, kwargs = mock_detect.call_args
    assert kwargs.get("conf") == pytest.approx(0.8)
