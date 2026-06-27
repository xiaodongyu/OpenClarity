"""
Integration test for the main pipeline.

All external dependencies (camera, PaddleOCR, pyttsx3) are mocked.
Verifies that run_cycle() returns expected text end-to-end with --no-tts.
"""
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Permanently inject fakes before pipeline is imported so cv2 loads once and stays cached.
sys.modules.setdefault("paddleocr", MagicMock())
sys.modules.setdefault("pyttsx3", MagicMock())

import src.pipeline  # noqa: E402
from src.pipeline import run_cycle  # noqa: E402


def _bbox(x, y, w=80, h=20):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


MOCK_TOKENS = [
    {"text": "OPEN", "confidence": 0.95, "bbox": _bbox(10, 10)},
    {"text": "HOURS", "confidence": 0.92, "bbox": _bbox(100, 10)},
    {"text": "9am-5pm", "confidence": 0.90, "bbox": _bbox(10, 50)},
]

_FRAME = np.ones((200, 400, 3), dtype=np.uint8) * 128


def test_run_cycle_no_tts(capsys):
    with (
        patch.object(src.pipeline, "capture_frame", return_value=_FRAME),
        patch.object(src.pipeline, "recognize", return_value=MOCK_TOKENS),
    ):
        result = run_cycle(roi=None, no_tts=True, lang="en")

    captured = capsys.readouterr()
    assert "OPEN" in result
    assert "HOURS" in result
    assert "9am-5pm" in result
    assert "OPEN" in captured.out


def test_run_cycle_with_roi(capsys):
    with (
        patch.object(src.pipeline, "capture_frame", return_value=_FRAME),
        patch.object(src.pipeline, "recognize", return_value=MOCK_TOKENS),
    ):
        result = run_cycle(roi=(0, 0, 400, 200), no_tts=True, lang="en")

    assert "OPEN" in result


def test_run_cycle_empty_ocr(capsys):
    with (
        patch.object(src.pipeline, "capture_frame", return_value=_FRAME),
        patch.object(src.pipeline, "recognize", return_value=[]),
    ):
        result = run_cycle(roi=None, no_tts=True, lang="en")

    assert result == ""
