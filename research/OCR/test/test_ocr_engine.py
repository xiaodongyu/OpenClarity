"""
Tests for ocr_engine.recognize().

PaddleOCR is mocked so the tests verify filtering, sorting, and output
structure without requiring model weights or a GPU.
"""
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_mock_ocr(raw_output):
    mock_ocr = MagicMock()
    mock_ocr.ocr.return_value = raw_output
    return mock_ocr


# PaddleOCR bbox: list of 4 [x, y] corners (top-left, top-right, bottom-right, bottom-left)
_BBOX_TOP = [[10, 10], [200, 10], [200, 30], [10, 30]]
_BBOX_BOTTOM = [[10, 50], [200, 50], [200, 70], [10, 70]]
_BBOX_RIGHT = [[300, 10], [500, 10], [500, 30], [300, 30]]


RAW_TWO_LINES = [
    [
        [_BBOX_BOTTOM, ("World", 0.95)],
        [_BBOX_TOP, ("Hello", 0.90)],
    ]
]

RAW_LOW_CONF = [
    [
        [_BBOX_TOP, ("noise", 0.30)],
        [_BBOX_BOTTOM, ("signal", 0.85)],
    ]
]

RAW_TWO_COLUMNS = [
    [
        [_BBOX_RIGHT, ("Right", 0.80)],
        [_BBOX_TOP, ("Left", 0.80)],
    ]
]


def _recognize_with_mock(raw, img=None, **kwargs):
    import src.ocr_engine as engine

    engine._ocr = _make_mock_ocr(raw)
    if img is None:
        img = np.zeros((100, 200, 3), dtype=np.uint8)
    return engine.recognize(img, **kwargs)


def setup_function():
    import src.ocr_engine as engine

    engine._ocr = None


def test_tokens_sorted_top_to_bottom():
    tokens = _recognize_with_mock(RAW_TWO_LINES)
    assert len(tokens) == 2
    assert tokens[0]["text"] == "Hello"
    assert tokens[1]["text"] == "World"


def test_confidence_filtering():
    tokens = _recognize_with_mock(RAW_LOW_CONF, conf_threshold=0.6)
    assert len(tokens) == 1
    assert tokens[0]["text"] == "signal"


def test_tokens_sorted_left_to_right():
    tokens = _recognize_with_mock(RAW_TWO_COLUMNS)
    assert tokens[0]["text"] == "Left"
    assert tokens[1]["text"] == "Right"


def test_output_structure():
    tokens = _recognize_with_mock(RAW_TWO_LINES)
    for tok in tokens:
        assert "text" in tok
        assert "confidence" in tok
        assert "bbox" in tok
        assert isinstance(tok["confidence"], float)


def test_empty_result():
    tokens = _recognize_with_mock([[]])
    assert tokens == []


def test_none_result():
    tokens = _recognize_with_mock(None)
    assert tokens == []
