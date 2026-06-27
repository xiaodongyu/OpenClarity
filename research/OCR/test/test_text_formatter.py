import pytest
from src.text_formatter import format_for_speech, structure_text

# bbox corners: [[x0,y0],[x1,y1],[x2,y2],[x3,y3]] (TL, TR, BR, BL)
def bbox(x, y, w=80, h=20):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


FIXTURE_TOKENS = [
    {"text": "Name:", "confidence": 0.95, "bbox": bbox(10, 10)},
    {"text": "John",  "confidence": 0.92, "bbox": bbox(100, 10)},
    {"text": "Age:",  "confidence": 0.90, "bbox": bbox(10, 50)},
    {"text": "30",    "confidence": 0.88, "bbox": bbox(100, 50)},
]


def test_structure_text_reading_order():
    result = structure_text(FIXTURE_TOKENS)
    lines = result.split("\n")
    assert lines[0] == "Name: John"
    assert lines[1] == "Age: 30"


def test_structure_text_empty():
    assert structure_text([]) == ""


def test_structure_text_single_token():
    tokens = [{"text": "Hello", "confidence": 0.9, "bbox": bbox(10, 10)}]
    assert structure_text(tokens) == "Hello"


def test_structure_text_two_columns_same_row():
    tokens = [
        {"text": "Right", "confidence": 0.9, "bbox": bbox(300, 10)},
        {"text": "Left",  "confidence": 0.9, "bbox": bbox(10, 10)},
    ]
    result = structure_text(tokens)
    assert result == "Left Right"


def test_format_for_speech_newlines_replaced():
    text = "Name: John\nAge: 30"
    result = format_for_speech(text)
    assert "\n" not in result
    assert "Name: John" in result
    assert "Age: 30" in result


def test_format_for_speech_strips_extra_spaces():
    text = "Hello   World"
    result = format_for_speech(text)
    assert "  " not in result


def test_format_for_speech_empty():
    assert format_for_speech("") == ""
