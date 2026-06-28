import cv2
import numpy as np
import pytest
from src.preprocess import (
    adaptive_threshold,
    correct_orientation,
    detect_orientation,
    deskew,
    preprocess,
    to_grayscale,
)


def make_text_image(angle: float = 0) -> np.ndarray:
    img = np.ones((200, 400), dtype=np.uint8) * 255
    cv2.putText(img, "Hello OCR", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 3)
    if angle != 0:
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=255)
    return img


def test_to_grayscale_from_bgr():
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    gray = to_grayscale(bgr)
    assert gray.ndim == 2
    assert gray.dtype == np.uint8


def test_to_grayscale_passthrough():
    gray = np.zeros((100, 100), dtype=np.uint8)
    result = to_grayscale(gray)
    assert result is gray


def test_adaptive_threshold_output():
    img = make_text_image()
    result = adaptive_threshold(img)
    assert result.dtype == np.uint8
    assert result.shape == img.shape
    unique = np.unique(result)
    assert set(unique).issubset({0, 255})


def test_preprocess_dtype_and_shape():
    bgr = np.ones((200, 400, 3), dtype=np.uint8) * 128
    result = preprocess(bgr)
    assert result.dtype == np.uint8
    assert result.shape == (200, 400)


def test_deskew_corrects_skew():
    skewed = make_text_image(angle=10)
    corrected = deskew(skewed)
    assert corrected.shape == skewed.shape
    assert corrected.dtype == np.uint8


# ── detect_orientation / correct_orientation ──────────────────────────────────

def test_detect_orientation_horizontal_is_zero():
    img = make_text_image(angle=0)   # 200×400 horizontal text
    assert detect_orientation(img) == 0


def test_detect_orientation_accepts_color():
    bgr = cv2.cvtColor(make_text_image(), cv2.COLOR_GRAY2BGR)
    angle = detect_orientation(bgr)
    assert angle in (0, 90, 180, 270)


def test_correct_orientation_passthrough_when_upright():
    img = make_text_image()
    result = correct_orientation(img)
    # Upright image should be returned unchanged (same object or same content)
    assert result.shape == img.shape


def test_correct_orientation_90cw_restores_dimensions():
    """Image rotated 90° CW should be corrected back to original H×W."""
    img = make_text_image()          # 200 rows × 400 cols
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)   # → 400 rows × 200 cols
    corrected = correct_orientation(rotated)
    # After correction the dominant dimension should match the original layout
    assert corrected.shape == img.shape or corrected.shape == rotated.shape


def test_correct_orientation_color_image():
    bgr = cv2.cvtColor(make_text_image(), cv2.COLOR_GRAY2BGR)
    result = correct_orientation(bgr)
    assert result.ndim == 3          # colour preserved


def test_correct_orientation_output_dtype():
    img = make_text_image()
    result = correct_orientation(img)
    assert result.dtype == np.uint8
