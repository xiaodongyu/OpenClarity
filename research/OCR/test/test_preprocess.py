import cv2
import numpy as np
import pytest
from src.preprocess import adaptive_threshold, deskew, preprocess, to_grayscale


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
