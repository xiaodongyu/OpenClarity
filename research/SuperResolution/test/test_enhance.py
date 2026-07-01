import cv2
import numpy as np
from src.enhance import adaptive_sharpen, bilinear_upscale, enhance, laplacian_edge_enhance


def make_step_edge(size: int = 100) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.uint8)
    img[:, size // 2 :] = 200
    return img


def gradient_magnitude(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1)
    return float(np.max(np.sqrt(gx**2 + gy**2)))


def test_bilinear_upscale_shape():
    img = np.zeros((50, 60, 3), dtype=np.uint8)
    result = bilinear_upscale(img, 4)
    assert result.shape == (200, 240, 3)


def test_adaptive_sharpen_dtype_and_shape():
    img = make_step_edge()
    result = adaptive_sharpen(img)
    assert result.dtype == np.uint8
    assert result.shape == img.shape


def test_adaptive_sharpen_increases_edge_gradient():
    img = make_step_edge()
    result = adaptive_sharpen(img)
    assert gradient_magnitude(result) > gradient_magnitude(img)


def test_adaptive_sharpen_leaves_flat_region_mostly_unchanged():
    flat = np.full((100, 100), 128, dtype=np.uint8)
    result = adaptive_sharpen(flat)
    assert np.max(np.abs(result.astype(int) - flat.astype(int))) < 5


def test_laplacian_edge_enhance_dtype_and_shape():
    img = make_step_edge()
    result = laplacian_edge_enhance(img)
    assert result.dtype == np.uint8
    assert result.shape == img.shape


def test_laplacian_edge_enhance_increases_edge_gradient():
    img = make_step_edge()
    result = laplacian_edge_enhance(img)
    assert gradient_magnitude(result) > gradient_magnitude(img)


def test_enhance_composed_dtype_and_shape():
    img = make_step_edge()
    result = enhance(img)
    assert result.dtype == np.uint8
    assert result.shape == img.shape


def test_enhance_color_image():
    bgr = cv2.cvtColor(make_step_edge(), cv2.COLOR_GRAY2BGR)
    result = enhance(bgr)
    assert result.ndim == 3
    assert result.dtype == np.uint8
