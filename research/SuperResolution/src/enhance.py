import cv2
import numpy as np


def bilinear_upscale(img: np.ndarray, scale: int) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)


def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _broadcast_like(map2d: np.ndarray, img: np.ndarray) -> np.ndarray:
    return cv2.merge([map2d] * 3) if img.ndim == 3 else map2d


def adaptive_sharpen(img: np.ndarray, amount: float = 1.0, radius: int = 9) -> np.ndarray:
    """Unsharp mask whose strength scales with local contrast.

    Flat/low-contrast regions (little to gain, noise-prone) are left mostly
    untouched, while high-local-variance regions (text and object edges) get
    the full sharpening amount.
    """
    gray = _to_gray(img).astype(np.float32)
    local_mean = cv2.blur(gray, (radius, radius))
    local_var = np.clip(cv2.blur(gray * gray, (radius, radius)) - local_mean**2, 0, None)
    weight = amount * (local_var / (local_var.max() + 1e-6))

    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    detail = img.astype(np.float32) - blurred.astype(np.float32)
    sharpened = img.astype(np.float32) + _broadcast_like(weight, img) * detail
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def laplacian_edge_enhance(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
    gray = _to_gray(img)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    enhanced = img.astype(np.float32) - strength * _broadcast_like(laplacian, img)
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def enhance(img: np.ndarray) -> np.ndarray:
    """Fallback / post-processing enhancement: adaptive sharpen + Laplacian edge boost."""
    return laplacian_edge_enhance(adaptive_sharpen(img))
