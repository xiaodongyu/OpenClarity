import cv2
import numpy as np

# cv2.rotate codes for each clockwise angle
_ORIENT_CODES: dict[int, int | None] = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def adaptive_threshold(img: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )


def deskew(img: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    if lines is None:
        return img
    angles = []
    for rho, theta in lines[:, 0]:
        angle = np.degrees(theta) - 90
        if abs(angle) <= 45:
            angles.append(angle)
    if not angles:
        return img
    skew = float(np.median(angles))
    if abs(skew) <= 1.0:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)


def orientation_scores(img: np.ndarray) -> dict[int, float]:
    """Return the horizontal projection-profile variance for each candidate rotation.

    Keys are clockwise rotation angles (0, 90, 180, 270).  Higher variance
    indicates that the text rows are horizontal at that rotation, so the angle
    with the highest score is the one to apply.

    Note: 0°/180° and 90°/270° pairs are mathematically guaranteed to produce
    identical variance values (reversing row order does not change variance).
    Use an external tiebreaker such as OCR confidence to choose between them.
    """
    gray = to_grayscale(img) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    scale = min(1.0, 600.0 / max(h, w, 1))
    if scale < 1.0:
        gray = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))))
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return {
        angle: float(np.var(np.sum(
            cv2.rotate(binary, code) if code is not None else binary,
            axis=1, dtype=np.float64,
        )))
        for angle, code in _ORIENT_CODES.items()
    }


def detect_orientation(img: np.ndarray) -> int:
    """Return the clockwise rotation angle (0, 90, 180, or 270) to apply to make
    text horizontal and upright, chosen by maximising horizontal projection-profile
    variance on a downsampled binary image.
    """
    scores = orientation_scores(img)
    return max(scores, key=scores.__getitem__)


def correct_orientation(img: np.ndarray) -> np.ndarray:
    """Rotate *img* clockwise by the detected orientation angle to make text upright.
    Works on both colour (BGR) and grayscale images.
    """
    angle = detect_orientation(img)
    if angle == 0:
        return img
    return cv2.rotate(img, _ORIENT_CODES[angle])


def preprocess(img: np.ndarray) -> np.ndarray:
    gray = to_grayscale(img)
    thresh = adaptive_threshold(gray)
    return deskew(thresh)
