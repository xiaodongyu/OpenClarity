import sys
import time
from typing import Any

import cv2
import numpy as np

# Lazy singleton — initialised on first call to recognize() to avoid import-time cost.
_ocr = None


def _get_ocr(lang: str = "en") -> Any:
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR  # noqa: PLC0415

        _ocr = PaddleOCR(use_angle_cls=True, lang=lang)
    return _ocr


def _resolve_rotation_ambiguity(img: np.ndarray, angle_a: int, angle_b: int) -> int:
    """Choose between two 180°-apart orientations using OCR confidence as tiebreaker.

    Run a fast, downsampled OCR pass *without* the angle classifier on each
    candidate.  When text is right-side-up the recogniser returns high
    confidence; when it is 180° inverted the confidence drops sharply.
    The angle with the higher total confidence wins.
    """
    from src.preprocess import _ORIENT_CODES

    ocr = _get_ocr()
    best_angle, best_score = angle_a, -1.0

    for angle in (angle_a, angle_b):
        code = _ORIENT_CODES[angle]
        candidate = cv2.rotate(img, code) if code is not None else img.copy()

        # Aggressively downsample — we only need a confidence signal, not accuracy.
        h, w = candidate.shape[:2]
        if max(h, w) > 800:
            s = 800 / max(h, w)
            candidate = cv2.resize(candidate, (int(w * s), int(h * s)))
        if candidate.ndim == 2:
            candidate = cv2.cvtColor(candidate, cv2.COLOR_GRAY2BGR)

        try:
            raw = ocr.ocr(candidate, cls=False)  # no angle correction → inverted text scores low
        except Exception:
            continue

        score = sum(
            conf
            for page in (raw or []) if page
            for _, (_, conf) in page
            if conf > 0.3
        )
        print(
            f"[ocr_engine] ambiguity tiebreak angle={angle}° confidence_sum={score:.2f}",
            file=sys.stderr,
        )
        if score > best_score:
            best_score, best_angle = score, angle

    return best_angle


def recognize(img: np.ndarray, conf_threshold: float = 0.6, lang: str = "en") -> list[dict]:
    """Return OCR tokens sorted top-to-bottom, left-to-right, above conf_threshold."""
    from src.preprocess import orientation_scores, _ORIENT_CODES

    # Determine orientation from projection-profile variance.
    scores = orientation_scores(img)
    sorted_angles = sorted(scores, key=scores.__getitem__, reverse=True)
    best, second = sorted_angles[0], sorted_angles[1]

    # 0°/180° and 90°/270° are symmetric under row-reversal, so their variance
    # scores are always equal.  When those two tie for first place, use a quick
    # OCR pass (without angle correction) to pick the truly upright rotation.
    if abs(best - second) == 180 and scores[best] > 0:
        rel_gap = (scores[best] - scores[second]) / scores[best]
        if rel_gap < 0.02:
            best = _resolve_rotation_ambiguity(img, best, second)

    if best != 0:
        img = cv2.rotate(img, _ORIENT_CODES[best])

    # PaddleOCR performs best on colour images
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    ocr = _get_ocr(lang)
    t0 = time.perf_counter()
    raw = ocr.ocr(img, cls=True)
    elapsed = time.perf_counter() - t0
    print(f"[ocr_engine] recognition time: {elapsed:.3f}s", file=sys.stderr)

    results = []
    if raw is None:
        return results
    for page in raw:
        if page is None:
            continue
        for line in page:
            bbox, (text, conf) = line
            if conf < conf_threshold:
                continue
            results.append({"text": text, "confidence": float(conf), "bbox": bbox})

    # Sort top-to-bottom (min y of bbox), then left-to-right (min x of bbox)
    results.sort(key=lambda r: (min(pt[1] for pt in r["bbox"]), min(pt[0] for pt in r["bbox"])))
    return results
