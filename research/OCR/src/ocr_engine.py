import sys
import time
from typing import Any

import numpy as np

# Lazy singleton — initialised on first call to recognize() to avoid import-time cost.
_ocr = None


def _get_ocr(lang: str = "en") -> Any:
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR  # noqa: PLC0415

        _ocr = PaddleOCR(use_angle_cls=True, lang=lang)
    return _ocr


def recognize(img: np.ndarray, conf_threshold: float = 0.6, lang: str = "en") -> list[dict]:
    """Return OCR tokens sorted top-to-bottom, left-to-right, above conf_threshold."""
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
