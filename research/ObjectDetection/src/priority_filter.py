import os

MAX_DETECTIONS = int(os.environ.get("MAX_DETECTIONS", "4"))


def top_n(detections: list[dict], n: int = MAX_DETECTIONS) -> list[dict]:
    return sorted(detections, key=lambda d: d["confidence"], reverse=True)[:n]
