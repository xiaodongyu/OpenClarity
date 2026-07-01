import re
from dataclasses import dataclass
from pathlib import Path

import cv2

from src.detector import detect

SUPPORTED_OBJECT_ALIASES: dict[str, str] = {
    "backpack": "backpack",
    "bottle": "bottle",
    "cell phone": "cell phone",
    "cellphone": "cell phone",
    "chair": "chair",
    "cup": "cup",
    "laptop": "laptop",
    "phone": "cell phone",
    "suitcase": "suitcase",
    "table": "dining table",
    "toothbrush": "toothbrush",
    "water cup": "cup",
}


@dataclass(frozen=True)
class QueryExample:
    image_name: str
    prompt: str


@dataclass(frozen=True)
class GroundTruthExample:
    image_name: str
    label: str
    location: str


@dataclass(frozen=True)
class QueryResult:
    image_name: str
    prompt: str
    target_label: str | None
    predicted_location: str
    expected_location: str | None
    is_supported: bool
    is_correct: bool
    bbox: tuple[int, int, int, int] | None = None  # (x1, y1, x2, y2) of best detection


def extract_query_object(prompt: str) -> str | None:
    match = re.search(r"where is (?:my |the )?(.+?)\?\s*$", prompt.strip(), re.IGNORECASE)
    if match is None:
        return None
    return match.group(1).strip().lower()


def normalize_query_object(query_object: str | None) -> str | None:
    if query_object is None:
        return None
    normalized = re.sub(r"\s+", " ", query_object.strip().lower())
    return SUPPORTED_OBJECT_ALIASES.get(normalized)


def centroid_to_location(centroid: tuple[int, int], frame_shape: tuple[int, int, int] | tuple[int, int]) -> str:
    frame_height, frame_width = frame_shape[:2]
    cx, cy = centroid

    x_idx = min(2, max(0, int((cx * 3) / max(frame_width, 1))))
    y_idx = min(2, max(0, int((cy * 3) / max(frame_height, 1))))

    horizontal = ("left", "middle", "right")[x_idx]
    vertical = ("top", "middle", "bottom")[y_idx]
    return f"{vertical}-{horizontal}"


def answer_query(
    frame,
    prompt: str,
    conf: float = 0.5,
    detect_fn=detect,
) -> tuple[str, str | None, bool, tuple[int, int, int, int] | None]:
    target_label = normalize_query_object(extract_query_object(prompt))
    if target_label is None:
        return ("N/A", None, False, None)

    detections = detect_fn(frame, conf=conf)
    matches = [d for d in detections if d["label"] == target_label]
    if not matches:
        return ("N/A", target_label, True, None)

    best = max(matches, key=lambda d: d["confidence"])
    location = centroid_to_location(best["centroid"], frame.shape)
    return (location, target_label, True, best["bbox"])


def load_prompt_examples(path: str | Path) -> list[QueryExample]:
    examples = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        image_name, prompt = line.split(":", 1)
        examples.append(QueryExample(image_name=image_name.strip(), prompt=prompt.strip()))
    return examples


def load_ground_truth(path: str | Path) -> dict[tuple[str, str], str]:
    ground_truth = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        image_name, label, location = [part.strip() for part in line.split(":", 2)]
        ground_truth[(image_name, label)] = location
    return ground_truth


def evaluate_queries(
    prompts_path: str | Path,
    ground_truth_path: str | Path,
    image_dir: str | Path,
    conf: float = 0.5,
    detect_fn=detect,
    image_loader=cv2.imread,
) -> list[QueryResult]:
    prompt_examples = load_prompt_examples(prompts_path)
    ground_truth = load_ground_truth(ground_truth_path)

    results = []
    for example in prompt_examples:
        image_path = Path(image_dir) / example.image_name
        frame = image_loader(str(image_path))
        if frame is None:
            raise RuntimeError(f"Failed to load image: {image_path}")

        predicted_location, target_label, is_supported, bbox = answer_query(
            frame,
            example.prompt,
            conf=conf,
            detect_fn=detect_fn,
        )
        expected_location = None if target_label is None else ground_truth.get((example.image_name, target_label))
        is_correct = predicted_location == expected_location
        results.append(
            QueryResult(
                image_name=example.image_name,
                prompt=example.prompt,
                target_label=target_label,
                predicted_location=predicted_location,
                expected_location=expected_location,
                is_supported=is_supported,
                is_correct=is_correct,
                bbox=bbox,
            )
        )
    return results


def summarize_results(results: list[QueryResult]) -> dict[str, float | int]:
    total = len(results)
    correct = sum(result.is_correct for result in results)
    supported = sum(result.is_supported for result in results)
    return {
        "total": total,
        "correct": correct,
        "supported": supported,
        "accuracy": (correct / total) if total else 0.0,
    }
