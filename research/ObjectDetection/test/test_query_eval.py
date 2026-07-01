from pathlib import Path

import numpy as np
import pytest

from src.query_eval import (
    SUPPORTED_OBJECT_ALIASES,
    answer_query,
    centroid_to_location,
    evaluate_queries,
    extract_query_object,
    load_ground_truth,
    load_prompt_examples,
    normalize_query_object,
    summarize_results,
)


def test_extract_query_object():
    assert extract_query_object("where is the cup?") == "cup"
    assert extract_query_object("Where is my toothbrush?") == "toothbrush"


def test_normalize_query_object():
    assert normalize_query_object("water cup") == "cup"
    assert normalize_query_object("phone") == "cell phone"
    assert normalize_query_object("rice cooker") is None


def test_supported_aliases_include_home_scene_targets():
    for label in ("cup", "toothbrush", "laptop", "chair"):
        assert SUPPORTED_OBJECT_ALIASES[label] == label


def test_centroid_to_location():
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    assert centroid_to_location((20, 20), frame.shape) == "top-left"
    assert centroid_to_location((150, 150), frame.shape) == "middle-middle"
    assert centroid_to_location((250, 260), frame.shape) == "bottom-right"


def test_answer_query_returns_na_for_unsupported_prompt():
    frame = np.zeros((300, 300, 3), dtype=np.uint8)

    def detect_fn(_frame, conf=0.5):
        raise AssertionError("detect_fn should not be called for unsupported prompts")

    location, target_label, is_supported, bbox = answer_query(frame, "where is the rice cooker?", detect_fn=detect_fn)
    assert location == "N/A"
    assert target_label is None
    assert is_supported is False


def test_answer_query_returns_na_when_supported_object_missing():
    frame = np.zeros((300, 300, 3), dtype=np.uint8)

    def detect_fn(_frame, conf=0.5):
        return [{"label": "chair", "confidence": 0.8, "bbox": (0, 0, 100, 100), "centroid": (50, 50)}]

    location, target_label, is_supported, bbox = answer_query(frame, "where is the laptop?", detect_fn=detect_fn)
    assert location == "N/A"
    assert target_label == "laptop"
    assert is_supported is True


def test_answer_query_selects_highest_confidence_match():
    frame = np.zeros((300, 300, 3), dtype=np.uint8)

    def detect_fn(_frame, conf=0.5):
        return [
            {"label": "cup", "confidence": 0.6, "bbox": (0, 0, 100, 100), "centroid": (50, 50)},
            {"label": "cup", "confidence": 0.9, "bbox": (100, 100, 200, 200), "centroid": (150, 150)},
        ]

    location, target_label, is_supported, bbox = answer_query(frame, "where is the cup?", detect_fn=detect_fn)
    assert location == "middle-middle"
    assert target_label == "cup"
    assert is_supported is True


def test_load_fixture_files():
    root = Path("test/fixtures/home_scene")
    prompts = load_prompt_examples(root / "object_detection_prompt.txt")
    ground_truth = load_ground_truth(root / "object_detection_groundtruth.txt")

    assert len(prompts) == 6
    assert ("IMG_4272.JPG", "laptop") in ground_truth
    assert ground_truth[("IMG_4272.JPG", "laptop")] == "top-left"


def test_evaluate_queries_and_summary():
    root = Path("test/fixtures/home_scene")

    def image_loader(path: str):
        return np.zeros((300, 300, 3), dtype=np.uint8) + (hash(path) % 2)

    path_to_centroid = {
        "IMG_4269.JPG": {
            "cup": (50, 150),
            "toothbrush": (150, 150),
        },
        "IMG_4270.JPG": {
            "laptop": (250, 250),
        },
        "IMG_4271.JPG": {
            "chair": (150, 250),
        },
        "IMG_4272.JPG": {
            "chair": (150, 250),
            "laptop": (50, 50),
        },
    }

    def detect_fn(frame, conf=0.5):
        image_name = current_image_name[0]
        detections = []
        for label, centroid in path_to_centroid[image_name].items():
            detections.append(
                {
                    "label": label,
                    "confidence": 0.9,
                    "bbox": (0, 0, 100, 100),
                    "centroid": centroid,
                }
            )
        return detections

    current_image_name = [None]

    def tracking_image_loader(path: str):
        current_image_name[0] = Path(path).name
        return image_loader(path)

    results = evaluate_queries(
        root / "object_detection_prompt.txt",
        root / "object_detection_groundtruth.txt",
        root,
        detect_fn=detect_fn,
        image_loader=tracking_image_loader,
    )

    assert len(results) == 6
    assert all(result.is_correct for result in results)

    summary = summarize_results(results)
    assert summary["total"] == 6
    assert summary["correct"] == 6
    assert summary["supported"] == 6
    assert summary["accuracy"] == pytest.approx(1.0)
