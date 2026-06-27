import pytest
from src.priority_filter import top_n


def _make_detections(confidences: list[float]) -> list[dict]:
    return [{"label": "obj", "confidence": c, "bbox": (0, 0, 1, 1), "centroid": (0, 0)}
            for c in confidences]


def test_top_n_returns_four_from_ten():
    detections = _make_detections([0.1 * i for i in range(1, 11)])
    result = top_n(detections, n=4)
    assert len(result) == 4
    assert [d["confidence"] for d in result] == pytest.approx([1.0, 0.9, 0.8, 0.7])


def test_top_n_sorted_descending():
    detections = _make_detections([0.3, 0.9, 0.5, 0.7, 0.1])
    result = top_n(detections, n=3)
    confs = [d["confidence"] for d in result]
    assert confs == sorted(confs, reverse=True)


def test_top_n_fewer_than_n():
    detections = _make_detections([0.8, 0.6])
    result = top_n(detections, n=4)
    assert len(result) == 2


def test_top_n_empty():
    assert top_n([], n=4) == []
