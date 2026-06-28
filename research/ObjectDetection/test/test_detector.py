import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _make_mock_results(detections: list[dict]):
    """Build a fake ultralytics Results object from a list of detection dicts."""
    names = {i: d["label"] for i, d in enumerate(detections)}

    boxes_list = []
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d["bbox"]
        box = MagicMock()
        box.cls = MagicMock(**{"__int__": lambda self, _i=i: _i})
        box.conf = MagicMock(**{"__float__": lambda self, _c=d["confidence"]: _c})
        xyxy_row = MagicMock()
        xyxy_row.tolist.return_value = [x1, y1, x2, y2]
        box.xyxy = [xyxy_row]
        boxes_list.append(box)

    result = MagicMock()
    result.names = names
    result.boxes = boxes_list
    return [result]


@pytest.fixture(autouse=True)
def reset_model():
    import src.detector as detector
    detector._model = None
    yield
    detector._model = None


@patch("src.detector.YOLO")
def test_detect_returns_known_object(mock_yolo_cls):
    mock_model = MagicMock()
    mock_model.return_value = _make_mock_results([
        {"label": "person", "confidence": 0.9, "bbox": (10, 20, 110, 220)},
    ])
    mock_yolo_cls.return_value = mock_model

    from src.detector import detect
    import cv2
    frame = cv2.imread("test/fixtures/indoor_scene.jpg")
    detections = detect(frame, conf=0.5)

    assert len(detections) == 1
    d = detections[0]
    assert d["label"] == "person"
    assert 0.0 < d["confidence"] <= 1.0
    x1, y1, x2, y2 = d["bbox"]
    assert x2 > x1 and y2 > y1
    cx, cy = d["centroid"]
    assert x1 <= cx <= x2
    assert y1 <= cy <= y2


@patch("src.detector.YOLO")
def test_ignored_classes_filtered(mock_yolo_cls):
    mock_model = MagicMock()
    mock_model.return_value = _make_mock_results([
        {"label": "person", "confidence": 0.8, "bbox": (0, 0, 100, 100)},
        {"label": "sports ball", "confidence": 0.9, "bbox": (200, 200, 300, 300)},
    ])
    mock_yolo_cls.return_value = mock_model

    from src.detector import detect
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detect(frame)

    labels = [d["label"] for d in detections]
    assert "sports ball" not in labels
    assert "person" in labels


@patch("src.detector.YOLO")
def test_detect_empty_frame(mock_yolo_cls):
    mock_model = MagicMock()
    mock_model.return_value = _make_mock_results([])
    mock_yolo_cls.return_value = mock_model

    from src.detector import detect
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert detect(frame) == []
