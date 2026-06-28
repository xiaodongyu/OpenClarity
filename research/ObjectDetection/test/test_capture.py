import numpy as np
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def reset_cap():
    import src.capture as capture
    capture._cap = None
    yield
    capture.release()


def _make_mock_cap(w=640, h=480):
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.read.return_value = (True, np.zeros((h, w, 3), dtype=np.uint8))
    cap.get.side_effect = lambda prop: {
        0: float(w),  # CAP_PROP_FRAME_WIDTH  == 3
        1: float(h),  # CAP_PROP_FRAME_HEIGHT == 4
    }.get(prop % 2, 0.0)  # simplification: width prop is even index
    return cap


def _make_cap_prop_mock(w=640, h=480):
    """Return a cap mock whose .get() responds to the real OpenCV prop IDs."""
    import cv2
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.read.return_value = (True, np.zeros((h, w, 3), dtype=np.uint8))
    prop_map = {
        cv2.CAP_PROP_FRAME_WIDTH: float(w),
        cv2.CAP_PROP_FRAME_HEIGHT: float(h),
    }
    cap.get.side_effect = lambda p: prop_map.get(p, 0.0)
    return cap


@patch("cv2.VideoCapture")
def test_capture_frame_shape(mock_vc):
    mock_vc.return_value = _make_cap_prop_mock()
    from src.capture import capture_frame
    frame = capture_frame()
    assert frame.shape == (480, 640, 3)


@patch("cv2.VideoCapture")
def test_get_frame_dims(mock_vc):
    mock_vc.return_value = _make_cap_prop_mock()
    from src.capture import get_frame_dims
    assert get_frame_dims() == (640, 480)


@patch("cv2.VideoCapture")
def test_capture_frame_failed_read(mock_vc):
    cap = _make_cap_prop_mock()
    cap.read.return_value = (False, None)
    mock_vc.return_value = cap
    from src.capture import capture_frame
    with pytest.raises(RuntimeError, match="Failed to read frame"):
        capture_frame()


@patch("cv2.VideoCapture")
def test_device_not_opened(mock_vc):
    cap = MagicMock()
    cap.isOpened.return_value = False
    mock_vc.return_value = cap
    import src.capture as capture
    capture._cap = None
    with pytest.raises(RuntimeError, match="Cannot open webcam"):
        capture.capture_frame()
