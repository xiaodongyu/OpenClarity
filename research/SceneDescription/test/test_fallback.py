"""
Tests for fallback.py — requests is mocked so no Ollama instance needed.
"""
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.modules.setdefault("requests", MagicMock())

from src.fallback import describe_scene_local  # noqa: E402

IMAGE_B64 = "aGVsbG8="


def _mock_requests_post(response_text: str):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": response_text}
    mock_resp.raise_for_status.return_value = None
    mock_requests = sys.modules["requests"]
    mock_requests.post.side_effect = None  # clear any leftover side_effect
    mock_requests.post.return_value = mock_resp
    mock_requests.exceptions.ConnectionError = ConnectionError
    mock_requests.exceptions.Timeout = TimeoutError
    return mock_requests


def test_returns_description():
    _mock_requests_post("A crowded lobby with people walking.")
    result = describe_scene_local(IMAGE_B64)
    assert result == "A crowded lobby with people walking."


def test_sends_image_b64_in_payload():
    mock_requests = _mock_requests_post("desc")
    describe_scene_local(IMAGE_B64)

    call_kwargs = mock_requests.post.call_args.kwargs
    assert IMAGE_B64 in call_kwargs["json"]["images"]


def test_uses_llava_model():
    mock_requests = _mock_requests_post("desc")
    describe_scene_local(IMAGE_B64)

    call_kwargs = mock_requests.post.call_args.kwargs
    assert call_kwargs["json"]["model"] == "llava:7b"


def test_connection_error_raises_runtime_error():
    mock_requests = sys.modules["requests"]
    mock_requests.exceptions.ConnectionError = ConnectionError
    mock_requests.exceptions.Timeout = TimeoutError
    mock_requests.post.side_effect = ConnectionError("refused")

    with pytest.raises(RuntimeError, match="Ollama is not running"):
        describe_scene_local(IMAGE_B64)


def test_timeout_raises_timeout_error():
    mock_requests = sys.modules["requests"]
    mock_requests.exceptions.ConnectionError = ConnectionError
    mock_requests.exceptions.Timeout = TimeoutError
    mock_requests.post.side_effect = TimeoutError("timed out")

    with pytest.raises(TimeoutError):
        describe_scene_local(IMAGE_B64)


def test_strips_whitespace():
    _mock_requests_post("  A hallway.  ")
    result = describe_scene_local(IMAGE_B64)
    assert result == "A hallway."
