"""
Integration tests for pipeline.py.

All external dependencies are mocked. Verifies that three full cycles
complete without error and that fallback/cache paths work correctly.
"""
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Inject fakes before pipeline import to prevent cv2/SDK import failures
sys.modules.setdefault("anthropic", MagicMock())
sys.modules.setdefault("openai", MagicMock())
sys.modules.setdefault("pyttsx3", MagicMock())
sys.modules.setdefault("sounddevice", MagicMock())
sys.modules.setdefault("requests", MagicMock())

import src.pipeline as pipeline  # noqa: E402
from src.pipeline import run_cycle  # noqa: E402

_FRAME = np.ones((480, 640, 3), dtype=np.uint8) * 100
_B64 = "aGVsbG8="
_DESCRIPTION = "A busy conference hall with people nearby."


@pytest.fixture(autouse=True)
def reset_cache():
    pipeline._DEMO_CACHE = []
    yield


def test_three_cycles_no_tts(capsys):
    with (
        patch("src.pipeline.capture_frame", return_value=_FRAME),
        patch("src.pipeline.encode_jpeg", return_value=_B64),
        patch("src.pipeline.describe_scene", return_value=_DESCRIPTION),
    ):
        for _ in range(3):
            result = run_cycle(no_tts=True, backend=None, interval=0)
            assert result == _DESCRIPTION

    captured = capsys.readouterr()
    assert captured.out.count(_DESCRIPTION) == 3


def test_fallback_to_local_on_timeout(capsys):
    with (
        patch("src.pipeline.capture_frame", return_value=_FRAME),
        patch("src.pipeline.encode_jpeg", return_value=_B64),
        patch("src.pipeline.describe_scene", side_effect=TimeoutError("timeout")),
        patch("src.pipeline.describe_scene_local", return_value="Fallback description."),
    ):
        result = run_cycle(no_tts=True, backend=None, interval=0)

    assert result == "Fallback description."


def test_fallback_to_cache_when_all_fail(capsys):
    pipeline._DEMO_CACHE = [
        {"scene": "indoor", "response": "You are in a large hall."}
    ]
    with (
        patch("src.pipeline.capture_frame", return_value=_FRAME),
        patch("src.pipeline.encode_jpeg", return_value=_B64),
        patch("src.pipeline.describe_scene", side_effect=RuntimeError("down")),
        patch("src.pipeline.describe_scene_local", side_effect=RuntimeError("no ollama")),
    ):
        result = run_cycle(no_tts=True, backend=None, interval=0)

    assert result == "You are in a large hall."


def test_backend_local_skips_remote():
    with (
        patch("src.pipeline.capture_frame", return_value=_FRAME),
        patch("src.pipeline.encode_jpeg", return_value=_B64),
        patch("src.pipeline.describe_scene_local", return_value="Local desc.") as mock_local,
        patch("src.pipeline.describe_scene") as mock_remote,
    ):
        result = run_cycle(no_tts=True, backend="local", interval=0)

    assert result == "Local desc."
    mock_remote.assert_not_called()
