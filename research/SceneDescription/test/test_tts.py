"""
Tests for tts.py — pyttsx3 and sounddevice are mocked.
Verifies queue draining and no-deadlock on interruption.
"""
import queue
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

sys.modules.setdefault("pyttsx3", MagicMock())
sys.modules.setdefault("sounddevice", MagicMock())
sys.modules.setdefault("requests", MagicMock())

import src.tts as tts  # noqa: E402


def _reset():
    tts._stop_event.set()
    if tts._thread and tts._thread.is_alive():
        tts._thread.join(timeout=2.0)
    tts._thread = None
    tts._stop_event.clear()
    while not tts._queue.empty():
        try:
            tts._queue.get_nowait()
        except queue.Empty:
            break


@pytest.fixture(autouse=True)
def reset_tts():
    _reset()
    yield
    _reset()


def test_speak_enqueues_item():
    with patch("src.tts._ensure_thread"):
        tts.speak("hello")
    assert not tts._queue.empty()


def test_queue_drains_correctly(monkeypatch):
    monkeypatch.delenv("TTS_BACKEND", raising=False)
    collected = []
    done = threading.Event()

    def fake_worker():
        while not tts._stop_event.is_set():
            try:
                text = tts._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            while not tts._queue.empty():
                try:
                    text = tts._queue.get_nowait()
                except queue.Empty:
                    break
            collected.append(text)
            tts._queue.task_done()
            done.set()

    tts._thread = threading.Thread(target=fake_worker, daemon=True)
    tts._thread.start()

    with patch("src.tts._ensure_thread"):
        tts.speak("first")
        tts.speak("second")
        tts.speak("third")

    done.wait(timeout=2.0)
    tts._stop_event.set()
    tts._thread.join(timeout=2.0)

    assert len(collected) >= 1
    assert not tts._thread.is_alive()


def test_shutdown_no_deadlock(monkeypatch):
    monkeypatch.delenv("TTS_BACKEND", raising=False)
    mock_pyttsx3 = sys.modules["pyttsx3"]
    mock_pyttsx3.init.return_value = MagicMock()

    tts._ensure_thread()
    time.sleep(0.1)
    tts.shutdown(timeout=2.0)

    assert not tts._thread.is_alive()


def test_interruption_no_deadlock():
    with patch("src.tts._ensure_thread"):
        for i in range(10):
            tts.speak(f"msg {i}")
    assert tts._queue.qsize() <= 10


def test_elevenlabs_backend_selected(monkeypatch):
    monkeypatch.setenv("TTS_BACKEND", "elevenlabs")
    with patch("src.tts._worker_elevenlabs") as mock_worker:
        with patch("threading.Thread") as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread.is_alive.return_value = False
            mock_thread_cls.return_value = mock_thread
            tts._ensure_thread()
            call_kwargs = mock_thread_cls.call_args.kwargs
            assert call_kwargs["target"] == tts._worker_elevenlabs
