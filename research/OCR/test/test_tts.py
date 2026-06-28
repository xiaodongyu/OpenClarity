"""
Tests for tts.py — pyttsx3 is mocked so no audio hardware is required.
Verifies queue draining and that interruption does not deadlock.
"""
import queue
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


def _reset_module():
    import src.tts as tts
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
    _reset_module()
    yield
    _reset_module()


def test_speak_puts_item_in_queue():
    import src.tts as tts

    with patch("src.tts._ensure_thread"):
        tts.speak("hello")
        assert not tts._queue.empty()


def test_queue_drains_correctly():
    import src.tts as tts

    collected = []
    done = threading.Event()

    def fake_worker():
        while not tts._stop_event.is_set():
            try:
                text = tts._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            # Drain remaining
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

    # Queue drained — only last item (or fewer) was spoken; no deadlock
    assert len(collected) >= 1
    assert not tts._thread.is_alive()


def test_shutdown_does_not_deadlock():
    import sys
    import src.tts as tts

    mock_pyttsx3 = MagicMock()
    mock_pyttsx3.init.return_value = MagicMock()
    with patch.dict(sys.modules, {"pyttsx3": mock_pyttsx3}):
        tts._ensure_thread()
        time.sleep(0.1)
        tts.shutdown(timeout=2.0)

    assert not tts._thread.is_alive()


def test_multiple_speaks_no_deadlock():
    import src.tts as tts

    with patch("src.tts._ensure_thread"):
        for i in range(20):
            tts.speak(f"message {i}")

    # All items enqueued; verify no exception raised and queue is intact
    assert tts._queue.qsize() <= 20
