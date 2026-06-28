import os
import queue
import threading
from typing import Optional


_engine = None
_queue: queue.Queue = queue.Queue()
_thread: Optional[threading.Thread] = None
_lock = threading.Lock()
_stop_event = threading.Event()


def _get_rate() -> int:
    return int(os.environ.get("TTS_RATE", "175"))


def _worker():
    import pyttsx3

    engine = pyttsx3.init()
    engine.setProperty("rate", _get_rate())

    while not _stop_event.is_set():
        try:
            text = _queue.get(timeout=0.1)
        except queue.Empty:
            continue

        # Drain queue — only speak the latest pending item
        while not _queue.empty():
            try:
                text = _queue.get_nowait()
            except queue.Empty:
                break

        engine.say(text)
        engine.runAndWait()
        _queue.task_done()


def _ensure_thread():
    global _thread
    with _lock:
        if _thread is None or not _thread.is_alive():
            _stop_event.clear()
            _thread = threading.Thread(target=_worker, daemon=True, name="tts-worker")
            _thread.start()


def speak(text: str) -> None:
    _ensure_thread()
    _queue.put(text)


def shutdown(timeout: float = 2.0) -> None:
    _stop_event.set()
    if _thread is not None:
        _thread.join(timeout=timeout)
