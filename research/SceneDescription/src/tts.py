import os
import queue
import threading
from typing import Optional

_queue: queue.Queue = queue.Queue()
_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()
_lock = threading.Lock()


def _worker_pyttsx3():
    import pyttsx3

    engine = pyttsx3.init()
    while not _stop_event.is_set():
        try:
            text = _queue.get(timeout=0.1)
        except queue.Empty:
            continue
        # Drain — only speak the latest pending item (interruption behaviour)
        while not _queue.empty():
            try:
                text = _queue.get_nowait()
            except queue.Empty:
                break
        engine.say(text)
        engine.runAndWait()
        _queue.task_done()


def _worker_elevenlabs():
    import requests
    import sounddevice as sd
    import numpy as np

    api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"

    while not _stop_event.is_set():
        try:
            text = _queue.get(timeout=0.1)
        except queue.Empty:
            continue
        while not _queue.empty():
            try:
                text = _queue.get_nowait()
            except queue.Empty:
                break
        try:
            resp = requests.post(
                url,
                headers={"xi-api-key": api_key, "Content-Type": "application/json"},
                json={"text": text, "model_id": "eleven_monolingual_v1"},
                stream=True,
                timeout=10,
            )
            audio = b"".join(resp.iter_content(chunk_size=4096))
            arr = np.frombuffer(audio, dtype=np.int16)
            sd.play(arr, samplerate=22050, blocking=True)
        except Exception:
            pass
        _queue.task_done()


def _ensure_thread():
    global _thread
    backend = os.environ.get("TTS_BACKEND", "pyttsx3").lower()
    worker = _worker_elevenlabs if backend == "elevenlabs" else _worker_pyttsx3
    with _lock:
        if _thread is None or not _thread.is_alive():
            _stop_event.clear()
            _thread = threading.Thread(target=worker, daemon=True, name="tts-worker")
            _thread.start()


def speak(text: str) -> None:
    _ensure_thread()
    _queue.put(text)


def shutdown(timeout: float = 2.0) -> None:
    _stop_event.set()
    if _thread is not None:
        _thread.join(timeout=timeout)
