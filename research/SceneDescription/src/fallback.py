import json
import sys
import time

_OLLAMA_URL = "http://localhost:11434/api/generate"
_MODEL = "llava:7b"


def describe_scene_local(image_b64: str) -> str:
    """Call the local Ollama LLaVA model and return a scene description."""
    import requests

    payload = {
        "model": _MODEL,
        "prompt": (
            "You are a visual assistant for a blind user. "
            "Describe what is in front of the user in 1-2 sentences. "
            "Prioritize people, obstacles, text, and wayfinding cues. "
            "Speak directly without saying 'I see' or 'the image shows'."
        ),
        "images": [image_b64],
        "stream": False,
    }

    t0 = time.perf_counter()
    try:
        resp = requests.post(_OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError("Ollama is not running at localhost:11434") from exc
    except requests.exceptions.Timeout as exc:
        raise TimeoutError("Ollama request timed out") from exc

    elapsed = time.perf_counter() - t0
    print(f"[fallback] ollama latency: {elapsed:.3f}s", file=sys.stderr)

    data = resp.json()
    return data.get("response", "").strip()
