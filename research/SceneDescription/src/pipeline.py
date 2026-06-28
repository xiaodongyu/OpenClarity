"""
Main SceneDescription pipeline.

Usage:
    python src/pipeline.py [--backend anthropic|openai|local] [--interval FLOAT] [--no-tts]

Loop: capture → encode → describe → speak, at CAPTURE_INTERVAL_SEC cadence.
Falls back to local Ollama on TimeoutError/network error, then to demo_cache.json.
"""
import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import cv2

from src.capture import capture_frame, encode_jpeg
from src.fallback import describe_scene_local
from src.vlm_client import describe_scene


_DEMO_CACHE: list[dict] = []


def _load_demo_cache() -> list[dict]:
    cache_path = Path(__file__).parent.parent / "demo_cache.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return []


def _describe_with_fallback(image_b64: str, backend_override: str | None) -> str:
    if backend_override == "local":
        return describe_scene_local(image_b64)

    try:
        if backend_override:
            old = os.environ.get("VLM_BACKEND")
            os.environ["VLM_BACKEND"] = backend_override
        result = describe_scene(image_b64)
        return result
    except (TimeoutError, OSError, Exception) as exc:
        print(f"[pipeline] remote failed ({exc}), trying local fallback", file=sys.stderr)
    finally:
        if backend_override and old is not None:
            os.environ["VLM_BACKEND"] = old
        elif backend_override:
            os.environ.pop("VLM_BACKEND", None)

    try:
        return describe_scene_local(image_b64)
    except Exception as exc:
        print(f"[pipeline] local fallback failed ({exc}), using demo cache", file=sys.stderr)

    if _DEMO_CACHE:
        # Cycle through cache entries
        entry = _DEMO_CACHE[int(time.time()) % len(_DEMO_CACHE)]
        return entry["response"]

    return "Unable to describe the scene at this time."


def run_cycle(no_tts: bool, backend: str | None, interval: float) -> str:
    frame = capture_frame()
    image_b64 = encode_jpeg(frame)
    description = _describe_with_fallback(image_b64, backend)

    if no_tts:
        print(description)
    else:
        from src.tts import speak
        speak(description)

    return description


def main(argv=None):
    global _DEMO_CACHE
    _DEMO_CACHE = _load_demo_cache()

    parser = argparse.ArgumentParser(description="Scene description pipeline")
    parser.add_argument("--backend", choices=["anthropic", "openai", "local"], default=None)
    parser.add_argument("--interval", type=float, default=float(os.environ.get("CAPTURE_INTERVAL_SEC", "2.0")))
    parser.add_argument("--no-tts", action="store_true")
    args = parser.parse_args(argv)

    print("[pipeline] Starting. Ctrl+C to quit.", file=sys.stderr)
    try:
        while True:
            t0 = time.perf_counter()
            run_cycle(no_tts=args.no_tts, backend=args.backend, interval=args.interval)
            elapsed = time.perf_counter() - t0
            remaining = args.interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\n[pipeline] Exiting.", file=sys.stderr)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
