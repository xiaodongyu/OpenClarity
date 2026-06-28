"""
Main OCR pipeline.

Usage:
    python src/pipeline.py [--continuous] [--no-tts] [--lang en]

Modes:
    default      Wait for Space keypress to trigger one capture→OCR→speak cycle.
    --continuous Repeat automatically every CAPTURE_INTERVAL_SEC seconds (default: 3).

Flags:
    --no-tts     Print recognised text to stdout instead of speaking (useful for testing).
    --lang CODE  Language code passed to PaddleOCR (default: en).
"""
import argparse
import os
import sys
import time

import cv2

from src.capture import capture_frame, crop, select_roi
from src.ocr_engine import recognize
from src.text_formatter import format_for_speech, structure_text


def run_cycle(roi, no_tts: bool, lang: str) -> str:
    frame = capture_frame()
    if roi is not None:
        frame = crop(frame, roi)
    # Orientation correction is applied inside recognize() before calling PaddleOCR
    tokens = recognize(frame, lang=lang)
    text = structure_text(tokens)
    speech_text = format_for_speech(text)

    if no_tts:
        print(speech_text)
    else:
        from src.tts import speak
        speak(speech_text)

    return speech_text


def main(argv=None):
    parser = argparse.ArgumentParser(description="On-device OCR pipeline")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--no-tts", action="store_true", help="Print text instead of speaking")
    parser.add_argument("--lang", default="en", help="OCR language (default: en)")
    args = parser.parse_args(argv)

    interval = float(os.environ.get("CAPTURE_INTERVAL_SEC", "3"))

    # Warm-up: grab one frame to set ROI
    print("[pipeline] Starting camera...", file=sys.stderr)
    frame = capture_frame()
    roi = select_roi(frame)
    print("[pipeline] ROI selected. Press Space to capture (Ctrl+C to quit).", file=sys.stderr)

    try:
        if args.continuous:
            while True:
                run_cycle(roi, args.no_tts, args.lang)
                time.sleep(interval)
        else:
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord(" "):
                    run_cycle(roi, args.no_tts, args.lang)
    except KeyboardInterrupt:
        print("\n[pipeline] Exiting.", file=sys.stderr)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
