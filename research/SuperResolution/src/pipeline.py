"""
Main super-resolution zoom pipeline.

Usage:
    python -m src.pipeline [--scale 4] [--model espcn|fsrcnn] [--weights PATH] [--no-sr]

Flags:
    --scale INT    Magnification factor (default: 4).
    --model NAME   SR architecture to use (default: espcn).
    --weights PATH Path to trained model weights (.pt). Falls back to random
                   init if omitted — pass real weights for usable output.
    --no-sr        Skip the SR model; use bilinear + adaptive sharpen/Laplacian
                   enhancement only (for low-power devices or missing weights).
    --device NAME  torch device (default: cpu).
"""
import argparse
import sys

import cv2
import numpy as np

from src.capture import capture_frame, crop, select_zoom_region
from src.enhance import bilinear_upscale, enhance


def run_cycle(frame: np.ndarray, roi, scale: int, model=None) -> np.ndarray:
    region = crop(frame, roi) if roi is not None else frame

    if model is not None:
        upscaled = model.upscale(region)
        if model.scale != scale:
            h, w = region.shape[:2]
            upscaled = cv2.resize(upscaled, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    else:
        upscaled = bilinear_upscale(region, scale)

    return enhance(upscaled)


def main(argv=None):
    parser = argparse.ArgumentParser(description="On-device super-resolution zoom pipeline")
    parser.add_argument("--scale", type=int, default=4, help="Magnification factor")
    parser.add_argument("--model", default="espcn", choices=["espcn", "fsrcnn"])
    parser.add_argument("--weights", default=None, help="Path to trained model weights (.pt)")
    parser.add_argument("--no-sr", action="store_true", help="Bilinear + enhancement only")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    model = None
    if not args.no_sr:
        try:
            from src.sr_model import SRModel

            model = SRModel(
                scale=args.scale, arch=args.model, weights_path=args.weights, device=args.device
            )
        except Exception as exc:
            print(
                f"[pipeline] SR model unavailable ({exc}); falling back to bilinear+enhance",
                file=sys.stderr,
            )

    print("[pipeline] Starting camera...", file=sys.stderr)
    frame = capture_frame()
    roi = select_zoom_region(frame)
    print("[pipeline] Zoom region selected. Press Space to capture (Ctrl+C to quit).", file=sys.stderr)

    try:
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord(" "):
                frame = capture_frame()
                result = run_cycle(frame, roi, args.scale, model)
                cv2.imshow("Super-resolved zoom", result)
    except KeyboardInterrupt:
        print("\n[pipeline] Exiting.", file=sys.stderr)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
