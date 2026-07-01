"""
Quality/latency benchmark: bilinear vs. adaptive-sharpen vs. SR models vs.
hybrid (SR + enhancement), measured against ground-truth HR crops.

Usage:
    python -m src.benchmark --hr-dir test/fixtures/hr_crops --scale 4 \\
        --models espcn=src/weights/espcn_x4.pt --report docs/benchmark_report.md
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np

from src.enhance import adaptive_sharpen, bilinear_upscale, enhance
from src.train import psnr


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) if b.ndim == 3 else b
    return float(structural_similarity(gray_a, gray_b, data_range=255))


def _downscale(hr: np.ndarray, scale: int) -> np.ndarray:
    h, w = hr.shape[:2]
    return cv2.resize(hr, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)


def build_methods(scale: int, sr_models: dict) -> dict:
    methods = {
        "bilinear": lambda img: bilinear_upscale(img, scale),
        "adaptive_sharpen": lambda img: adaptive_sharpen(bilinear_upscale(img, scale)),
    }
    for name, model in sr_models.items():
        methods[name] = model.upscale
        methods[f"{name}_hybrid"] = (lambda img, m=model: enhance(m.upscale(img)))
    return methods


def evaluate(hr_dir: str, scale: int = 4, sr_models: dict | None = None) -> list[dict]:
    methods = build_methods(scale, sr_models or {})
    results = []

    for path in sorted(glob.glob(os.path.join(hr_dir, "*"))):
        hr = cv2.imread(path)
        if hr is None:
            continue
        h, w = hr.shape[:2]
        h, w = h - h % scale, w - w % scale
        hr = hr[:h, :w]
        lr = _downscale(hr, scale)

        for name, fn in methods.items():
            start = time.perf_counter()
            out = fn(lr)
            elapsed_ms = (time.perf_counter() - start) * 1000
            if out.shape[:2] != (h, w):
                out = cv2.resize(out, (w, h))
            results.append(
                {
                    "image": os.path.basename(path),
                    "method": name,
                    "psnr": psnr(out, hr),
                    "ssim": ssim(out, hr),
                    "latency_ms": elapsed_ms,
                }
            )
    return results


def summarize(results: list[dict]) -> dict:
    by_method: dict[str, list[dict]] = {}
    for r in results:
        by_method.setdefault(r["method"], []).append(r)
    return {
        method: {
            "psnr": float(np.mean([r["psnr"] for r in rows])),
            "ssim": float(np.mean([r["ssim"] for r in rows])),
            "latency_ms": float(np.mean([r["latency_ms"] for r in rows])),
        }
        for method, rows in by_method.items()
    }


def write_report(summary: dict, output_path: str) -> None:
    lines = ["# Super-Resolution Benchmark Report", "", "| Method | PSNR (dB) | SSIM | Latency (ms) |", "|---|---|---|---|"]
    for method, stats in sorted(summary.items(), key=lambda kv: -kv[1]["psnr"]):
        lines.append(f"| {method} | {stats['psnr']:.2f} | {stats['ssim']:.4f} | {stats['latency_ms']:.1f} |")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark SR methods against bilinear baseline")
    parser.add_argument("--hr-dir", required=True, help="Directory of ground-truth HR crops")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="name=weights_path pairs, e.g. espcn=src/weights/espcn_x4.pt",
    )
    parser.add_argument("--report", default="docs/benchmark_report.md")
    args = parser.parse_args(argv)

    sr_models = {}
    if args.models:
        from src.sr_model import SRModel

        for spec in args.models:
            name, weights_path = spec.split("=", 1)
            sr_models[name] = SRModel(scale=args.scale, arch=name, weights_path=weights_path)

    results = evaluate(args.hr_dir, scale=args.scale, sr_models=sr_models)
    summary = summarize(results)
    write_report(summary, args.report)
    print(f"[benchmark] wrote {args.report}")


if __name__ == "__main__":
    main()
