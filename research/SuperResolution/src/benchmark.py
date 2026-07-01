"""
Quality/latency benchmark: bilinear vs. adaptive-sharpen vs. SR models vs.
hybrid (SR + enhancement), measured against ground-truth HR crops.

LR is synthesised with PIL's BICUBIC filter, not OpenCV's. This matters:
the public FSRCNN/ESPCN checkpoints ported in src/convert_pretrained.py were
trained on PIL-bicubic-degraded LR, and these small SR nets are sensitive to
the exact downsampling kernel used. Using cv2's bicubic kernel here would
make a correctly-converted pretrained checkpoint look *worse than bilinear*
-- a kernel-mismatch artifact, not a real deficiency. See
docs/algorithm_readme.md §2.5 for measured numbers.

Each model is specified as "label:arch:scale:weights_path" so different
labels can share an arch at different scales (e.g. compare fsrcnn at x2/x3/x4)
or share a scale across archs (e.g. fsrcnn_x3 vs espcn_x3) in one report.

Usage:
    python -m src.benchmark --hr-dir test/fixtures/medicine_package \\
        --models fsrcnn_x4:fsrcnn:4:src/weights/fsrcnn_x4_pretrained.pt \\
                 espcn_x3:espcn:3:src/weights/espcn_x3_pretrained.pt \\
        --report docs/benchmark_report.md
"""
import argparse
import glob
import os
import time
from dataclasses import dataclass

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
    """PIL-bicubic downsample -- see module docstring for why not cv2."""
    import PIL.Image as pil_image

    rgb = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    lr = pil_image.fromarray(rgb).resize((w // scale, h // scale), resample=pil_image.BICUBIC)
    return cv2.cvtColor(np.array(lr), cv2.COLOR_RGB2BGR)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    arch: str
    scale: int
    weights_path: str

    @classmethod
    def parse(cls, spec: str) -> "ModelSpec":
        parts = spec.split(":", 3)
        if len(parts) != 4:
            raise ValueError(f"Model spec must be 'label:arch:scale:weights_path', got {spec!r}")
        label, arch, scale, weights_path = parts
        return cls(label, arch, int(scale), weights_path)


def load_models(model_specs: list[ModelSpec]) -> dict[int, dict[str, object]]:
    """Instantiate SRModel for each spec, grouped by scale. Requires torch."""
    from src.sr_model import SRModel

    models_by_scale: dict[int, dict[str, object]] = {}
    for spec in model_specs:
        models_by_scale.setdefault(spec.scale, {})[spec.label] = SRModel(
            scale=spec.scale, arch=spec.arch, weights_path=spec.weights_path
        )
    return models_by_scale


def build_methods_for_scale(scale: int, sr_models: dict) -> dict:
    methods = {
        f"bilinear_x{scale}": lambda img: bilinear_upscale(img, scale),
        f"adaptive_sharpen_x{scale}": lambda img: adaptive_sharpen(bilinear_upscale(img, scale)),
    }
    for label, model in sr_models.items():
        methods[label] = model.upscale
        methods[f"{label}_hybrid"] = (lambda img, m=model: enhance(m.upscale(img)))
    return methods


def evaluate(
    hr_dir: str,
    models_by_scale: dict[int, dict[str, object]] | None = None,
    baseline_scales: list[int] = (),
    keep_images: bool = False,
) -> list[dict]:
    """Evaluate bilinear/adaptive-sharpen/SR/hybrid methods against HR images
    in `hr_dir`. `models_by_scale` maps scale -> {label: model} where each
    model exposes `.upscale(lr) -> np.ndarray` (e.g. an `SRModel` instance, or
    a fake for testing). Pass pre-built models (via `load_models`) rather than
    specs so this function itself never needs torch.
    """
    models_by_scale = models_by_scale or {}
    scales = sorted(set(baseline_scales) | set(models_by_scale))
    if not scales:
        raise ValueError("no scales to evaluate: pass models_by_scale or baseline_scales")

    results = []
    for path in sorted(glob.glob(os.path.join(hr_dir, "*"))):
        hr_full = cv2.imread(path)
        if hr_full is None:
            continue

        for scale in scales:
            h, w = hr_full.shape[:2]
            h, w = h - h % scale, w - w % scale
            hr = hr_full[:h, :w]
            lr = _downscale(hr, scale)
            methods = build_methods_for_scale(scale, models_by_scale.get(scale, {}))

            for name, fn in methods.items():
                start = time.perf_counter()
                out = fn(lr)
                elapsed_ms = (time.perf_counter() - start) * 1000
                if out.shape[:2] != (h, w):
                    out = cv2.resize(out, (w, h))

                row = {
                    "image": os.path.basename(path),
                    "scale": scale,
                    "method": name,
                    "psnr": psnr(out, hr),
                    "ssim": ssim(out, hr),
                    "latency_ms": elapsed_ms,
                }
                if keep_images:
                    row.update(output=out, hr=hr, lr=lr)
                results.append(row)
    return results


def summarize(results: list[dict]) -> dict:
    by_key: dict[tuple[int, str], list[dict]] = {}
    for r in results:
        by_key.setdefault((r["scale"], r["method"]), []).append(r)
    return {
        key: {
            "psnr": float(np.mean([r["psnr"] for r in rows])),
            "ssim": float(np.mean([r["ssim"] for r in rows])),
            "latency_ms": float(np.mean([r["latency_ms"] for r in rows])),
        }
        for key, rows in by_key.items()
    }


def write_report(summary: dict, output_path: str) -> None:
    lines = ["# Super-Resolution Benchmark Report"]
    for scale in sorted({s for s, _ in summary}):
        lines += ["", f"## Scale x{scale}", "", "| Method | PSNR (dB) | SSIM | Latency (ms) |", "|---|---|---|---|"]
        rows = {method: stats for (s, method), stats in summary.items() if s == scale}
        for method, stats in sorted(rows.items(), key=lambda kv: -kv[1]["psnr"]):
            lines.append(f"| {method} | {stats['psnr']:.2f} | {stats['ssim']:.4f} | {stats['latency_ms']:.1f} |")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark SR methods against bilinear baseline")
    parser.add_argument("--hr-dir", required=True, help="Directory of ground-truth HR images")
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="label:arch:scale:weights_path entries, e.g. fsrcnn_x4:fsrcnn:4:src/weights/fsrcnn_x4_pretrained.pt",
    )
    parser.add_argument(
        "--baseline-scales",
        nargs="*",
        type=int,
        default=[],
        help="Extra scales to report bilinear/adaptive-sharpen baselines for, even without a model at that scale",
    )
    parser.add_argument("--report", default="docs/benchmark_report.md")
    args = parser.parse_args(argv)

    model_specs = [ModelSpec.parse(s) for s in args.models]
    models_by_scale = load_models(model_specs)

    results = evaluate(args.hr_dir, models_by_scale=models_by_scale, baseline_scales=args.baseline_scales)
    summary = summarize(results)
    write_report(summary, args.report)
    print(f"[benchmark] wrote {args.report}")


if __name__ == "__main__":
    main()
