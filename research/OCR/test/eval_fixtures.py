"""
Evaluate OCR accuracy against all fixture sets that contain a ground_truth.txt.

Usage (from the research/OCR directory):
    python test/eval_fixtures.py                    # all fixture subfolders
    python test/eval_fixtures.py --fixture medicine_package

For each fixture subfolder the script:
  - Runs OCR via src.ocr_engine.recognize() (includes orientation correction).
  - Computes per-image Character Error Rate (CER).
  - Prints a detailed report to stdout.
  - Saves an artifact JSON to test/fixtures/<subfolder>/eval_<commit>.json.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

# Allow running from the project root without installing the package.
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

FIXTURE_ROOT = _PROJECT_ROOT / "test" / "fixtures"
PASS_THRESHOLD = 0.30  # mean CER below this is considered a pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _cer(ref: str, hyp: str) -> float:
    ref, hyp = ref.lower(), hyp.lower()
    r, h = len(ref), len(hyp)
    if r == 0:
        return 0.0 if h == 0 else 1.0
    dp = list(range(h + 1))
    for i in range(1, r + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, h + 1):
            dp[j] = prev[j - 1] if ref[i - 1] == hyp[j - 1] else 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[h] / r


def _load_ground_truth(gt_path: Path) -> dict[str, str]:
    gt = {}
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fname, _, text = line.partition(": ")
            gt[fname.strip()] = text.strip()
    return gt


def _ocr_image(img_path: Path) -> tuple[str, list[dict]]:
    import cv2
    from src.ocr_engine import recognize
    from src.text_formatter import format_for_speech, structure_text

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    h, w = img.shape[:2]
    max_side = 2000
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    tokens = recognize(img)
    text = format_for_speech(structure_text(tokens))
    return text, tokens


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _explain(fixture_name: str, image_results: list[dict], mean_cer: float) -> str:
    passed = [r for r in image_results if r["passed"]]
    failed = [r for r in image_results if not r["passed"]]

    if not failed:
        best = min(image_results, key=lambda r: r["cer"])
        worst = max(image_results, key=lambda r: r["cer"])
        return (
            f"All {len(image_results)} images passed the {PASS_THRESHOLD:.0%} CER threshold. "
            f"Best: {best['filename']} ({best['cer']:.1%}). "
            f"Worst: {worst['filename']} ({worst['cer']:.1%})."
        )

    fail_names = ", ".join(f"{r['filename']} ({r['cer']:.1%})" for r in failed)
    pass_names = (
        ", ".join(f"{r['filename']} ({r['cer']:.1%})" for r in passed)
        if passed else "none"
    )

    # Identify failure modes from per-image data.
    modes = []
    empty = [r for r in failed if r["token_count"] == 0]
    order_errors = [r for r in failed if r["token_count"] > 0 and r["cer"] > 0.60]
    partial = [r for r in failed if r["token_count"] > 0 and r["cer"] <= 0.60]

    if empty:
        modes.append(f"no tokens detected ({', '.join(r['filename'] for r in empty)})")
    if order_errors:
        modes.append(
            "high CER despite tokens detected — likely reading-order errors on curved/rotated surfaces "
            f"({', '.join(r['filename'] for r in order_errors)})"
        )
    if partial:
        modes.append(
            f"partial recognition — some tokens missing or substituted ({', '.join(r['filename'] for r in partial)})"
        )

    mode_str = "; ".join(modes) if modes else "undiagnosed"

    return (
        f"Mean CER {mean_cer:.1%} exceeds the {PASS_THRESHOLD:.0%} threshold. "
        f"Failed: {fail_names}. "
        f"Passed: {pass_names}. "
        f"Failure mode(s): {mode_str}."
    )


# ---------------------------------------------------------------------------
# Per-fixture evaluation
# ---------------------------------------------------------------------------

def evaluate_fixture(fixture_dir: Path, commit: str) -> dict:
    gt = _load_ground_truth(fixture_dir / "ground_truth.txt")

    image_results = []
    total_cer = 0.0

    for fname, expected in gt.items():
        img_path = fixture_dir / fname
        t0 = time.perf_counter()
        try:
            recognized, tokens = _ocr_image(img_path)
            error = None
        except Exception as exc:
            recognized, tokens, error = "", [], str(exc)
        elapsed = time.perf_counter() - t0

        c = _cer(expected, recognized)
        total_cer += c

        image_results.append({
            "filename": fname,
            "expected": expected,
            "recognized": recognized,
            "token_count": len(tokens),
            "tokens": [
                {
                    "text": t["text"],
                    "confidence": round(t["confidence"], 4),
                    "bbox": t["bbox"],
                }
                for t in tokens
            ],
            "cer": round(c, 4),
            "passed": c < PASS_THRESHOLD,
            "elapsed_s": round(elapsed, 3),
            **({"error": error} if error else {}),
        })

    mean_cer = total_cer / len(image_results) if image_results else 0.0
    verdict = "PASS" if mean_cer < PASS_THRESHOLD else "NEEDS IMPROVEMENT"
    explanation = _explain(fixture_dir.name, image_results, mean_cer)

    return {
        "fixture": fixture_dir.name,
        "commit": commit,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "threshold_cer": PASS_THRESHOLD,
        "mean_cer": round(mean_cer, 4),
        "verdict": verdict,
        "explanation": explanation,
        "images": image_results,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_report(result: dict) -> None:
    w = 72
    print("=" * w)
    print(f"{'Fixture: ' + result['fixture']:^{w}}")
    print(f"{'Commit: ' + result['commit']:^{w}}")
    print("=" * w)

    for img in result["images"]:
        print(f"\n{'─' * w}")
        print(f"  Image      : {img['filename']}")
        print(f"  Expected   : {img['expected']}")
        print(f"  Recognized : {img['recognized'] or '(empty)'}")
        print(f"  Tokens     : {img['token_count']} detected")
        if img["tokens"]:
            for tok in img["tokens"]:
                print(f"               [{tok['confidence']:.2f}] {tok['text']!r}")
        print(f"  CER        : {img['cer']:.1%}  {'✓ PASS' if img['passed'] else '✗ FAIL'}")
        print(f"  Elapsed    : {img['elapsed_s']:.2f}s")
        if "error" in img:
            print(f"  Error      : {img['error']}")

    print(f"\n{'─' * w}")
    print(f"  Mean CER   : {result['mean_cer']:.1%}")
    print(f"  Verdict    : {result['verdict']}  (threshold < {result['threshold_cer']:.0%})")
    print(f"\n  {result['explanation']}")
    print("=" * w)


def _save_artifact(fixture_dir: Path, result: dict) -> Path:
    out_path = fixture_dir / f"eval_{result['commit']}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate OCR against fixture ground truth")
    parser.add_argument(
        "--fixture",
        help="Run only this fixture subfolder name (default: all with ground_truth.txt)",
    )
    args = parser.parse_args(argv)

    # Initialise PaddleOCR once (warm up the lazy singleton).
    from paddleocr import PaddleOCR
    import src.ocr_engine as engine_mod
    print("Initialising PaddleOCR (loading cached weights)…", file=sys.stderr)
    engine_mod._ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

    commit = _git_commit()

    if args.fixture:
        dirs = [FIXTURE_ROOT / args.fixture]
    else:
        dirs = sorted(
            d for d in FIXTURE_ROOT.iterdir()
            if d.is_dir() and (d / "ground_truth.txt").exists()
        )

    for fixture_dir in dirs:
        result = evaluate_fixture(fixture_dir, commit)
        _print_report(result)
        artifact_path = _save_artifact(fixture_dir, result)
        print(f"\nArtifact saved → {artifact_path.relative_to(_PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()
