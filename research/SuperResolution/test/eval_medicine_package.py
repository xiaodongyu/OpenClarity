"""
Evaluate SR methods (bilinear / adaptive-sharpen / pretrained FSRCNN & ESPCN /
hybrid) against the medicine_package fixture set.

Usage (from research/SuperResolution):
    .venv/bin/python test/eval_medicine_package.py

Writes eval_<YYYYMMDD>_<NNN>.json and .html to test/fixtures/medicine_package/,
where NNN is the 1-based sequence number of runs on that calendar date in
that folder.

Each source photo is centre-cropped to CROP_SIZE, then downscaled by
BASE_DOWNSCALE before evaluation -- these are full 12MP phone photos, and at
CROP_SIZE alone the "ground truth" is still sharp enough that x2-x4 SR
barely looks different from the original. Shrinking the reference first
means the same nominal scale factors correspond to a harder, more visible
reconstruction problem.

LR is synthesised with PIL's BICUBIC filter (via src.benchmark._downscale),
matching the degradation convention the pretrained checkpoints were trained
on -- see docs/algorithm_readme.md §2.5.
"""
import base64
import html as _html
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.benchmark import ModelSpec, _downscale, evaluate, load_models, ssim
from src.train import psnr

FIXTURE_DIR = _PROJECT_ROOT / "test" / "fixtures" / "medicine_package"
WEIGHTS_DIR = _PROJECT_ROOT / "src" / "weights"

CROP_SIZE = (800, 600)  # (w, h) centred crop -- see module docstring
BASE_DOWNSCALE = 4  # shrink the crop by this factor before using it as ground truth

MODEL_SPECS = [
    ModelSpec("fsrcnn_x2", "fsrcnn", 2, str(WEIGHTS_DIR / "fsrcnn_x2_pretrained.pt")),
    ModelSpec("fsrcnn_x3", "fsrcnn", 3, str(WEIGHTS_DIR / "fsrcnn_x3_pretrained.pt")),
    ModelSpec("espcn_x3", "espcn", 3, str(WEIGHTS_DIR / "espcn_x3_pretrained.pt")),
    ModelSpec("fsrcnn_x4", "fsrcnn", 4, str(WEIGHTS_DIR / "fsrcnn_x4_pretrained.pt")),
]
BASELINE_ONLY_SCALES = []

# Method display order within a card, and which ones count as "SR-based" for
# the improved-over-bilinear verdict.
_BASELINE_PREFIXES = ("bilinear_x", "adaptive_sharpen_x")


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(["git"] + cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return ""


def _is_dirty() -> bool:
    """True if there are uncommitted changes -- including to this eval script
    itself. Commit first, then run the eval: otherwise the commit stamped in
    the report describes code that predates whatever actually produced it."""
    return bool(_git(["status", "--porcelain"]))


def _commit_info() -> dict:
    short = _git(["rev-parse", "--short", "HEAD"])
    full = _git(["rev-parse", "HEAD"])
    subj = _git(["log", "-1", "--format=%s"])
    body = _git(["log", "-1", "--format=%b"])
    if _is_dirty():
        short += "-dirty"
        subj += "  [WARNING: uncommitted changes present at eval time]"
    return {"commit": short, "commit_full": full, "commit_subject": subj, "commit_body": body}


def _next_run_id(date_str: str) -> str:
    existing = sorted(FIXTURE_DIR.glob(f"eval_{date_str}_*.json"))
    seq = len(existing) + 1
    return f"eval_{date_str}_{seq:03d}"


# ---------------------------------------------------------------------------
# Fixture preparation
# ---------------------------------------------------------------------------

def _center_crop(img, size):
    w, h = size
    ih, iw = img.shape[:2]
    w, h = min(w, iw), min(h, ih)
    x0 = (iw - w) // 2
    y0 = (ih - h) // 2
    return img[y0 : y0 + h, x0 : x0 + w]


def _prepare_cropped_fixtures(tmp_dir: Path) -> list[str]:
    names = []
    for path in sorted(FIXTURE_DIR.glob("*")):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        cropped = _center_crop(img, CROP_SIZE)
        reference = _downscale(cropped, BASE_DOWNSCALE)
        cv2.imwrite(str(tmp_dir / path.name), reference)
        names.append(path.name)
    return names


# ---------------------------------------------------------------------------
# Result grouping
# ---------------------------------------------------------------------------

def _group_by_card(results: list[dict]) -> dict:
    cards = {}
    for r in results:
        cards.setdefault((r["image"], r["scale"]), []).append(r)
    return cards


def _card_verdict(rows: list[dict]) -> tuple[bool, float, str] | None:
    """Returns (improved, baseline_psnr, best_method), or None if this card has
    no SR model to compare (a baseline-only scale, e.g. x8 -- no pretrained
    checkpoint exists there, see BASELINE_ONLY_SCALES)."""
    baseline = next(r for r in rows if r["method"].startswith("bilinear_x"))
    sr_rows = [r for r in rows if not r["method"].startswith(_BASELINE_PREFIXES)]
    if not sr_rows:
        return None
    best = max(sr_rows, key=lambda r: r["psnr"])
    return bool(best["psnr"] > baseline["psnr"]), baseline["psnr"], best["method"]


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def _build_json(run_id: str, commit: dict, ts: str, cards: dict) -> dict:
    card_summaries = []
    improved = 0
    scored = 0
    for (image, scale), rows in sorted(cards.items()):
        verdict = _card_verdict(rows)
        if verdict is not None:
            is_improved, _, best_method = verdict
            improved += int(is_improved)
            scored += 1
        else:
            is_improved, best_method = None, None
        card_summaries.append(
            {
                "image": image,
                "scale": scale,
                "improved_over_bilinear": is_improved,
                "best_method": best_method,
                "methods": [
                    {"method": r["method"], "psnr": r["psnr"], "ssim": r["ssim"], "latency_ms": r["latency_ms"]}
                    for r in sorted(rows, key=lambda r: -r["psnr"])
                ],
            }
        )

    return {
        "fixture": "medicine_package",
        **commit,
        "timestamp": ts,
        "total": scored,
        "improved": improved,
        "baseline_only_cards": len(card_summaries) - scored,
        "verdict": "PASS" if improved == scored else "NEEDS IMPROVEMENT",
        "cards": card_summaries,
    }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, -apple-system, sans-serif; background: #f3f4f6; color: #1f2937; }

header { background: #111827; color: #f9fafb; padding: 1.25rem 2rem; }
header h1 { font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem; }
.commit-block { display: flex; flex-direction: column; gap: 0.2rem; font-size: 0.8rem; color: #9ca3af; }
.commit-block .commit-subject { color: #e5e7eb; font-weight: 600; font-size: 0.85rem; }
.commit-block .commit-hash { font-family: monospace; color: #6b7280; font-size: 0.75rem; }
.commit-block .meta { color: #6b7280; }

.container { max-width: 1280px; margin: 0 auto; padding: 1.5rem; }

.summary {
  border-radius: 10px; padding: 1.25rem 1.5rem; margin-bottom: 1.75rem;
  border-left: 5px solid; display: flex; flex-direction: column; gap: 0.5rem;
}
.summary.pass { background: #f0fdf4; border-color: #22c55e; }
.summary.fail { background: #fff7ed; border-color: #f97316; }
.summary-top { display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; }
.verdict-badge {
  font-size: 0.7rem; font-weight: 800; letter-spacing: 0.08em;
  padding: 0.2rem 0.55rem; border-radius: 4px; text-transform: uppercase;
}
.verdict-badge.pass { background: #22c55e; color: #fff; }
.verdict-badge.fail { background: #f97316; color: #fff; }
.pass-count { font-size: 2rem; font-weight: 800; line-height: 1; }
.pass-count.pass { color: #16a34a; }
.pass-count.fail { color: #ea580c; }
.threshold-note { font-size: 0.8rem; color: #6b7280; }

.image-card {
  background: white; border: 1px solid #e5e7eb; border-radius: 10px;
  margin-bottom: 1.5rem; overflow: hidden;
}
.card-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.6rem 1rem; background: #f9fafb; border-bottom: 1px solid #e5e7eb;
}
.filename { font-weight: 700; font-family: monospace; font-size: 0.9rem; }
.match-badge { font-size: 0.75rem; font-weight: 700; padding: 0.2rem 0.6rem; border-radius: 5px; }
.match-badge.yes { background: #dcfce7; color: #15803d; }
.match-badge.no  { background: #ffedd5; color: #c2410c; }
.match-badge.na  { background: #e5e7eb; color: #4b5563; }

.card-body { padding: 1rem; }
.filmstrip { display: flex; gap: 0.75rem; overflow-x: auto; padding-bottom: 0.75rem; }
.thumb { flex: 0 0 auto; width: 200px; text-align: center; }
.thumb img { width: 100%; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,0.15); }
.thumb .caption { font-size: 0.75rem; margin-top: 0.35rem; }
.thumb .method-name { font-weight: 700; font-family: monospace; display: block; }
.thumb .metric { color: #6b7280; }
.thumb.best .method-name { color: #16a34a; }
.thumb.baseline .method-name { color: #6b7280; }

table.metrics { width: 100%; border-collapse: collapse; margin-top: 0.75rem; font-size: 0.8rem; }
table.metrics th, table.metrics td { text-align: left; padding: 0.3rem 0.6rem; border-bottom: 1px solid #e5e7eb; }
table.metrics th { color: #6b7280; font-weight: 700; text-transform: uppercase; font-size: 0.7rem; }
"""


def _b64_jpeg(img) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("imencode failed")
    return base64.b64encode(bytes(buf)).decode()


def _thumbnail(img, width=200, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    return cv2.resize(img, (width, int(h * width / w)), interpolation=interpolation)


def _render_card(image: str, scale: int, rows: list[dict]) -> str:
    h = _html.escape
    verdict = _card_verdict(rows)
    if verdict is not None:
        is_improved, _, best_method = verdict
        match_cls = "yes" if is_improved else "no"
        match_text = "Improved" if is_improved else "No improvement"
    else:
        best_method = None
        match_cls, match_text = "na", "Baseline only (no pretrained model at this scale)"

    hr = rows[0]["hr"]
    lr = rows[0]["lr"]
    hh, hw = hr.shape[:2]
    lh, lw = lr.shape[:2]

    # Nearest-neighbor upscale of the LR input: the crudest possible
    # reconstruction, quantifying how much detail SR/bilinear actually recover.
    nn_full = cv2.resize(lr, (hw, hh), interpolation=cv2.INTER_NEAREST)
    nn_psnr, nn_ssim = psnr(nn_full, hr), ssim(nn_full, hr)

    lines = [
        "",
        '  <div class="image-card">',
        '    <div class="card-header">',
        f'      <span class="filename">{h(image)} &mdash; x{scale}</span>',
        f'      <span class="match-badge {match_cls}">{match_text}</span>',
        "    </div>",
        '    <div class="card-body">',
        '      <div class="filmstrip">',
        f'        <div class="thumb baseline"><img src="data:image/jpeg;base64,{_b64_jpeg(_thumbnail(hr))}" alt="ground truth"><div class="caption"><span class="method-name">ground truth</span><span class="metric">{hw}&times;{hh}</span></div></div>',
        f'        <div class="thumb baseline"><img src="data:image/jpeg;base64,{_b64_jpeg(_thumbnail(nn_full, interpolation=cv2.INTER_NEAREST))}" alt="low-res input (before)"><div class="caption"><span class="method-name">low-res input</span><span class="metric">{lw}&times;{lh} native &mdash; {nn_psnr:.2f} dB / {nn_ssim:.3f}</span></div></div>',
    ]
    for r in rows:
        cls = "best" if r["method"] == best_method else ("baseline" if r["method"].startswith(_BASELINE_PREFIXES) else "")
        lines += [
            f'        <div class="thumb {cls}">',
            f'          <img src="data:image/jpeg;base64,{_b64_jpeg(_thumbnail(r["output"]))}" alt="{h(r["method"])}">',
            '          <div class="caption">',
            f'            <span class="method-name">{h(r["method"])}</span>',
            f'            <span class="metric">{r["psnr"]:.2f} dB / {r["ssim"]:.3f}</span>',
            "          </div>",
            "        </div>",
        ]
    lines += ["      </div>", '      <table class="metrics">', "        <tr><th>Method</th><th>PSNR (dB)</th><th>SSIM</th><th>Latency (ms)</th></tr>"]
    for r in sorted(rows, key=lambda r: -r["psnr"]):
        lines.append(f'        <tr><td>{h(r["method"])}</td><td>{r["psnr"]:.2f}</td><td>{r["ssim"]:.4f}</td><td>{r["latency_ms"]:.1f}</td></tr>')
    lines += ["      </table>", "    </div>", "  </div>"]
    return "\n".join(lines)


def _build_html(run_id: str, commit: dict, ts: str, cards: dict) -> str:
    h = _html.escape
    verdicts = [_card_verdict(rows) for rows in cards.values()]
    scored = [v for v in verdicts if v is not None]
    total = len(scored)
    improved = sum(1 for v in scored if v[0])
    verdict_cls = "pass" if improved == total else "fail"
    verdict_text = "PASS" if improved == total else "NEEDS IMPROVEMENT"

    lines = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="UTF-8">',
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        f"  <title>SuperResolution Eval &mdash; medicine_package &mdash; {run_id}</title>",
        f"  <style>{_CSS}</style>",
        "</head>",
        "<body>",
        "<header>",
        "  <h1>Super-Resolution Evaluation &mdash; medicine_package</h1>",
        '  <div class="commit-block">',
        f'    <span class="commit-subject">{h(commit["commit_subject"])}</span>',
        f'    <span class="commit-hash">{h(commit["commit_full"])}</span>',
        f'    <span class="meta">{h(run_id)} &nbsp;&middot;&nbsp; {h(ts)}</span>',
        "  </div>",
        "</header>",
        "",
        '<div class="container">',
        f'  <div class="summary {verdict_cls}">',
        '    <div class="summary-top">',
        f'      <span class="verdict-badge {verdict_cls}">{verdict_text}</span>',
        f'      <span class="pass-count {verdict_cls}">{improved} / {total}</span>',
        '      <span class="threshold-note">image&times;scale combinations where the best SR/hybrid method beat bilinear PSNR</span>',
        "    </div>",
        "  </div>",
    ]

    for (image, scale), rows in sorted(cards.items()):
        lines.append(_render_card(image, scale, rows))

    lines += ["</div>", "</body>", "</html>"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ts_obj = datetime.now(timezone.utc)
    ts = ts_obj.isoformat()
    date_str = ts_obj.strftime("%Y%m%d")
    run_id = _next_run_id(date_str)
    commit = _commit_info()

    print(f"Running eval: {run_id}")
    print(f"  Fixtures: {FIXTURE_DIR}")
    print()

    tmp_dir = Path(tempfile.mkdtemp(prefix="sr_eval_"))
    try:
        names = _prepare_cropped_fixtures(tmp_dir)
        ref_w, ref_h = CROP_SIZE[0] // BASE_DOWNSCALE, CROP_SIZE[1] // BASE_DOWNSCALE
        print(
            f"  Cropped {len(names)} fixture images to {CROP_SIZE[0]}x{CROP_SIZE[1]}, "
            f"then downscaled x{BASE_DOWNSCALE} to {ref_w}x{ref_h} ground truth"
        )

        models_by_scale = load_models(MODEL_SPECS)
        results = evaluate(
            str(tmp_dir), models_by_scale=models_by_scale, baseline_scales=BASELINE_ONLY_SCALES, keep_images=True
        )
        cards = _group_by_card(results)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    total = 0
    improved = 0
    for (image, scale), rows in sorted(cards.items()):
        verdict = _card_verdict(rows)
        if verdict is None:
            print(f"  .  {image} x{scale}: baseline only (no pretrained model at this scale)")
            continue
        is_improved, baseline_psnr, best_method = verdict
        total += 1
        improved += int(is_improved)
        mark = "✓" if is_improved else "✗"
        print(f"  {mark}  {image} x{scale}: best={best_method} vs bilinear={baseline_psnr:.2f} dB")

    print()
    print(f"Improved: {improved}/{total} ({improved / total:.0%})")
    print()

    json_path = FIXTURE_DIR / f"{run_id}.json"
    report = _build_json(run_id, commit, ts, cards)
    json_path.write_text(json.dumps(report, indent=2))
    print(f"  JSON -> {json_path.relative_to(_PROJECT_ROOT)}")

    html_path = FIXTURE_DIR / f"{run_id}.html"
    html_src = _build_html(run_id, commit, ts, cards)
    html_path.write_text(html_src)
    print(f"  HTML -> {html_path.relative_to(_PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
