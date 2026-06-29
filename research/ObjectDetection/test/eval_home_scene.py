"""
Evaluate object-location accuracy against the home_scene fixture set.

Usage (from research/ObjectDetection):
    .venv/bin/python test/eval_home_scene.py

Writes eval_<YYYYMMDD>_<NNN>.json and .html to
test/fixtures/home_scene/, where NNN is the 1-based sequence number
of runs on that calendar date in that folder.

The HTML report embeds annotated images: each image is overlaid with
a 3×3 grid (2 vertical + 2 horizontal lines), the detected object's
bounding box, and a location label.  A per-query table shows prompt,
predicted location, expected location, and match (Yes/No).
"""

import base64
import html as _html
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.query_eval import QueryResult, evaluate_queries, summarize_results

FIXTURE_DIR = _PROJECT_ROOT / "test" / "fixtures" / "home_scene"
PROMPTS_FILE = FIXTURE_DIR / "object_detection_prompt.txt"
GROUNDTRUTH_FILE = FIXTURE_DIR / "object_detection_groundtruth.txt"

_GRID_COLOR = (255, 255, 0)     # BGR cyan — scene divider
_BBOX_COLOR = (0, 220, 0)       # BGR green — correct match
_MISS_COLOR = (0, 0, 255)       # BGR red — wrong/no detection
_LABEL_BG   = (0, 0, 0)
_LABEL_FG   = (255, 255, 255)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(["git"] + cmd, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return ""


def _commit_info() -> dict:
    short = _git(["rev-parse", "--short", "HEAD"])
    full  = _git(["rev-parse", "HEAD"])
    subj  = _git(["log", "-1", "--format=%s"])
    body  = _git(["log", "-1", "--format=%b"])
    return {"commit": short, "commit_full": full,
            "commit_subject": subj, "commit_body": body}


# ---------------------------------------------------------------------------
# Next eval run ID
# ---------------------------------------------------------------------------

def _next_run_id(date_str: str) -> str:
    existing = sorted(FIXTURE_DIR.glob(f"eval_{date_str}_*.json"))
    seq = len(existing) + 1
    return f"eval_{date_str}_{seq:03d}"


# ---------------------------------------------------------------------------
# Image annotation
# ---------------------------------------------------------------------------

def _annotate(image_path: Path, result: QueryResult) -> bytes:
    """Return JPEG bytes of the image annotated with 3×3 grid + bbox."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Cannot load {image_path}")
    h, w = img.shape[:2]

    # 3×3 grid lines
    for i in (1, 2):
        x = w * i // 3
        y = h * i // 3
        cv2.line(img, (x, 0), (x, h - 1), _GRID_COLOR, 4, cv2.LINE_AA)
        cv2.line(img, (0, y), (w - 1, y), _GRID_COLOR, 4, cv2.LINE_AA)

    if result.bbox is not None:
        x1, y1, x2, y2 = result.bbox
        color = _BBOX_COLOR if result.is_correct else _MISS_COLOR
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 6)

        label = result.predicted_location
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.6, 1
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)
        ty = max(y1 - 6, th + baseline)
        cv2.rectangle(img, (x1, ty - th - baseline), (x1 + tw + 4, ty + baseline), _LABEL_BG, -1)
        cv2.putText(img, label, (x1 + 2, ty), font, scale, _LABEL_FG, thick, cv2.LINE_AA)
    elif result.predicted_location == "N/A":
        cv2.putText(img, "N/A (not detected)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, _MISS_COLOR, 2, cv2.LINE_AA)

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("imencode failed")
    return bytes(buf)


def _b64_jpeg(image_path: Path, result: QueryResult) -> str:
    data = _annotate(image_path, result)
    return base64.b64encode(data).decode()


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def _build_json(run_id: str, commit: dict, ts: str,
                results: list[QueryResult], summary: dict) -> dict:
    correct = summary["correct"]
    total   = summary["total"]
    verdict = "PASS" if correct == total else "NEEDS IMPROVEMENT"

    images = []
    for r in results:
        images.append({
            "image_name": r.image_name,
            "prompt": r.prompt,
            "target_label": r.target_label,
            "predicted_location": r.predicted_location,
            "expected_location": r.expected_location,
            "is_supported": r.is_supported,
            "is_correct": r.is_correct,
            "bbox": list(r.bbox) if r.bbox else None,
        })

    return {
        "fixture": "home_scene",
        **commit,
        "timestamp": ts,
        "total": total,
        "correct": correct,
        "accuracy": summary["accuracy"],
        "verdict": verdict,
        "images": images,
    }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, -apple-system, sans-serif; background: #f3f4f6; color: #1f2937; }

header {
  background: #111827; color: #f9fafb;
  padding: 1.25rem 2rem;
}
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
.match-badge {
  font-size: 0.75rem; font-weight: 700; padding: 0.2rem 0.6rem; border-radius: 5px;
}
.match-badge.yes { background: #dcfce7; color: #15803d; }
.match-badge.no  { background: #ffedd5; color: #c2410c; }

.card-body { display: flex; min-height: 0; }
.image-col {
  flex: 0 0 55%; padding: 1rem; border-right: 1px solid #e5e7eb;
  display: flex; align-items: flex-start; justify-content: center; background: #f9fafb;
}
.image-col img { max-width: 100%; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,0.15); }
.text-col { flex: 1; padding: 1rem; display: flex; flex-direction: column; gap: 0.9rem; }
.field label {
  display: block; font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.07em; color: #9ca3af; margin-bottom: 0.3rem;
}
.field p { font-size: 0.875rem; line-height: 1.6; }
.pred { color: #2563eb; font-weight: 600; }
.expected { color: #111827; }
.na { color: #9ca3af; font-style: italic; }

@media (max-width: 800px) {
  .card-body { flex-direction: column; }
  .image-col { flex: none; border-right: none; border-bottom: 1px solid #e5e7eb; }
}
"""


def _build_html(run_id: str, commit: dict, ts: str,
                results: list[QueryResult], summary: dict) -> str:
    correct = summary["correct"]
    total   = summary["total"]
    verdict_cls  = "pass" if correct == total else "fail"
    verdict_text = "PASS" if correct == total else "NEEDS IMPROVEMENT"

    h = _html.escape
    lines = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="UTF-8">',
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        f'  <title>ObjectDetection Eval — home_scene — {run_id}</title>',
        f"  <style>{_CSS}</style>",
        "</head>",
        "<body>",
        "<header>",
        "  <h1>ObjectDetection Evaluation &mdash; home_scene</h1>",
        '  <div class="commit-block">',
        f'    <span class="commit-subject">{h(commit["commit_subject"])}</span>',
        f'    <span class="commit-hash">{h(commit["commit_full"])}</span>',
        f'    <span class="meta">{h(run_id)} &nbsp;·&nbsp; {h(ts)}</span>',
        "  </div>",
        "</header>",
        "",
        '<div class="container">',
        f'  <div class="summary {verdict_cls}">',
        '    <div class="summary-top">',
        f'      <span class="verdict-badge {verdict_cls}">{verdict_text}</span>',
        f'      <span class="pass-count {verdict_cls}">{correct} / {total}</span>',
        '      <span class="threshold-note">queries correct</span>',
        "    </div>",
        "  </div>",
    ]

    for r in results:
        match_cls  = "yes" if r.is_correct else "no"
        match_text = "Yes" if r.is_correct else "No"
        img_src = _b64_jpeg(FIXTURE_DIR / r.image_name, r)

        pred_cls = "na" if r.predicted_location == "N/A" else "pred"

        lines += [
            "",
            '  <div class="image-card">',
            '    <div class="card-header">',
            f'      <span class="filename">{h(r.image_name)}</span>',
            '      <div style="display:flex;align-items:center;gap:0.75rem;">',
            f'        <span class="match-badge {match_cls}">Match: {match_text}</span>',
            "      </div>",
            "    </div>",
            '    <div class="card-body">',
            '      <div class="image-col">',
            f'        <img src="data:image/jpeg;base64,{img_src}" alt="{h(r.image_name)}">',
            "      </div>",
            '      <div class="text-col">',
            '        <div class="field">',
            '          <label>Prompt</label>',
            f'          <p>{h(r.prompt)}</p>',
            "        </div>",
            '        <div class="field">',
            '          <label>Predicted Location</label>',
            f'          <p class="{pred_cls}">{h(r.predicted_location)}</p>',
            "        </div>",
            '        <div class="field">',
            '          <label>Expected Location</label>',
            f'          <p class="expected">{h(str(r.expected_location))}</p>',
            "        </div>",
            '        <div class="field">',
            '          <label>Target Label</label>',
            f'          <p>{h(str(r.target_label))}</p>',
            "        </div>",
            '        <div class="field">',
            '          <label>Supported by YOLO</label>',
            f'          <p>{"Yes" if r.is_supported else "No"}</p>',
            "        </div>",
            "      </div>",
            "    </div>",
            "  </div>",
        ]

    lines += [
        "</div>",
        "</body>",
        "</html>",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ts_obj  = datetime.now(timezone.utc)
    ts      = ts_obj.isoformat()
    date_str = ts_obj.strftime("%Y%m%d")
    run_id   = _next_run_id(date_str)
    commit   = _commit_info()

    print(f"Running eval: {run_id}")
    print(f"  Prompts:     {PROMPTS_FILE}")
    print(f"  Ground truth:{GROUNDTRUTH_FILE}")
    print()

    results = evaluate_queries(
        prompts_path=PROMPTS_FILE,
        ground_truth_path=GROUNDTRUTH_FILE,
        image_dir=FIXTURE_DIR,
    )
    summary = summarize_results(results)

    # Print per-query results
    for r in results:
        mark = "✓" if r.is_correct else "✗"
        print(f"  {mark}  {r.image_name}: {r.prompt}")
        print(f"       predicted={r.predicted_location}  expected={r.expected_location}")

    print()
    print(f"Accuracy: {summary['correct']}/{summary['total']} "
          f"({summary['accuracy']:.0%})")
    print()

    # Write JSON
    json_path = FIXTURE_DIR / f"{run_id}.json"
    report    = _build_json(run_id, commit, ts, results, summary)
    json_path.write_text(json.dumps(report, indent=2))
    print(f"  JSON → {json_path.relative_to(_PROJECT_ROOT)}")

    # Write HTML
    html_path = FIXTURE_DIR / f"{run_id}.html"
    html_src  = _build_html(run_id, commit, ts, results, summary)
    html_path.write_text(html_src)
    print(f"  HTML → {html_path.relative_to(_PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
