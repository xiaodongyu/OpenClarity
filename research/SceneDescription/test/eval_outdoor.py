"""
Evaluate SceneDescription accuracy against the outdoor fixture set.

Calls describe_scene() on each test image, then uses an LLM judge to score
semantic similarity against a one-sentence ground truth (0.0–1.0).

Usage (from research/SceneDescription):
    python test/eval_outdoor.py

Writes eval_<YYYYMMDD>_<NNN>.json and .html to
test/fixtures/outdoor/, where NNN is the 1-based sequence number
of runs on that calendar date in that folder.
"""

import base64
import html as _html
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env from project root if present (ANTHROPIC_API_KEY etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

from src.capture import encode_jpeg
from src.vlm_client import describe_scene

FIXTURE_DIR  = _PROJECT_ROOT / "test" / "fixtures" / "outdoor"
GT_FILE      = FIXTURE_DIR / "scene_description_groundtruth.txt"
PASS_THRESHOLD = 0.70   # semantic similarity threshold


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git"] + cmd, text=True, stderr=subprocess.DEVNULL,
            cwd=_PROJECT_ROOT,
        ).strip()
    except subprocess.CalledProcessError:
        return ""


def _commit_info() -> dict:
    return {
        "commit":         _git(["rev-parse", "--short", "HEAD"]),
        "commit_full":    _git(["rev-parse", "HEAD"]),
        "commit_subject": _git(["log", "-1", "--format=%s"]),
        "commit_body":    _git(["log", "-1", "--format=%b"]),
    }


# ---------------------------------------------------------------------------
# Run ID
# ---------------------------------------------------------------------------

def _next_run_id(date_str: str) -> str:
    existing = sorted(FIXTURE_DIR.glob(f"eval_{date_str}_*.json"))
    return f"eval_{date_str}_{len(existing) + 1:03d}"


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def _load_ground_truth() -> dict[str, str]:
    gt: dict[str, str] = {}
    with open(GT_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fname, _, text = line.partition(": ")
            gt[fname.strip()] = text.strip()
    return gt


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

def _judge(generated: str, ground_truth: str) -> tuple[float, str]:
    """Return (similarity_score 0.0–1.0, one-sentence rationale) via Claude."""
    import anthropic
    import os

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    prompt = (
        "Compare these two scene descriptions and rate their semantic similarity.\n\n"
        f"Ground truth: {ground_truth}\n"
        f"Generated:    {generated}\n\n"
        "Score 0.0–1.0 where 1.0 = same essential information, "
        "0.7 = mostly matches with minor omissions, "
        "0.4 = partial match, 0.0 = completely different scene.\n\n"
        'Reply with JSON only: {"score": <float 0.0-1.0>, "rationale": "<one sentence>"}'
    )
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=128,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    data = json.loads(raw)
    score = float(data["score"])
    rationale = str(data.get("rationale", ""))
    return round(min(max(score, 0.0), 1.0), 3), rationale


# ---------------------------------------------------------------------------
# Per-image eval
# ---------------------------------------------------------------------------

def _eval_image(fname: str, ground_truth: str) -> dict:
    img_path = FIXTURE_DIR / fname
    img = cv2.imread(str(img_path))
    if img is None:
        return {
            "filename": fname,
            "ground_truth": ground_truth,
            "generated": "",
            "score": 0.0,
            "rationale": "",
            "passed": False,
            "elapsed_s": 0.0,
            "error": f"Cannot read image: {img_path}",
        }

    t0 = time.perf_counter()
    try:
        image_b64 = encode_jpeg(img)
        generated = describe_scene(image_b64)
        error = None
    except Exception as exc:
        generated = ""
        error = str(exc)
    elapsed = round(time.perf_counter() - t0, 3)

    if generated:
        score, rationale = _judge(generated, ground_truth)
    else:
        score, rationale = 0.0, ""

    entry: dict = {
        "filename": fname,
        "ground_truth": ground_truth,
        "generated": generated,
        "score": score,
        "rationale": rationale,
        "passed": score >= PASS_THRESHOLD,
        "elapsed_s": elapsed,
    }
    if error:
        entry["error"] = error
    return entry


# ---------------------------------------------------------------------------
# Stdout report
# ---------------------------------------------------------------------------

def _print_report(run_id: str, commit: dict, results: list[dict]) -> None:
    w = 72
    print("=" * w)
    print(f"{'SceneDescription Eval — outdoor':^{w}}")
    print(f"{(commit['commit'] + '  ' + commit['commit_subject'])[:w]:^{w}}")
    print("=" * w)
    for r in results:
        sym = "✓" if r["passed"] else "✗"
        print(f"\n{'─' * w}")
        print(f"  {sym}  {r['filename']}")
        print(f"  Ground truth : {r['ground_truth']}")
        print(f"  Generated    : {r['generated'] or '(empty)'}")
        print(f"  Score        : {r['score']:.2f}  (threshold ≥ {PASS_THRESHOLD})")
        print(f"  Rationale    : {r['rationale']}")
        print(f"  Elapsed      : {r['elapsed_s']:.2f}s")
        if "error" in r:
            print(f"  Error        : {r['error']}")
    pass_count = sum(1 for r in results if r["passed"])
    total = len(results)
    verdict = "PASS" if pass_count == total else "NEEDS IMPROVEMENT"
    print(f"\n{'─' * w}")
    print(f"  Verdict : {verdict}  ({pass_count}/{total} passed)")
    print("=" * w)


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def _build_json(run_id: str, commit: dict, ts: str, results: list[dict]) -> dict:
    pass_count = sum(1 for r in results if r["passed"])
    total = len(results)
    mean_score = round(sum(r["score"] for r in results) / total, 3) if total else 0.0
    verdict = "PASS" if pass_count == total else "NEEDS IMPROVEMENT"
    return {
        "fixture": "outdoor",
        **commit,
        "timestamp": ts,
        "threshold": PASS_THRESHOLD,
        "total": total,
        "passed": pass_count,
        "mean_score": mean_score,
        "verdict": verdict,
        "images": results,
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
.commit-block .commit-body { font-family: monospace; color: #6b7280; font-size: 0.75rem; white-space: pre-wrap; margin-top: 0.15rem; }
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
.mean-score { font-size: 0.875rem; color: #4b5563; }

.image-card {
  background: white; border: 1px solid #e5e7eb; border-radius: 10px;
  margin-bottom: 1.5rem; overflow: hidden;
}
.card-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.6rem 1rem; background: #f9fafb; border-bottom: 1px solid #e5e7eb;
}
.filename { font-weight: 700; font-family: monospace; font-size: 0.9rem; }
.card-header-right { display: flex; align-items: center; gap: 0.75rem; }
.elapsed { font-size: 0.75rem; color: #9ca3af; }
.score-badge { font-size: 0.8rem; font-weight: 700; padding: 0.2rem 0.6rem; border-radius: 5px; }
.score-badge.pass { background: #dcfce7; color: #15803d; }
.score-badge.fail { background: #ffedd5; color: #c2410c; }

.card-body { display: flex; min-height: 0; }
.image-col {
  flex: 0 0 46%; padding: 1rem; border-right: 1px solid #e5e7eb;
  display: flex; align-items: flex-start; justify-content: center; background: #f9fafb;
}
.image-col img { max-width: 100%; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,0.15); }
.image-col .no-image { color: #9ca3af; font-style: italic; font-size: 0.85rem; align-self: center; }
.text-col { flex: 1; padding: 1rem; display: flex; flex-direction: column; gap: 0.9rem; overflow: auto; }
.field label {
  display: block; font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.07em; color: #9ca3af; margin-bottom: 0.3rem;
}
.field p { font-size: 0.875rem; line-height: 1.6; }
.field .generated { color: #2563eb; font-weight: 500; }
.field .empty { color: #9ca3af; font-style: italic; }
.field .rationale { color: #6b7280; font-style: italic; }
.error-box {
  background: #fef2f2; border: 1px solid #fecaca; border-radius: 6px;
  padding: 0.5rem 0.75rem; font-size: 0.8rem; color: #b91c1c; font-family: monospace;
}

.score-bar-wrap { display: flex; align-items: center; gap: 0.5rem; }
.score-bar-bg { flex: 1; height: 6px; background: #e5e7eb; border-radius: 3px; overflow: hidden; }
.score-bar-fill { height: 100%; border-radius: 3px; }
.score-bar-fill.pass { background: #22c55e; }
.score-bar-fill.fail { background: #f97316; }
.score-label { font-size: 0.8rem; font-weight: 700; min-width: 3rem; text-align: right; }
.score-label.pass { color: #16a34a; }
.score-label.fail { color: #ea580c; }

@media (max-width: 800px) {
  .card-body { flex-direction: column; }
  .image-col { flex: none; border-right: none; border-bottom: 1px solid #e5e7eb; }
}
"""


def _image_to_b64(img_path: Path) -> str:
    img = cv2.imread(str(img_path))
    if img is None:
        return ""
    h, w = img.shape[:2]
    if max(h, w) > 1000:
        scale = 1000 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode()


def _render_card(r: dict) -> str:
    h = _html.escape
    fname = r["filename"]
    passed = r["passed"]
    cls = "pass" if passed else "fail"
    sym = "✓" if passed else "✗"

    b64 = _image_to_b64(FIXTURE_DIR / fname)
    img_tag = (
        f'<img src="data:image/jpeg;base64,{b64}" alt="{h(fname)}">'
        if b64 else '<span class="no-image">Image unavailable</span>'
    )

    generated_html = (
        f'<p class="generated">{h(r["generated"])}</p>'
        if r["generated"]
        else '<p class="empty">(no description generated)</p>'
    )

    error_html = (
        f'<div class="error-box">Error: {h(r["error"])}</div>'
        if "error" in r else ""
    )

    score_pct = int(r["score"] * 100)
    rationale_html = (
        f'<p class="rationale">{h(r["rationale"])}</p>'
        if r["rationale"] else ""
    )

    return f"""
<div class="image-card">
  <div class="card-header">
    <span class="filename">{h(fname)}</span>
    <div class="card-header-right">
      <span class="elapsed">{r['elapsed_s']:.2f}s</span>
      <span class="score-badge {cls}">{sym} {r['score']:.2f}</span>
    </div>
  </div>
  <div class="card-body">
    <div class="image-col">{img_tag}</div>
    <div class="text-col">
      <div class="field">
        <label>Ground Truth</label>
        <p>{h(r['ground_truth'])}</p>
      </div>
      <div class="field">
        <label>Generated</label>
        {generated_html}
      </div>
      <div class="field">
        <label>Semantic Similarity</label>
        <div class="score-bar-wrap">
          <div class="score-bar-bg">
            <div class="score-bar-fill {cls}" style="width:{score_pct}%"></div>
          </div>
          <span class="score-label {cls}">{r['score']:.2f}</span>
        </div>
        {rationale_html}
      </div>
      {error_html}
    </div>
  </div>
</div>"""


def _build_html(run_id: str, commit: dict, ts: str,
                results: list[dict], report: dict) -> str:
    h = _html.escape
    passed = report["verdict"] == "PASS"
    cls = "pass" if passed else "fail"
    ts_disp = ts.replace("T", " ").replace("+00:00", " UTC")

    body_html = (
        f'<span class="commit-body">{h(commit["commit_body"])}</span>'
        if commit.get("commit_body") else ""
    )

    cards = "\n".join(_render_card(r) for r in results)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SceneDescription Eval — outdoor — {h(run_id)}</title>
  <style>{_CSS}</style>
</head>
<body>
<header>
  <h1>SceneDescription Evaluation &mdash; outdoor</h1>
  <div class="commit-block">
    <span class="commit-subject">{h(commit['commit_subject'])}</span>
    <span class="commit-hash">{h(commit['commit_full'])}</span>
    {body_html}
    <span class="meta">{h(run_id)} &nbsp;·&nbsp; {h(ts_disp)}</span>
  </div>
</header>

<div class="container">

  <div class="summary {cls}">
    <div class="summary-top">
      <span class="verdict-badge {cls}">{h(report['verdict'])}</span>
      <span class="pass-count {cls}">{report['passed']} / {report['total']}</span>
      <span class="threshold-note">images passed (threshold ≥ {report['threshold']})</span>
    </div>
    <p class="mean-score">Mean semantic similarity: {report['mean_score']:.2f}</p>
  </div>

  {cards}

</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ts_obj   = datetime.now(timezone.utc)
    ts       = ts_obj.isoformat()
    date_str = ts_obj.strftime("%Y%m%d")
    run_id   = _next_run_id(date_str)
    commit   = _commit_info()

    print(f"Running eval: {run_id}", file=sys.stderr)
    print(f"  Fixture : {FIXTURE_DIR}", file=sys.stderr)
    print(f"  GT file : {GT_FILE}", file=sys.stderr)
    print(file=sys.stderr)

    gt = _load_ground_truth()
    results: list[dict] = []

    for fname, ground_truth in gt.items():
        print(f"  → {fname}", file=sys.stderr)
        result = _eval_image(fname, ground_truth)
        results.append(result)
        sym = "✓" if result["passed"] else "✗"
        print(f"    {sym}  score={result['score']:.2f}  elapsed={result['elapsed_s']:.2f}s",
              file=sys.stderr)

    _print_report(run_id, commit, results)

    report = _build_json(run_id, commit, ts, results)

    json_path = FIXTURE_DIR / f"{run_id}.json"
    json_path.write_text(json.dumps(report, indent=2))

    html_path = FIXTURE_DIR / f"{run_id}.html"
    html_path.write_text(_build_html(run_id, commit, ts, results, report))

    print(f"\nArtifacts saved ({run_id}) →", file=sys.stderr)
    print(f"  JSON : {json_path.relative_to(_PROJECT_ROOT)}", file=sys.stderr)
    print(f"  HTML : {html_path.relative_to(_PROJECT_ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
