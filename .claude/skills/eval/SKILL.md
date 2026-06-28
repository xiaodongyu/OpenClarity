# Skill: eval

## Purpose

Commit any pending source changes, run the project's evaluation or test suite,
then commit the output artifacts — so every artifact file permanently records
the exact git commit it was produced from.

## When to invoke

Use when the user says any of:
- "run the eval" / "run eval" / "evaluate"
- "run tests" / "run the fixture tests"
- `/eval`
- `/eval <project>` — target a specific sub-project (e.g. `/eval OCR`)
- `/eval <project> <fixture>` — target a fixture subset (e.g. `/eval OCR medicine_package`)

## Steps

### 1. Determine the target project

If the user provided a project name, map it to its directory under `research/`.

If no project was specified:
- Run `git status --short` and identify which `research/<Project>/` subtree
  has modified or untracked source files.
- If exactly one project has changes, use it without asking.
- If multiple projects have changes, ask the user which to run.

### 2. Discover the eval command

Look inside the project directory for, in order of preference:

| Signal | Command |
|--------|---------|
| `test/eval_*.py` exists | `<python> test/eval_*.py [args]` run from the project root |
| `pytest.ini` / `pyproject.toml` / `setup.cfg` present | `pytest [args]` |
| `Makefile` with `eval` or `test` target | `make eval` or `make test` |
| README "Testing" / "Evaluation" section | follow those instructions |

If the project has a virtual environment, use its interpreter; otherwise fall
back to the system Python. For this repo:
- `research/OCR/` → `/usr/bin/python3.10` (has paddleocr; no venv)
- `research/SceneDescription/` → `/usr/bin/python3.10`
- `research/ObjectDetection/` → `research/ObjectDetection/.venv/bin/python3`

If the command is still ambiguous after inspecting the above, ask the user.

### 3. Classify uncommitted files

Run `git status --short`. For each changed file under the target project,
classify as either:

- **Source** — anything that affects what the eval tests: `src/**`, `test/*.py`,
  `test/fixtures/**/ground_truth.txt`, `docs/**`, config files, `*.md` outside
  output directories.
- **Artifact** — output files from a prior eval run. Heuristics:
  - Filename starts with `eval_`, `results_`, or `report_`
  - Extension is `.json`, `.html`, `.csv`, or `.log` inside a `fixtures/` or
    `results/` subdirectory.

Artifacts must NOT appear in the source commit.

### 4. Commit source changes (if any)

If source files are uncommitted:

a. Stage only source files:
```bash
git add <source files>
```

b. Read `git diff --cached` and draft a concise commit message (subject ≤ 72
   chars) describing what changed. Do not mention the eval run itself.

c. Commit:
```bash
git commit -m "$(cat <<'EOF'
<subject line>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

Record the resulting hash — this is the **source commit** the eval will test.

If there are no uncommitted source changes, HEAD is already the source commit.

### 5. Run the eval

From the project root, execute the command discovered in Step 2. Pass any
fixture or subset argument the user supplied. Suppress noisy framework output:

```bash
cd <project-dir> && <eval-command> [args] 2>/dev/null
```

Capture stdout for the report in Step 7.

### 6. Commit output artifacts

Run `git status --short` again to detect newly written or modified files.
Stage artifact files (skip any source files that changed incidentally):

```bash
git add <artifact files>
```

Commit with a message that names the project, scope, and source commit:
```bash
git commit -m "$(cat <<'EOF'
[eval] <Project> <scope or "all"> — tested at <source-commit-hash>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

If the eval produced no output files, skip this commit.

### 7. Report to the user

Summarise:
- **Source commit**: hash + subject
- **Scope**: project, fixture/subset (if any)
- **Results**: per-test verdict and score (✓/✗), any aggregate metric
- **Artifacts**: relative paths to generated HTML/JSON reports
- **Regressions**: flag anything worse than the previous run if visible in the
  captured stdout

## Conventions

- **Two-commit rule**: source changes in one commit, eval artifacts in another.
  This keeps `git log` readable — "what changed" is separate from "what the
  eval measured."
- Always commit source *before* running so the artifact's embedded commit hash
  refers to the code under test, not a later revision.
- Never force-push or amend commits that already contain eval artifacts; add a
  new eval run instead.
