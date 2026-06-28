# Skill: session-report

## Purpose

Produce a dated Markdown document summarising all source improvements and
evaluation results from the current working session, then commit it.

## When to invoke

Use when the user says any of:
- "write a session report" / "write a summary report"
- "summarise what we did today" / "document today's work"
- `/session-report`
- `/session-report <project>` — limit scope to one sub-project

## Steps

### 1. Determine scope

If the user named a project (e.g. `/session-report OCR`), focus on
`research/<Project>/`. Otherwise include all sub-projects that have commits
since the session began (use `git log` to find them).

### 2. Gather commits made this session

```bash
git log --oneline <since-ref>..HEAD
```

If no reference point is known, use commits from today's date in UTC:
```bash
git log --oneline --since="$(date -u +%Y-%m-%d) 00:00" --until="$(date -u +%Y-%m-%d) 23:59"
```

Group commits by type:
- **Source improvement** — commits that change `src/`, `test/*.py`,
  `test/fixtures/**/ground_truth.txt`, `docs/`, config, or skill files.
- **Eval artifact** — commits whose subject starts with `[eval]`.
- **Report** — commits that add a `revision_and_test_report_*.md` file
  (skip these to avoid self-reference).

### 3. Gather eval results

For each `[eval]` commit, read the stdout captured in the artifact JSON
(`test/fixtures/**/eval_*.json`) to extract per-image verdicts and scores.
If no JSON is available, note that the eval was run but results are unavailable.

### 4. Draft the report

Write to `research/<Project>/docs/revision_and_test_report_YYYYMMDD.md`
where `YYYYMMDD` is today's UTC date. If the file already exists for today,
append a `_2`, `_3`, … suffix.

Structure:

```markdown
# <Project> — Revision and Test Report — YYYY-MM-DD

Summary of improvements made and evaluation results recorded during this session.

---

## Improvements

### N. <Short title>
**Commit:** `<hash>`  
**Files:** `<affected files>`

<What the problem was, what the fix was, why it works.>

---

(repeat for each source improvement commit)

## Test Results

Eval artifact commit: `<hash>` — tested at source commit `<hash>`.

### <fixture name> (Mode — threshold)

| Image | Expected | Score | Verdict |
...

**N / Total passed**

<Per-failure root-cause note if any failed.>

---

(repeat for each fixture that was evaluated)

## Known Limitations

| Issue | Root cause | Fix path |
...
```

Rules:
- One `##` section per source improvement commit; one `###` subsection per
  evaluated fixture.
- Include commit hash for every improvement so the reader can `git show` it.
- For failures, always explain the root cause (not just "failed").
- Known-limitations table is optional; include it if any failures have a clear
  structural cause that a future fix should address.
- Do not invent metrics — pull numbers directly from the JSON artifacts or
  captured stdout.

### 5. Commit the report

```bash
git add research/<Project>/docs/revision_and_test_report_YYYYMMDD.md
git commit -m "$(cat <<'EOF'
[<Project>] Add session revision and test report YYYY-MM-DD

<One-line summary: N improvements, eval results for fixture A (X/Y) and B (X/Y).>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

### 6. Report to the user

State the file path, commit hash, and a one-sentence summary of what the
report covers.

## Conventions

- Date is always UTC so it matches the eval artifact filenames (which also use
  UTC via `datetime.now(timezone.utc)`).
- Never fabricate metrics. If a number is unavailable, write "not recorded".
- The report is a permanent record — write it for a reader who was not present
  in the session and has no conversation context.
