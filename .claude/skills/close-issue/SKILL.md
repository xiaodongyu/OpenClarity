# Skill: close-issue

## Purpose

After completing work for a GitHub issue and verifying all acceptance criteria are met, post a closing comment summarising what was done and close the issue.

## When to invoke

Use this skill when the user says any of:
- "close issue N"
- "mark issue N as done"
- "add commit to issue N and close it"
- Automatically after finishing work on an issue if the user said "work on issue N" and all acceptance criteria pass

## Steps

### 1. Gather context

Collect the following before composing the comment:
- **Issue number** — from the user's instruction or the current task
- **Repo** — from `git remote get-url origin`, parse `owner/repo`
- **Commit hash + message** — `git log --oneline -1`
- **Test results** — the output of the test run (copy the summary line, e.g. `4 passed in 0.21s`)
- **Acceptance criteria** — from the issue body (`gh issue view N --repo owner/repo`)

### 2. Post a closing comment

```bash
gh issue comment {N} --repo {owner}/{repo} --body "$(cat <<'EOF'
## Implementation Complete

**Commit**: {short_hash} — _{commit message}_

### What was done
{bullet list of files created/modified and what each does}

### Test results
\`\`\`
{test runner output summary}
\`\`\`
All acceptance criteria met — closing.
EOF
)"
```

Rules for the comment body:
- List every file created or modified with a one-line description
- Copy the exact test summary line from the test run output
- Map each acceptance criterion from the issue to a ✅ result
- Keep it factual — no filler phrases

### 3. Close the issue

```bash
gh issue close {N} --repo {owner}/{repo}
```

### 4. Report

Confirm to the user: issue number, comment URL, and closed status.

## Conventions

- Always post the comment **before** closing (GitHub shows comments in creation order)
- If tests did not pass, do **not** close the issue — report the failure instead
- If the issue has no explicit acceptance criteria, document the observable behaviour that confirms correctness
- `owner/repo` must be derived from the actual remote, not hard-coded
