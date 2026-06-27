# Skill: github-project-setup

## Purpose

Given a research sub-project with a `docs/dev_plan.md`, create a GitHub milestone and one issue per Phase, so the project is immediately trackable on GitHub.

## When to invoke

Use this skill when the user says any of:
- "set up GitHub issues for [project]"
- "create milestone and issues from dev_plan"
- "push the dev plan to GitHub"

## Steps

### 1. Read the dev plan

```
Read {project}/docs/dev_plan.md
```

Identify:
- **Project name** — the subfolder name (e.g. `ObjectDetection`, `OCR`, `SceneDescription`)
- **Phases** — every `### Phase N — <Title>` heading and the block of content beneath it (up to the next `---` or heading)

### 2. Create the GitHub milestone

```bash
gh api repos/{owner}/{repo}/milestones \
  --method POST \
  --field title="{ProjectName}" \
  --field description="{one-line description from dev_plan Goal section}" \
  --jq '.number'
```

Note the returned milestone number.

### 3. Create one issue per Phase

For each phase:

**Title format**: `[{ProjectName} Phase {N}] {Phase title}`
- Example: `[ObjectDetection Phase 1] Camera Capture (src/capture.py)`

**Body**: the full markdown content of that phase section (tasks, code blocks, acceptance criteria).

```bash
gh issue create \
  --repo {owner}/{repo} \
  --title "[{ProjectName} Phase {N}] {Phase title}" \
  --milestone "{ProjectName}" \
  --body "$(cat <<'EOF'
{phase content}
EOF
)"
```

Create issues sequentially (not in parallel) to preserve phase ordering in the issue list.

### 4. Report

List each created issue URL grouped by milestone.

## Conventions

- Milestone title = project subfolder name exactly (case-sensitive)
- Issue numbers are auto-assigned by GitHub; do not hard-code them
- Phase body should be copied verbatim from the dev plan — no summarising
- If a milestone with the same title already exists, reuse it (fetch with `gh api repos/{owner}/{repo}/milestones --jq '.[] | select(.title=="{ProjectName}") | .number'`)
