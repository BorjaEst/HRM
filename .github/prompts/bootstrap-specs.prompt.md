---
description: "Bootstrap canonical specs when spec/ is missing or empty; discover current repo structure, propose improvements, and ask for user inputs before writing."
agent: "agent"
tools:
  - search
  - search/codebase
  - edit/editFiles
argument-hint: "Use when spec/ is missing, incomplete, or placeholder-only"
---

# Bootstrap Canonical Specs (when spec/ is empty)

## Mission

Bootstrap the canonical spec set for this repository when `spec/` is missing, incomplete, or
contains placeholder/empty files. This is an interactive workflow: you MUST assess the current
project state, propose a spec structure, ask for user decisions, then generate the canonical spec
files.

## Scope & Preconditions

### Detect whether bootstrapping is needed

1. Check whether `spec/spec-manifest.toml` exists.
2. If it exists, read it and check whether every path in `required.files` exists.
3. Determine whether the spec set is “empty”:
   - `spec/` missing, OR
   - manifest missing, OR
   - any required file missing, OR
   - required files exist but are clearly placeholder-only (e.g., mostly template text, or missing
     core sections like Purpose/Scope/Requirements/Acceptance Criteria).

### Behavior by case

- If bootstrapping is **NOT** needed (manifest + required specs exist and are non-placeholder):
  - STOP and tell the user to use the existing specification update workflow (e.g., the
    “Update Specification” prompt) instead of overwriting canonical specs.

- If bootstrapping **IS** needed:
  - Continue with discovery + clarification + generation below.

## Inputs

- `${input:STRICT_MODE}`: `true|false` (default: `true`)
- `${input:PLANS_ROOT}`: Path (default: `.copilot-tracking/plans`)
- `${input:PLAN_DETAILS_ROOT}`: Path (default: `.copilot-tracking/details`)
- `${input:PLAN_CHANGES_ROOT}`: Path (default: `.copilot-tracking/changes`)
- `${input:DOCS_ROOT}`: Path (default: `docs`)
- `${input:REPO_SUMMARY}`: Short plain-English summary (optional; if missing, infer from repo)

## Workflow

### 1) Repository Discovery (evidence-first)

Scan the repository to collect evidence (do not invent anything):

- Identify entry points (scripts/notebooks/CLIs).
- Identify configuration patterns and where configuration lives.
- Identify major modules/folders and their responsibilities as implemented today.
- Identify how datasets are built and how training/evaluation consumes them (based on code).
- Identify dependencies (requirements files, lockfiles, env markers).
- Identify existing documentation and where it currently lives (README, docs/, etc.).

Produce a short “Observed Structure” summary that names concrete folders/files.

### 2) Improvement Opportunities (non-destructive)

Based on the observed structure, list improvement opportunities (no refactoring in this prompt):

- Inconsistencies in naming / structure
- Missing boundaries / unclear responsibilities
- Where specs would reduce ambiguity for future work
- Any risky assumptions you would otherwise have to make

Keep this as a candidate backlog; do NOT change code.

### 3) Ask the User for Decisions (mandatory)

Ask for confirmation/inputs before writing files. At minimum ask:

1. Project intent: “What is the repository’s primary purpose and target users?”
2. Strict mode: confirm `${input:STRICT_MODE}` should be enforced for artifact generation.
3. Canonical locations: confirm plans/docs roots (defaults provided).
4. Architecture taxonomy: confirm the top-level components you inferred are acceptable.
5. Any forbidden taxonomies/frameworks to avoid (e.g., generic web-app layering).
6. Any special external integrations that must be named as dependencies.

If the user’s answers conflict with observed code, surface the conflict and ask which to treat as
authoritative.

### 4) Propose Spec Outline (await approval)

Propose the outline/content you will write for the canonical spec set:

- `spec/spec-manifest.toml`
- `spec/spec-architecture.md`
- `spec/spec-requirements.md`
- `spec/spec-standards.md`

Include the exact required.files list and the precedence order you will encode.

Await user approval or edits.

### 5) Generate Canonical Specs

After approval, create/update the following files:

1. `spec/spec-manifest.toml`
   - `strict = ${input:STRICT_MODE}`
   - `required.files` MUST include:
     - `spec/spec-manifest.toml`
     - `spec/spec-architecture.md`
     - `spec/spec-requirements.md`
     - `spec/spec-standards.md`
   - Add canonical paths section:
     - `plans_root = ${input:PLANS_ROOT}`
     - `plan_details_root = ${input:PLAN_DETAILS_ROOT}`
     - `plan_changes_root = ${input:PLAN_CHANGES_ROOT}`
     - `docs_root = ${input:DOCS_ROOT}`

2. `spec/spec-architecture.md`
   - Define repo-native components based on discovery evidence.
   - Define allowed vocabulary and forbidden frameworks if required.
   - Include exemplar categorization guidance tied to the taxonomy.

3. `spec/spec-requirements.md`
   - Define strict gating requirements for agents generating artifacts.
   - Define environment and data-handling constraints grounded in repo reality.
   - Include “do not invent schemas/contracts” constraint unless specified.

4. `spec/spec-standards.md`
   - Define coding/testing/doc standards relevant to the repo.
   - Define artifact expectations (plans/specs/docs) and scannability norms.
   - Define exemplar criteria for what “good” looks like in this codebase.

All specs must be internally consistent and reference each other consistently.

### 6) Validation (mandatory)

Before finishing, verify:

- All required files exist and are non-placeholder.
- Manifest strict mode matches the chosen setting.
- Paths in the manifest match the chosen canonical locations.
- No sections contain unresolved placeholder text.

### 7) Hand-off

Summarize:

- What you inferred about the repo structure (with file/folder references).
- What you wrote (list files).
- What the next recommended prompts/workflows are (e.g., update prompts to enforce spec gate,
  create implementation plan, documentation improvements).

## Output Expectations

- The output is the creation/update of the canonical spec set in `spec/`.
- The prompt must not refactor application code.
- No external web browsing unless the user provides a link and requests it.
