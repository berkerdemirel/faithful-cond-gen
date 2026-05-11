# CLAUDE.md (how to work in this repo)

## Non-negotiables
- Be clean and concise. No overengineering, no noisy try/except, no unnecessary asserts.
- Minimal diffs. No refactors unless explicitly requested.
- Prefer existing project conventions (scripts vs src, configs in configs/).
- Use `PYTHONPATH=src uv run python ...` for commands (package not pip-installed).

## Source of truth
- PROJECT_SNAPSHOT.md is the only factual source of truth for goals and evaluation.
- Do not “assume” commands, config names, or architecture details. Verify by inspecting the repo.

## Reset / discovery pass (required before edits)
Before changing any code:
1) Inspect repo structure and identify the current phase (what runs end-to-end).
2) Fill the “Repo discovery” section of PROJECT_SNAPSHOT.md with:
   - canonical commands (train/sample/eval)
   - key paths (splits, metrics, scoring)
   - outputs
   - tests/sanity checks
3) Summarize understanding in 8–12 bullets, and list uncertainties explicitly.

Only after the user confirms/corrects the snapshot: proceed to planning/implementation.

## Planning format (modular and editable)
For any task >3 steps:
- Provide a plan with <=10 steps.
- For each step include:
  - goal
  - files to touch
  - exact commands to run
  - expected outputs
  - done-when check
  - risks/fallback
If larger, propose a v1 slice first.

## Execution rules (graspable)
- Implement ONE plan step at a time and stop.
- After each step: run verification commands and paste key output lines.
- Update notes/<task>.md with what changed + how verified + new assumptions.

## Subagents (use liberally, with boundaries)
Use subagents for:
- Planner: propose plan + risks
- Implementer: produce minimal diff for step k
- Verifier: try to break it and run checks
Main agent consolidates into one coherent diff + verification story.

## Stuck protocol (avoid wasting prompts)
If the same error happens twice:
- Stop.
- Produce a debug bundle:
  - command(s) run
  - full error
  - relevant snippets/paths
  - environment info (python/uv, CUDA if relevant)
  - what was tried
Wait for user input before further attempts.

## Training objective changes (guardrail)
When adding a new loss/regularizer:
- Put it behind a config flag with **default OFF**
- Log activation status and basic stats (loss value, gradient norm) to wandb
- Do NOT change existing defaults or baseline behavior
- Add a minimal ablation plan entry in `notes/ablations.md`

## Lessons
Maintain lessons.md only for recurring mistakes (repeat twice).
Each entry: trigger → rule → 2–4 bullet checklist.
