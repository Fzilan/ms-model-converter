---
name: convert-model
description: Convert a Hugging Face PyTorch model into MindOne with the standard migration workflow.
metadata:
  short-description: Convert HF model to MindOne
---

# Skill: convert-model

## Purpose
End-to-end conversion of a Hugging Face PyTorch model to `mindone.transformers`.

## Use when
The task is to migrate or convert a model from `transformers/` into MindOne.

## Inputs to collect
Use `.claude/snippets/00-intake.md` as the intake prompt.

## Steps
1. Use `.claude/snippets/01-auto-convert.md` for auto-convert commands.
2. Use `.claude/snippets/02-post-convert-checklist.md` for manual fixes.
3. Use `.claude/snippets/03-auto-mapping.md` for registrations and exports.
4. Use `.claude/snippets/04-test-template.md` for the test scaffold.

## Constraints
- Follow the guardrails in `CLAUDE.md`.

## Output
- Files changed and why.
- Tests run or suggested.
- Any remaining TODOs or risks.
