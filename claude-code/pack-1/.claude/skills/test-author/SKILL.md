---
name: test-author
description: Create minimal MindSpore tests for migrated models to validate basic forward passes.
metadata:
  short-description: Author minimal model tests
---

# Skill: test-author

## Purpose
Create minimal MindSpore tests for a migrated model.

## Use when
A model has been migrated and needs a smoke or parity test scaffold.

## Inputs to collect
Use `.claude/snippets/04-test-template.md` as the scaffold.

## Steps
1. Follow `.claude/snippets/04-test-template.md`.
2. Align inputs with the model's forward signature.

## Constraints
- Keep tests minimal and fast.
- If inputs are unclear, add a TODO with required fields.

## Output
- Test file path.
- Required inputs and any assumptions.
