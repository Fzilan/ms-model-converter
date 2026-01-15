---
name: register-auto-maps
description: Register migrated models in MindOne auto-mapping and export files for AutoModel support.
metadata:
  short-description: Update MindOne auto maps
---

# Skill: register-auto-maps

## Purpose
Register a migrated model in MindOne auto-mapping and export files.

## Use when
The model code exists but is not discoverable via `AutoModel` or imports.

## Inputs to collect
Use `.claude/snippets/03-auto-mapping.md` as the checklist.

## Steps
1. Follow `.claude/snippets/03-auto-mapping.md`.
2. Use HF auto files as the ordering reference.

## Constraints
- Preserve ordering to match HF auto maps.

## Output
- Mapping entries added.
- Files updated.
- Any follow-up tasks.
