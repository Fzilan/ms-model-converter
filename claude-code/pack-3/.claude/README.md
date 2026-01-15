# Claude Code Conversion Pack

This folder is a self-contained Claude Code pack for PyTorch to MindSpore model migration.

## Placement (recommended)
Copy this folder into your project root as `.claude/`. Commands in this pack assume you run them from the project root and reference tools under `.claude/`.

## Contents
- `CLAUDE.md` - Claude Code instructions tailored to the conversion workflow.
- `WORKFLOW.md` - Step-by-step SOP with checklists and file touchpoints.
- `snippets/` - Copy/paste prompt snippets and command templates.
- `tools/` - Local scripts used by this pack.
- `ENV.md` - Setup, tooling, and test commands.

## Source of truth
- Files in this folder

## Quick start
1. Open `CLAUDE.md` at the repo root as your Claude Code instructions.
2. Use `.claude/snippets/00-intake.md` to gather model inputs.
3. Run the auto-convert step, then follow `.claude/WORKFLOW.md`.
4. See `.claude/ENV.md` for setup, tooling, and test commands.
