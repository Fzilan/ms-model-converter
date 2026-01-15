# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Authority and references
This is the authoritative instruction file. Supporting docs live under `.claude-option3/`:
- `.claude-option3/ENV.md` for setup, tooling, and test commands
- `.claude-option3/WORKFLOW.md` for the step-by-step SOP
- `.claude-option3/snippets/` for prompts and command templates

When you use `.claude-option3/` files, explicitly name which files you consulted.

## Repository overview
This monorepo contains the core libraries used in PyTorch-to-MindSpore conversion:

1. **mindone/** - MindSpore ONE: multimodal framework and MindSpore-backed transformers
2. **transformers/** - Hugging Face Transformers (PyTorch reference models)
3. **.claude-option3/** - Conversion workflow, snippets, and tools

## Conversion workflow
Use `.claude-option3/WORKFLOW.md` for the full SOP, plus `.claude-option3/snippets/` for command templates.

## Guardrails
- Do not migrate `configuration_*.py`, `tokenization_*.py`, or `*_fast.py`
- Only migrate processing files if they manipulate torch tensors; otherwise use HF implementations
- Avoid custom compatibility wrappers unless required
- Use diff-based insertion when updating auto maps
- Keep changes minimal and aligned with existing MindOne patterns

## Response expectations
- List `.claude-option3/` references consulted (file names)
- Summarize edits and note any risks or TODOs
- Suggest next tests when appropriate
