# MindSpore Model Converter — mini-swe agent

This folder contains configs, knowledgebase docs and guidance for converting PyTorch models to MindSpore using the mini-swe agent.

## Structure

- `mini-swe/knowledgebase/` — the core knowledgebase containing documentation and examples for both automated and manual conversion.
    - `API_MAPPING.md` — a comprehensive PyTorch → MindSpore API mapping table. Includes recommended replacements, notes about behavioral differences, and short code snippets to illustrate changes.
    - `AUTO_CONVERT.md` — instructions for the automatic conversion workflow. Includes command-line examples, expected inputs/outputs, log locations, and common error diagnostics.
    - `CONVERSION_RULES.md` — rules and best practices for manual fixes and complex cases (for example: custom layers, optimizer/gradient differences, tensor layout changes, and numerical stability issues).
    - `STRUCTURAL_GUIDE.md` — recommended repository structure and organization for converted models, expected config file locations, and naming conventions.
    - `TESTING_STANDARD.md` — testing and validation checklist to confirm functional parity after conversion, including unit test examples and numeric tolerance recommendations.
    - `auto_convert.py` — the conversion utility script (if present). See `AUTO_CONVERT.md` for usage examples and invocation details.
