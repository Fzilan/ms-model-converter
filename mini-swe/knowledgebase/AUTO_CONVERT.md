# Auto-Convert First

Run the fixed mapping conversion script before manual edits. It handles the bulk of API replacements using the mapping table.

## Usage

Folder conversion:
```
python mindspore-coder/src/mindsporecoder/knowledgebase/transformers/auto_convert.py --src_root /path/to/src --dst_root /path/to/dst
```

Single-file in-place conversion:
```
python mindspore-coder/src/mindsporecoder/knowledgebase/transformers/auto_convert.py --src_file /path/to/file.py --inplace
```

## What to do after auto-convert
- Scan for remaining `torch.` or `F.` usage and fix them using `CONVERSION_RULES.md`.
- Remove PyTorch-only decorators and their imports.
- Clean docstrings/comments that still mention `torch.Tensor`, `torch.device`, `cuda`, or `mps`.
