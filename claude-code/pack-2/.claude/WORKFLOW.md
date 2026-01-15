# PyTorch to MindSpore Migration Workflow

This SOP provides a single execution path for Claude Code.

## 1. Intake checklist
- Model name and task type.
- Source path: `transformers/src/transformers/models/{model_name}/`.
- Target path: `mindone/mindone/transformers/models/{model_name}/`.
- Any processing files to migrate: `processing_*.py`, `image_processing_*.py`, `video_processing_*.py`.
- Reference HF tests for minimal test shapes.

## 2. Auto-convert (required first)
Run the fixed mapping script before manual edits.
```
python .claude/tools/auto_convert.py \
  --src_root transformers/src/transformers/models/{model_name} \
  --dst_root mindone/mindone/transformers/models/{model_name}
```
Single file in-place:
```
python .claude/tools/auto_convert.py \
  --src_file path/to/file.py --inplace
```

## 3. Manual fix checklist
### Structural and API
- `torch.nn.Module` -> `mindspore.nn.Cell`.
- `forward` -> `construct`.
- `torch.nn.Parameter` -> `mindspore.Parameter`.
- Remove device code: `.to(device)`, `.cuda()`, `torch.device`, `mps` branches.
- Replace `torch` and `torch.nn.functional` usages with `mindspore.mint` or `mindspore.ops`.
- Prefer `mindspore.mint`, then `mindspore.ops`, then `mindspore.nn`.
- Drop unsupported `inplace=True` args.

### Imports and decorators
- Keep config/tokenizer imports from HF `transformers`.
- Use `mindone.transformers.modeling_utils` for modeling utilities.
- Remove decorators: `@torch.jit.script`, `@auto_docstring`, kernel hub decorators.

### Tensors and shapes
- Use `mindspore.Tensor` in docstrings.
- Wrap shape arguments in tuples, e.g. `.view((b, s, h))`.

## 4. Registration and exports
- Add config to `mindone/mindone/transformers/models/auto/configuration_auto.py`.
- Add model class to `mindone/mindone/transformers/models/auto/modeling_auto.py`.
- Update processor maps if processor files are migrated.
- Export chain updates:
  - `mindone/mindone/transformers/models/{model_name}/__init__.py`
  - `mindone/mindone/transformers/models/__init__.py`
  - `mindone/mindone/transformers/__init__.py`

## 5. Tests and minimal validation
- Create tests under `mindone/tests/transformers_tests/models/{model_name}/`.
- Use `ms.set_context(mode=ms.PYNATIVE_MODE)`.
- Keep configs tiny to minimize runtime.

## 6. Done criteria
- Model imports cleanly in MindOne.
- Auto mappings and exports are updated.
- At least one smoke test passes with dummy inputs.

## Reference map
- Auto-convert usage: `.claude/snippets/01-auto-convert.md`
- Post-convert checklist: `.claude/snippets/02-post-convert-checklist.md`
- Auto-mapping updates: `.claude/snippets/03-auto-mapping.md`
- Test template: `.claude/snippets/04-test-template.md`
