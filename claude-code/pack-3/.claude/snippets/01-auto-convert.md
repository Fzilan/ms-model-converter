# Auto-convert commands

Run this first, before manual edits.

Folder conversion:
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
