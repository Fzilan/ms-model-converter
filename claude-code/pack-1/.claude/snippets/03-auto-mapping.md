# Auto-mapping updates

Update these files to register the model:
- `mindone/mindone/transformers/models/auto/configuration_auto.py`
- `mindone/mindone/transformers/models/auto/modeling_auto.py`
- `mindone/mindone/transformers/models/auto/processing_auto.py` (if needed)
- `mindone/mindone/transformers/models/auto/image_processing_auto.py` (if needed)
- `mindone/mindone/transformers/models/auto/video_processing_auto.py` (if needed)

Also update exports:
- `mindone/mindone/transformers/models/{model_name}/__init__.py`
- `mindone/mindone/transformers/models/__init__.py`
- `mindone/mindone/transformers/__init__.py`

Tip: use HF auto files as a reference to insert in the correct order.
