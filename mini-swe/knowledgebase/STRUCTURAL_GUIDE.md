# Structural & Auto-Mapping Guide

## 1. Migration Workflow Overview
The following table illustrates the relationship between the migration steps, the target files, and the reference documents in your knowledge base.

| Stage | Target Files | Reference |
| :--- | :--- | :--- |
| **Logic** | `modeling_{model_name}.py`, `processing_*.py`, `image_processing_*.py`, `video_processing_*.py` | `AUTO_CONVERT.md`, `API_MAPPING.md`, `CONVERSION_RULES.md` |
| **Export** | `models/{model_name}/__init__.py` | `STRUCTURAL_GUIDE.md` |
| **Auto** | `models/auto/modeling_auto.py`, `configuration_auto.py` | `STRUCTURAL_GUIDE.md` |
| **Test** | `tests/transformers_tests/models/{model_name}/test_modeling_{model_name}.py` | `TESTING_STANDARD.md` |


## 2. Directory Organization
All migrated models must be placed within the `mindone` library structure to ensure compatibility with the internal registry and automated testing tools.

Source Directory (HF transformers)
* **Model Directory**: `transformers/src/transformers/models/{model_name}/`
* **Modeling Script**: `transformers/src/transformers/models/{model_name}/modeling_{model_name}.py`
* **Init Script**: `transformers/src/transformers/models/{model_name}/__init__.py`
* **Processor**: any `processing_*.py`, `image_processing_*.py`, or `video_processing_*.py` under `{model_name}`folder
* **Pre-check**: Verify `transformers/src/transformers/models/{model_name}/` exists; ask for a correct model name if missing.

Target Directory (mindone transformers)
* **Model Directory**: `mindone/mindone/transformers/models/{model_name}/`
* **Modeling Script**: `mindone/mindone/transformers/models/{model_name}/modeling_{model_name}.py`
* **Init Script**: `mindone/mindone/transformers/models/{model_name}/__init__.py`
* **Processor**: any `processing_*.py`, `image_processing_*.py`, or `video_processing_*.py` under `{model_name}`folder
* **Pre-check**: Locate the `mindone/mindone/transformers/models/` first, it MUST already exists the `mindone` project root. and then verify `mindone/mindone/transformers/models/{model_name}/` exists, create it if missing.

## 3. Import Strategy (HF vs. MindOne)
To ensure the migrated model reuses official configuration and tokenization logic while replacing the execution backend, follow these strict import rules:

* **From Hugging Face (`transformers`)**:
    * **DO** import `Configuration` classes, `Tokenizer` classes, and `FeatureExtractors` directly from the original `transformers` library.
    * **DO NOT** migrate or copy `configuration_*.py`, `tokenization_*.py`.
    * **DO NOT** migrate or copy `processing_*.py` unless they require MindSpore-specific tensor operations.
    * **DO NOT** migrate `tokenization_*_fast.py` or `modular_*.py`.
* **From MindOne (`mindone.transformers`)**:
    * **Internal Imports**: Any import pointing to a `modeling_*.py` file must be redirected to `mindone.transformers.models`.
    * **Utilities**: Replace `transformers.modeling_utils` with `mindone.transformers.modeling_utils`.

## 3.1 Reuse vs. Migrate (Practical Rules)
Use Hugging Face originals for configuration and tokenization, unless they require MindSpore tensors:
* **Reuse (do not copy)**: `configuration_*.py`, `tokenization_*.py`, `tokenization_*_fast.py`, `modular_*.py`.
* **Migrate (copy + refactor)**: `modeling_*.py` and any processor/image/video processing file that uses Torch tensors/ops.
* **Decorators**: If a decorator is not migrated, remove its import and usage to avoid dead references.

## 4. Auto-Mapping Registration
To enable `AutoModel.from_pretrained()` support, you must register the model in the auto-mapping files located in `mindone/transformers/models/auto/`:

* **`configuration_auto.py`**:
    * Add the model type mapping to `CONFIG_MAPPING_NAMES`.
    * *Example*: `"my_model": "MyModelConfig"`
* **`modeling_auto.py`**:
    * Add the model class mapping to `MODEL_MAPPING_NAMES` (or the specific task mapping like `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES`).
    * *Example*: `"my_model": "MyModel"`
* **Processor auto-maps (when applicable)**:
    * If `processing_*.py` is migrated, update `PROCESSOR_MAPPING_NAMES` in `processing_auto.py`.
    * If `image_processing_*.py` is migrated, update `IMAGE_PROCESSOR_MAPPING_NAMES` in `image_processing_auto.py`.
    * If `video_processing_*.py` is migrated, update `VIDEO_PROCESSOR_MAPPING_NAMES` in `video_processing_auto.py`.

## 4.1 Diff-Based Insertion (Recommended)
For auto-mapping and export updates, use the HF `transformers` auto files as a reference:
* Open the corresponding HF auto file and locate the model name entry.
* Insert into the MindOne auto file at the matching block, preserving ordering and structure.
This avoids incorrect placement and reduces manual search steps.

A reference mapping file/fodler from hf and mindone:
- `transformers/src/transformers/models/__init__.py` -> `mindone/mindone/transformers/models/__init__.py`
- the auto folders, auto-maps files should be correspond one by one: `transformers/src/transformers/models/auto/` -> `mindone/mindone/transformers/models/auto/`

## 5. Export Chain (__init__.py)
The model must be exposed through the library's namespace. Update the following files:

1.  **Local Level**: `mindone/transformers/models/{model_name}/__init__.py`
    ```python
    from .modeling_{model_name} import *
    ```
2.  **Models Group**: `mindone/transformers/models/__init__.py`
    * Add `from . import {model_name}` to the imports.
    * Append `{model_name}` to the `__all__` list.
3.  **Library Root**: `mindone/transformers/__init__.py`
    * Ensure the models are correctly exported to the top-level namespace.

## 6. Testing Requirements
* **Location**: Tests must be placed in `tests/transformers_tests/models/{model_name}/test_modeling_{model_name}.py`.
* **Standard**: Reference the original Hugging Face test file for test cases (e.g., hidden state shapes, attention masks).
* **Parity**: Use `ms.set_context(mode=ms.PYNATIVE_MODE)` during testing to ensure compatibility with imperative PyTorch logic.

## 7. Processing and Feature Migration
If `processing_*.py`, `image_processing_*.py`, or `video_processing_*.py` exist and require Torch tensors/ops, migrate them into `mindone/transformers/models/{model_name}/` and register in the corresponding auto-mapping files.

## 8. Minimal Validation
After migration, do a minimal import/load check to ensure the module can be imported without errors.
