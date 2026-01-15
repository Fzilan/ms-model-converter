# Testing Standard (Minimal Template)

Use `mindone/mindone/tests/transformers_tests/models/cohere` as the layout example and learn the template of the model test.
For small configs and input shapes, refer to the original HF test file at
`transformers/tests/transformers_tests/models/{model_name}/test_modeling_{model_name}.py`.
If no fast test exists there, create a minimal config by shrinking model dimensions
(e.g., hidden size, heads, layers) and use tiny dummy inputs that satisfy required shapes.

## Minimal checks
1. Import the migrated model and config class from HF transformers.
2. Instantiate the model with a small config.
3. Run a forward pass with dummy inputs.
4. Assert output shapes and key fields.

If no testing guidance is available, create the file in the correct location and leave a TODO with required inputs.
