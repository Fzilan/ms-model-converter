# Model Conversion Rules

This file is the primary post-auto-convert checklist. It focuses on what the agent must still fix after running `auto_convert.py`.

## 1. Fundamental Class Structure

The core architectural shift involves moving from PyTorch's imperative style to MindSpore's cell-based structure.
Inheritance: `torch.nn.Module` to `mindspore.nn.Cell`.
Execution Method: `forward` to `construct`.

## 2. Parameter and Buffer Management
MindSpore currently uses the concept of Parameter for all persistent data that needs to be tracked or saved.
Model Parameters: Replace torch.nn.Parameter with mindspore.Parameter.
Buffers: MindSpore now supports `register_buffer` as Torch does.
Parameter Access: Replace named_parameters() with cells_and_names() or get_parameters(). Use trainable_params() to get parameters that require gradient updates.

## 3. Execution Control and Modes
Device Handling: Remove torch.device, .to(device), and .cuda() calls. 
Should check the device related code and remove it.
Examples like:

<BEFORE>

```python
    def _dynamic_frequency_update(self, position_ids, device):
    	seq_len = mint.max(position_ids) + 1
    	if seq_len > self.max_seq_len_cached:  # growth
        	inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
        	self.max_seq_len_cached = seq_len
 
    	if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
        	self.max_seq_len_cached = self.original_max_seq_len
        ...
    	device_type = x.device.type
    	device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"	
```

<AFTER>

```python
def _dynamic_frequency_update(self, position_ids):
    seq_len = mint.max(position_ids) + 1
    if seq_len > self.max_seq_len_cached:  # growth
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, seq_len=seq_len)
        self.inv_freq = inv_freq  # TODO joao: may break with compilation
        self.max_seq_len_cached = seq_len

    if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
        self.inv_freq = self.original_inv_freq
        self.max_seq_len_cached = self.original_max_seq_len
    ...
    # all device related code should be removed  
```

## 4. Imports and Decorators
Remove unused or PyTorch-only imports that are not migrated. This avoids false positives later.
Remove decorators, likes:
- `@torch.jit.script` - PyTorch-specific
- `@auto_docstring` - do not migrate and use
Remove decorators belows as we use mindspore local implementation:
- `@use_kernel_func_from_hub`
- `@use_kernelized_func`
- `@use_kernel_forward_from_hub`
If a decorator is not migrated, delete both its import and usage.

## 5. Docstrings and Comments
Remove or update docstrings/comments that still mention PyTorch-only concepts:
- `torch.Tensor`, `torch.device`, `cuda`, `mps`
Docstrings should refer to `mindspore.Tensor` or omit backend-specific text.

## 6. Tensor Methods and Functional Ops
After auto-convert, scan for remaining `torch.` or `F.` usage and handle by rule:
- Convert tensor methods that were not covered by the script.
- Replace functional ops with `mindspore.mint` or `mindspore.ops` equivalents.
- Drop unsupported inplace parameters (e.g., `inplace=True`) and adjust logic if needed.
- Enclose all shape arguments with parentheses to convert them to tuples. MindSpore requires tuple-form shape arguments; PyTorch-style positional dimension arguments are not supported.

## 7. Code Cleanup and Formatting
Types: Replace torch.FloatTensor, torch.LongTensor, etc., with generic mindspore.Tensor and specify the dtype if necessary.
Align naming, docstrings, and exports with existing MindSpore model folders.
