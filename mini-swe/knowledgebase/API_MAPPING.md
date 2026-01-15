# PyTorch to MindSpore API Mapping Reference

This document is a reference for the fixed mapping table used by `auto_convert.py`.
Manual conversion should focus on remaining issues in `CONVERSION_RULES.md` after the script runs.

## 1. Conversion Priority Rules
When migrating PyTorch scripts, the agent must strictly follow this order of preference for selecting MindSpore APIs:

Level 1: mindspore.mint – Primary choice. These APIs are designed for 1:1 functional parity with PyTorch and are optimized for high-performance execution on Ascend hardware.

Level 2: mindspore.ops – Secondary choice. Use only if a mint equivalent does not exist.

Level 3: mindspore.nn – Use specifically for structural components like layers and cells, only if a mindspore.mint.nn equivalent does not exist.

## 2. Global Structural Logic
Before mapping individual operators, apply these fundamental framework changes:

Base Class: Replace torch.nn.Module with mindspore.nn.Cell.

Execution Method: Rename forward method to construct.

Tensors: Map torch.Tensor to mindspore.Tensor.

Context/Devices: Remove torch.device and .to(device) calls; MindSpore manages hardware targets via mindspore.set_context. Remove all device related code like cpu, cuda, mps compatibility.

## 3. Important Constraints
No API Creation: The agent must not attempt to create custom wrapper functions to mimic PyTorch behavior unless explicitly defined in CONVERSION_RULES.md.

No inplace Operations: MindSpore does not support the inplace=True parameter for many operators; these must be removed during conversion.

Signature Consistency: Ensure that parameter names in construct match the original PyTorch forward signatures to maintain compatibility with existing Hugging Face configuration classes.

## 4. Strict API Mapping Table
Constraint: Do not invent or hallucinate MindSpore APIs. If an operator is not in the following tables or cannot be found in mint or ops, flag it for manual review.

Refer to the tables belows for mapping reference.

### mint_map

| Torch API | MindSpore API |
| --- | --- |
| torch.arange | mindspore.mint.arange |
| torch.ge | mindspore.mint.ge |
| torch.bernoulli | mindspore.mint.bernoulli |
| torch.isnan | mindspore.mint.isnan |
| torch.bincount | mindspore.mint.bincount |
| torch.clone | mindspore.mint.clone |
| torch.eye | mindspore.mint.eye |
| torch.einsum | mindspore.mint.einsum |
| torch.empty | mindspore.mint.empty |
| torch.empty_like | mindspore.mint.empty_like |
| torch.full_like | mindspore.mint.full_like |
| torch.linspace | mindspore.mint.linspace |
| torch.ones | mindspore.mint.ones |
| torch.ones_like | mindspore.mint.ones_like |
| torch.randint | mindspore.mint.randint |
| torch.randint_like | mindspore.mint.randint_like |
| torch.randn | mindspore.mint.randn |
| torch.randn_like | mindspore.mint.randn_like |
| torch.randperm | mindspore.mint.randperm |
| torch.zeros | mindspore.mint.zeros |
| torch.zeros_like | mindspore.mint.zeros_like |
| torch.cat | mindspore.mint.cat |
| torch.chunk | mindspore.mint.chunk |
| torch.concat | mindspore.mint.concat |
| torch.count_nonzero | mindspore.mint.count_nonzero |
| torch.gather | mindspore.mint.gather |
| torch.index_add | mindspore.mint.index_add |
| torch.index_select | mindspore.mint.index_select |
| torch.masked_select | mindspore.mint.masked_select |
| torch.permute | mindspore.mint.permute |
| torch.reshape | mindspore.mint.reshape |
| torch.scatter | mindspore.mint.scatter |
| torch.scatter_add | mindspore.mint.scatter_add |
| torch.split | mindspore.mint.split |
| torch.narrow | mindspore.mint.narrow |
| torch.nonzero | mindspore.mint.nonzero |
| torch.tile | mindspore.mint.tile |
| torch.tril | mindspore.mint.tril |
| torch.select | mindspore.mint.select |
| torch.squeeze | mindspore.mint.squeeze |
| torch.stack | mindspore.mint.stack |
| torch.swapaxes | mindspore.mint.swapaxes |
| torch.transpose | mindspore.mint.transpose |
| torch.triu | mindspore.mint.triu |
| torch.unbind | mindspore.mint.unbind |
| torch.unique_consecutive | mindspore.mint.unique_consecutive |
| torch.unsqueeze | mindspore.mint.unsqueeze |
| torch.where | mindspore.mint.where |
| torch.multinomial | mindspore.mint.multinomial |
| torch.normal | mindspore.mint.normal |
| torch.rand_like | mindspore.mint.rand_like |
| torch.rand | mindspore.mint.rand |
| torch.abs | mindspore.mint.abs |
| torch.add | mindspore.mint.add |
| torch.addmv | mindspore.mint.addmv |
| torch.acos | mindspore.mint.acos |
| torch.acosh | mindspore.mint.acosh |
| torch.arccos | mindspore.mint.arccos |
| torch.arccosh | mindspore.mint.arccosh |
| torch.arcsin | mindspore.mint.arcsin |
| torch.arcsinh | mindspore.mint.arcsinh |
| torch.arctan | mindspore.mint.arctan |
| torch.arctan2 | mindspore.mint.arctan2 |
| torch.arctanh | mindspore.mint.arctanh |
| torch.asin | mindspore.mint.asin |
| torch.asinh | mindspore.mint.asinh |
| torch.atan | mindspore.mint.atan |
| torch.atan2 | mindspore.mint.atan2 |
| torch.atanh | mindspore.mint.atanh |
| torch.bitwise_and | mindspore.mint.bitwise_and |
| torch.bitwise_or | mindspore.mint.bitwise_or |
| torch.bitwise_xor | mindspore.mint.bitwise_xor |
| torch.ceil | mindspore.mint.ceil |
| torch.clamp | mindspore.mint.clamp |
| torch.cos | mindspore.mint.cos |
| torch.cosh | mindspore.mint.cosh |
| torch.cross | mindspore.mint.cross |
| torch.diff | mindspore.mint.diff |
| torch.div | mindspore.mint.div |
| torch.divide | mindspore.mint.divide |
| torch.erf | mindspore.mint.erf |
| torch.erfc | mindspore.mint.erfc |
| torch.erfinv | mindspore.mint.erfinv |
| torch.exp | mindspore.mint.exp |
| torch.exp2 | mindspore.mint.exp2 |
| torch.expm1 | mindspore.mint.expm1 |
| torch.fix | mindspore.mint.fix |
| torch.float_power | mindspore.mint.float_power |
| torch.floor | mindspore.mint.floor |
| torch.fmod | mindspore.mint.fmod |
| torch.frac | mindspore.mint.frac |
| torch.lerp | mindspore.mint.lerp |
| torch.log | mindspore.mint.log |
| torch.log1p | mindspore.mint.log1p |
| torch.log2 | mindspore.mint.log2 |
| torch.log10 | mindspore.mint.log10 |
| torch.logaddexp | mindspore.mint.logaddexp |
| torch.logaddexp2 | mindspore.mint.logaddexp2 |
| torch.logical_and | mindspore.mint.logical_and |
| torch.logical_not | mindspore.mint.logical_not |
| torch.logical_or | mindspore.mint.logical_or |
| torch.logical_xor | mindspore.mint.logical_xor |
| torch.mul | mindspore.mint.mul |
| torch.mv | mindspore.mint.mv |
| torch.nansum | mindspore.mint.nansum |
| torch.nan_to_num | mindspore.mint.nan_to_num |
| torch.neg | mindspore.mint.neg |
| torch.negative | mindspore.mint.negative |
| torch.pow | mindspore.mint.pow |
| torch.polar | mindspore.mint.polar |
| torch.ravel | mindspore.mint.ravel |
| torch.reciprocal | mindspore.mint.reciprocal |
| torch.remainder | mindspore.mint.remainder |
| torch.roll | mindspore.mint.roll |
| torch.round | mindspore.mint.round |
| torch.rsqrt | mindspore.mint.rsqrt |
| torch.sigmoid | mindspore.mint.sigmoid |
| torch.sign | mindspore.mint.sign |
| torch.sin | mindspore.mint.sin |
| torch.sinc | mindspore.mint.sinc |
| torch.sinh | mindspore.mint.sinh |
| torch.softmax | mindspore.mint.softmax |
| torch.sqrt | mindspore.mint.sqrt |
| torch.square | mindspore.mint.square |
| torch.sub | mindspore.mint.sub |
| torch.t | mindspore.mint.t |
| torch.tan | mindspore.mint.tan |
| torch.tanh | mindspore.mint.tanh |
| torch.trunc | mindspore.mint.trunc |
| torch.xlogy | mindspore.mint.xlogy |
| torch.amax | mindspore.mint.amax |
| torch.amin | mindspore.mint.amin |
| torch.argmax | mindspore.mint.argmax |
| torch.argmin | mindspore.mint.argmin |
| torch.argsort | mindspore.mint.argsort |
| torch.all | mindspore.mint.all |
| torch.any | mindspore.mint.any |
| torch.cumprod | mindspore.mint.cumprod |
| torch.histc | mindspore.mint.histc |
| torch.logsumexp | mindspore.mint.logsumexp |
| torch.max | mindspore.mint.max |
| torch.mean | mindspore.mint.mean |
| torch.median | mindspore.mint.median |
| torch.min | mindspore.mint.min |
| torch.norm | mindspore.mint.norm |
| torch.prod | mindspore.mint.prod |
| torch.sum | mindspore.mint.sum |
| torch.std | mindspore.mint.std |
| torch.std_mean | mindspore.mint.std_mean |
| torch.unique | mindspore.mint.unique |
| torch.var | mindspore.mint.var |
| torch.var_mean | mindspore.mint.var_mean |
| torch.allclose | mindspore.mint.allclose |
| torch.eq | mindspore.mint.eq |
| torch.equal | mindspore.mint.equal |
| torch.greater | mindspore.mint.greater |
| torch.greater_equal | mindspore.mint.greater_equal |
| torch.gt | mindspore.mint.gt |
| torch.isclose | mindspore.mint.isclose |
| torch.isfinite | mindspore.mint.isfinite |
| torch.isinf | mindspore.mint.isinf |
| torch.isneginf | mindspore.mint.isneginf |
| torch.le | mindspore.mint.le |
| torch.less | mindspore.mint.less |
| torch.less_equal | mindspore.mint.less_equal |
| torch.lt | mindspore.mint.lt |
| torch.maximum | mindspore.mint.maximum |
| torch.minimum | mindspore.mint.minimum |
| torch.ne | mindspore.mint.ne |
| torch.not_equal | mindspore.mint.not_equal |
| torch.topk | mindspore.mint.topk |
| torch.sort | mindspore.mint.sort |
| torch.addbmm | mindspore.mint.addbmm |
| torch.addmm | mindspore.mint.addmm |
| torch.baddbmm | mindspore.mint.baddbmm |
| torch.bmm | mindspore.mint.bmm |
| torch.dot | mindspore.mint.dot |
| torch.inverse | mindspore.mint.inverse |
| torch.matmul | mindspore.mint.matmul |
| torch.meshgrid | mindspore.mint.meshgrid |
| torch.mm | mindspore.mint.mm |
| torch.outer | mindspore.mint.outer |
| torch.trace | mindspore.mint.trace |
| torch.broadcast_to | mindspore.mint.broadcast_to |
| torch.cdist | mindspore.mint.cdist |
| torch.cummax | mindspore.mint.cummax |
| torch.cummin | mindspore.mint.cummin |
| torch.cumsum | mindspore.mint.cumsum |
| torch.diag | mindspore.mint.diag |
| torch.flatten | mindspore.mint.flatten |
| torch.flip | mindspore.mint.flip |
| torch.repeat_interleave | mindspore.mint.repeat_interleave |
| torch.searchsorted | mindspore.mint.searchsorted |
| torch.tril | mindspore.mint.tril |
| torch.triangular_solve | mindspore.mint.triangular_solve |
| torch.clip | mindspore.mint.clamp |
| torch.concatenate | mindspore.mint.cat |
| torch.log_softmax | mindspore.mint.nn.functional.log_softmax |

### mint_nn_map
| Torch API | MindSpore API |
| --- | --- |
| torch.nn.Conv2d | mindspore.mint.nn.Conv2d |
| torch.nn.Conv3d | mindspore.mint.nn.Conv3d |
| torch.nn.ConvTranspose2d | mindspore.mint.nn.ConvTranspose2d |
| torch.nn.Fold | mindspore.mint.nn.Fold |
| torch.nn.Unfold | mindspore.mint.nn.Unfold |
| torch.nn.BatchNorm1d | mindspore.mint.nn.BatchNorm1d |
| torch.nn.BatchNorm2d | mindspore.mint.nn.BatchNorm2d |
| torch.nn.BatchNorm3d | mindspore.mint.nn.BatchNorm3d |
| torch.nn.GroupNorm | mindspore.mint.nn.GroupNorm |
| torch.nn.LayerNorm | mindspore.mint.nn.LayerNorm |
| torch.nn.SyncBatchNorm | mindspore.mint.nn.SyncBatchNorm |
| torch.nn.ELU | mindspore.mint.nn.ELU |
| torch.nn.GELU | mindspore.mint.nn.GELU |
| torch.nn.GLU | mindspore.mint.nn.GLU |
| torch.nn.Hardshrink | mindspore.mint.nn.Hardshrink |
| torch.nn.Hardsigmoid | mindspore.mint.nn.Hardsigmoid |
| torch.nn.Hardswish | mindspore.mint.nn.Hardswish |
| torch.nn.LogSigmoid | mindspore.mint.nn.LogSigmoid |
| torch.nn.LogSoftmax | mindspore.mint.nn.LogSoftmax |
| torch.nn.Mish | mindspore.mint.nn.Mish |
| torch.nn.PReLU | mindspore.mint.nn.PReLU |
| torch.nn.ReLU | mindspore.mint.nn.ReLU |
| torch.nn.ReLU6 | mindspore.mint.nn.ReLU6 |
| torch.nn.SELU | mindspore.mint.nn.SELU |
| torch.nn.SiLU | mindspore.mint.nn.SiLU |
| torch.nn.Sigmoid | mindspore.mint.nn.Sigmoid |
| torch.nn.Softmax | mindspore.mint.nn.Softmax |
| torch.nn.Softshrink | mindspore.mint.nn.Softshrink |
| torch.nn.Tanh | mindspore.mint.nn.Tanh |
| torch.nn.Embedding | mindspore.mint.nn.Embedding |
| torch.nn.Linear | mindspore.mint.nn.Linear |
| torch.nn.Dropout | mindspore.mint.nn.Dropout |
| torch.nn.Dropout2d | mindspore.mint.nn.Dropout2d |
| torch.nn.AdaptiveAvgPool1d | mindspore.mint.nn.AdaptiveAvgPool1d |
| torch.nn.AdaptiveAvgPool2d | mindspore.mint.nn.AdaptiveAvgPool2d |
| torch.nn.AdaptiveAvgPool3d | mindspore.mint.nn.AdaptiveAvgPool3d |
| torch.nn.AdaptiveMaxPool1d | mindspore.mint.nn.AdaptiveMaxPool1d |
| torch.nn.AvgPool2d | mindspore.mint.nn.AvgPool2d |
| torch.nn.AvgPool3d | mindspore.mint.nn.AvgPool3d |
| torch.nn.MaxUnpool2d | mindspore.mint.nn.MaxUnpool2d |
| torch.nn.ConstantPad1d | mindspore.mint.nn.ConstantPad1d |
| torch.nn.ConstantPad2d | mindspore.mint.nn.ConstantPad2d |
| torch.nn.ConstantPad3d | mindspore.mint.nn.ConstantPad3d |
| torch.nn.ReflectionPad1d | mindspore.mint.nn.ReflectionPad1d |
| torch.nn.ReflectionPad2d | mindspore.mint.nn.ReflectionPad2d |
| torch.nn.ReflectionPad3d | mindspore.mint.nn.ReflectionPad3d |
| torch.nn.ReplicationPad1d | mindspore.mint.nn.ReplicationPad1d |
| torch.nn.ReplicationPad2d | mindspore.mint.nn.ReplicationPad2d |
| torch.nn.ReplicationPad3d | mindspore.mint.nn.ReplicationPad3d |
| torch.nn.ZeroPad1d | mindspore.mint.nn.ZeroPad1d |
| torch.nn.ZeroPad2d | mindspore.mint.nn.ZeroPad2d |
| torch.nn.ZeroPad3d | mindspore.mint.nn.ZeroPad3d |
| torch.nn.BCELoss | mindspore.mint.nn.BCELoss |
| torch.nn.BCEWithLogitsLoss | mindspore.mint.nn.BCEWithLogitsLoss |
| torch.nn.CrossEntropyLoss | mindspore.mint.nn.CrossEntropyLoss |
| torch.nn.KLDivLoss | mindspore.mint.nn.KLDivLoss |
| torch.nn.L1Loss | mindspore.mint.nn.L1Loss |
| torch.nn.MSELoss | mindspore.mint.nn.MSELoss |
| torch.nn.NLLLoss | mindspore.mint.nn.NLLLoss |
| torch.nn.SmoothL1Loss | mindspore.mint.nn.SmoothL1Loss |
| torch.nn.PixelShuffle | mindspore.mint.nn.PixelShuffle |
| torch.nn.Upsample | mindspore.mint.nn.Upsample |
| torch.nn.Identity | mindspore.mint.nn.Identity |
| torch.nn.functional.conv2d | mindspore.mint.nn.functional.conv2d |
| torch.nn.functional.conv3d | mindspore.mint.nn.functional.conv3d |
| torch.nn.functional.conv_transpose2d | mindspore.mint.nn.functional.conv_transpose2d |
| torch.nn.functional.fold | mindspore.mint.nn.functional.fold |
| torch.nn.functional.unfold | mindspore.mint.nn.functional.unfold |
| torch.nn.functional.adaptive_avg_pool1d | mindspore.mint.nn.functional.adaptive_avg_pool1d |
| torch.nn.functional.adaptive_avg_pool2d | mindspore.mint.nn.functional.adaptive_avg_pool2d |
| torch.nn.functional.adaptive_avg_pool3d | mindspore.mint.nn.functional.adaptive_avg_pool3d |
| torch.nn.functional.adaptive_max_pool1d | mindspore.mint.nn.functional.adaptive_max_pool1d |
| torch.nn.functional.avg_pool1d | mindspore.mint.nn.functional.avg_pool1d |
| torch.nn.functional.avg_pool2d | mindspore.mint.nn.functional.avg_pool2d |
| torch.nn.functional.avg_pool3d | mindspore.mint.nn.functional.avg_pool3d |
| torch.nn.functional.max_pool2d | mindspore.mint.nn.functional.max_pool2d |
| torch.nn.functional.max_unpool2d | mindspore.mint.nn.functional.max_unpool2d |
| torch.nn.functional.batch_norm | mindspore.mint.nn.functional.batch_norm |
| torch.nn.functional.elu | mindspore.mint.nn.functional.elu |
| torch.nn.functional.elu_ | mindspore.mint.nn.functional.elu_ |
| torch.nn.functional.gelu | mindspore.mint.nn.functional.gelu |
| torch.nn.functional.glu | mindspore.mint.nn.functional.glu |
| torch.nn.functional.group_norm | mindspore.mint.nn.functional.group_norm |
| torch.nn.functional.hardshrink | mindspore.mint.nn.functional.hardshrink |
| torch.nn.functional.hardsigmoid | mindspore.mint.nn.functional.hardsigmoid |
| torch.nn.functional.hardswish | mindspore.mint.nn.functional.hardswish |
| torch.nn.functional.layer_norm | mindspore.mint.nn.functional.layer_norm |
| torch.nn.functional.leaky_relu | mindspore.mint.nn.functional.leaky_relu |
| torch.nn.functional.log_softmax | mindspore.mint.nn.functional.log_softmax |
| torch.nn.functional.logsigmoid | mindspore.mint.nn.functional.logsigmoid |
| torch.nn.functional.mish | mindspore.mint.nn.functional.mish |
| torch.nn.functional.prelu | mindspore.mint.nn.functional.prelu |
| torch.nn.functional.relu | mindspore.mint.nn.functional.relu |
| torch.nn.functional.relu6 | mindspore.mint.nn.functional.relu6 |
| torch.nn.functional.relu_ | mindspore.mint.nn.functional.relu_ |
| torch.nn.functional.selu | mindspore.mint.nn.functional.selu |
| torch.nn.functional.sigmoid | mindspore.mint.nn.functional.sigmoid |
| torch.nn.functional.silu | mindspore.mint.nn.functional.silu |
| torch.nn.functional.softmax | mindspore.mint.nn.functional.softmax |
| torch.nn.functional.softplus | mindspore.mint.nn.functional.softplus |
| torch.nn.functional.softshrink | mindspore.mint.nn.functional.softshrink |
| torch.nn.functional.tanh | mindspore.mint.nn.functional.tanh |
| torch.nn.functional.normalize | mindspore.mint.nn.functional.normalize |
| torch.nn.functional.linear | mindspore.mint.nn.functional.linear |
| torch.nn.functional.dropout | mindspore.mint.nn.functional.dropout |
| torch.nn.functional.dropout2d | mindspore.mint.nn.functional.dropout2d |
| torch.nn.functional.embedding | mindspore.mint.nn.functional.embedding |
| torch.nn.functional.one_hot | mindspore.mint.nn.functional.one_hot |
| torch.nn.functional.cross_entropy | mindspore.mint.nn.functional.cross_entropy |
| torch.nn.functional.binary_cross_entropy | mindspore.mint.nn.functional.binary_cross_entropy |
| torch.nn.functional.binary_cross_entropy_with_logits | mindspore.mint.nn.functional.binary_cross_entropy_with_logits |
| torch.nn.functional.kl_div | mindspore.mint.nn.functional.kl_div |
| torch.nn.functional.l1_loss | mindspore.mint.nn.functional.l1_loss |
| torch.nn.functional.mse_loss | mindspore.mint.nn.functional.mse_loss |
| torch.nn.functional.nll_loss | mindspore.mint.nn.functional.nll_loss |
| torch.nn.functional.smooth_l1_loss | mindspore.mint.nn.functional.smooth_l1_loss |
| torch.nn.functional.interpolate | mindspore.mint.nn.functional.interpolate |
| torch.nn.functional.grid_sample | mindspore.mint.nn.functional.grid_sample |
| torch.nn.functional.pad | mindspore.mint.nn.functional.pad |
| torch.nn.functional.pixel_shuffle | mindspore.mint.nn.functional.pixel_shuffle |
| torch.nn.Module | mindspore.nn.Cell |
| torch.nn.Sequential | mindspore.nn.SequentialCell |
| torch.nn.ModuleList | mindspore.nn.CellList |
| torch.nn.ModuleDict | mindspore.nn.CellDict |
| torch.nn.Flatten | mindspore.nn.Flatten |
| torch.nn.CTCLoss | mindspore.nn.CTCLoss |

### ops_map
| Torch API | MindSpore API |
| --- | --- |
| torch.addcmul | mindspore.ops.addcmul |
| torch.argwhere | mindspore.ops.argwhere |
| torch.bucketize | mindspore.ops.bucketize |
| torch.conj | mindspore.ops.conj |
| torch.cosine_similarity | mindspore.ops.cosine_similarity |
| torch.deg2rad | mindspore.ops.deg2rad |
| torch.hann_window | mindspore.ops.hann_window |
| torch.hstack | mindspore.ops.hstack |
| torch.masked_fill | mindspore.ops.masked_fill |
| torch.multiply | mindspore.ops.multiply |
| torch.numel | mindspore.ops.numel |
| torch.range | mindspore.ops.range |
| torch.relu | mindspore.ops.relu |
| torch.nn.functional.ctc_loss | mindspore.ops.ctc_loss |
| torch.nn.functional.gumbel_softmax | mindspore.ops.gumbel_softmax |
| torch.full | mindspore.ops.full |
| torch.fill | mindspore.ops.fill |

### others
| Torch API | MindSpore API |
| --- | --- |
| torch.Tensor | mindspore.Tensor |
| torch.tensor | mindspore.Tensor |
| torch.ByteTensor | mindspore.Tensor |
| torch.IntTensor | mindspore.Tensor |
| torch.FloatTensor | mindspore.Tensor |
| torch.LongTensor | mindspore.Tensor |
| torch.BoolTensor | mindspore.Tensor |
| torch.float | mindspore.float32 |
| torch.double | mindspore.float64 |
| torch.float32 | mindspore.float32 |
| torch.float64 | mindspore.float64 |
| torch.float16 | mindspore.float16 |
| torch.bfloat16 | mindspore.bfloat16 |
| torch.int8 | mindspore.int8 |
| torch.uint8 | mindspore.uint8 |
| torch.int16 | mindspore.int16 |
| torch.int | mindspore.int32 |
| torch.int32 | mindspore.int32 |
| torch.int64 | mindspore.int64 |
| torch.long | mindspore.int64 |
| torch.bool | mindspore.bool_ |
| torch.dtype | mindspore.dtype |
| torch.Generator | mindspore.Generator |
| torch.complex64 | mindspore.complex64 |
| torch.no_grad | mindspore._no_grad |
| torch.version | mindspore.version |
| torch.vmap | mindspore.vmap |
| torch.nn.Parameter | mindspore.Parameter |
| torch.from_numpy | mindspore.Tensor.from_numpy |

### Tensor api
| Torch API | MindSpore API | Alternatives |
| --- | --- | --- |
| torch.Tensor.__len__ | mindspore.Tensor.__len__ |  |
| torch.Tensor.abs | mindspore.Tensor.abs |  |
| torch.Tensor.add | mindspore.Tensor.add |  |
| torch.Tensor.add_ | mindspore.Tensor.add_ |  |
| torch.Tensor.all | mindspore.Tensor.all |  |
| torch.Tensor.amax | mindspore.Tensor.amax |  |
| torch.Tensor.amin | mindspore.Tensor.amin |  |
| torch.Tensor.angle | mindspore.Tensor.angle |  |
| torch.Tensor.any | mindspore.Tensor.any |  |
| torch.Tensor.arctan | mindspore.Tensor.arctan |  |
| torch.Tensor.arctan2 | mindspore.Tensor.arctan2 |  |
| torch.Tensor.argmax | mindspore.Tensor.argmax |  |
| torch.Tensor.argmin | mindspore.Tensor.argmin |  |
| torch.Tensor.argsort | mindspore.Tensor.argsort |  |
| torch.Tensor.argwhere | mindspore.Tensor.argwhere |  |
| torch.Tensor.atan | mindspore.Tensor.atan |  |
| torch.Tensor.atan2 | mindspore.Tensor.atan2 |  |
| torch.Tensor.bernoulli_ | mindspore.Tensor.bernoulli |  |
| torch.Tensor.bincount | mindspore.Tensor.bincount |  |
| torch.Tensor.bitwise_and | mindspore.Tensor.bitwise_and |  |
| torch.Tensor.bmm | mindspore.Tensor.bmm |  |
| torch.Tensor.bool | mindspore.Tensor.bool |  |
| torch.Tensor.ceil | mindspore.Tensor.ceil |  |
| torch.Tensor.chunk | mindspore.Tensor.chunk |  |
| torch.Tensor.clamp | mindspore.Tensor.clamp |  |
| torch.Tensor.clamp_ | mindspore.Tensor.clamp_ |  |
| torch.Tensor.clamp_max |  | mindspore.mint.clamp |
| torch.Tensor.clip | mindspore.Tensor.clip |  |
| torch.Tensor.clone | mindspore.Tensor.clone |  |
| torch.Tensor.contiguous | mindspore.Tensor.contiguous |  |
| torch.Tensor.copy_ | mindspore.Tensor.copy_ |  |
| torch.Tensor.cos | mindspore.Tensor.cos |  |
| torch.Tensor.count_nonzero | mindspore.Tensor.count_nonzero |  |
| torch.Tensor.cumsum | mindspore.Tensor.cumsum |  |
| torch.Tensor.data |  | mindspore.ops.stop_gradient<br/>mindspore.Parameter.requires_grad = False |
| torch.Tensor.deg2rad | mindspore.Tensor.deg2rad |  |
| torch.Tensor.detach |  | mindspore.ops.stop_gradient<br/>mindspore.Parameter.requires_grad = False |
| torch.Tensor.device | mindspore.Tensor.device |  |
| torch.Tensor.diff | mindspore.Tensor.diff |  |
| torch.Tensor.dim | mindspore.Tensor.dim |  |
| torch.Tensor.div | mindspore.Tensor.div |  |
| torch.Tensor.div_ | mindspore.Tensor.div_ |  |
| torch.Tensor.divide | mindspore.Tensor.divide |  |
| torch.Tensor.dot | mindspore.Tensor.dot |  |
| torch.Tensor.double | mindspore.Tensor.double |  |
| torch.Tensor.dtype | mindspore.Tensor.dtype |  |
| torch.Tensor.eq | mindspore.Tensor.eq |  |
| torch.Tensor.exp | mindspore.Tensor.exp |  |
| torch.Tensor.expand | mindspore.Tensor.expand |  |
| torch.Tensor.expand_as | mindspore.Tensor.expand_as |  |
| torch.Tensor.fill_ | mindspore.Tensor.fill_ |  |
| torch.Tensor.flatten | mindspore.Tensor.flatten |  |
| torch.Tensor.flip | mindspore.Tensor.flip |  |
| torch.Tensor.float | mindspore.Tensor.float |  |
| torch.Tensor.floor | mindspore.Tensor.floor |  |
| torch.Tensor.gather | mindspore.Tensor.gather |  |
| torch.Tensor.ge | mindspore.Tensor.ge |  |
| torch.Tensor.get_device |  | mindspore.context.get_context("device_id") |
| torch.Tensor.grad |  | mindspore.grad |
| torch.Tensor.greater | mindspore.Tensor.greater |  |
| torch.Tensor.half | mindspore.Tensor.half |  |
| torch.Tensor.histogram |  | mindspore.Tensor.histo |
| torch.Tensor.index_select | mindspore.Tensor.index_select |  |
| torch.Tensor.int | mindspore.Tensor.int |  |
| torch.Tensor.inverse | mindspore.Tensor.inverse |  |
| torch.Tensor.is_contiguous | mindspore.Tensor.is_contiguous |  |
| torch.Tensor.is_floating_point | mindspore.Tensor.is_floating_point |  |
| torch.Tensor.is_sparse |  |  |
| torch.Tensor.isclose | mindspore.Tensor.isclose |  |
| torch.Tensor.isinf | mindspore.Tensor.isinf |  |
| torch.Tensor.isnan | mindspore.Tensor.isnan |  |
| torch.Tensor.isneginf | mindspore.Tensor.isneginf |  |
| torch.Tensor.item | mindspore.Tensor.item |  |
| torch.Tensor.layout |  |  |
| torch.Tensor.lcm | mindspore.Tensor.lcm |  |
| torch.Tensor.log | mindspore.Tensor.log |  |
| torch.Tensor.log2 | mindspore.Tensor.log2 |  |
| torch.Tensor.logical_not | mindspore.Tensor.logical_not |  |
| torch.Tensor.logical_or | mindspore.Tensor.logical_or |  |
| torch.Tensor.long | mindspore.Tensor.long |  |
| torch.Tensor.lt | mindspore.Tensor.lt |  |
| torch.Tensor.masked_fill | mindspore.Tensor.masked_fill |  |
| torch.Tensor.masked_fill_ | mindspore.Tensor.masked_fill_ |  |
| torch.Tensor.matmul | mindspore.Tensor.matmul |  |
| torch.Tensor.max | mindspore.Tensor.max |  |
| torch.Tensor.maximum | mindspore.Tensor.maximum |  |
| torch.Tensor.mean | mindspore.Tensor.mean |  |
| torch.Tensor.min | mindspore.Tensor.min |  |
| torch.Tensor.minimum | mindspore.Tensor.minimum |  |
| torch.Tensor.mode |  |  |
| torch.Tensor.mul | mindspore.Tensor.mul |  |
| torch.Tensor.mul_ | mindspore.Tensor.mul_ |  |
| torch.Tensor.multiply | mindspore.Tensor.multiply |  |
| torch.Tensor.nan_to_num | mindspore.Tensor.nan_to_num |  |
| torch.Tensor.nbytes | mindspore.Tensor.nbytes |  |
| torch.Tensor.ndim | mindspore.Tensor.ndim |  |
| torch.Tensor.ndim | mindspore.Tensor.ndim |  |
| torch.Tensor.ndimension | mindspore.Tensor.ndimension |  |
| torch.Tensor.nelement | mindspore.Tensor.nelement |  |
| torch.Tensor.new |  | ms.Tensor() |
| torch.Tensor.new_full | mindspore.Tensor.new_full |  |
| torch.Tensor.new_ones | mindspore.Tensor.new_ones |  |
| torch.Tensor.new_tensor |  | ms.Tensor() |
| torch.Tensor.new_zeros | mindspore.Tensor.new_zeros |  |
| torch.Tensor.nonzero | mindspore.Tensor.nonzero |  |
| torch.Tensor.norm | mindspore.Tensor.norm |  |
| torch.Tensor.normal_ | mindspore.Tensor.normal_ |  |
| torch.Tensor.numel | mindspore.Tensor.numel |  |
| torch.Tensor.numpy | mindspore.Tensor.numpy | mindspore.Tensor.asnumpy |
| torch.Tensor.permute | mindspore.Tensor.permute |  |
| torch.Tensor.pow | mindspore.Tensor.pow |  |
| torch.Tensor.put |  |  |
| torch.Tensor.qr |  | mindspore.mint.linalg.qr |
| torch.Tensor.rad2deg | mindspore.Tensor.rad2deg |  |
| torch.Tensor.relu |  |  |
| torch.Tensor.repeat | mindspore.Tensor.repeat |  |
| torch.Tensor.repeat | mindspore.Tensor.repeat |  |
| torch.Tensor.requires_grad |  | mindspore.grad |
| torch.Tensor.reshape | mindspore.Tensor.reshape |  |
| torch.Tensor.resize | mindspore.Tensor.resize |  |
| torch.Tensor.roll | mindspore.Tensor.roll |  |
| torch.Tensor.round | mindspore.Tensor.round |  |
| torch.Tensor.scatter | mindspore.Tensor.scatter |  |
| torch.Tensor.scatter_ | mindspore.Tensor.scatter_add |  |
| torch.Tensor.shape | mindspore.Tensor.shape |  |
| torch.Tensor.sigmoid | mindspore.Tensor.sigmoid |  |
| torch.Tensor.sigmoid_ |  | mindspore.Tensor.sigmoid |
| torch.Tensor.sin | mindspore.Tensor.sin |  |
| torch.Tensor.size | mindspore.Tensor.shape |  |
| torch.Tensor.softmax | mindspore.Tensor.softmax |  |
| torch.Tensor.sort | mindspore.Tensor.sort |  |
| torch.Tensor.split | mindspore.Tensor.split |  |
| torch.Tensor.sqrt | mindspore.Tensor.sqrt |  |
| torch.Tensor.square | mindspore.Tensor.square |  |
| torch.Tensor.squeeze | mindspore.Tensor.squeeze |  |
| torch.Tensor.std | mindspore.Tensor.std |  |
| torch.Tensor.stride | mindspore.Tensor.stride |  |
| torch.Tensor.sub | mindspore.Tensor.sub |  |
| torch.Tensor.sum | mindspore.Tensor.sum |  |
| torch.Tensor.swapaxes | mindspore.Tensor.swapaxes |  |
| torch.Tensor.t | mindspore.Tensor.t |  |
| torch.Tensor.tan | mindspore.Tensor.tan |  |
| torch.Tensor.tanh | mindspore.Tensor.tanh |  |
| torch.Tensor.tanh_ |  | mindspore.Tensor.tanh |
| torch.Tensor.tile | mindspore.Tensor.tile |  |
| torch.Tensor.to | mindspore.Tensor.to |  |
| torch.Tensor.tolist | mindspore.Tensor.tolist |  |
| torch.Tensor.topk | mindspore.Tensor.topk |  |
| torch.Tensor.transpose | mindspore.Tensor.transpose |  |
| torch.Tensor.type | mindspore.Tensor.type |  |
| torch.Tensor.type_as | mindspore.Tensor.type_as |  |
| torch.Tensor.unbind | mindspore.Tensor.unbind |  |
| torch.Tensor.unfold | mindspore.Tensor.unfold |  |
| torch.Tensor.uniform_ | mindspore.Tensor.uniform_ |  |
| torch.Tensor.unique | mindspore.Tensor.unique |  |
| torch.Tensor.unsqueeze | mindspore.Tensor.unsqueeze |  |
| torch.Tensor.values |  |  |
| torch.Tensor.view | mindspore.Tensor.view |  |
| torch.Tensor.where | mindspore.Tensor.where |  |
| torch.Tensor.zero_ | mindspore.Tensor.zero_ |  |
| torch.Tensor.backward |  | mindspore.grad |


