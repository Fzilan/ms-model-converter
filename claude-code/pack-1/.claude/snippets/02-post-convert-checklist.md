# Post-convert checklist

- `torch.nn.Module` -> `mindspore.nn.Cell`
- `forward` -> `construct`
- `torch.nn.Parameter` -> `mindspore.Parameter`
- Remove device handling (`.to`, `.cuda`, `torch.device`, `mps`)
- Prefer `mindspore.mint`, then `mindspore.ops`, then `mindspore.nn`
- Remove unsupported `inplace=True` args
- Drop PyTorch-only decorators and imports
- Update docstrings to `mindspore.Tensor`
- Wrap shape args in tuples, e.g. `.view((b, s, h))`
