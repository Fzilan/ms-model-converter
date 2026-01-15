# Minimal test template

```python
import numpy as np
import mindspore as ms
from transformers import <ConfigClass>
from mindone.transformers.models.<model_name>.modeling_<model_name> import <ModelClass>

ms.set_context(mode=ms.PYNATIVE_MODE)

def test_forward_smoke():
    config = <ConfigClass>(...)
    model = <ModelClass>(config)
    input_ids = ms.Tensor(np.zeros((2, 4), dtype=np.int32))
    outputs = model(input_ids=input_ids)
    assert outputs.last_hidden_state.shape == (2, 4, config.hidden_size)
```

Notes:
- Shrink config sizes for fast tests.
- Adjust inputs and expected fields to match the model.
