# Getting started

## Install

```bash
pip install pytest-torch          # or: uv add --dev pytest-torch
```

PyTorch is not installed for you: pick the build that matches your hardware. The plugin
registers itself through pytest's `pytest11` entry point, and importing it never imports
`torch`, so projects that don't test models pay nothing.

Optional extras:

| Extra | Adds | Needed for |
|---|---|---|
| `hypothesis` | `hypothesis` | `pytest_torch.strategies` |
| `onnx` | `onnx`, `onnxruntime`, `onnxscript` | `assert_onnx_parity` |
| `hf` | `transformers` | `ModelLoader.hf`, `assert_generation_equal` with HF models |
| `timm`, `torchvision` | the respective package | `ModelLoader.timm`, `ModelLoader.torchvision` |
| `plots` | `matplotlib` | mismatch heatmaps (`--torch-failure-artifacts`) |

## First test

```python
# tests/test_model.py
import pytest
from pytest_torch import check_inference_contract

@pytest.fixture
def model(torch_seed, torch_device):
    return MyModel().to(torch_device).eval()

def test_contract(model, make_tensor):
    check_inference_contract(model, make_tensor(8, 16))
```

```bash
pytest                           # runs on cpu, cuda, xpu and mps, whichever exist
pytest --torch-device=cpu        # pin a device
```

- `torch_device` parametrizes the test over the available devices.
- `torch_seed` seeds `random`, `numpy` and `torch`; the seed is printed when a test fails.
- `make_tensor` draws reproducible tensors on the CPU and moves them to the device, so
  every device sees identical inputs.

Model inputs can be a tensor, a tuple of positional arguments, or a dict of keyword
arguments; outputs can be any nested structure of dicts, tuples, dataclasses or Hugging
Face `ModelOutput`s.

## Next

- Pick a path on the [home page](index.md#start-here): check one prediction, check a model
  change, or check a release.
- Get a whole suite from one class: [Generated test suites](guides/model-suite.md).
- Browse runnable examples in the repository's
  [`examples/`](https://github.com/AKJ7/pytest-torch/tree/main/examples) folder. Most of
  them carry a "Try ..." comment: a one-line bug that makes the test fail.
- See it on realistic projects:
  [`examples/integration/`](https://github.com/AKJ7/pytest-torch/tree/main/examples/integration)
  has a transformer text classifier, an image classifier with its deployment pipeline, a
  small GPT with a KV cache, and a navigation policy tested once per YAML config, each laid
  out like a real repository and tested end to end.
