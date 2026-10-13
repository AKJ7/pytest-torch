# pytest-torch

**Test PyTorch models like you mean it.** `pytest-torch` is a pytest plugin that turns the
tedious, easy-to-get-wrong parts of model testing into one-liners: device parametrization,
reproducible inputs, tolerance-aware comparisons, invariance checks, deployment parity,
training sanity checks, performance baselines and readable failures.

```python
from pytest_torch import ModelSuite

class TestClassifier(ModelSuite):
    checks = ("contract", "compile", "export", "training")

    def make_model(self):
        return Classifier()

    def make_inputs(self, make_tensor):
        return make_tensor(8, 16)

    def loss(self, output, inputs):
        return output.logsumexp(-1).mean()
```

That class is a test suite: eval mode, finite outputs, determinism, batch invariance,
cross-device parity, `torch.compile` and `torch.export` parity, finite gradients, every
parameter learning, and the model overfitting one batch, on every available device.

## What it catches

| You want to catch... | Use |
|---|---|
| A whole suite for a model, from one class | [`ModelSuite`](guides/model-suite.md) |
| Train mode left on, NaN/Inf, nondeterminism, batch-dependent output, mutated buffers | [`check_inference_contract`](guides/inference.md#inference-contract) |
| CPU and GPU (or fp32 and bf16) disagree: *where?* | [`localize_divergence`](guides/inference.md#find-the-first-diverging-layer) |
| Padding or masked tokens leak; a set model depends on order | [invariance checks](guides/inference.md#invariance-and-metamorphic-tests) |
| `torch.compile` / `torch.export` / AOTInductor / ONNX / autocast differ from eager | [deployment parity](guides/deployment.md) |
| Dead layers, NaN gradients, a broken backward, a model that cannot learn | [training sanity](guides/training.md) |
| Latency, memory, graph breaks, recompiles, host syncs, flaky GPU outputs | [performance and determinism](guides/performance.md) |

!!! note "Status: alpha"
    The API may still change. Requires Python 3.9+, pytest 7+ and PyTorch 2.0+
    (`torch.export` features need 2.2+, AOTInductor 2.6+).
