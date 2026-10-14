# pytest-torch { .visually-hidden }

![pytest-torch](assets/logo.png#only-light){ .hero-logo }
![pytest-torch](assets/logo-dark.png#only-dark){ .hero-logo }

**Test PyTorch models like you mean it.** `pytest-torch` is a pytest plugin with small,
composable helpers for the tests every model needs, and failure messages that say where a
mismatch is and how large it is.

```bash
pip install pytest-torch
```

Or try it on any model first, with no test file:

```bash
pytest --torch-model=hf:google-bert/bert-base-uncased     # or timm:NAME, torchvision:NAME, mypackage.models:Net
```

See [test any model with no code](guides/zero-code.md).

## Start here

Pick the question you're asking. Each path is one or two calls; every snippet is runnable in
[`examples/test_start_here.py`](https://github.com/AKJ7/pytest-torch/blob/main/examples/test_start_here.py).

### 1. Check one prediction

*Does the model behave like an inference model, and do its outputs look right?*

```python
from pytest_torch import Probabilities, TensorSpec, assert_output_spec, check_inference_contract

SPEC = {"logits": TensorSpec(("B", 3)), "probs": Probabilities(("B", 3))}

def test_one_prediction(model, make_tensor):
    x = make_tensor(8, 16)
    check_inference_contract(model, x)       # eval mode, finite, deterministic, batch-invariant, devices
    with torch.inference_mode():
        assert_output_spec(model(x), SPEC, B=8)   # shapes, dtypes, ranges, probabilities
```

Go further: [inference correctness](guides/inference.md) (contract, invariance, padding and
masks), [output specifications](guides/output-specs.md),
[devices, dtypes and unusual inputs](guides/matrix.md).

### 2. Check a model change

*Does the refactored, ported, compiled or compressed model still match the original?*

```python
from pytest_torch import assert_matches_reference, assert_models_agree, localize_divergence

def test_refactor_matches_the_original(original, refactored, make_tensor):
    x = make_tensor(8, 16)
    assert_matches_reference(refactored, original, x)              # same outputs, path by path
    localize_divergence(original, refactored, x).raise_if_diverged()  # ...and layer by layer

def test_bf16_model_stays_close(original, make_tensor):
    x = make_tensor(256, 16)
    bf16 = copy.deepcopy(original).to(torch.bfloat16)
    assert_models_agree(original, bf16, x, inputs_candidate=x.to(torch.bfloat16), min_top1=0.95)
```

Go further: [deployment parity](guides/deployment.md) (`torch.compile`, `torch.export`,
AOTInductor, ONNX, autocast), [stateful and streaming models](guides/streaming.md).

### 3. Check a release

*Is the model we're about to ship as good, as compatible and as fast as the last one?*

```python
from pytest_torch import assert_checkpoint_loads

def accuracy(output, batch):
    return (output["logits"].argmax(-1) == batch[1]).float()   # per sample: paired comparisons

def test_release_accuracy(model, val_loader, metric_baseline):
    metric_baseline(model, val_loader, {"accuracy": accuracy}, minimum={"accuracy": 0.9},
                    context={"dataset": "synthetic", "split": "validation"})

def test_last_release_checkpoint_loads():
    assert_checkpoint_loads(Classifier, "tests/data/classifier_v1.pt")

def test_speed(model, make_tensor, perf_baseline):
    perf_baseline(model, make_tensor(32, 16))
```

Record baselines once with `--torch-update-metrics` and `--torch-update-perf`, commit them,
and CI compares against them. Go further: [dataset metrics](guides/dataset-metrics.md),
[checkpoints](guides/checkpoints.md), [performance and determinism](guides/performance.md),
[failure reproduction](guides/repro.md).

## Or generate the tests

[`ModelSuite`](guides/model-suite.md) turns one class into a test suite: pick the checks
(contract, compile, export, output spec, checkpoint, training, ...), and each becomes a test
on every device. The helpers it uses stay available on their own.

!!! note "Requirements and stability"
    Python 3.9+, pytest 7+ and PyTorch 2.0+. `torch.export` checks skip on torch 2.0 and
    work best on recent versions (CI exercises them from 2.2); AOTInductor needs 2.6+. From 1.0, pytest-torch follows semantic
    versioning: breaking changes to the public API (the names exported from `pytest_torch`,
    the fixtures, markers, command-line options and configuration) only come with a new major
    version.
