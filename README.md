<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/AKJ7/pytest-torch/main/docs/assets/logo-dark.png">
    <img alt="pytest-torch" src="https://raw.githubusercontent.com/AKJ7/pytest-torch/main/docs/assets/logo.png" width="520">
  </picture>
</h1>

**Test PyTorch models like you mean it.** `pytest-torch` is a pytest plugin with small,
composable helpers for the tests every model needs, and failure messages that say where a
mismatch is and how large it is.

> **Status: alpha.** The API may still change. Requires Python 3.9+, pytest 7+ and PyTorch 2.0+.
> **Documentation:** <https://akj7.github.io/pytest-torch/>

```bash
pip install pytest-torch
```

## Start here

Pick the question you're asking. Each path is one or two calls; every snippet is runnable in
[`examples/test_start_here.py`](examples/test_start_here.py).

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

`model` is your fixture; `make_tensor` and `torch_device` come with the plugin, so the test
runs on every available device with seeded inputs. Go further:
[invariance](https://akj7.github.io/pytest-torch/guides/inference/) (padding, masks,
permutations), [output specs](https://akj7.github.io/pytest-torch/guides/output-specs/),
[unusual inputs and dtypes](https://akj7.github.io/pytest-torch/guides/matrix/).

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

Go further: [deployment parity](https://akj7.github.io/pytest-torch/guides/deployment/)
(`torch.compile`, `torch.export`, AOTInductor, ONNX, autocast),
[streaming and stateful models](https://akj7.github.io/pytest-torch/guides/streaming/),
[tolerances](https://akj7.github.io/pytest-torch/guides/tolerances/).

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
and CI compares against them. Go further:
[dataset metrics](https://akj7.github.io/pytest-torch/guides/dataset-metrics/) (paired
comparisons with confidence intervals),
[checkpoints](https://akj7.github.io/pytest-torch/guides/checkpoints/),
[performance](https://akj7.github.io/pytest-torch/guides/performance/),
[failure reproduction](https://akj7.github.io/pytest-torch/guides/repro/).

## Or generate the tests

`ModelSuite` turns one class into a test suite: pick the checks, and each becomes a test on
every device.

```python
from pytest_torch import ModelSuite

class TestClassifier(ModelSuite):
    checks = ("contract", "compile", "export", "output_spec", "checkpoint", "training")
    output_spec = {"logits": TensorSpec(("B", 3))}

    def make_model(self):
        return Classifier()

    def make_inputs(self, make_tensor):
        return make_tensor(8, 16)

    def loss(self, output, inputs):          # only needed for "training"
        return output["logits"].logsumexp(-1).mean()
```

See [generated test suites](https://akj7.github.io/pytest-torch/guides/model-suite/) for
every check and option. The helpers it uses stay available on their own.

## Everything else

| Topic | Guide |
|---|---|
| Fixtures: devices, seeds, reproducible tensors, golden files | [Fixtures](https://akj7.github.io/pytest-torch/reference/fixtures/) |
| Project-wide tolerances in `pyproject.toml` | [Tolerances](https://akj7.github.io/pytest-torch/guides/tolerances/) |
| Gradients, learning, overfitting one batch, gradcheck | [Training sanity](https://akj7.github.io/pytest-torch/guides/training/) |
| Latency, memory, graph breaks, host syncs, nondeterminism, xdist | [Performance and determinism](https://akj7.github.io/pytest-torch/guides/performance/) |
| Hypothesis strategies for edge-case inputs | [Devices, dtypes and unusual inputs](https://akj7.github.io/pytest-torch/guides/matrix/) |
| Hugging Face, timm and torchvision models | [Recipes](https://akj7.github.io/pytest-torch/guides/recipes/) |
| Command-line options and markers | [Reference](https://akj7.github.io/pytest-torch/reference/cli/) |

## Installation

```bash
pip install pytest-torch          # or: uv add --dev pytest-torch
```

PyTorch is not installed for you (pick the build that matches your hardware). The plugin
registers itself through pytest's `pytest11` entry point, and importing it never imports
`torch`. Optional extras:

| Extra | Adds | Needed for |
|---|---|---|
| `hypothesis` | `hypothesis` | `pytest_torch.strategies` |
| `onnx` | `onnx`, `onnxruntime`, `onnxscript` | `assert_onnx_parity` |
| `hf` | `transformers` | `ModelLoader.hf`, `assert_generation_equal` with HF models |
| `timm`, `torchvision` | the respective package | `ModelLoader.timm`, `ModelLoader.torchvision` |
| `plots` | `matplotlib` | mismatch heatmaps (`--torch-failure-artifacts`) |

## Examples

Runnable test files in [`examples/`](examples/) (`pytest examples/ -v`). Most have a "Try ..."
comment: a one-line bug that makes the test fail.

| Topic | Files |
|---|---|
| Start with | [`test_start_here.py`](examples/test_start_here.py) (the three paths above), [`test_model_suite.py`](examples/test_model_suite.py), [`test_markers.py`](examples/test_markers.py) |
| One prediction | [`test_output_specs.py`](examples/test_output_specs.py), [`test_metamorphic.py`](examples/test_metamorphic.py), [`test_unusual_inputs.py`](examples/test_unusual_inputs.py), [`test_dtype_matrix.py`](examples/test_dtype_matrix.py), [`test_property_based.py`](examples/test_property_based.py) |
| Model changes | [`test_reimplementation.py`](examples/test_reimplementation.py), [`test_deployment.py`](examples/test_deployment.py), [`test_streaming.py`](examples/test_streaming.py) |
| Releases | [`test_dataset_metrics.py`](examples/test_dataset_metrics.py), [`test_checkpoints.py`](examples/test_checkpoints.py), [`test_failure_repro.py`](examples/test_failure_repro.py), [`test_tiny_classifier.py`](examples/test_tiny_classifier.py) |
| More | [`test_training_sanity.py`](examples/test_training_sanity.py), [`test_nondeterminism.py`](examples/test_nondeterminism.py) |

[`examples/integration/`](examples/integration/) has three small but realistic projects, laid
out like real repositories and tested end to end: a transformer
[text classifier](examples/integration/text_classification/), an
[image classifier with its deployment pipeline](examples/integration/image_classification/),
and a [small GPT with a KV cache](examples/integration/language_model/).

## Development

```bash
uv sync                          # environment, lockfile, dev dependencies
uv run pytest                    # the plugin's own tests
uv run pytest examples/ -v       # runnable examples of every feature
uv run pytest -m "not slow"      # skip AOTInductor compilation and other slow tests
uv run --group docs mkdocs serve # documentation at http://127.0.0.1:8000
uv build                         # sdist and wheel
```

CI (`.github/workflows/ci.yml`) runs the tests and the examples over operating systems, Python
versions and torch versions with CPU-only wheels, plus the slow tests once on the newest torch.
`docs.yml` builds the docs on every PR and publishes them from `main`; `release.yml` publishes
to PyPI from a GitHub release.

## License

[MIT](LICENSE)
