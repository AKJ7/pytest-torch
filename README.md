<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/AKJ7/pytest-torch/main/docs/assets/logo-dark.png">
    <img alt="pytest-torch" src="https://raw.githubusercontent.com/AKJ7/pytest-torch/main/docs/assets/logo.png" width="520">
  </picture>
</h1>

<div align="center">

[![PyPI](https://img.shields.io/badge/-PyPI-blue.svg?logo=pypi&labelColor=555555&style=for-the-badge)](https://pypi.org/project/pytest-torch/)
[![Download count](https://img.shields.io/pypi/dm/pytest-torch?color=brightgreen&style=for-the-badge)](https://pypistats.org/packages/pytest-torch)
[![PyPI license](https://img.shields.io/pypi/l/pytest-torch?color=brightgreen&style=for-the-badge)](https://github.com/AKJ7/pytest-torch/blob/main/LICENSE)
[![Latest version](https://img.shields.io/github/v/tag/AKJ7/pytest-torch?color=brightgreen&style=for-the-badge)](https://github.com/AKJ7/pytest-torch/releases)

</div>

**Test PyTorch models like you mean it.** `pytest-torch` is a pytest plugin with small,
composable helpers for the tests every model needs, and failure messages that say where a
mismatch is and how large it is.

> **Stable** (1.x, [semantic versioning](https://github.com/AKJ7/pytest-torch/blob/main/CHANGELOG.md#stability)). Requires Python 3.9+, pytest 7+ and PyTorch 2.0+.
> **Documentation:** <https://akj7.github.io/pytest-torch/>

```bash
pip install pytest-torch
```

Try it on any model first, with no test file:

```bash
pytest --torch-model=hf:google-bert/bert-base-uncased     # or timm:NAME, torchvision:NAME, mypackage.models:Net
```

It builds the model (random weights; only the config is downloaded), infers inputs, and runs
the inference contract, padding and mask invariance, a checkpoint round-trip and finite
gradients on every available device. See
[test any model with no code](https://akj7.github.io/pytest-torch/guides/zero-code/).

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

## Features

Every feature, grouped by the question it answers. Each group links to its guide.

**Check one prediction** · [inference](https://akj7.github.io/pytest-torch/guides/inference/), [output specs](https://akj7.github.io/pytest-torch/guides/output-specs/), [devices and inputs](https://akj7.github.io/pytest-torch/guides/matrix/)

| Feature | API |
|---|---|
| One-call inference contract: eval mode, finite outputs, determinism, batch invariance, device and dtype parity, no state mutation, with a hint per failure | `check_inference_contract` → `ContractReport` of `CheckResult`s |
| Tensor and nested-output comparisons with dtype-aware tolerances, exact integers, complex support, `equal_nan`, structure checks and per-path messages | `assert_tensors_close`, `assert_outputs_close`, `assert_finite`, `assert_batch_invariant` |
| Output specifications: shapes with symbolic dims, dtypes, finiteness, ranges, probabilities and log-probabilities | `assert_output_spec`, `TensorSpec`, `Probabilities` |
| Padding, mask and permutation invariance, custom metamorphic relations | `assert_padding_invariant`, `assert_mask_invariant`, `assert_permutation_invariant`, `assert_permutation_equivariant`, `assert_invariant`, `assert_equivariant` |
| Memory-layout invariance: non-contiguous and channels_last inputs | `assert_layout_invariant` |
| Every test on every device (CPU, CUDA, Intel XPU, Apple MPS) and, on request, every dtype | `torch_device`, `torch_dtype` fixtures, `--torch-device`, `--torch-dtype`, `@pytest.mark.torch_dtypes`, `available_devices` |
| Reproducible inputs, identical on every device | `make_tensor`, `torch_generator` fixtures |
| Hypothesis strategies: edge-case tensors, any memory layout, ragged padded batches | `pytest_torch.strategies`: `tensors`, `shapes`, `padded_batches` |

**Check a model change** · [deployment parity](https://akj7.github.io/pytest-torch/guides/deployment/), [streaming](https://akj7.github.io/pytest-torch/guides/streaming/)

| Feature | API |
|---|---|
| Compare with a reference implementation or saved outputs | `assert_matches_reference` |
| Find the first layer where two models diverge, and whether the error starts inside it | `localize_divergence` → `DivergenceReport` of `LayerDiff`s |
| Accuracy budgets for quantized, pruned, distilled or low-precision models | `assert_models_agree`, `compare_outputs` → `AgreementMetrics` |
| Greedy decoding gives the same tokens, first difference per row | `assert_generation_equal` |
| Deployment parity: `torch.compile`, `torch.export` (dynamic batch), AOTInductor, TorchScript, ONNX Runtime, autocast | `assert_compile_parity`, `assert_export_parity`, `assert_aoti_parity`, `assert_torchscript_parity`, `assert_onnx_parity`, `assert_autocast_parity` |
| `torch.compile` health: one graph, no recompiles | `assert_no_graph_breaks`, `assert_no_recompiles`, `is_compile_supported` |
| Stateful and streaming models: chunked vs full, resets, request isolation, user-defined equivalence | `assert_streaming_equivalent`, `StreamingEquivalence` → `StreamingReport` |

**Check a release** · [dataset metrics](https://akj7.github.io/pytest-torch/guides/dataset-metrics/), [checkpoints](https://akj7.github.io/pytest-torch/guides/checkpoints/), [performance](https://akj7.github.io/pytest-torch/guides/performance/), [benchmarks](https://akj7.github.io/pytest-torch/guides/benchmarks/), [failure reproduction](https://akj7.github.io/pytest-torch/guides/repro/)

| Feature | API |
|---|---|
| Dataset metrics against recorded baselines that remember what they measured (context, data fingerprint, samples, metric config, checkpoint) | `metric_baseline` fixture → `MetricResult`; `--torch-update-metrics`, `--torch-metrics-strict`, `--torch-metrics-file` |
| Paired comparisons with bootstrap confidence intervals: noise on a small set isn't a regression | `paired_difference`, `evaluate`, `evaluate_detailed` |
| Checkpoints: save/reload preserves predictions; old checkpoints still load into new code | `assert_checkpoint_roundtrip`, `assert_checkpoint_loads` |
| Latency budgets and per-machine performance baselines, with peak memory on CUDA and XPU | `latency`, `perf_baseline` fixtures, `@pytest.mark.torch_max_latency`, `--torch-update-perf`, `--torch-perf-strict`, `--torch-perf-file`, `measure_latency`, `measure_peak_memory`, `check_perf` → `PerfResult`, `PerfStore` |
| Benchmark suites over batch sizes, dtypes and eager/`torch.compile`: median and p90 latency, throughput, peak memory, compile time, per-machine baselines, an end-of-run report and JSON | `BenchmarkSuite` → `BenchmarkResult`, `@pytest.mark.torch_benchmark`, `--torch-benchmark`, `--torch-benchmark-json`, `compare_perf` |
| Golden (snapshot) files for tensors and nested outputs | `golden` fixture, `@pytest.mark.torch_golden`, `--torch-update-golden` |
| Failure reproduction bundles: record a failing test's seed, inputs and environment, replay it anywhere | `--torch-repro-dir`, `--torch-repro`, `--torch-repro-max-weights-mb`, `record_repro`, `load_repro` |

**Training** · [training sanity](https://akj7.github.io/pytest-torch/guides/training/), [training loops](https://akj7.github.io/pytest-torch/guides/training-loops/)

| Feature | API |
|---|---|
| Finite gradients, every parameter learns, loss decreases, one batch overfits | `assert_gradients_finite`, `assert_all_params_receive_grad`, `assert_loss_decreases`, `assert_overfits_one_batch` → `TrainingRun` |
| Custom backward passes against finite differences | `assert_gradcheck` |
| Frozen parameters stay frozen through real optimizer steps, including BatchNorm statistics and weight decay | `assert_frozen` |
| Short seeded runs of your own training step (custom losses, several optimizers) with per-step telemetry: non-finite values, exploding gradients, skipped updates, unexpected learning rates | `training_runner` fixture, `run_training` → `TrainingRun` of `StepRecord`s, `TrainingSetup`, `standard_step` |
| Every trainable parameter is in exactly one optimizer, and the optimizer belongs to this copy of the model | `assert_optimizer_coverage` |
| Resuming from a checkpoint reproduces uninterrupted training, naming the missing optimizer, scheduler, scaler or RNG state | `assert_resume_equivalent` → `ResumeReport`, `save_training_state`, `load_training_state` |
| Gradient accumulation matches one full-batch step: loss scaling, gradient resets, step timing | `assert_accumulation_equivalent`, `split_batch` |
| Mixed-precision training uses the GradScaler correctly and doesn't keep overflowing | `assert_mixed_precision_training` |

**Determinism, configuration and CI** · [performance and determinism](https://akj7.github.io/pytest-torch/guides/performance/), [tolerances](https://akj7.github.io/pytest-torch/guides/tolerances/), [reference](https://akj7.github.io/pytest-torch/reference/cli/)

| Feature | API |
|---|---|
| Seeding with per-test seeds, printed on failure; deterministic algorithms | `torch_seed` fixture, `@pytest.mark.torch_seed`, `--torch-seed`, `--torch-seed-mode`, `--torch-autoseed`, `--torch-deterministic`, `seed_everything`, `deterministic_algorithms` |
| Nondeterminism diagnosis: RNG use vs nondeterministic kernels | `assert_deterministic`, `diagnose_nondeterminism` → `NondeterminismReport`, `measure_nondeterminism` → `VarianceResult` |
| Hidden host-device synchronizations (`.item()`, `.cpu()`, `if tensor:`) | `assert_no_host_sync`, `forbid_host_sync` |
| Project-wide tolerances per dtype, check and device | `[tool.pytest_torch.tolerance]`, `@pytest.mark.torch_tolerance`, `tolerance_overrides` |
| pytest-xdist: workers spread over GPUs, memory split, exclusive GPU tests | `--torch-xdist-gpus`, `--torch-gpu-memory-fraction`, `@pytest.mark.torch_gpu_exclusive`, `FileLock` |
| Mismatch heatmaps and GitHub Actions annotations for failing tests | `--torch-failure-artifacts`, `--torch-github-annotations`, `save_mismatch_heatmap`, `github_annotation` |
| Session-cached Hugging Face, timm and torchvision models | `model_loader` fixture, `ModelLoader` |
| Run a test body without autograd | `inference_mode` fixture |
| Zero-code testing of any model: Hugging Face, timm, torchvision or your own, with inferred inputs | `--torch-model`, `--torch-model-checks`, `--torch-model-inputs`, `--torch-model-pretrained` |
| Generated test suites: contract, compile, export, TorchScript, ONNX, AOTInductor, autocast, golden, output spec, checkpoint, layouts, frozen, training | `ModelSuite` |
| The generated suite once per config file (YAML, JSON, Hydra-style), with overrides to shrink each config | `ModelSuite.configs`, `config_overrides`, `model_config` fixture |

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
| Releases | [`test_dataset_metrics.py`](examples/test_dataset_metrics.py), [`test_checkpoints.py`](examples/test_checkpoints.py), [`test_benchmarks.py`](examples/test_benchmarks.py), [`test_failure_repro.py`](examples/test_failure_repro.py), [`test_tiny_classifier.py`](examples/test_tiny_classifier.py) |
| More | [`test_training_sanity.py`](examples/test_training_sanity.py), [`test_training_loops.py`](examples/test_training_loops.py), [`test_nondeterminism.py`](examples/test_nondeterminism.py) |

[`examples/integration/`](examples/integration/) has four small but realistic projects, laid
out like real repositories and tested end to end: a transformer
[text classifier](examples/integration/text_classification/), an
[image classifier with its deployment pipeline](examples/integration/image_classification/),
a [small GPT with a KV cache](examples/integration/language_model/), and a
[navigation policy tested once per YAML config](examples/integration/navigation_policy/)
(habitat-baselines style, with a frozen encoder and a transformer memory).

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
