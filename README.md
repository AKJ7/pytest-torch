<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/AKJ7/pytest-torch/main/docs/assets/logo-dark.png">
    <img alt="pytest-torch" src="https://raw.githubusercontent.com/AKJ7/pytest-torch/main/docs/assets/logo.png" width="520">
  </picture>
</h1>

**Test PyTorch models like you mean it.** `pytest-torch` is a pytest plugin that turns the
tedious, easy-to-get-wrong parts of model testing into one-liners: device parametrization,
reproducible inputs, tolerance-aware comparisons, invariance checks, deployment parity,
training sanity checks, performance baselines and readable failures.

> **Status: alpha.** The API may still change. Requires Python 3.9+, pytest 7+ and PyTorch 2.0+.
> **Documentation:** <https://akj7.github.io/pytest-torch/>

```python
from pytest_torch import ModelSuite

class TestClassifier(ModelSuite):           # a whole test suite from one class
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

## A bug it catches

A classification head that averages over *all* positions, padding included, looks fine on a
single unpadded example and silently degrades in production:

```python
from pytest_torch import assert_mask_invariant

def test_padding_does_not_leak(model, batch):
    # batch = {"input_ids": ..., "attention_mask": ...}
    assert_mask_invariant(model, batch)
```

```
AssertionError: Mask invariance violated (masked tokens leak): 1 problem(s)
  output['pooled']: 11/24 elements differ beyond tolerance (rtol=1.3e-06, atol=1e-05)
    max abs error: 2.1e-01 at index (1, 5) (actual=..., expected=...)
    max rel error: ...
```
*(abridged, numbers illustrative)*. The failure names the exact output (`output['pooled']`),
where it is worst, and how bad it is, for any nested output structure.

## At a glance

| You want to catch... | Use |
|---|---|
| All of the below for one model, without writing the tests | [`ModelSuite`](#generated-test-suites) |
| Model left in train mode, NaN/Inf, nondeterminism, batch-dependent output, mutated buffers | [`check_inference_contract`](#inference-contract) |
| CPU and GPU (or fp32 and bf16, or your port and the original) disagree: *where?* | [`localize_divergence`](#find-the-first-diverging-layer) |
| Padding or masked tokens leak into real tokens; a "set" model depends on order | [`assert_padding_invariant`, `assert_mask_invariant`, `assert_permutation_invariant`](#invariance-and-metamorphic-tests) |
| `torch.compile` / `torch.export` / AOTInductor / TorchScript / ONNX output differs from eager | [`assert_compile_parity`, `assert_export_parity`, `assert_aoti_parity`, ...](#parity-with-deployment-targets-and-references) |
| Mixed precision (`torch.autocast`) drifts too far from fp32 | [`assert_autocast_parity`](#parity-with-deployment-targets-and-references) |
| Dead layers, NaN gradients, a wrong custom backward, a model that cannot learn | [`assert_all_params_receive_grad`, `assert_gradcheck`, `assert_overfits_one_batch`, ...](#training-sanity) |
| A quantized or distilled model is too inaccurate | [`assert_models_agree`](#parity-with-deployment-targets-and-references) |
| Your reimplementation differs from the reference; greedy decoding diverges | [`assert_matches_reference`, `assert_generation_equal`](#parity-with-deployment-targets-and-references) |
| Flaky GPU outputs | [`diagnose_nondeterminism`](#nondeterminism) |
| Latency or memory regressions | [`perf_baseline`](#performance-regressions) |
| Graph breaks, recompilations, hidden host-device syncs | [`assert_no_graph_breaks`, `assert_no_recompiles`, `assert_no_host_sync`](#torchcompile-health-and-host-device-syncs) |
| Inputs you didn't think of (zeros, extremes, odd shapes, ragged padding, odd memory layouts) | [Hypothesis strategies](#hypothesis-strategies), [`assert_layout_invariant`](#output-specs-layouts-and-the-dtype-matrix) |
| Wrong output shapes, dtypes, ranges, probabilities that don't sum to 1 | [`assert_output_spec`](#output-specs-layouts-and-the-dtype-matrix) |
| A checkpoint that doesn't reload the same model; yesterday's checkpoint no longer loading | [`assert_checkpoint_roundtrip`, `assert_checkpoint_loads`](#checkpoints) |
| Accuracy (or any dataset metric) silently dropping | [`metric_baseline`](#dataset-metrics) |
| A failure you can't reproduce locally | [`--torch-repro-dir`, `--torch-repro`](#failure-reproduction) |

## Examples

Runnable, self-contained test files in [`examples/`](examples/) (`pytest examples/ -v`). Most
have a "Try ..." comment showing a one-line bug that makes the test fail.

| File | Shows |
|---|---|
| [`test_model_suite.py`](examples/test_model_suite.py) | `ModelSuite`: contract, compile, export with dynamic batch, autocast and training checks from one class |
| [`test_markers.py`](examples/test_markers.py) | Every marker (`torch_seed`, `torch_golden`, `torch_max_latency`, `torch_gpu_exclusive`, `torch_tolerance`) on a test, a class and a module, and combined |
| [`test_tiny_classifier.py`](examples/test_tiny_classifier.py) | Inference contract, golden files, latency budget, CPU vs GPU layer diff |
| [`test_metamorphic.py`](examples/test_metamorphic.py) | Point-cloud order invariance, token-tag equivariance, flip/rotation invariance, scaling relations |
| [`test_deployment.py`](examples/test_deployment.py) | `torch.compile`, `torch.export`, AOTInductor, TorchScript and ONNX parity; a pruned model's accuracy budget; bf16 and autocast |
| [`test_output_specs.py`](examples/test_output_specs.py) | Shapes with symbolic dims, dtypes, ranges and probabilities for a token tagger |
| [`test_checkpoints.py`](examples/test_checkpoints.py) | Save/reload round-trip with tied weights; a v1 checkpoint loading into renamed v2 code |
| [`test_unusual_inputs.py`](examples/test_unusual_inputs.py) | Memory-layout invariance, ragged padding with Hypothesis, non-contiguous edge-case tensors |
| [`test_dataset_metrics.py`](examples/test_dataset_metrics.py) | Accuracy and loss baselines with thresholds, macro F1 with `update`/`compute` |
| [`test_dtype_matrix.py`](examples/test_dtype_matrix.py) | The same tests across devices and dtypes, with dtype-aware tolerances |
| [`test_failure_repro.py`](examples/test_failure_repro.py) | Recording and replaying failure bundles |
| [`test_training_sanity.py`](examples/test_training_sanity.py) | A two-tower model: finite gradients, every parameter learns, loss decreases, one batch overfits; gradcheck of a custom backward |
| [`test_reimplementation.py`](examples/test_reimplementation.py) | Hand-written attention vs `nn.MultiheadAttention`, BatchNorm folding, a renamed refactor checked layer by layer, incremental vs full greedy decoding |
| [`test_property_based.py`](examples/test_property_based.py) | Hypothesis edge cases: softmax overflow, zero-variance normalization, NaN inputs |
| [`test_nondeterminism.py`](examples/test_nondeterminism.py) | Bit-exact eval, a forgotten `.eval()` caught by the contract, MC dropout diagnosed as RNG-driven |

## Installation

```bash
pip install pytest-torch          # or: uv add --dev pytest-torch
```

PyTorch is not installed for you (pick the build that matches your hardware). Optional extras:

| Extra | Adds | Needed for |
|---|---|---|
| `hypothesis` | `hypothesis` | `pytest_torch.strategies` |
| `onnx` | `onnx`, `onnxruntime`, `onnxscript` | `assert_onnx_parity` |
| `hf` | `transformers` | `ModelLoader.hf`, `assert_generation_equal` with HF models |
| `timm`, `torchvision` | the respective package | `ModelLoader.timm`, `ModelLoader.torchvision` |
| `plots` | `matplotlib` | mismatch heatmaps (`--torch-failure-artifacts`) |

The plugin registers itself through the `pytest11` entry point; there is nothing to configure.
Importing it never imports `torch`, so projects that don't test models pay nothing.

## Quick start

```python
# tests/test_model.py
import pytest
from pytest_torch import check_inference_contract, assert_batch_invariant

@pytest.fixture
def model(torch_seed, torch_device):
    return MyModel().to(torch_device).eval()

def test_contract(model, make_tensor):
    check_inference_contract(model, make_tensor(8, 16))

def test_batch_invariance(model, make_tensor):
    assert_batch_invariant(model, make_tensor(8, 16))
```

```bash
pytest                           # every test runs on cpu, cuda, xpu and mps, whichever exist
pytest --torch-device=cpu        # pin it
```

Model inputs can be a tensor, a tuple of positional arguments, or a dict of keyword arguments;
outputs can be any nested structure of dicts, tuples, dataclasses or Hugging Face `ModelOutput`s.

## Generated test suites

Subclass `ModelSuite`, describe the model and its inputs, and pick the checks. Each check
becomes a test, on every device:

| Check | Generated tests |
|---|---|
| `contract` *(default)* | `test_inference_contract` |
| `compile` *(default)*, `export` *(default)*, `torchscript`, `onnx`, `aoti`, `autocast` | `test_<name>_parity` |
| `golden` | `test_golden` (file `__golden__/<ClassName>.pt`) |
| `training` (needs `loss()`) | `test_gradients_finite`, `test_all_params_receive_grad`, `test_loss_decreases`, `test_overfits_one_batch` |

```python
class TestTextClassifier(ModelSuite):
    checks = ("contract", "export", "autocast", "training")
    dynamic_batch = True                                   # exported program must accept other batch sizes
    check_options = {                                      # kwargs for the assertion behind a test
        "inference_contract": {"dtypes": (torch.bfloat16,)},
        "overfits_one_batch": {"steps": 300},
    }
    ...
```

Checks you don't list don't exist (no wall of skips); generated tests are ordinary methods
you can override or add to. Configuration mistakes fail at collection with a clear message.

## Tolerances

Set default tolerances once, for every comparison that doesn't pass `rtol`/`atol` itself:

```toml
# pyproject.toml
[tool.pytest_torch.tolerance]
bfloat16 = { rtol = 2e-2, atol = 1e-2 }          # any comparison of bf16 tensors
"cuda.float32" = { rtol = 1e-4, atol = 1e-5 }    # only when testing on CUDA (TF32)
compile_parity = { rtol = 1e-3, atol = 1e-3 }    # default of assert_compile_parity
```

Keys are a dtype, a check (`compile_parity`, `export_parity`, `aoti_parity`,
`torchscript_parity`, `onnx_parity`, `autocast_parity`, `device_parity`, `dtype_parity`,
`divergence`, `gradcheck`), or either prefixed with a device. For one test, class or module:
`@pytest.mark.torch_tolerance(bfloat16=(5e-2, 1e-2))`; for a block of code:
`with tolerance_overrides(...)`. An explicit argument always wins, then the marker, then
`pyproject.toml`, then the built-in default. Failure messages print the tolerance that was used.

## Fixtures

| Fixture | What it gives you |
|---|---|
| `torch_device` | Parametrizes the test over the available devices (skips unavailable ones) |
| `torch_seed` | Seeds `random`, `numpy` and `torch`; yields the seed. It is printed in the report of a failing test |
| `torch_generator` | A seeded **CPU** `Generator`, so the same seed gives identical values on every device |
| `make_tensor` | Reproducible tensors, generated on the CPU and moved to the device |
| `inference_mode` | Runs the test under `torch.inference_mode()` |
| `golden` | Snapshot testing of tensors and nested outputs |
| `latency` | Median latency with warmup and device sync, with an optional budget marker |
| `perf_baseline` | Latency/memory regression checks against a stored baseline |
| `model_loader` | Session-cached Hugging Face / timm / torchvision loading |
| `torch_dtype` | Parametrizes the test over dtypes (default float32; `--torch-dtype`, `@pytest.mark.torch_dtypes`) |
| `metric_baseline` | Dataset metrics compared with a stored baseline and thresholds |

### Reproducible inputs

```python
def test_logits(model, make_tensor):
    x = make_tensor(4, 16)                                   # normal(0, 1), float32
    ids = make_tensor(4, 8, dtype=torch.long, low=1, high=50)
    u = make_tensor(4, 16, kind="uniform", low=0, high=1)
```

Seed priority: `@pytest.mark.torch_seed(n)` > `--torch-seed-mode=per-test` (a stable seed derived
from the test id, so tests are independent but reproducible) > `--torch-seed`. Add
`--torch-autoseed` to seed before every test, even those that don't request the fixture.

### Golden files

```python
@pytest.mark.torch_golden("classifier_v1")
def test_matches_golden(model, make_tensor, golden):
    golden.check(model(make_tensor(8, 16)), atol=1e-4)
```

```bash
pytest --torch-update-golden --torch-device=cpu    # (re)generate a portable reference on CPU
```

A missing golden file **fails** (CI must never pass vacuously). Goldens store nested outputs
by path and record the torch version, which is shown when a comparison fails.

### Latency budgets

```python
@pytest.mark.torch_max_latency(ms=50)
def test_latency(model, make_tensor, latency):
    latency(model, make_tensor(32, 16))
```

## Assertions

```python
from pytest_torch import (
    assert_tensors_close, assert_outputs_close, assert_finite, assert_batch_invariant,
)

assert_tensors_close(actual, expected, rtol=1e-4)      # dtype-aware default tolerances
assert_outputs_close(out, reference_out)               # nested outputs, reports the path of each mismatch
assert_finite(out, "output")                           # NaN/Inf anywhere in the structure
assert_batch_invariant(model, batch)                   # model(batch)[i] == model(batch[i:i+1])[0]
```

Failures say how many elements differ, where the worst one is, and the absolute and relative
error, instead of a bare `False`.

## Inference contract

One call runs the checks every inference model should satisfy and explains failures:

```python
report = check_inference_contract(model, x, raise_on_failure=False)
print(report)
```

```
Inference contract: 4 passed, 1 failed, 2 skipped
  [PASS] eval_mode
  [PASS] finite_outputs
  [PASS] deterministic
  [FAIL] batch_invariance
         per-sample results differ from the batched run
         ...
         hint: Output depends on batch composition: check batch norm in training mode, ...
  [SKIP] device_parity (no other device available)
  [SKIP] dtype_parity (no dtypes requested, e.g. dtypes=(torch.bfloat16,))
  [PASS] no_state_mutation
```

Checks: `eval_mode`, `finite_outputs`, `deterministic`, `batch_invariance`, `device_parity`,
`dtype_parity` (opt-in via `dtypes=(torch.bfloat16,)`) and `no_state_mutation`. Select with
`checks=[...]` / `skip=[...]`; tune tolerances with `rtol`, `device_rtol`, `dtype_rtol`, ...
`device_parity` compares against the other available devices, or only the other ones pinned
with `--torch-device` (`--torch-device=cpu` keeps it off every GPU).

## Find the first diverging layer

When two models disagree, *which layer* is the culprit?

```python
from pytest_torch import localize_divergence

report = localize_divergence(model_cpu, model_gpu, x)
report.raise_if_diverged()
```

```
Divergence: 3 of 12 layers exceed tolerance (rtol=0.0001, atol=0.0001). First: 'encoder.1' (LayerNorm)
  layer       type        max abs   max rel  inputs
  encoder.1   LayerNorm  3.2e-02   1.1e-01  ok
  ...
The first diverging layer receives matching inputs: the error originates inside it.
```

It records every layer's inputs and outputs with forward hooks and compares them in execution
order. `inputs=ok` means the error started *inside* that layer; `inputs=DIFF` means it entered
between layers (a functional op or residual add). Use `module_map` when layer names differ.

## Invariance and metamorphic tests

```python
from pytest_torch import (
    assert_padding_invariant, assert_mask_invariant,
    assert_permutation_invariant, assert_permutation_equivariant,
    assert_invariant, assert_equivariant,
)

assert_padding_invariant(model, batch, extra=4, side="right")  # padding must not change real tokens
assert_mask_invariant(model, batch)                            # masked ids must not leak into the rest
assert_permutation_invariant(set_model, points, dim=1)         # order of a set must not matter
assert_permutation_equivariant(per_token_model, tokens)        # permuted in -> permuted out
assert_invariant(model, x, lambda t: -t, name="sign flip")     # any custom relation
```

`batch` is a dict like `{"input_ids": ..., "attention_mask": ...}`. Sequence-aligned outputs are
compared only on real tokens; pooled outputs are compared as is.

## Parity with deployment targets and references

```python
from pytest_torch import (
    assert_compile_parity, assert_torchscript_parity, assert_onnx_parity,
    assert_models_agree, assert_matches_reference, assert_generation_equal,
)

assert_compile_parity(model, x, backend="eager")        # torch.compile vs eager, run twice
assert_export_parity(model, x, x_other_batch, dynamic_batch=True)  # torch.export, one program for all inputs
assert_aoti_parity(model, x)                            # AOTInductor .pt2 package (torch 2.6+, needs a C++ compiler)
assert_autocast_parity(model, x, dtype=torch.bfloat16)  # mixed precision vs fp32
assert_torchscript_parity(model, x, mode="trace")       # or mode="script" (TorchScript is deprecated upstream)
assert_onnx_parity(model, x)                            # export, run in ONNX Runtime, compare

# fp32 vs quantized / pruned / distilled, with an accuracy budget:
metrics = assert_models_agree(fp32, quantized, x, min_top1=0.99, min_cosine=0.999)

# your port vs the official implementation:
assert_matches_reference(my_bert, hf_bert, batch, select=lambda o: o.last_hidden_state)

# greedy decoding must produce identical tokens:
assert_generation_equal(my_gpt, hf_gpt, input_ids, max_new_tokens=16, pad_token_id=0)
```

`assert_models_agree` returns the measured metrics (max/mean abs error, cosine similarity,
top-1 agreement) so you can log them. `assert_generation_equal` reports the first diverging
token per row. If `torch.export` fails, the assertion carries torch's reason and a hint.

## Training sanity

Every helper takes `(model, inputs, loss_fn)` with `loss_fn(output) -> scalar`; helpers that
take optimizer steps train a deep copy, so your model is never modified.

```python
from pytest_torch import (
    assert_gradients_finite, assert_all_params_receive_grad,
    assert_loss_decreases, assert_overfits_one_batch, assert_gradcheck,
)

loss_fn = lambda out: F.cross_entropy(out, labels)
assert_gradients_finite(model, x, loss_fn)            # NaN/Inf in loss or any gradient, per parameter
assert_all_params_receive_grad(model, x, loss_fn)     # unused layers, .detach() in forward, ignored heads
assert_loss_decreases(model, x, loss_fn, steps=20)    # the training loop is wired up
assert_overfits_one_batch(model, x, loss_fn)          # the model can memorize one batch
assert_gradcheck(MyFunction.apply, small_x)           # custom backward vs finite differences
```

```
AssertionError: 4 trainable parameter(s) receive no useful gradient:
  doc.0.weight: no gradient (not connected to the loss)
  ...
```

## Output specs, layouts and the dtype matrix

```python
from pytest_torch import Probabilities, TensorSpec, assert_output_spec, assert_layout_invariant

spec = {"logits": TensorSpec(("B", "T", 5), dtype=torch.float32), "probs": Probabilities(("B", "T", 5))}
assert_output_spec(model(ids), spec, B=4)      # symbolic dims, dtypes, finite, ranges, sums to 1
assert_layout_invariant(model, images)         # contiguous == non-contiguous == channels_last

@pytest.mark.torch_dtypes(torch.float32, torch.bfloat16)
def test_any_precision(torch_device, torch_dtype, make_tensor):   # [cpu-float32], [cpu-bfloat16], ...
    x = make_tensor(8, 16)                     # follows torch_dtype; tolerances follow the dtype too
```

Every spec problem is reported with its output path. Device/dtype combinations a device
can't run are skipped. `ModelSuite` has matching checks: `output_spec`, `layouts`.

## Checkpoints

```python
from pytest_torch import assert_checkpoint_roundtrip, assert_checkpoint_loads

assert_checkpoint_roundtrip(make_model, inputs)        # save, reload into a fresh model, same outputs
assert_checkpoint_loads(make_model, "tests/data/v1.pt") # yesterday's checkpoint, today's code
```

The round-trip rebuilds the model under a different seed, so anything not restored shows up,
and names the culprit (plain tensor attributes, non-persistent buffers, tied weights that came
apart). The compatibility check lists every missing, unexpected and reshaped key at once,
runs the model's own compatibility shims, and suggests `strip_prefix="module."` for DDP.

## Dataset metrics

```python
def test_accuracy(model, val_loader, metric_baseline):
    metric_baseline(model, val_loader, {"accuracy": accuracy}, minimum={"accuracy": 0.9})
```

Record once with `--torch-update-metrics` (`.torch-metric-baselines.json`), then a metric
that gets worse than its baseline by more than `tolerance` fails; thresholds always apply.
Metrics are averaged weighted by batch size, or use `update`/`compute` objects (F1, AUC).
`evaluate(model, loader, metrics)` is the same loop without a baseline.

## Failure reproduction

```bash
pytest --torch-repro-dir=repro                        # failing tests write repro/<test>.pt
pytest --torch-repro=repro/<test>.pt                  # replay: same test, seed and inputs
pytest --torch-repro=repro/<test>.pt --torch-device=cpu
```

A bundle holds the test id and parameters, seed, device, every `make_tensor` output
(replayed bit for bit), values passed to `record_repro(name, value)`, tolerances and the
environment; weights only up to `--torch-repro-max-weights-mb`. Upload the directory as a
CI artifact and replay the CI failure locally.

## Hypothesis strategies

```python
from hypothesis import given, settings
from pytest_torch.strategies import tensors, shapes

MODEL = build_model().eval()

@settings(max_examples=50, deadline=None)
@given(x=tensors(shape=(4, 16)))              # zeros, constants, extremes, tiny values, ...
def test_finite_on_edge_cases(x):
    assert torch.isfinite(MODEL(x)).all()
```

`tensors(...)` takes `dtype`, `low`/`high`, `edge_cases=True`, and `allow_nan`/`allow_inf` to
inject non-finite values. Build models at module scope: Hypothesis rejects function-scoped
fixtures in `@given` tests by default.

## Model zoo

```python
def test_bert(model_loader):                                  # session-cached
    model = model_loader.hf("bert-base-uncased", auto_class="AutoModel")
    resnet = model_loader.torchvision("resnet18")             # random init, no download
    vit = model_loader.timm("vit_tiny_patch16_224")
```

Models come back in eval mode and are cached by every argument. Cached instances are shared:
`copy.deepcopy` before mutating. Pretrained weights are downloaded on first use.

## Performance regressions

```python
def test_speed(model, make_tensor, perf_baseline):
    perf_baseline(model, make_tensor(32, 64), latency_tolerance=0.3)
```

```bash
pytest --torch-update-perf     # record baselines once per machine
pytest                         # later runs compare against them
```

Baselines live in `.torch-perf-baselines.json`, keyed by test and **hardware fingerprint**, so
numbers from different machines never mix. A regression is
`new > baseline * (1 + tolerance) + slack`; faster is never a failure. A missing baseline skips
the test (`--torch-perf-strict` fails it instead). Peak memory is tracked on CUDA and XPU.

## `torch.compile` health and host-device syncs

```python
from pytest_torch import assert_no_graph_breaks, assert_no_recompiles, assert_no_host_sync

assert_no_graph_breaks(model, x)                        # one graph, with torch's reason if not
assert_no_recompiles(model, x)                          # repeated calls hit the cache
assert_no_recompiles(model, x, x_bigger, max_recompiles=1)   # a new shape costs one recompile
assert_no_host_sync(lambda: model(x_cuda))              # .item(), .cpu(), `if tensor:` raise (CUDA)
```

## Nondeterminism

```python
from pytest_torch import assert_deterministic, diagnose_nondeterminism

assert_deterministic(model, x, runs=5)

print(diagnose_nondeterminism(model, x).verdict)
# Randomness-driven: the variance disappears when the RNG is reseeded before each run.
# Look for dropout in training mode, sampling, or noise inside forward().
```

The diagnosis separates **RNG usage** from **kernel-level** variance (e.g. atomic-add
reductions) and checks whether `torch.use_deterministic_algorithms(True)` fixes it
(`pytest --torch-deterministic` applies it to `torch_seed`).

## Running in parallel (pytest-xdist)

```bash
pytest -n 4 --torch-xdist-gpus=0,1 --torch-gpu-memory-fraction=auto
```

- Workers are spread round-robin over GPUs (CUDA or Intel XPU). `torch_device` maps `"cuda"`
  (or `"xpu"`) to the worker's `cuda:N` (`xpu:N`) while test ids stay identical (xdist requires that).
- `--torch-gpu-memory-fraction=auto` splits each GPU's memory between the workers sharing it.
- `@pytest.mark.torch_gpu_exclusive` serializes heavy tests per GPU across workers.

## Better failures

```bash
pytest --torch-github-annotations --torch-failure-artifacts=failure-artifacts
```

- A **heatmap PNG** of every tensor mismatch is saved (red = outside tolerance, large tensors
  are max-pooled so a single bad element stays visible) and its path is added to the error.
- **GitHub Actions annotations** (`::error file=...,line=...`) mark failing tests on the PR.

## Reference

### Command-line options

| Option | Purpose |
|---|---|
| `--torch-device` | Device(s) for `torch_device` (repeatable). Default: all available. Also limits the contract's `device_parity` |
| `--torch-seed`, `--torch-seed-mode={fixed,per-test}`, `--torch-autoseed` | Seeding |
| `--torch-deterministic` | Use deterministic algorithms inside `torch_seed` |
| `--torch-update-golden` | Rewrite golden files |
| `--torch-update-perf`, `--torch-perf-strict`, `--torch-perf-file` | Performance baselines |
| `--torch-xdist-gpus`, `--torch-gpu-memory-fraction` | GPU handling under xdist |
| `--torch-dtype` | Dtype(s) for `torch_dtype` (repeatable). Default: float32 |
| `--torch-update-metrics`, `--torch-metrics-strict`, `--torch-metrics-file` | Dataset metric baselines |
| `--torch-repro-dir=DIR`, `--torch-repro=BUNDLE`, `--torch-repro-max-weights-mb` | Failure reproduction bundles |
| `--torch-failure-artifacts=DIR`, `--torch-github-annotations` | Failure output |

### Markers

`torch_seed(n)`, `torch_golden(name)`, `torch_max_latency(ms=...)`, `torch_gpu_exclusive(timeout=...)`,
`torch_tolerance(**entries)`, `torch_dtypes(*dtypes)`.

Skip compile/export tests on platforms without dynamo with
`pytest.mark.skipif(not is_compile_supported(), ...)`; `ModelSuite` does it for you.

## Good to know

- **TorchScript and ONNX** accept only positional tensor inputs; `trace` records the control
  flow of the inputs you pass. The ONNX exporter's defaults differ between torch versions, so
  pin them with `export_kwargs` (e.g. `{"dynamo": False, "opset_version": 17}`).
- **Left padding:** models with absolute position ids can legitimately differ; use
  `side="right"` or pass `position_ids`.
- **Devices:** `cpu`, `cuda`, `xpu` (Intel GPUs, through `torch.xpu`) and `mps`. Peak memory is
  tracked on CUDA and XPU (XPU memory stats need a recent torch). Host-sync detection is
  CUDA-only and skips elsewhere.
- **Perf baselines under xdist:** record them without parallel workers.
- `torch.compile` checks use the `eager` backend by default, so no C++ compiler is required.

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
versions and torch versions with CPU-only wheels, plus the slow tests once on the newest torch,
and uses the plugin's own annotations and failure artifacts. `docs.yml` builds the docs on
every PR and publishes them from `main`; `release.yml` publishes to PyPI from a GitHub release.

## License

[MIT](LICENSE)