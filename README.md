# pytest-torch

**Test PyTorch inference like you mean it.** `pytest-torch` is a pytest plugin that turns the
tedious, easy-to-get-wrong parts of model testing into one-liners: device parametrization,
reproducible inputs, tolerance-aware comparisons, invariance checks, deployment parity,
performance baselines and readable failures.

> **Status: alpha.** The API may still change. Requires Python 3.9+, pytest 7+ and PyTorch 2.0+.

```python
from pytest_torch import check_inference_contract

def test_model(model, make_tensor):
    check_inference_contract(model, make_tensor(8, 16))
```

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
| Model left in train mode, NaN/Inf, nondeterminism, batch-dependent output, mutated buffers | [`check_inference_contract`](#inference-contract) |
| CPU and GPU (or fp32 and bf16, or your port and the original) disagree: *where?* | [`localize_divergence`](#find-the-first-diverging-layer) |
| Padding or masked tokens leak into real tokens; a "set" model depends on order | [`assert_padding_invariant`, `assert_mask_invariant`, `assert_permutation_invariant`](#invariance-and-metamorphic-tests) |
| `torch.compile` / TorchScript / ONNX output differs from eager | [`assert_compile_parity`, `assert_torchscript_parity`, `assert_onnx_parity`](#parity-with-deployment-targets-and-references) |
| A quantized or distilled model is too inaccurate | [`assert_models_agree`](#parity-with-deployment-targets-and-references) |
| Your reimplementation differs from the reference; greedy decoding diverges | [`assert_matches_reference`, `assert_generation_equal`](#parity-with-deployment-targets-and-references) |
| Flaky GPU outputs | [`diagnose_nondeterminism`](#nondeterminism) |
| Latency or memory regressions | [`perf_baseline`](#performance-regressions) |
| Graph breaks, recompilations, hidden host-device syncs | [`assert_no_graph_breaks`, `assert_no_recompiles`, `assert_no_host_sync`](#torchcompile-health-and-host-device-syncs) |
| Inputs you didn't think of (zeros, extremes, odd shapes) | [Hypothesis strategies](#hypothesis-strategies) |

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
pytest                           # every test runs on cpu, cuda and mps, whichever exist
pytest --torch-device=cpu        # pin it
```

Model inputs can be a tensor, a tuple of positional arguments, or a dict of keyword arguments;
outputs can be any nested structure of dicts, tuples, dataclasses or Hugging Face `ModelOutput`s.

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
assert_torchscript_parity(model, x, mode="trace")       # or mode="script"
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
token per row.

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
the test (`--torch-perf-strict` fails it instead). Peak memory is tracked on CUDA.

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

- Workers are spread round-robin over GPUs. `torch_device` maps `"cuda"` to the worker's
  `cuda:N` while test ids stay identical (xdist requires that).
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
| `--torch-device` | Device(s) for `torch_device` (repeatable). Default: all available |
| `--torch-seed`, `--torch-seed-mode={fixed,per-test}`, `--torch-autoseed` | Seeding |
| `--torch-deterministic` | Use deterministic algorithms inside `torch_seed` |
| `--torch-update-golden` | Rewrite golden files |
| `--torch-update-perf`, `--torch-perf-strict`, `--torch-perf-file` | Performance baselines |
| `--torch-xdist-gpus`, `--torch-gpu-memory-fraction` | GPU handling under xdist |
| `--torch-failure-artifacts=DIR`, `--torch-github-annotations` | Failure output |

### Markers

`torch_seed(n)`, `torch_golden(name)`, `torch_max_latency(ms=...)`, `torch_gpu_exclusive(timeout=...)`.

## Good to know

- **TorchScript and ONNX** accept only positional tensor inputs; `trace` records the control
  flow of the inputs you pass. The ONNX exporter's defaults differ between torch versions, so
  pin them with `export_kwargs` (e.g. `{"dynamo": False, "opset_version": 17}`).
- **Left padding:** models with absolute position ids can legitimately differ; use
  `side="right"` or pass `position_ids`.
- **Peak memory and host-sync detection** are CUDA-only. Host-sync checks skip elsewhere.
- **Perf baselines under xdist:** record them without parallel workers.
- `torch.compile` checks use the `eager` backend by default, so no C++ compiler is required.

## Development

```bash
uv sync                          # environment, lockfile, dev dependencies
uv run pytest                    # the plugin's own tests
uv run pytest examples/ -v       # runnable examples of every feature
uv build                         # sdist and wheel
```

CI (`.github/workflows/ci.yml`) runs a matrix over operating systems, Python versions and torch
versions with CPU-only wheels, and uses the plugin's own annotations and failure artifacts.

## License

To be added before the first release.