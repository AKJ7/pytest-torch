# Markers

pytest-torch registers these markers, so `--strict-markers` accepts them; there is nothing
to import besides `pytest`. Every one of them is used in
[`examples/test_markers.py`](https://github.com/AKJ7/pytest-torch/blob/main/examples/test_markers.py).

| Purpose | Markers |
|---|---|
| [Seeds and golden files](#seeds-and-golden-files) | `torch_seed`, `torch_golden` |
| [Devices and dtypes](#devices-and-dtypes) | `torch_dtypes`, `torch_requires`, `torch_skip_on`, `torch_xfail_on` |
| [Modes for one test](#modes-for-one-test) | `torch_deterministic`, `torch_default_dtype`, `torch_matmul_precision`, `torch_no_host_sync`, `torch_detect_anomaly`, `torch_threads` |
| [Tolerances](#tolerances) | `torch_tolerance` |
| [Speed and memory](#speed-and-memory) | `torch_max_latency`, `torch_max_memory`, `torch_memory_leak`, `torch_benchmark` |
| [Parallel runs](#parallel-runs) | `torch_gpu_exclusive` |

The examples below assume `import pytest`, `import torch` and `from torch import nn`.

## Where markers go

A marker applies to a test, to every test of a class, or to a whole module:

```python
pytestmark = pytest.mark.torch_seed(1234)        # every test in the module


@pytest.mark.torch_deterministic                 # every test in the class
class TestTraining:
    @pytest.mark.torch_seed(0)                   # this test: overrides the module's seed
    def test_step(self, torch_seed): ...
```

The closest marker wins: a test's own overrides its class's, which overrides the module's.
Two markers combine instead: every `torch_requires` that applies must hold, and
`torch_tolerance` merges all levels, the closest winning per key.

## Seeds and golden files

### `torch_seed(n)`

```python
@pytest.mark.torch_seed(42)
def test_init(torch_seed, make_tensor):
    assert torch_seed == 42
```

The seed the `torch_seed` fixture (and through it `make_tensor` and `make_batch`) uses, instead
of `--torch-seed` or the per-test seed of `--torch-seed-mode=per-test`.

### `torch_golden(name)`

```python
@pytest.mark.torch_golden("classifier_v1")
def test_matches_golden(model, make_tensor, golden):
    golden.check(model(make_tensor(8, 16)), atol=1e-4)
```

The golden file the `golden` fixture uses: `__golden__/classifier_v1.pt` next to the test
file. Without the marker, the name is the test's; with it, several tests can share one file.

## Devices and dtypes

### `torch_dtypes(*dtypes)`

```python
@pytest.mark.torch_dtypes(torch.float32, torch.bfloat16)
def test_batch_invariance(torch_device, torch_dtype, make_tensor):
    ...
```

Runs a test that requests `torch_dtype` once per dtype: `[cpu-float32]`, `[cpu-bfloat16]`, ...
Without it, the dtypes come from `--torch-dtype`, else float32 only. Combinations a device
can't run are skipped with the reason.

### `torch_requires(**requirements)`

```python
@pytest.mark.torch_requires(cuda=True, cuda_capability=(8, 0), torch=">=2.4")
def test_flash_attention(): ...

@pytest.mark.torch_requires(packages=["onnxruntime"], python=">=3.10")
def test_onnx_export(): ...
```

```
SKIPPED [1] tests/test_attention.py:12: needs CUDA
```

Skips unless every requirement holds:

| Requirement | Holds when |
|---|---|
| `cuda=True`, `xpu=True`, `mps=True` | That device is available |
| `gpu=True` | CUDA, XPU or MPS is available |
| `torch=">=2.4"`, `python=">=3.10"` | The version matches the specifier (any [PEP 440](https://peps.python.org/pep-0440/) specifier, e.g. `">=2.2,<2.6"`) |
| `compile=True` | `torch.compile` works on this platform and Python version |
| `gpu_memory_gb=16` | The test's GPU (or the first one) has that much memory |
| `cuda_capability=(8, 0)` | The CUDA GPU's compute capability is at least that (Ampere for `(8, 0)`) |
| `packages=["onnxruntime"]` | Every package is importable (a string for one) |

Requirements are checked at collection, and the reason is reported at the test's location.
An unknown keyword stops the run, so a typo can't silently skip a test.

### `torch_skip_on(*devices, dtype=None, reason=...)`

```python
@pytest.mark.torch_skip_on("mps", reason="float64 isn't supported on MPS")
def test_float64(torch_device):
    ...
```

Skips this test's runs on these devices only; the other devices still run it. `"cuda"`
matches every CUDA device (`cuda:1` too), and `dtype=` (one or a list) narrows it to some
`torch_dtype` runs. The test must use the `torch_device` fixture.

### `torch_xfail_on(*devices, dtype=None, reason=..., raises=None, strict=True)`

```python
@pytest.mark.torch_xfail_on("mps", dtype=torch.bfloat16, reason="missing kernel", raises=NotImplementedError)
@pytest.mark.torch_dtypes(torch.float32, torch.bfloat16)
def test_layer(torch_device, torch_dtype):
    ...
```

```
XFAIL tests/test_layers.py::test_layer[mps-bfloat16] - missing kernel
```

An expected failure on these devices (and dtypes) only. It's strict by default: when a
torch release fixes the operation, the run reports `XPASS(strict)` as a failure, so the
marker doesn't outlive the bug. With `raises=`, any other error stays a real failure.

## Modes for one test

These markers set a mode for the test, function-scoped fixtures included (so a model built
in a fixture is affected), and restore the previous mode afterwards.

### `torch_deterministic`

```python
@pytest.mark.torch_deterministic
def test_reproducible_step(model, batch):
    ...
```

`torch.use_deterministic_algorithms(True)` during the test: a nondeterministic kernel raises
instead of varying silently. `--torch-deterministic` does the same for every test using
`torch_seed`.

### `torch_default_dtype(dtype)`

```python
@pytest.mark.torch_default_dtype(torch.float64)
def test_numerics():
    assert nn.Linear(2, 2).weight.dtype == torch.float64
```

`torch.set_default_dtype` during the test, e.g. float64 for numerical and gradient checks.
Takes a dtype or its name.

### `torch_matmul_precision(precision)`

```python
@pytest.mark.torch_matmul_precision("highest")
def test_exact_float32(model, make_tensor):
    ...
```

`torch.set_float32_matmul_precision` during the test: `"highest"`, `"high"` or `"medium"`.
`"highest"` also turns off TF32 in cuDNN convolutions. On Ampere GPUs and newer, TF32 is
the usual reason exact float32 comparisons fail on CUDA but pass on CPU.

### `torch_no_host_sync`

```python
@pytest.mark.torch_no_host_sync
def test_forward_never_syncs(model, make_tensor, torch_device):
    model(make_tensor(8, 16))
```

On CUDA, operations that synchronize the host with the GPU (`.item()`, `.cpu()`,
`if tensor:`) raise during the test, with the offending line in the
traceback. Such syncs stall the GPU pipeline in production. On other devices the test runs
normally and its report says it wasn't checked.

### `torch_detect_anomaly`

```python
@pytest.mark.torch_detect_anomaly
def test_gradients(model, batch):
    model(batch).sum().backward()
```

Autograd anomaly detection during the test: a NaN produced in the backward pass raises,
naming the backward function (e.g. `SqrtBackward0`) and showing where its forward
operation ran. It slows autograd down, so use it to investigate rather than everywhere.

### `torch_threads(n)`

```python
@pytest.mark.torch_threads(1)
def test_single_threaded_timing(latency, model, make_tensor):
    ...
```

Torch uses `n` CPU threads during the test. `--torch-num-threads` sets the count for the
whole run; under pytest-xdist its default divides the cores between the workers.

## Tolerances

### `torch_tolerance(**entries)`

```python
@pytest.mark.torch_tolerance(bfloat16=(5e-2, 1e-2), compile_parity={"atol": 1e-3})
def test_low_precision(model, make_tensor):
    ...

@pytest.mark.torch_tolerance(**{"cuda.float32": (1e-3, 1e-4)})   # device keys need **
class TestOnAmpere: ...
```

Default tolerances for comparisons that don't pass `rtol`/`atol`, with the keys of
`[tool.pytest_torch.tolerance]`: a dtype, a check, or either prefixed with a device. Values
are `(rtol, atol)` or a dict with either. See [Tolerances](../guides/tolerances.md).

## Speed and memory

### `torch_max_latency(ms=...)`

```python
@pytest.mark.torch_max_latency(ms=50)
def test_latency(model, make_tensor, latency):
    latency(model, make_tensor(32, 16))
```

The `latency` fixture fails when the median latency exceeds the budget.

### `torch_max_memory(mb=...)`

```python
@pytest.mark.torch_max_memory(mb=512)
def test_inference_fits(model, make_tensor):
    with torch.inference_mode():
        model(make_tensor(64, 3, 224, 224))
```

```
Failed: peak memory 812.4 MB exceeds the budget of 512 MB on cuda:0 (memory allocated during the test, above the 98.0 MB allocated before it)
```

Fails when the test body's peak memory, above what was allocated before it, exceeds `mb`.
Only CUDA and XPU keep the allocator statistics this needs; elsewhere the test runs
normally and its report says the budget wasn't checked.

### `torch_memory_leak(iterations=5)`

```python
HISTORY = []

@pytest.mark.torch_memory_leak
def test_training_step(model, batch):
    loss = model(batch).sum()
    HISTORY.append(loss)          # should be loss.item()
```

```
Memory leak: every run of the test leaves more behind
  live tensors after each run: 7, 10, 13, 16, 19, 22
  new: () float32 on cpu with an autograd graph (SumBackward0), held by a list of 6 item(s)
Hint: tensors that still hold their autograd graph are being kept: store loss.item() or loss.detach() instead of loss.
```

Runs the test body once to warm up (lazy initialization and caches may allocate once),
then `iterations` more times with the same fixtures. It fails when the number of live
tensors, or the device memory allocated, grows after every run, and describes the new
tensors: shape, dtype, device, autograd graph, and the container holding them. Counting
tensors works on every device. Since the body runs several times, keep its side effects
repeatable. With `--torch-repro-dir`, a failure bundle holds the inputs of one run.

### `torch_benchmark`

Marks a benchmark: deselected unless `--torch-benchmark` is given. Every
[`BenchmarkSuite`](../guides/benchmarks.md) carries it; mark your own slow measurements with
it to keep them out of normal runs.

## Parallel runs

### `torch_gpu_exclusive(timeout=1800)`

```python
@pytest.mark.torch_gpu_exclusive
def test_full_size_model(model_loader, torch_device): ...
```

Under pytest-xdist, only one such test runs at a time on each GPU, across workers. The
others wait for it; after `timeout` seconds a waiting test errors with a `TimeoutError`. Use it for tests that need most of a GPU's memory. See
[running in parallel](../guides/performance.md#running-in-parallel-pytest-xdist).
