# Performance and determinism

## Latency budgets

```python
@pytest.mark.torch_max_latency(ms=50)
def test_latency(model, make_tensor, latency):
    latency(model, make_tensor(32, 16))
```

## Performance regressions

```python
def test_speed(model, make_tensor, perf_baseline):
    perf_baseline(model, make_tensor(32, 64), latency_tolerance=0.3)
```

```bash
pytest --torch-update-perf     # record baselines once per machine
pytest                         # later runs compare against them
```

Baselines live in `.torch-perf-baselines.json`, keyed by test and **hardware fingerprint**,
so numbers from different machines never mix. A regression is
`new > baseline * (1 + tolerance) + slack`; faster is never a failure. A missing baseline
skips the test (`--torch-perf-strict` fails it instead). Peak memory is tracked on CUDA and
Intel XPU.

To benchmark a model over batch sizes, dtypes and `torch.compile`, with an end-of-run
report, see [Benchmark suites](benchmarks.md).

## Host-device syncs

```python
from pytest_torch import assert_no_host_sync

assert_no_host_sync(lambda: model(x_cuda))   # .item(), .cpu(), `if tensor:` raise (CUDA only)
```

## Nondeterminism

```python
from pytest_torch import assert_deterministic, diagnose_nondeterminism

assert_deterministic(model, x, runs=5)
print(diagnose_nondeterminism(model, x).verdict)
# Randomness-driven: the variance disappears when the RNG is reseeded before each run.
```

The diagnosis separates **RNG usage** from **kernel-level** variance and checks whether
`torch.use_deterministic_algorithms(True)` fixes it (`pytest --torch-deterministic` applies
it to `torch_seed`).

## Running in parallel (pytest-xdist)

```bash
pytest -n 4 --torch-xdist-gpus=0,1 --torch-gpu-memory-fraction=auto
```

- Workers are spread round-robin over GPUs (CUDA or Intel XPU); `torch_device` maps `"cuda"`
  (or `"xpu"`) to the worker's `cuda:N` (`xpu:N`) while test ids stay identical.
- `--torch-gpu-memory-fraction=auto` splits each GPU's memory between the workers sharing it.
- `@pytest.mark.torch_gpu_exclusive` serializes heavy tests per GPU across workers.
- Record performance baselines without parallel workers: timings taken while other workers
  share the machine are not a fair reference.

## Better failures in CI

```bash
pytest --torch-github-annotations --torch-failure-artifacts=failure-artifacts
```

A heatmap PNG of every tensor mismatch is saved and its path added to the error, and
GitHub Actions annotations mark failing tests on the pull request.
