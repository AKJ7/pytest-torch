# Command line and markers

## Options

| Option | Purpose |
|---|---|
| `--torch-dtype` | Dtype(s) for `torch_dtype` (repeatable): `float32`, `bfloat16`, `float16`, `float64`. Default: float32 |
| `--torch-device` | Device(s) for `torch_device` (repeatable), e.g. `cpu`, `cuda`, `xpu`, `mps`, `cuda:1`. Default: all available. The contract's `device_parity` then compares only among these devices |
| `--torch-seed` | Base seed used by `torch_seed` (default 0) |
| `--torch-seed-mode={fixed,per-test}` | `per-test` derives a stable seed from the test id |
| `--torch-autoseed` | Seed before every test, even those that don't request `torch_seed` |
| `--torch-deterministic` | Use deterministic algorithms inside `torch_seed` |
| `--torch-update-golden` | Rewrite golden files instead of comparing |
| `--torch-update-perf` | Record performance baselines instead of comparing |
| `--torch-perf-strict` | Fail (instead of skip) when a performance baseline is missing |
| `--torch-perf-file` | Baseline file (default `<rootdir>/.torch-perf-baselines.json`) |
| `--torch-xdist-gpus` | Under xdist: CUDA or XPU GPUs to spread workers over: `auto`, `none`, or e.g. `0,1` |
| `--torch-gpu-memory-fraction` | Per-process CUDA memory cap under xdist: `none`, `auto` or a float |
| `--torch-update-metrics` | Record metric baselines (`metric_baseline`) instead of comparing |
| `--torch-metrics-strict` | Fail (instead of skip) when a metric baseline is missing |
| `--torch-metrics-file` | Metric baseline file (default `<rootdir>/.torch-metric-baselines.json`) |
| `--torch-repro-dir=DIR` | Write a reproduction bundle for every failing test |
| `--torch-repro=BUNDLE` | Replay a bundle: only its test, same seed and inputs |
| `--torch-repro-max-weights-mb` | Store weights passed to `record_repro` up to this size (default 0) |
| `--torch-failure-artifacts=DIR` | Save a heatmap PNG for every tensor mismatch (needs matplotlib) |
| `--torch-github-annotations` | Print GitHub Actions `::error` annotations for failing tests |

## Markers

| Marker | Effect |
|---|---|
| `@pytest.mark.torch_seed(n)` | Seed used by `torch_seed` for this test |
| `@pytest.mark.torch_golden(name)` | Golden file name used by the `golden` fixture |
| `@pytest.mark.torch_max_latency(ms=...)` | Median latency budget enforced by `latency` |
| `@pytest.mark.torch_gpu_exclusive(timeout=1800)` | One such test at a time per GPU, across xdist workers |
| `@pytest.mark.torch_dtypes(*dtypes)` | Dtypes the `torch_dtype` fixture runs this test with |
| `@pytest.mark.torch_tolerance(**entries)` | Default tolerances for this test, class or module; see [Tolerances](../guides/tolerances.md) |
