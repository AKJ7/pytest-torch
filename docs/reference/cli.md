# Command line

All options are in the `pytest-torch` group of `pytest --help`. For markers, see
[Markers](markers.md).

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
| `--torch-benchmark` | Run [benchmark suites](../guides/benchmarks.md) (deselected otherwise) and print the benchmark report |
| `--torch-benchmark-json=PATH` | Also write the benchmark results as JSON |
| `--torch-xdist-gpus` | Under xdist: CUDA or XPU GPUs to spread workers over: `auto`, `none`, or e.g. `0,1` |
| `--torch-num-threads` | CPU threads per test process: an integer, `none`, or `auto` (default): under pytest-xdist, the cores divided between the workers unless `OMP_NUM_THREADS` or `MKL_NUM_THREADS` is set; otherwise torch's default |
| `--torch-gpu-memory-fraction` | Per-process CUDA memory cap under xdist: `none`, `auto` or a float |
| `--torch-update-metrics` | Record metric baselines (`metric_baseline`) instead of comparing |
| `--torch-metrics-strict` | Fail (instead of skip) when a metric baseline is missing |
| `--torch-metrics-file` | Metric baseline file (default `<rootdir>/.torch-metric-baselines.json`) |
| `--torch-model=SPEC` | Test a model with no test file (repeatable): `hf:NAME[@AutoClass]`, `hf-tiny:NAME[@AutoClass]`, `timm:NAME`, `torchvision:NAME`, `package.module:Factory`, `path/to/file.py:Factory` |
| `--torch-model-checks` | Checks for `--torch-model`: `contract`, `invariance`, `checkpoint`, `gradients` (default), `export`, `compile` |
| `--torch-model-inputs` | Inputs for `--torch-model` when they can't be inferred, e.g. `4x3x32x32` or `input_ids=long:4x16:1000,attention_mask=mask:4x16` |
| `--torch-model-pretrained` | Load pretrained weights for `--torch-model` (default: random init) |
| `--torch-repro-dir=DIR` | Write a reproduction bundle for every failing test |
| `--torch-repro=BUNDLE` | Replay a bundle: only its test, same seed and inputs |
| `--torch-repro-max-weights-mb` | Store weights passed to `record_repro` up to this size (default 0) |
| `--torch-failure-artifacts=DIR` | Save a heatmap PNG for every tensor mismatch (needs matplotlib) |
| `--torch-github-annotations` | Print GitHub Actions `::error` annotations for failing tests |
