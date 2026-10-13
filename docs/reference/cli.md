# Command line and markers

## Options

| Option | Purpose |
|---|---|
| `--torch-device` | Device(s) for `torch_device` (repeatable). Default: all available |
| `--torch-seed` | Base seed used by `torch_seed` (default 0) |
| `--torch-seed-mode={fixed,per-test}` | `per-test` derives a stable seed from the test id |
| `--torch-autoseed` | Seed before every test, even those that don't request `torch_seed` |
| `--torch-deterministic` | Use deterministic algorithms inside `torch_seed` |
| `--torch-update-golden` | Rewrite golden files instead of comparing |
| `--torch-update-perf` | Record performance baselines instead of comparing |
| `--torch-perf-strict` | Fail (instead of skip) when a performance baseline is missing |
| `--torch-perf-file` | Baseline file (default `<rootdir>/.torch-perf-baselines.json`) |
| `--torch-xdist-gpus` | Under xdist: `auto`, `none`, or GPUs such as `0,1` |
| `--torch-gpu-memory-fraction` | Per-process CUDA memory cap under xdist: `none`, `auto` or a float |
| `--torch-failure-artifacts=DIR` | Save a heatmap PNG for every tensor mismatch (needs matplotlib) |
| `--torch-github-annotations` | Print GitHub Actions `::error` annotations for failing tests |

## Markers

| Marker | Effect |
|---|---|
| `@pytest.mark.torch_seed(n)` | Seed used by `torch_seed` for this test |
| `@pytest.mark.torch_golden(name)` | Golden file name used by the `golden` fixture |
| `@pytest.mark.torch_max_latency(ms=...)` | Median latency budget enforced by `latency` |
| `@pytest.mark.torch_gpu_exclusive(timeout=1800)` | One such test at a time per GPU, across xdist workers |
