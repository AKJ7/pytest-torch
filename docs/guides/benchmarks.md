# Benchmark suites

A benchmark suite measures a model's latency, throughput and memory over a matrix of batch
sizes, dtypes and execution modes. It compares every case with this machine's baseline and
prints one table at the end of the run.

```python
import torch
from pytest_torch import BenchmarkSuite


class TestEncoderBenchmark(BenchmarkSuite):
    batch_sizes = (1, 8, 32)
    dtypes = (torch.float32, torch.bfloat16)
    modes = ("eager", "compile")

    def make_model(self):
        return TextEncoder()

    def make_inputs(self, batch_size, make_tensor):
        return {
            "input_ids": make_tensor(batch_size, 32, dtype=torch.long, low=0, high=1000),
            "attention_mask": make_tensor(batch_size, 32, dtype=torch.long, low=1, high=2),
        }
```

Each combination is one test on each device, e.g.
`test_benchmark[cpu-bfloat16-compile-b8]`. `make_inputs` returns a tensor, a tuple of
positional arguments or a dict of keyword arguments. The model and the floating-point inputs
are cast to each benchmarked dtype for you; integer inputs keep their dtype.

## Running

Benchmarks are deselected in a normal `pytest` run, so they never slow down the test suite:

```bash
pytest --torch-benchmark                          # measure and report
pytest --torch-benchmark --torch-update-perf      # record this machine's baselines
pytest --torch-benchmark                          # compare against them
pytest --torch-benchmark --torch-benchmark-json=bench.json   # also write the results as JSON
pytest --torch-benchmark -k Encoder --torch-device=cuda      # one suite, one device
```

```
============================ pytest-torch benchmarks =============================
suite                 device  dtype     mode     batch  median ms  p90 ms  samples/s  peak MB  compile s  vs baseline
TestEncoderBenchmark  cuda    float32   eager        8      1.003   1.014      7,975     41.2          -        -2.1%
TestEncoderBenchmark  cuda    float32   compile      8      0.531   0.545     15,055     38.9       0.63        +1.4%
TestEncoderBenchmark  cuda    bfloat16  eager        8      1.402   1.506      5,706     22.0          -  +38% REGRESSED
```

| Column | Meaning |
|---|---|
| median ms, p90 ms | Per-call latency, with warmup and device synchronization |
| samples/s | `batch / median` |
| peak MB | Peak memory of one call (CUDA and Intel XPU; `-` elsewhere) |
| compile s | Time of the first `torch.compile` call, including compilation |
| vs baseline | Change of the median against the recorded baseline; `recorded` when baselines were just written |

## Baselines and regressions

Baselines share the file and rules of [performance baselines](performance.md#performance-regressions):
`.torch-perf-baselines.json` (or `--torch-perf-file`), keyed by test, case and hardware
fingerprint, so numbers from different machines never mix. A case fails when
`median > baseline * (1 + latency_tolerance) + slack`, or when peak memory grows beyond
`memory_tolerance`. Faster is never a failure. A case without a baseline is measured and
reported, and passes; `--torch-perf-strict` fails it instead.

## Settings

| Attribute | Default | Meaning |
|---|---|---|
| `batch_sizes` | `(1, 8, 32)` | Positive integers |
| `dtypes` | `("float32",)` | Torch dtypes or their names; dtypes the device can't run are skipped |
| `modes` | `("eager",)` | `"eager"`, `"compile"`; compile is skipped where `torch.compile` doesn't work |
| `compile_options` | `{}` | Keyword arguments for `torch.compile`, e.g. `{"mode": "max-autotune"}` |
| `warmup`, `iters` | `5`, `20` | Calls before and during each measurement |
| `rounds` | `3` | Measurements per case; the fastest is kept |
| `latency_tolerance` | `0.25` | Allowed slowdown of the median |
| `memory_tolerance` | `0.10` | Allowed growth of peak memory |

An unknown mode or an invalid batch size raises `TypeError` when the class is defined.

## Noisy machines

Interference from other processes only ever slows a run down, so each case is measured
`rounds` times and the fastest round is kept. A round whose p90 exceeds twice its median is
marked `(noisy)` in the report, and recording a baseline from it warns you to re-record
on an idle machine. Record baselines without pytest-xdist workers, and on the hardware that
runs the comparisons.

## Using the results elsewhere

`--torch-benchmark-json` writes one object per case with the fields of `BenchmarkResult`:
`suite`, `device`, `hardware`, `dtype`, `mode`, `batch_size`, `median_ms`, `p90_ms`,
`min_ms`, `throughput`, `peak_memory_mb`, `compile_s`, `baseline_ms`, `status`
(`measured`, `baseline updated`, `regressed`) and `noisy`. Use it to chart results over
releases or to post them on a pull request.

To compare your own measurements with stored baselines outside a suite, call
`compare_perf(store, key, hardware, latency_ms, peak_memory_mb)`; it applies the same rules
and returns a `PerfResult`.
