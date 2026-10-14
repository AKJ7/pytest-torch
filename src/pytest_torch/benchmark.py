"""Benchmark suites: a model's latency, throughput and memory across a matrix of batch
sizes, dtypes and execution modes, with per-machine baselines and an end-of-run report.

    class TestResNetBenchmark(BenchmarkSuite):
        batch_sizes = (1, 8, 32)
        dtypes = (torch.float32, torch.bfloat16)
        modes = ("eager", "compile")

        def make_model(self):
            return ResNet()

        def make_inputs(self, batch_size, make_tensor):
            return make_tensor(batch_size, 3, 224, 224)

    pytest --torch-benchmark                       # run benchmarks (skipped otherwise)
    pytest --torch-benchmark --torch-update-perf   # record this machine's baselines
    pytest --torch-benchmark --torch-benchmark-json=bench.json
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

import pytest

MODES = ("eager", "compile")
NOISY_SPREAD = 2.0


@dataclass
class BenchmarkCase:
    dtype: str
    mode: str
    batch_size: int

    @property
    def id(self) -> str:
        return f"{self.dtype}-{self.mode}-b{self.batch_size}"


@dataclass
class BenchmarkResult:
    """One measured case; collected for the end-of-run report and ``--torch-benchmark-json``."""

    suite: str
    device: str
    hardware: str
    dtype: str
    mode: str
    batch_size: int
    median_ms: float
    p90_ms: float
    min_ms: float
    throughput: float  # samples per second, from the median
    peak_memory_mb: float | None
    compile_s: float | None
    baseline_ms: float | None = None
    status: str = "measured"  # "measured", "baseline updated", "regressed"
    #: p90 more than NOISY_SPREAD x the median: something else competed for the machine
    noisy: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


#: results of the running session, filled by BenchmarkSuite tests
results: list[BenchmarkResult] = []


def cases(suite: type[BenchmarkSuite]) -> list[BenchmarkCase]:
    """The matrix of a suite class, in a stable order."""
    for mode in suite.modes:
        if mode not in MODES:
            raise TypeError(f"{suite.__name__}: unknown mode {mode!r}; choose from {list(MODES)}")
    if not suite.batch_sizes or any(int(b) < 1 for b in suite.batch_sizes):
        raise TypeError(f"{suite.__name__}: batch_sizes must be positive integers")
    return [
        BenchmarkCase(str(dtype).replace("torch.", ""), mode, int(batch))
        for dtype in suite.dtypes
        for mode in suite.modes
        for batch in suite.batch_sizes
    ]


class BenchmarkSuite:
    """Base class for a benchmark suite (it is not collected itself).

    Override :meth:`make_model` and :meth:`make_inputs`, and set the matrix with
    ``batch_sizes``, ``dtypes`` and ``modes`` (``"eager"``, ``"compile"``). Every
    combination becomes one test on every device, measuring median and p90 latency,
    throughput, peak memory (CUDA, XPU) and, in ``"compile"`` mode, compilation time.
    Latency is measured ``rounds`` times (``warmup`` + ``iters`` calls each) and the
    fastest round is kept; a round whose p90 exceeds twice its median is flagged noisy.
    Results are compared with this machine's baseline (record with
    ``--torch-update-perf``; tolerances: ``latency_tolerance``, ``memory_tolerance``).
    Benchmarks only run with ``--torch-benchmark``.
    """

    pytestmark: ClassVar[list] = [pytest.mark.torch_benchmark]

    batch_sizes: ClassVar[Sequence[int]] = (1, 8, 32)
    dtypes: ClassVar[Sequence[Any]] = ("float32",)
    modes: ClassVar[Sequence[str]] = ("eager",)
    warmup: ClassVar[int] = 5
    iters: ClassVar[int] = 20
    #: measure this many times and keep the fastest round: interference only slows runs down
    rounds: ClassVar[int] = 3
    latency_tolerance: ClassVar[float] = 0.25
    memory_tolerance: ClassVar[float] = 0.10
    #: keyword arguments for torch.compile in "compile" mode
    compile_options: ClassVar[dict[str, Any]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cases(cls)  # validate the matrix at import time

    def make_model(self):
        raise NotImplementedError(f"{type(self).__name__} must implement make_model()")

    def make_inputs(self, batch_size: int, make_tensor) -> Any:
        """Inputs for one batch of ``batch_size`` (tensor, tuple of args or dict of kwargs)."""
        raise NotImplementedError(f"{type(self).__name__} must implement make_inputs(batch_size, make_tensor)")

    def test_benchmark(self, request, torch_seed, torch_device, make_tensor, benchmark_case) -> None:
        import torch

        from .perf import PerfStore, compare_perf, hardware_id, measure_latency, measure_peak_memory
        from .plugin import _unsupported
        from .utils import call_model, cast_floating, is_compile_supported

        case: BenchmarkCase = benchmark_case
        dtype = getattr(torch, case.dtype)
        reason = _unsupported(torch_device, dtype)
        if reason:
            pytest.skip(f"{case.dtype} is not supported on {torch_device}: {reason}")
        model = self.make_model().to(device=torch_device, dtype=dtype).eval()
        inputs = cast_floating(self.make_inputs(case.batch_size, make_tensor), dtype)

        compile_s = None
        runner: Any = model
        if case.mode == "compile":
            if not is_compile_supported():
                pytest.skip("torch.compile is not supported here")
            runner = torch.compile(model, **self.compile_options)
            start = time.perf_counter()
            try:
                with torch.inference_mode():
                    call_model(runner, inputs)  # compiles
            except Exception as exc:  # e.g. no C++ compiler for inductor
                pytest.skip(f"torch.compile failed here: {type(exc).__name__}: {str(exc).splitlines()[0][:150]}")
            compile_s = time.perf_counter() - start

        stats = min(
            (measure_latency(call_model, runner, inputs, device=torch_device, warmup=self.warmup, iters=self.iters)
             for _ in range(max(1, self.rounds))),
            key=lambda s: s.median_ms,
        )
        peak = measure_peak_memory(call_model, runner, inputs, device=torch_device)
        hardware = hardware_id(torch_device)
        result = BenchmarkResult(
            suite=type(self).__name__, device=str(torch_device), hardware=hardware, dtype=case.dtype,
            mode=case.mode, batch_size=case.batch_size, median_ms=stats.median_ms, p90_ms=stats.p90_ms,
            min_ms=stats.min_ms, throughput=case.batch_size / (stats.median_ms / 1e3) if stats.median_ms else 0.0,
            peak_memory_mb=peak, compile_s=compile_s,
            noisy=stats.p90_ms > NOISY_SPREAD * stats.median_ms,
        )
        results.append(result)

        config = request.config
        path = config.getoption("torch_perf_file") or config.rootpath / ".torch-perf-baselines.json"
        key = f"{request.node.nodeid.split('[')[0]}[{case.id}]"
        try:
            perf = compare_perf(
                PerfStore(path), key, hardware, stats.median_ms, peak,
                update=config.getoption("torch_update_perf"), strict=config.getoption("torch_perf_strict"),
                latency_tolerance=self.latency_tolerance, memory_tolerance=self.memory_tolerance,
            )
        except AssertionError:
            result.status = "regressed"
            raise
        except pytest.skip.Exception:
            return  # measured, no baseline yet: reported, not compared
        result.baseline_ms = perf.baseline_latency_ms
        if perf.updated:
            result.status = "baseline updated"
            if result.noisy:
                import warnings

                warnings.warn(
                    f"{key}: recorded a baseline from a noisy measurement (p90 {stats.p90_ms:.3f} ms vs "
                    f"median {stats.median_ms:.3f} ms); re-record on an idle machine",
                    stacklevel=1,
                )


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def format_table(rows: list[BenchmarkResult]) -> list[str]:
    def num(value, fmt):
        return format(value, fmt) if value is not None else "-"

    header = ("suite", "device", "dtype", "mode", "batch", "median ms", "p90 ms", "samples/s",
              "peak MB", "compile s", "vs baseline")
    table = []
    for r in rows:
        change = "-"
        if r.status == "baseline updated":
            change = "recorded"
        elif r.baseline_ms:
            pct = (r.median_ms / r.baseline_ms - 1) * 100
            change = f"{pct:+.1f}%" if abs(pct) >= 99.5 or abs(pct) < 10 else f"{pct:+.0f}%"
        table.append((r.suite, r.device, r.dtype, r.mode, str(r.batch_size), num(r.median_ms, ".3f"),
                      num(r.p90_ms, ".3f"), num(r.throughput, ",.0f"), num(r.peak_memory_mb, ".1f"),
                      num(r.compile_s, ".2f"),
                      change + (" REGRESSED" if r.status == "regressed" else "") + (" (noisy)" if r.noisy else "")))
    widths = [max(len(h), *(len(row[i]) for row in table)) for i, h in enumerate(header)]
    numeric = set(range(4, len(header)))

    def line(cells):
        return "  ".join(c.rjust(w) if i in numeric else c.ljust(w) for i, (c, w) in enumerate(zip(cells, widths)))

    return [line(header), *(line(row) for row in table)]
