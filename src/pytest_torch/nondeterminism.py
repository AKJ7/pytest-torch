"""Quantify and diagnose nondeterminism: RNG usage vs. kernel-level variance."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import torch

from .utils import (
    deterministic_algorithms,
    flatten,
    model_device,
    run_inference,
    seed_everything,
    to_device,
)

_INF = float("inf")


def _first_line(text: str) -> str:
    lines = text.strip().splitlines()
    return lines[0][:200] if lines else ""


@dataclass
class LeafVariance:
    path: str
    max_abs: float = 0.0
    frac_elements: float = 0.0  # worst run: fraction of elements that differ
    runs_differing: int = 0


@dataclass
class VarianceResult:
    runs: int
    leaves: list[LeafVariance]
    error: str | None = None  # e.g. an op without a deterministic implementation
    reseed: int | None = None

    @property
    def max_abs(self) -> float:
        return max((leaf.max_abs for leaf in self.leaves), default=0.0)

    @property
    def deterministic(self) -> bool:
        return self.error is None and self.max_abs == 0.0

    def format(self) -> str:
        seeding = (
            "no reseeding" if self.reseed is None else f"reseeded with {self.reseed} before every run"
        )
        if self.error:
            return f"Run failed under deterministic algorithms: {_first_line(self.error)}"
        varying = [leaf for leaf in self.leaves if leaf.max_abs > 0]
        if not varying:
            return f"Bit-identical outputs over {self.runs} runs ({seeding})"
        lines = [
            f"Outputs vary over {self.runs} runs ({seeding}):",
            f"  {'output':<36} {'max abs diff':>13} {'elements differing':>19} {'runs differing':>15}",
        ]
        for leaf in varying[:10]:
            ratio = f"{leaf.runs_differing}/{self.runs - 1}"
            lines.append(
                f"  {leaf.path:<36} {leaf.max_abs:>13.3e} {leaf.frac_elements:>19.2%} {ratio:>15}"
            )
        if len(varying) > 10:
            lines.append(f"  ... and {len(varying) - 10} more")
        return "\n".join(lines)

    __str__ = format


def _snapshot(output: Any) -> dict[str, torch.Tensor]:
    snap = {}
    for path, leaf in flatten(output):
        if isinstance(leaf, torch.Tensor):
            tensor = leaf.detach()
            if tensor.is_quantized:
                tensor = tensor.dequantize()
            snap[path] = tensor.to("cpu", torch.float64)
    return snap


def measure_nondeterminism(
    model,
    inputs: Any,
    *,
    runs: int = 10,
    reseed: int | None = None,
    deterministic: bool = False,
) -> VarianceResult:
    """Run the model ``runs`` times and measure how much the outputs differ from run 1.

    ``reseed`` reseeds all RNGs before every run (isolates kernel-level variance from RNG
    usage); ``deterministic=True`` runs under ``torch.use_deterministic_algorithms(True)``.
    """
    if runs < 2:
        raise ValueError("runs must be >= 2")
    inputs = to_device(inputs, model_device(model))
    reference: dict[str, torch.Tensor] | None = None
    stats: dict[str, LeafVariance] = {}
    context = deterministic_algorithms(True) if deterministic else contextlib.nullcontext()
    try:
        with context:
            for _ in range(runs):
                if reseed is not None:
                    seed_everything(reseed)
                current = _snapshot(run_inference(model, inputs))
                if reference is None:
                    reference = current
                    stats = {path: LeafVariance(path) for path in current}
                    continue
                for path, ref in reference.items():
                    leaf = stats[path]
                    cur = current.get(path)
                    if cur is None or cur.shape != ref.shape:
                        leaf.max_abs, leaf.frac_elements = _INF, 1.0
                        leaf.runs_differing += 1
                        continue
                    differs = ~((cur == ref) | (torch.isnan(cur) & torch.isnan(ref)))
                    if differs.any():
                        diff = torch.where(differs, (cur - ref).abs(), torch.zeros_like(cur))
                        diff = torch.nan_to_num(diff, nan=_INF, posinf=_INF)
                        leaf.max_abs = max(leaf.max_abs, float(diff.max()))
                        leaf.frac_elements = max(leaf.frac_elements, float(differs.double().mean()))
                        leaf.runs_differing += 1
    except RuntimeError as exc:
        if deterministic and "deterministic" in str(exc).lower():
            return VarianceResult(runs, [], error=str(exc), reseed=reseed)
        raise
    return VarianceResult(runs, list(stats.values()), reseed=reseed)


@dataclass
class NondeterminismReport:
    runs: int
    free: VarianceResult
    seeded: VarianceResult | None
    deterministic_mode: VarianceResult | None
    verdict: str

    @property
    def is_deterministic(self) -> bool:
        return self.free.deterministic

    def format(self) -> str:
        parts = [self.verdict, "", self.free.format()]
        if self.seeded is not None:
            parts += ["", self.seeded.format()]
        if self.deterministic_mode is not None:
            parts += ["", "With torch.use_deterministic_algorithms(True): "
                      + self.deterministic_mode.format()]
        return "\n".join(parts)

    __str__ = format


def diagnose_nondeterminism(model, inputs: Any, *, runs: int = 10, seed: int = 0) -> NondeterminismReport:
    """Find out *why* outputs vary: RNG usage, kernel-level nondeterminism, or neither."""
    free = measure_nondeterminism(model, inputs, runs=runs)
    if free.deterministic:
        return NondeterminismReport(
            runs, free, None, None, f"Deterministic: {runs} runs gave bit-identical outputs."
        )
    seeded = measure_nondeterminism(model, inputs, runs=runs, reseed=seed)
    if seeded.deterministic:
        return NondeterminismReport(
            runs,
            free,
            seeded,
            None,
            "Randomness-driven: the variance disappears when the RNG is reseeded before each "
            "run. Look for dropout in training mode, sampling, or noise inside forward().",
        )
    det = measure_nondeterminism(model, inputs, runs=runs, reseed=seed, deterministic=True)
    verdict = (
        "Kernel-level nondeterminism: outputs still vary with a fixed seed (e.g. atomic-add "
        "reductions on GPU, non-deterministic cuDNN algorithms, multi-threaded reductions)."
    )
    if det.error:
        verdict += " Deterministic mode is unavailable here: " + _first_line(det.error)
    elif det.deterministic:
        verdict += (
            " torch.use_deterministic_algorithms(True) fixes it "
            "(run pytest with --torch-deterministic)."
        )
    else:
        verdict += (
            " It persists with deterministic algorithms enabled: look at custom CUDA "
            "extensions or non-PyTorch libraries."
        )
    return NondeterminismReport(runs, free, seeded, det, verdict)


def assert_deterministic(
    model, inputs: Any, *, runs: int = 5, reseed: int | None = None, max_abs: float = 0.0
) -> None:
    """Assert the outputs vary by at most ``max_abs`` (default: bit-identical) over ``runs``."""
    result = measure_nondeterminism(model, inputs, runs=runs, reseed=reseed)
    if result.max_abs > max_abs:
        raise AssertionError(
            f"Model is not deterministic (budget: max abs diff {max_abs:g}):\n{result.format()}\n"
            "Hint: diagnose_nondeterminism(model, inputs) separates RNG usage from "
            "kernel-level nondeterminism."
        )
