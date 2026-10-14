"""Dataset evaluation: run a loader through a model, aggregate metrics, guard regressions.

    def accuracy(output, batch):
        return (output.argmax(-1) == batch[1]).float()        # per sample: enables paired tests

    def test_accuracy(model, val_loader, metric_baseline):
        metric_baseline(model, val_loader, {"accuracy": accuracy},
                        context={"dataset": "sst2", "split": "validation", "preprocessing": "v3"},
                        minimum={"accuracy": 0.9})

Record baselines with ``pytest --torch-update-metrics``. A baseline remembers what it
measured (context, input fingerprint, sample selection, metric configuration, checkpoint),
and a later run is only compared with a baseline that measured the same thing. Metrics with
per-sample values are compared *paired*, with a bootstrap confidence interval, so a small
drop on a small dataset isn't mistaken for a regression.
"""
from __future__ import annotations

import datetime
import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest
import torch

from .utils import call_model, find_batch_size, flatten, model_device, to_device

#: context keys that are recorded for interpretation but never block a comparison
RECORD_ONLY_CONTEXT = ("checkpoint",)


def default_inputs(batch: Any) -> Any:
    """``(x, y, ...)`` -> ``x``; a dict is passed to the model as keyword arguments."""
    if isinstance(batch, (list, tuple)) and not hasattr(batch, "_fields"):
        return batch[0]
    return batch


# --------------------------------------------------------------------------- #
# fingerprints
# --------------------------------------------------------------------------- #
def _tensor_bytes(t: torch.Tensor) -> bytes:
    flat = t.detach().cpu().contiguous().reshape(-1)
    if flat.dtype == torch.bool:
        flat = flat.to(torch.uint8)
    try:
        return flat.numpy().tobytes()
    except (RuntimeError, TypeError):  # no numpy, or a dtype numpy lacks (bfloat16)
        return bytes(flat.view(torch.uint8).tolist())


class _Fingerprint:
    """One running hash per tensor position (inputs, labels, ...), combined at the end: the
    same samples in the same order give the same fingerprint however they are batched."""

    def __init__(self):
        self._streams: dict[str, Any] = {}

    def add(self, obj: Any) -> None:
        for path, leaf in flatten(obj, "batch"):
            if isinstance(leaf, torch.Tensor):
                shape = tuple(leaf.shape[1:]) if leaf.dim() else ()
                stream = self._streams.setdefault(f"{path}|{leaf.dtype}|{shape}", hashlib.sha256())
                stream.update(_tensor_bytes(leaf))

    def hexdigest(self) -> str:
        combined = hashlib.sha256()
        for key in sorted(self._streams):
            combined.update(key.encode())
            combined.update(self._streams[key].digest())
        return combined.hexdigest()[:16]


def model_fingerprint(model) -> str | None:
    """Short hash of a model's parameters and buffers (what was evaluated)."""
    if not hasattr(model, "state_dict"):
        return None
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if isinstance(value, torch.Tensor):
            digest.update(f"{key}|{value.dtype}|{tuple(value.shape)}|".encode())
            digest.update(_tensor_bytes(value))
    return digest.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
@dataclass
class Evaluation:
    """Result of :func:`evaluate_detailed`."""

    values: dict[str, float]
    #: per-sample values for metrics that return one value per sample
    per_sample: dict[str, torch.Tensor]
    provenance: dict[str, Any]


def _is_stateful(metric: Any) -> bool:
    return hasattr(metric, "update") and hasattr(metric, "compute")


def _metric_description(metric: Any, per_sample: bool) -> dict[str, Any]:
    config = getattr(metric, "config", None)
    return {
        "kind": "update/compute" if _is_stateful(metric) else ("per-sample" if per_sample else "per-batch"),
        "config": dict(config) if isinstance(config, Mapping) else config,
        "function": getattr(metric, "__qualname__", type(metric).__qualname__),
    }


def evaluate_detailed(
    model,
    loader: Iterable,
    metrics: Mapping[str, Any],
    *,
    inputs: Callable[[Any], Any] = default_inputs,
    max_batches: int | None = None,
    context: Mapping[str, Any] | None = None,
    fingerprint_model: bool = True,
) -> Evaluation:
    """Like :func:`evaluate`, plus per-sample values and provenance.

    A metric ``fn(output, batch)`` returning a 1-D tensor with one value per sample is
    *per-sample* (enables paired comparisons); a scalar is a batch mean, weighted by batch
    size. Objects with ``update(output, batch)`` / ``compute()`` cover metrics that need
    every prediction; give them a ``config`` attribute (averaging, thresholds, ignored
    labels) so baselines record it. ``context`` describes the evaluation (dataset, split,
    preprocessing version, checkpoint label).
    """
    if not metrics:
        raise ValueError("pass at least one metric")
    device = model_device(model)
    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    sums = dict.fromkeys(metrics, 0.0)
    samples: dict[str, list[torch.Tensor]] = {name: [] for name in metrics}
    kinds: dict[str, bool] = {}
    count = batches = 0
    fingerprint = _Fingerprint()
    try:
        with torch.inference_mode():
            for index, batch in enumerate(loader):
                if max_batches is not None and index >= max_batches:
                    break
                fingerprint.add(batch)
                batch = to_device(batch, device)
                output = call_model(model, inputs(batch))
                size = find_batch_size(batch) or 1
                for name, metric in metrics.items():
                    if _is_stateful(metric):
                        metric.update(output, batch)
                        continue
                    value = metric(output, batch)
                    is_per_sample = isinstance(value, torch.Tensor) and value.dim() == 1
                    if kinds.setdefault(name, is_per_sample) != is_per_sample:
                        raise ValueError(f"metric {name!r} returned per-sample values for some batches only")
                    if is_per_sample:
                        if value.shape[0] != size:
                            raise ValueError(
                                f"metric {name!r} returned {value.shape[0]} values for a batch of {size}"
                            )
                        samples[name].append(value.detach().double().cpu())
                    else:
                        sums[name] += float(value) * size
                count += size
                batches += 1
    finally:
        if was_training:
            model.train()
    if count == 0:
        raise ValueError("the loader produced no batches")

    values: dict[str, float] = {}
    per_sample: dict[str, torch.Tensor] = {}
    for name, metric in metrics.items():
        if _is_stateful(metric):
            values[name] = float(metric.compute())
        elif kinds.get(name):
            per_sample[name] = torch.cat(samples[name])
            values[name] = float(per_sample[name].mean())
        else:
            values[name] = sums[name] / count
    context = dict(context or {})
    provenance = {
        "context": {k: v for k, v in context.items() if k not in RECORD_ONLY_CONTEXT},
        "samples": count,
        "batches": batches,
        "max_batches": max_batches,
        "inputs_fingerprint": fingerprint.hexdigest(),
        "metrics": {name: _metric_description(m, bool(kinds.get(name))) for name, m in metrics.items()},
        "checkpoint": {
            "label": context.get("checkpoint"),
            "fingerprint": model_fingerprint(model) if fingerprint_model else None,
        },
        "torch": str(torch.__version__),
        "recorded": datetime.date.today().isoformat(),
    }
    return Evaluation(values, per_sample, provenance)


def evaluate(
    model,
    loader: Iterable,
    metrics: Mapping[str, Any],
    *,
    inputs: Callable[[Any], Any] = default_inputs,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run ``loader`` through ``model`` (eval mode, inference mode, model's device).

    A metric is ``fn(output, batch)`` returning a scalar (averaged over the dataset weighted
    by batch size) or one value per sample (averaged over samples), or an object with
    ``update(output, batch)`` and ``compute()`` for metrics that need every prediction
    (F1, AUC, ...). ``inputs`` extracts the model inputs from a batch. ``max_batches``
    caps the work, e.g. in CI.
    """
    return evaluate_detailed(model, loader, metrics, inputs=inputs, max_batches=max_batches,
                             fingerprint_model=False).values


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #
@dataclass
class PairedDifference:
    """Paired change of one metric versus its baseline: ``drop > 0`` means worse."""

    drop: float
    low: float
    high: float
    confidence: float
    samples: int

    def __str__(self) -> str:
        return (f"drop {self.drop:+.4g} ({self.confidence:.0%} CI {self.low:+.4g} .. {self.high:+.4g}, "
                f"n={self.samples}, paired)")


@dataclass
class MetricResult:
    values: dict[str, float]
    baseline: dict[str, float] = field(default_factory=dict)
    paired: dict[str, PairedDifference] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    baseline_provenance: dict[str, Any] = field(default_factory=dict)
    updated: bool = False

    def __str__(self) -> str:
        parts = []
        for name, value in self.values.items():
            base = self.baseline.get(name)
            text = f"{name}={value:.6g}" + (f" (baseline {base:.6g})" if base is not None else "")
            if name in self.paired:
                text += f" {self.paired[name]}"
            parts.append(text)
        text = ", ".join(parts) + (" [baseline updated]" if self.updated else "")
        if self.baseline_provenance:
            text += "\n" + _describe_runs(self.baseline_provenance, self.provenance)
        return text


def _checkpoint_text(provenance: Mapping[str, Any]) -> str:
    checkpoint = provenance.get("checkpoint") or {}
    parts = [str(checkpoint[k]) for k in ("label", "fingerprint") if checkpoint.get(k)]
    return " / ".join(parts) or "unknown checkpoint"


def _describe_runs(baseline: Mapping[str, Any], current: Mapping[str, Any]) -> str:
    return (f"baseline: {_checkpoint_text(baseline)}, torch {baseline.get('torch', '?')}, "
            f"recorded {baseline.get('recorded', '?')}; now: {_checkpoint_text(current)}, "
            f"torch {current.get('torch', '?')}")


def comparability_problems(baseline: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """Why ``current`` measured something other than ``baseline`` (empty: comparable)."""
    problems = []
    keys = sorted(set(baseline.get("context", {})) | set(current.get("context", {})))
    for key in keys:
        old, new = baseline.get("context", {}).get(key), current.get("context", {}).get(key)
        if old != new:
            problems.append(f"context {key!r}: baseline {old!r}, now {new!r}")
    for key, what in (("samples", "sample count"), ("max_batches", "max_batches (sample selection)")):
        if baseline.get(key) != current.get(key):
            problems.append(f"{what}: baseline {baseline.get(key)!r}, now {current.get(key)!r}")
    if baseline.get("inputs_fingerprint") != current.get("inputs_fingerprint") and not any(
        p.startswith("sample count") for p in problems
    ):
        problems.append(
            f"input fingerprint: baseline {baseline.get('inputs_fingerprint')}, now "
            f"{current.get('inputs_fingerprint')} (the data, its order or its preprocessing changed)"
        )
    old_metrics, new_metrics = baseline.get("metrics", {}), current.get("metrics", {})
    for name in sorted(set(old_metrics) & set(new_metrics)):
        for part in ("kind", "config"):
            if old_metrics[name].get(part) != new_metrics[name].get(part):
                problems.append(
                    f"metric {name!r} {part}: baseline {old_metrics[name].get(part)!r}, "
                    f"now {new_metrics[name].get(part)!r}"
                )
    return problems


def paired_difference(
    current: torch.Tensor,
    baseline: torch.Tensor,
    *,
    higher_is_better: bool = True,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> PairedDifference:
    """Mean per-sample drop (``> 0``: worse) with a seeded percentile-bootstrap interval."""
    diff = (baseline - current) if higher_is_better else (current - baseline)
    diff = diff.double().cpu()
    n = diff.numel()
    generator = torch.Generator().manual_seed(seed)
    chunks = []
    for start in range(0, resamples, 256):  # chunks keep memory at 256 * n indices
        k = min(256, resamples - start)
        idx = torch.randint(0, n, (k, n), generator=generator)
        chunks.append(diff[idx].mean(dim=1))
    means = torch.cat(chunks)
    alpha = (1 - confidence) / 2
    low, high = torch.quantile(means, torch.tensor([alpha, 1 - alpha], dtype=torch.float64)).tolist()
    return PairedDifference(float(diff.mean()), low, high, confidence, n)


def _per_metric(option: Any, name: str, default: Any) -> Any:
    if isinstance(option, Mapping):
        return option.get(name, default)
    return default if option is None else option


def _samples_path(store_path: Path, key: str) -> Path:
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return store_path.with_suffix("") / f"{digest}.pt"


def check_metrics(
    store,
    key: str,
    values: Mapping[str, float] | Evaluation,
    *,
    update: bool = False,
    strict: bool = False,
    tolerance: float | Mapping[str, float] = 0.01,
    higher_is_better: bool | Mapping[str, bool] = True,
    minimum: Mapping[str, float] | None = None,
    maximum: Mapping[str, float] | None = None,
    confidence: float = 0.95,
    resamples: int = 2000,
) -> MetricResult:
    """Compare an evaluation with the stored baseline and with absolute thresholds.

    The baseline must have measured the same thing (context except ``checkpoint``, sample
    count and selection, input fingerprint, metric configuration); otherwise the check
    fails and says what changed. Then, per metric:

    - with per-sample values on both sides: a **paired** comparison. A regression is a mean
      drop larger than ``tolerance`` whose bootstrap confidence interval (``confidence``)
      excludes zero, so noise on a small dataset doesn't fail the test;
    - otherwise: a regression is a drop larger than ``tolerance`` (absolute).

    Improvements never fail. ``minimum``/``maximum`` apply on every run. A missing baseline
    skips the test unless thresholds were given (they still apply) or ``strict`` is set.
    """
    evaluation = values if isinstance(values, Evaluation) else Evaluation(dict(values), {}, {})
    values = evaluation.values
    result = MetricResult(dict(values), provenance=evaluation.provenance)
    problems = []
    for name, value in values.items():
        low = (minimum or {}).get(name)
        high = (maximum or {}).get(name)
        if low is not None and value < low:
            problems.append(f"{name} = {value:.6g} is below the minimum {low:.6g}")
        if high is not None and value > high:
            problems.append(f"{name} = {value:.6g} is above the maximum {high:.6g}")
    unknown = (set(minimum or {}) | set(maximum or {})) - set(values)
    if unknown:
        raise ValueError(f"thresholds for unknown metric(s) {sorted(unknown)}; metrics are {list(values)}")

    store_path = Path(getattr(store, "path", "metrics.json"))
    if update:
        entry = {"metrics": {k: round(float(v), 8) for k, v in values.items()},
                 "torch": str(torch.__version__), "provenance": evaluation.provenance}
        if evaluation.per_sample:
            path = _samples_path(store_path, key)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({k: v.float() for k, v in evaluation.per_sample.items()}, path)
            entry["per_sample_file"] = path.relative_to(store_path.parent).as_posix()
        store.set(key, entry)
        result.updated = True
    else:
        entry = store.get(key)
        if entry is None:
            message = f"no metric baseline for {key!r}; record one with --torch-update-metrics"
            if strict:
                problems.append(message)
            elif not (minimum or maximum):
                pytest.skip(message)
        else:
            result.baseline = {k: float(v) for k, v in entry["metrics"].items()}
            result.baseline_provenance = entry.get("provenance", {})
            mismatch = comparability_problems(result.baseline_provenance, evaluation.provenance) \
                if result.baseline_provenance and evaluation.provenance else []
            if mismatch:
                problems.append(
                    "the baseline measured something else, so it can't be compared:\n    "
                    + "\n    ".join(mismatch)
                )
            else:
                baseline_samples = {}
                if entry.get("per_sample_file") and evaluation.per_sample:
                    sample_path = store_path.parent / entry["per_sample_file"]
                    if sample_path.is_file():
                        baseline_samples = torch.load(sample_path, map_location="cpu", weights_only=True)
                for name, value in values.items():
                    base = result.baseline.get(name)
                    if base is None:
                        continue
                    tol = float(_per_metric(tolerance, name, 0.01))
                    up = bool(_per_metric(higher_is_better, name, True))
                    if name in evaluation.per_sample and name in baseline_samples:
                        paired = paired_difference(evaluation.per_sample[name], baseline_samples[name],
                                                   higher_is_better=up, confidence=confidence,
                                                   resamples=resamples)
                        result.paired[name] = paired
                        if paired.drop > tol and paired.low > 0:
                            problems.append(
                                f"{name} regressed from {base:.6g} to {value:.6g}: {paired}; "
                                f"allowed drop {tol:g}"
                            )
                        continue
                    worse = (base - value) if up else (value - base)
                    if worse > tol:
                        direction = "dropped" if up else "rose"
                        problems.append(
                            f"{name} {direction} from {base:.6g} to {value:.6g} "
                            f"({worse:+.6g} worse; allowed {tol:g})"
                        )
    if problems:
        footer = ""
        if result.baseline_provenance:
            footer = "\n  " + _describe_runs(result.baseline_provenance, evaluation.provenance)
        raise AssertionError(
            f"Metric regression in {key!r}:\n  " + "\n  ".join(problems) + footer
            + "\n  (re-record with --torch-update-metrics if the change is intended)"
        )
    return result
