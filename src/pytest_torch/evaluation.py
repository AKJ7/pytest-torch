"""Dataset evaluation: run a loader through a model, aggregate metrics, guard regressions.

    def accuracy(output, batch):
        return (output.argmax(-1) == batch[1]).float().mean()

    def test_accuracy(model, val_loader, metric_baseline):
        metric_baseline(model, val_loader, {"accuracy": accuracy}, minimum={"accuracy": 0.9})

Record baselines with ``pytest --torch-update-metrics``; later runs fail when a metric gets
worse than its baseline by more than ``tolerance``.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Callable

import pytest
import torch

from .utils import call_model, find_batch_size, model_device, to_device


def default_inputs(batch: Any) -> Any:
    """``(x, y, ...)`` -> ``x``; a dict is passed to the model as keyword arguments."""
    if isinstance(batch, (list, tuple)) and not hasattr(batch, "_fields"):
        return batch[0]
    return batch


def evaluate(
    model,
    loader: Iterable,
    metrics: Mapping[str, Any],
    *,
    inputs: Callable[[Any], Any] = default_inputs,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run ``loader`` through ``model`` (eval mode, inference mode, model's device).

    A metric is either ``fn(output, batch) -> scalar``, averaged over the dataset weighted
    by batch size (accuracy, mean loss, ...), or an object with ``update(output, batch)``
    and ``compute()`` for metrics that need every prediction (F1, AUC, ...). ``inputs``
    extracts the model inputs from a batch. ``max_batches`` caps the work, e.g. in CI.
    """
    if not metrics:
        raise ValueError("pass at least one metric")
    device = model_device(model)
    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    sums = dict.fromkeys(metrics, 0.0)
    count = 0
    try:
        with torch.inference_mode():
            for index, batch in enumerate(loader):
                if max_batches is not None and index >= max_batches:
                    break
                batch = to_device(batch, device)
                output = call_model(model, inputs(batch))
                weight = find_batch_size(batch) or 1
                for name, metric in metrics.items():
                    if hasattr(metric, "update") and hasattr(metric, "compute"):
                        metric.update(output, batch)
                    else:
                        sums[name] += float(metric(output, batch)) * weight
                count += weight
    finally:
        if was_training:
            model.train()
    if count == 0:
        raise ValueError("the loader produced no batches")
    results = {}
    for name, metric in metrics.items():
        if hasattr(metric, "update") and hasattr(metric, "compute"):
            results[name] = float(metric.compute())
        else:
            results[name] = sums[name] / count
    return results


@dataclass
class MetricResult:
    values: dict[str, float]
    baseline: dict[str, float] = field(default_factory=dict)
    updated: bool = False

    def __str__(self) -> str:
        parts = []
        for name, value in self.values.items():
            base = self.baseline.get(name)
            parts.append(f"{name}={value:.6g}" + (f" (baseline {base:.6g})" if base is not None else ""))
        return ", ".join(parts) + (" [baseline updated]" if self.updated else "")


def _per_metric(option: Any, name: str, default: Any) -> Any:
    if isinstance(option, Mapping):
        return option.get(name, default)
    return default if option is None else option


def check_metrics(
    store,
    key: str,
    values: Mapping[str, float],
    *,
    update: bool = False,
    strict: bool = False,
    tolerance: float | Mapping[str, float] = 0.01,
    higher_is_better: bool | Mapping[str, bool] = True,
    minimum: Mapping[str, float] | None = None,
    maximum: Mapping[str, float] | None = None,
) -> MetricResult:
    """Compare ``values`` with the stored baseline and with absolute thresholds.

    A regression is a change in the bad direction larger than ``tolerance`` (absolute, per
    metric or for all). Improvements never fail. ``minimum``/``maximum`` are checked on
    every run, baseline or not. A missing baseline skips the test, unless thresholds were
    given (they still apply) or ``strict`` is set (it fails).
    """
    result = MetricResult(dict(values))
    problems = []
    for name, value in values.items():
        low = (minimum or {}).get(name)
        high = (maximum or {}).get(name)
        if low is not None and value < low:
            problems.append(f"{name} = {value:.6g} is below the minimum {low:.6g}")
        if high is not None and value > high:
            problems.append(f"{name} = {value:.6g} is above the maximum {high:.6g}")
    unknown = set(minimum or {}) | set(maximum or {})
    unknown -= set(values)
    if unknown:
        raise ValueError(f"thresholds for unknown metric(s) {sorted(unknown)}; metrics are {list(values)}")

    if update:
        store.set(key, {"metrics": {k: round(float(v), 8) for k, v in values.items()}, "torch": str(torch.__version__)})
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
            for name, value in values.items():
                base = result.baseline.get(name)
                if base is None:
                    continue
                tol = float(_per_metric(tolerance, name, 0.01))
                up = bool(_per_metric(higher_is_better, name, True))
                worse = (base - value) if up else (value - base)
                if worse > tol:
                    direction = "dropped" if up else "rose"
                    problems.append(
                        f"{name} {direction} from {base:.6g} to {value:.6g} "
                        f"({worse:+.6g} worse; allowed {tol:g})"
                    )
    if problems:
        raise AssertionError(
            f"Metric regression in {key!r}:\n  " + "\n  ".join(problems)
            + "\n  (re-record with --torch-update-metrics if the change is intended)"
        )
    return result
