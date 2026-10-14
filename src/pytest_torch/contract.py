"""One-call inference contract check.

    report = check_inference_contract(model, example_inputs)

runs a battery of checks that every inference model should satisfy and reports
*which* failed and *why it likely failed*.
"""
from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, cast

import torch

from . import tolerances
from .assertions import (
    BatchInvarianceNotApplicable,
    assert_finite,
    assert_outputs_close,
    batch_invariance_problems,
    format_problems,
    leaves_diff,
)
from .utils import (
    call_model,
    cast_floating,
    flatten,
    model_device,
    to_device,
    tree_map,
)

# Execution order. ``no_state_mutation`` must stay last: it compares the state
# snapshot taken before the first forward pass with the state after all runs.
CHECKS = (
    "eval_mode",
    "finite_outputs",
    "deterministic",
    "batch_invariance",
    "device_parity",
    "dtype_parity",
    "no_state_mutation",
)
_NEEDS_MODULE = {"eval_mode", "device_parity", "dtype_parity", "no_state_mutation"}

_HINTS = {
    "eval_mode": "Call model.eval() before inference; dropout and batch norm behave "
    "differently in training mode.",
    "finite_outputs": "NaN/Inf often come from fp16 overflow, log(0), division by "
    "zero, or uninitialized/unnormalized inputs.",
    "deterministic": "Look for dropout in training mode, random ops in forward(), or "
    "nondeterministic GPU kernels (try --torch-deterministic).",
    "batch_invariance": "Output depends on batch composition: check batch norm in "
    "training mode, padding/masking, or ops that reduce over the batch dimension.",
    "device_parity": "Large cross-device differences point to a device-specific op, "
    "dtype promotion, or TF32. Use localize_divergence() to find the first diverging layer.",
    "dtype_parity": "Low-precision output drifted: look for overflow-prone ops "
    "(softmax, layer norm) or missing fp32 accumulation.",
    "no_state_mutation": "forward() modified parameters or buffers (e.g. batch norm "
    "running stats in training mode, or a counter updated in forward).",
}


class _Skip(Exception):
    pass


@dataclass
class CheckResult:
    name: str
    status: str  # "pass" | "fail" | "skip"
    message: str = ""
    hint: str = ""


@dataclass
class ContractReport:
    results: list[CheckResult] = field(default_factory=list)

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == "fail"]

    @property
    def ok(self) -> bool:
        return not self.failures

    def format(self) -> str:
        counts = {s: sum(r.status == s for r in self.results) for s in ("pass", "fail", "skip")}
        lines = [
            f"Inference contract: {counts['pass']} passed, "
            f"{counts['fail']} failed, {counts['skip']} skipped"
        ]
        tags = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}
        for r in self.results:
            suffix = f" ({r.message})" if r.status == "skip" and r.message else ""
            lines.append(f"  [{tags[r.status]}] {r.name}{suffix}")
            if r.status == "fail":
                lines.extend("         " + ln for ln in r.message.splitlines())
                if r.hint:
                    lines.append(f"         hint: {r.hint}")
        return "\n".join(lines)

    __str__ = format

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise AssertionError(self.format())


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _run(model: Callable, inputs: Any) -> Any:
    with torch.inference_mode():
        return call_model(model, inputs)


def _clone_model(model: torch.nn.Module, *, device=None, dtype=None) -> torch.nn.Module:
    clone = copy.deepcopy(model)
    if dtype is not None:
        clone = clone.to(dtype)
    if device is not None:
        clone = clone.to(device)
    return clone


def _snapshot_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
        if isinstance(v, torch.Tensor)
    }


def _default_parity_devices(current: torch.device) -> list[str]:
    """Devices to compare against: the other available ones, or, when the session pinned
    devices with ``--torch-device``, only the other pinned ones."""
    from . import _session
    from .devices import available_devices, is_available

    if _session.selected_devices is not None:
        candidates = [d for d in _session.selected_devices if is_available(d)]
        return [d for d in candidates if torch.device(d).type != current.type]
    if current.type != "cpu":
        return ["cpu"]
    return [d for d in available_devices() if d != "cpu"]


def _to_cpu(obj: Any) -> Any:
    return tree_map(lambda t: t.detach().cpu(), obj)


def _run_check(name: str, fn: Callable[[], None]) -> CheckResult:
    try:
        fn()
    except _Skip as skip:
        return CheckResult(name, "skip", str(skip))
    except AssertionError as exc:
        return CheckResult(name, "fail", str(exc), _HINTS.get(name, ""))
    except Exception as exc:  # the check itself blew up (e.g. unsupported op)
        return CheckResult(
            name, "fail", f"raised {type(exc).__name__}: {exc}", _HINTS.get(name, "")
        )
    return CheckResult(name, "pass")


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def check_inference_contract(
    model: Callable,
    inputs: Any,
    *,
    checks: Sequence[str] | None = None,
    skip: Sequence[str] = (),
    rtol: float | None = None,
    atol: float | None = None,
    devices: Sequence[str] | None = None,
    device_rtol: float | None = None,
    device_atol: float | None = None,
    dtypes: Sequence[torch.dtype] = (),
    dtype_rtol: float | None = None,
    dtype_atol: float | None = None,
    raise_on_failure: bool = True,
) -> ContractReport:
    """Run a battery of inference checks and return a :class:`ContractReport`.

    ``inputs`` may be a tensor, a tuple of positional args, or a dict of kwargs;
    they are moved to the model's device automatically. Outputs may be any pytree.

    Checks (see ``CHECKS``): eval mode, finite outputs, determinism, batch
    invariance, device parity (vs ``devices``; default: the other available
    devices), dtype parity (only for ``dtypes``, e.g. ``(torch.bfloat16,)``), and
    no parameter/buffer mutation. Use ``checks=`` / ``skip=`` to select.
    Tolerances: ``rtol``/``atol`` for same-device checks (dtype defaults when
    ``None``), ``device_*`` and ``dtype_*`` for the parity checks.
    """
    unknown = (set(checks or ()) | set(skip)) - set(CHECKS)
    if unknown:
        raise ValueError(f"unknown check(s) {sorted(unknown)}; choose from {CHECKS}")
    selected = [c for c in CHECKS if (checks is None or c in checks) and c not in skip]

    is_module = isinstance(model, torch.nn.Module)
    # The checks that need module methods only run when is_module (see _NEEDS_MODULE).
    module = cast(torch.nn.Module, model)
    device = model_device(model)
    device_rtol, device_atol = tolerances.for_check("device_parity", device, device_rtol, device_atol)
    dtype_rtol, dtype_atol = tolerances.for_check("dtype_parity", device, dtype_rtol, dtype_atol)
    inputs = to_device(inputs, device)
    snapshot = _snapshot_state(module) if is_module else {}
    reference = _run(model, inputs)  # if the forward pass itself fails, let it raise

    def eval_mode() -> None:
        training = [name or "<root>" for name, m in module.named_modules() if m.training]
        if training:
            raise AssertionError(
                f"{len(training)} module(s) in training mode (e.g. {', '.join(training[:3])})"
            )

    def finite_outputs() -> None:
        assert_finite(reference, "output")

    def deterministic() -> None:
        assert_outputs_close(_run(model, inputs), reference, rtol=0.0, atol=0.0)

    def batch_invariance() -> None:
        try:
            problems = batch_invariance_problems(
                model, inputs, batched=reference, rtol=rtol, atol=atol
            )
        except BatchInvarianceNotApplicable as exc:
            raise _Skip(str(exc)) from None
        if problems:
            raise AssertionError(
                format_problems("per-sample results differ from the batched run", problems)
            )

    def device_parity() -> None:
        targets = list(devices) if devices is not None else _default_parity_devices(device)
        if not targets:
            from . import _session

            pinned = _session.selected_devices is not None
            raise _Skip("no other device selected (--torch-device)" if pinned else "no other device available")
        failures = []
        ref_cpu = dict(flatten(_to_cpu(reference)))
        for target in targets:
            clone = _clone_model(module, device=target)
            out = _run(clone, to_device(inputs, target))
            problems = leaves_diff(
                dict(flatten(_to_cpu(out))),
                ref_cpu,
                rtol=device_rtol,
                atol=device_atol,
                check_device=False,
            )
            if problems:
                failures.append(format_problems(f"vs {target}:", problems, limit=5))
        if failures:
            raise AssertionError("\n".join(failures))

    def dtype_parity() -> None:
        if not dtypes:
            raise _Skip("no dtypes requested, e.g. dtypes=(torch.bfloat16,)")

        def as_f32(obj: Any) -> Any:
            return tree_map(
                lambda t: t.detach().to(torch.float32) if t.is_floating_point() else t.detach(),
                obj,
            )

        failures = []
        ref32 = dict(flatten(as_f32(_to_cpu(reference))))
        for dtype in dtypes:
            clone = _clone_model(module, dtype=dtype)
            out = _run(clone, cast_floating(inputs, dtype))
            problems = leaves_diff(
                dict(flatten(as_f32(_to_cpu(out)))),
                ref32,
                rtol=dtype_rtol,
                atol=dtype_atol,
            )
            if problems:
                failures.append(format_problems(f"vs {dtype}:", problems, limit=5))
        if failures:
            raise AssertionError("\n".join(failures))

    def no_state_mutation() -> None:
        current = module.state_dict()
        changed = [
            k
            for k, before in snapshot.items()
            if k not in current or not torch.equal(before, current[k])
        ]
        if changed:
            raise AssertionError(
                f"{len(changed)} parameter/buffer tensor(s) changed during inference "
                f"(e.g. {', '.join(changed[:3])})"
            )

    runners = {
        "eval_mode": eval_mode,
        "finite_outputs": finite_outputs,
        "deterministic": deterministic,
        "batch_invariance": batch_invariance,
        "device_parity": device_parity,
        "dtype_parity": dtype_parity,
        "no_state_mutation": no_state_mutation,
    }

    results = []
    for name in selected:
        if name in _NEEDS_MODULE and not is_module:
            results.append(CheckResult(name, "skip", "requires an nn.Module"))
            continue
        results.append(_run_check(name, runners[name]))

    report = ContractReport(results)
    if raise_on_failure:
        report.raise_if_failed()
    return report
