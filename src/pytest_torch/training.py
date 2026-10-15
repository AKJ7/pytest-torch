"""Training sanity checks: gradients flow, are finite, are correct; the model can learn.

Every helper takes ``(model, inputs, loss_fn)`` where ``loss_fn(output) -> scalar tensor``
(close over your targets, e.g. ``lambda out: F.cross_entropy(out, y)``). Helpers that
take optimizer steps train a deep copy, so the model you pass is never modified.
"""
from __future__ import annotations

import copy
import fnmatch
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from . import tolerances
from .assertions import format_problems, leaves_diff
from .utils import call_model, cast_floating, model_device, to_device, tree_map

LossFn = Callable[[Any], torch.Tensor]


def _loss(model, inputs: Any, loss_fn: LossFn) -> torch.Tensor:
    loss = loss_fn(call_model(model, inputs))
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        raise TypeError(
            f"loss_fn must return a scalar tensor, got {type(loss).__name__}"
            + (f" of shape {tuple(loss.shape)}" if isinstance(loss, torch.Tensor) else "")
        )
    return loss.reshape(())


def _backward(model, inputs: Any, loss_fn: LossFn) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    loss = _loss(model, to_device(inputs, model_device(model)), loss_fn)
    loss.backward()
    return loss


def _ignored(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


# --------------------------------------------------------------------------- #
# Gradients
# --------------------------------------------------------------------------- #
def assert_gradients_finite(model, inputs: Any, loss_fn: LossFn) -> torch.Tensor:
    """Run one backward pass and assert the loss and every gradient are finite.

    The model's ``.grad`` fields are overwritten. Returns the loss.
    """
    loss = _backward(model, inputs, loss_fn)
    if not torch.isfinite(loss):
        raise AssertionError(f"loss is not finite: {loss.item()}")
    bad = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        nan = int(torch.isnan(param.grad).sum())
        inf = int(torch.isinf(param.grad).sum())
        if nan or inf:
            bad.append(f"{name}: {nan} NaN and {inf} Inf of {param.grad.numel()} values")
    if bad:
        raise AssertionError(
            format_problems(f"Non-finite gradients in {len(bad)} parameter(s):", bad)
            + "\nHint: look for log(0), sqrt(0), division by a tiny norm, or fp16 overflow."
        )
    return loss


def assert_all_params_receive_grad(
    model,
    inputs: Any,
    loss_fn: LossFn,
    *,
    ignore: Sequence[str] = (),
    allow_zero: bool = False,
) -> None:
    """Assert every trainable parameter gets a gradient from ``loss_fn``.

    Catches layers that are defined but never used, outputs detached by mistake
    (``.detach()``, ``.item()``, ``torch.no_grad()`` in forward) and heads that the
    loss ignores. A gradient that is exactly zero everywhere also fails unless
    ``allow_zero=True``. ``ignore`` takes fnmatch patterns of parameter names, e.g.
    ``["pooler.*"]``. The model's ``.grad`` fields are overwritten.
    """
    _backward(model, inputs, loss_fn)
    missing, zero = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad or _ignored(name, ignore):
            continue
        if param.grad is None:
            missing.append(name)
        elif not allow_zero and not bool(param.grad.any()):
            zero.append(name)
    problems = [f"{n}: no gradient (not connected to the loss)" for n in missing]
    problems += [f"{n}: gradient is zero everywhere" for n in zero]
    if problems:
        raise AssertionError(
            format_problems(
                f"{len(problems)} trainable parameter(s) receive no useful gradient:", problems
            )
            + "\nHint: unused layers, .detach()/.item() in forward, or a loss that ignores an "
            "output. Freeze intentionally unused parameters (requires_grad=False) or pass ignore=."
        )


def assert_frozen(
    model,
    inputs: Any,
    loss_fn: LossFn,
    params: str | Sequence[str],
    *,
    optimizer: Callable | None = None,
    steps: int = 1,
    check_buffers: bool = True,
) -> None:
    """Assert the parameters matching ``params`` stay frozen through real training steps.

    ``params`` are fnmatch patterns of parameter names, e.g. ``"encoder.*"``. A deep copy
    of the model is trained in training mode for ``steps`` steps with ``optimizer`` (a
    ``fn(params) -> Optimizer``; default AdamW with weight decay, over *all* parameters like
    a typical training loop). Fails when a frozen parameter receives a gradient, is changed
    by the optimizer (e.g. weight decay on a zero gradient), or, with ``check_buffers``,
    when a buffer under the same names changes: a frozen BatchNorm left in training mode
    still updates its running statistics. The model you pass is never modified.
    """
    patterns = [params] if isinstance(params, str) else list(params)
    trainee = copy.deepcopy(model).train()
    named_params = dict(trainee.named_parameters())
    frozen = {n: p for n, p in named_params.items() if _ignored(n, patterns)}
    buffers = {n: b for n, b in trainee.named_buffers() if _ignored(n, patterns)} if check_buffers else {}
    if not frozen and not buffers:
        raise ValueError(
            f"no parameter or buffer matches {patterns}; names look like {list(named_params)[:5]}"
        )
    before = {n: t.detach().clone() for n, t in {**frozen, **buffers}.items()}
    opt = (optimizer(trainee.parameters()) if optimizer is not None
           else torch.optim.AdamW(trainee.parameters(), lr=1e-3, weight_decay=0.01))
    inputs = to_device(inputs, model_device(trainee))
    problems = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        _loss(trainee, inputs, loss_fn).backward()
        for name, param in frozen.items():
            if param.grad is not None and bool(param.grad.any()):
                problems.append(
                    f"{name}: received a gradient at step {step + 1} "
                    f"(max |grad| {float(param.grad.abs().max()):.3g})"
                )
        opt.step()
    for name in frozen:
        if not torch.equal(frozen[name].detach(), before[name]):
            problems.append(
                f"{name}: changed by the optimizer "
                f"(max change {float((frozen[name].detach() - before[name]).abs().max()):.3g})"
            )
    for name in buffers:
        if not torch.equal(buffers[name], before[name]):
            problems.append(f"{name}: buffer changed during training")
    if problems:
        hints = []
        if any("buffer changed" in p for p in problems):
            hints.append("frozen modules with BatchNorm/running statistics must stay in eval mode "
                         "(override train() to keep them in eval)")
        if any("received a gradient" in p for p in problems):
            hints.append("set requires_grad_(False) on the frozen parameters, or run the frozen part "
                         "under torch.no_grad()")
        if any("changed by the optimizer" in p for p in problems):
            hints.append("leave frozen parameters out of the optimizer, or keep their grad None "
                         "(weight decay moves parameters even with a zero gradient)")
        raise AssertionError(
            format_problems(f"{len(problems)} problem(s) with frozen parameters {patterns}:", problems)
            + "".join(f"\nHint: {h}." for h in dict.fromkeys(hints))
        )


def assert_gradcheck(
    fn: Callable,
    inputs: Any,
    *,
    eps: float = 1e-6,
    atol: float | None = None,
    rtol: float | None = None,
) -> None:
    """Compare analytical gradients (autograd) with finite differences.

    Use it for custom ``autograd.Function``s and hand-written backward passes. ``fn`` is
    a function or module; it runs in float64 (modules are deep-copied and converted).
    Gradients are checked with respect to the floating-point input tensors. Keep the
    inputs small: the cost grows with their number of elements.
    """
    rtol, atol = tolerances.for_check("gradcheck", "cpu", rtol, atol)
    if isinstance(fn, torch.nn.Module):
        fn = copy.deepcopy(fn).double()
    inputs = tree_map(
        lambda t: t.detach().clone().requires_grad_(t.is_floating_point()),
        cast_floating(inputs, torch.float64),
    )
    args = tuple(inputs) if isinstance(inputs, (list, tuple)) else (inputs,)
    if not any(isinstance(a, torch.Tensor) and a.requires_grad for a in args):
        raise ValueError("assert_gradcheck needs at least one floating-point input tensor")

    def flat_fn(*a):
        out = fn(*a)
        return out if isinstance(out, torch.Tensor) else tuple(
            t for t in out if isinstance(t, torch.Tensor)
        )

    try:
        torch.autograd.gradcheck(flat_fn, args, eps=eps, atol=atol, rtol=rtol)
    except Exception as exc:  # GradcheckError subclasses RuntimeError
        if "Jacobian mismatch" not in str(exc):
            raise
        analytical, numerical = _jacobians(flat_fn, args, eps)
        problems = leaves_diff(analytical, numerical, rtol=rtol, atol=atol, check_device=False)
        raise AssertionError(
            format_problems(
                "Analytical (autograd) and numerical (finite-difference) gradients differ:",
                problems,
            )
            + "\n(actual = autograd, expected = finite differences; "
            "jacobian[output_i][input_j] has shape output.shape + input.shape)"
        ) from None


def _jacobians(fn: Callable, args: tuple, eps: float):
    """Analytical and central-difference jacobians, keyed by readable paths."""
    out = fn(*args)
    outputs = (out,) if isinstance(out, torch.Tensor) else tuple(out)
    analytical, numerical = {}, {}
    for j, arg in enumerate(args):
        if not (isinstance(arg, torch.Tensor) and arg.requires_grad):
            continue
        for i, o in enumerate(outputs):
            key = f"jacobian[output_{i}][input_{j}]"
            jac = torch.zeros(*o.shape, *arg.shape, dtype=torch.float64)
            for idx in range(o.numel()):
                grad = torch.autograd.grad(
                    o.reshape(-1)[idx], arg, retain_graph=True, allow_unused=True
                )[0]
                if grad is not None:
                    jac.view(o.numel(), -1)[idx] = grad.reshape(-1)
            analytical[key] = jac

            num = torch.zeros_like(jac).view(o.numel(), -1)
            flat = arg.detach().reshape(-1)
            with torch.no_grad():
                for k in range(flat.numel()):
                    shifted = []
                    for sign in (1, -1):
                        bumped = flat.clone()
                        bumped[k] += sign * eps
                        call = list(args)
                        call[j] = bumped.view_as(arg)
                        res = fn(*call)
                        res = res if isinstance(res, torch.Tensor) else tuple(res)[i]
                        shifted.append(res.reshape(-1))
                    num[:, k] = (shifted[0] - shifted[1]) / (2 * eps)
            numerical[key] = num.view_as(jac)
    return analytical, numerical


# --------------------------------------------------------------------------- #
# Learning
# --------------------------------------------------------------------------- #
@dataclass
class StepRecord:
    """Telemetry of one training step, recorded by :func:`run_training`.

    Per-optimizer values are keyed by optimizer name (``"optimizer"`` for a single one).
    ``grad_norm``, ``lrs`` and ``nonfinite_grads`` are read when ``optimizer.step()``
    runs, so they describe the update that was applied.
    """

    step: int
    losses: dict[str, float] = field(default_factory=dict)
    optimizer_steps: dict[str, int] = field(default_factory=dict)
    grad_norm: dict[str, float] = field(default_factory=dict)  # total L2 norm before the step
    update_norm: dict[str, float] = field(default_factory=dict)  # L2 norm of the parameter change
    lrs: dict[str, list[float]] = field(default_factory=dict)  # per parameter group
    nonfinite_grads: list[str] = field(default_factory=list)  # parameters with NaN/Inf grads at a step
    loss_scale: float | None = None  # GradScaler scale after the step
    amp_problems: list[str] = field(default_factory=list)

    @property
    def loss(self) -> float | None:
        """The ``"loss"`` entry, or the first loss the step returned."""
        if "loss" in self.losses:
            return self.losses["loss"]
        return next(iter(self.losses.values()), None)

    @property
    def skipped(self) -> bool:
        """No optimizer stepped (not called, or skipped by a GradScaler after an overflow)."""
        return not any(self.optimizer_steps.values())


def _g(value: float | None) -> str:
    return "-" if value is None else f"{value:.4g}"


@dataclass
class TrainingRun:
    """Loss curve of a short training run (``losses[0]`` is before the first step).

    Runs from :func:`run_training` also hold per-step telemetry (``records``), the trained
    ``setup`` and the optimizer coverage problems found before training, and offer
    assertions: :meth:`assert_healthy`, :meth:`assert_finite`,
    :meth:`assert_no_skipped_updates`, :meth:`assert_grad_norm_below`, :meth:`assert_lr`
    and :meth:`assert_loss_decreases`. Each returns the run, so they chain.
    """

    losses: list[float] = field(default_factory=list)
    records: list[StepRecord] = field(default_factory=list)
    setup: Any = None
    coverage_problems: list[str] = field(default_factory=list)

    @property
    def initial(self) -> float:
        return self.losses[0]

    @property
    def final(self) -> float:
        return self.losses[-1]

    def summary(self, points: int = 6) -> str:
        n = len(self.losses)
        picks = sorted({round(i * (n - 1) / (points - 1)) for i in range(points)}) if n > 1 else [0]
        return ", ".join(f"step {i}: {self.losses[i]:.4g}" for i in picks)

    # ----------------------------------------------------------------------- #
    # telemetry
    # ----------------------------------------------------------------------- #
    def table(self, highlight: Sequence[int] = (), limit: int = 12) -> str:
        """Per-step telemetry; long runs show the first and last steps plus ``highlight``."""
        records = self.records
        shown: list[StepRecord | None]
        if len(records) > limit:
            keep = {*range(3), *range(len(records) - 3, len(records))}
            keep |= {i for i, r in enumerate(records) if r.step in set(highlight)}
            shown = []
            for i, record in enumerate(records):
                if i in keep:
                    shown.append(record)
                elif shown and shown[-1] is not None:
                    shown.append(None)
        else:
            shown = list(records)
        scaled = any(r.loss_scale is not None for r in records)
        header = ["step", "loss", "grad norm", "update", "lr", "opt steps"] + (["scale"] if scaled else [])
        rows = []
        for r in shown:
            if r is None:
                rows.append(["..."] + [""] * (len(header) - 1))
                continue
            names = list(r.optimizer_steps)
            row = [
                str(r.step),
                _g(r.loss),
                "/".join(_g(r.grad_norm.get(n)) for n in names),
                "/".join(_g(r.update_norm.get(n)) for n in names),
                "/".join(",".join(f"{lr:.3g}" for lr in r.lrs.get(n, [])) or "-" for n in names),
                "/".join(str(r.optimizer_steps[n]) for n in names) + (" skipped" if r.skipped else ""),
            ]
            if scaled:
                row.append(_g(r.loss_scale))
            rows.append(row)
        widths = [max(len(h), *(len(row[i]) for row in rows)) if rows else len(h) for i, h in enumerate(header)]
        lines = ["  ".join(c.ljust(w) for c, w in zip(header, widths)).rstrip()]
        lines += ["  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip() for row in rows]
        return "\n".join("  " + line for line in lines)

    def _fail(self, header: str, problems: list[str], hints: Sequence[str] = (),
              highlight: Sequence[int] = ()) -> None:
        text = format_problems(header, problems)
        if self.records:
            text += "\nTelemetry:\n" + self.table(highlight)
        text += "".join(f"\nHint: {h}" for h in dict.fromkeys(hints))
        raise AssertionError(text)

    def _need_records(self, what: str) -> None:
        if not self.records:
            raise ValueError(f"{what} needs per-step telemetry: use a run from run_training()")

    # ----------------------------------------------------------------------- #
    # assertions
    # ----------------------------------------------------------------------- #
    def assert_finite(self) -> TrainingRun:
        """Losses, the gradients applied by every optimizer step and the final parameters
        are finite."""
        self._need_records("assert_finite")
        problems, steps = [], []
        for r in self.records:
            bad = [k for k, v in r.losses.items() if not math.isfinite(v)]
            if bad:
                problems.append(f"step {r.step}: non-finite {', '.join(bad)} ({', '.join(_g(r.losses[k]) for k in bad)})")
                steps.append(r.step)
            if r.nonfinite_grads:
                names = ", ".join(r.nonfinite_grads[:3]) + (" ..." if len(r.nonfinite_grads) > 3 else "")
                problems.append(f"step {r.step}: optimizer.step() applied non-finite gradients ({names})")
                steps.append(r.step)
        if self.setup is not None:
            for name, param in self.setup.model.named_parameters():
                if not bool(torch.isfinite(param.detach()).all()):
                    problems.append(f"{name}: parameter is not finite after training")
        if problems:
            self._fail(f"Training produced non-finite values ({len(problems)} problem(s)):", problems, [
                "the first non-finite step is where to look: log(0), division by a tiny norm, a "
                "learning rate that is too high, or float16 overflow (use a GradScaler).",
            ], steps)
        return self

    def assert_no_skipped_updates(self, *, every: int = 1, max_skipped: int = 0) -> TrainingRun:
        """Every ``every``-th step updates the parameters: some optimizer steps, and every
        optimizer that steps changes them. Up to ``max_skipped`` steps may skip the update
        (e.g. early GradScaler overflows)."""
        self._need_records("assert_no_skipped_updates")
        skipped = [r for r in self.records if (r.step + 1) % every == 0 and r.skipped]
        problems = [f"step {r.step}: no optimizer step" for r in skipped] if len(skipped) > max_skipped else []
        hints = ["optimizer.step() isn't reached on every step (an early return, a wrong accumulation "
                 "condition), or a GradScaler skips it after inf/NaN gradients."] if problems else []
        for r in self.records:
            for name, norm in r.update_norm.items():
                if r.optimizer_steps.get(name) and norm == 0.0:
                    problems.append(f"step {r.step}: {name!r} stepped but no parameter changed")
                    hints.append("a learning rate of 0 (check the scheduler and warmup), gradients that are "
                                 "None, or parameters missing from the optimizer.")
        if problems:
            self._fail(f"{len(problems)} step(s) did not update the model:", problems, hints,
                       [r.step for r in skipped])
        return self

    def assert_grad_norm_below(self, max_norm: float) -> TrainingRun:
        """The total gradient norm stays below ``max_norm`` at every optimizer step."""
        self._need_records("assert_grad_norm_below")
        problems, steps = [], []
        for r in self.records:
            for name, norm in r.grad_norm.items():
                if not (norm <= max_norm):  # NaN fails too
                    problems.append(f"step {r.step}: {name!r} gradient norm {norm:.4g} > {max_norm:g}")
                    steps.append(r.step)
        if problems:
            self._fail("Exploding gradients:", problems, [
                "lower the learning rate, clip gradients (clip_grad_norm_), check the loss scale "
                "and normalization layers.",
            ], steps)
        return self

    def assert_lr(
        self,
        expected: Callable[[int], float] | Sequence[float],
        *,
        optimizer: str | None = None,
        group: int = 0,
        rtol: float = 1e-6,
    ) -> TrainingRun:
        """The learning rate used at each step matches ``expected``: a function of the step
        index or one value per step. ``optimizer`` names it when there are several."""
        self._need_records("assert_lr")
        names = list(self.records[0].lrs) or list(self.records[0].optimizer_steps)
        name = optimizer if optimizer is not None else names[0]
        if optimizer is None and len(names) > 1:
            raise ValueError(f"several optimizers: pass optimizer= one of {names}")
        problems, steps = [], []
        for i, r in enumerate(self.records):
            if not callable(expected) and i >= len(expected):
                break
            want = expected(r.step) if callable(expected) else expected[i]
            lrs = r.lrs.get(name)
            if lrs is None or group >= len(lrs):
                raise ValueError(f"no learning rate recorded for {name!r} group {group}")
            if not math.isclose(lrs[group], want, rel_tol=rtol, abs_tol=1e-12):
                problems.append(f"step {r.step}: lr {lrs[group]:.6g}, expected {want:.6g}")
                steps.append(r.step)
        if problems:
            self._fail(f"Unexpected learning rates for {name!r} group {group}:", problems, [
                "scheduler.step() called before optimizer.step(), once per batch instead of per epoch "
                "(or the reverse), or warmup configured in the wrong unit.",
            ], steps)
        return self

    def assert_loss_decreases(self, min_relative_decrease: float = 0.1) -> TrainingRun:
        """The mean loss of the last quarter of the steps is ``min_relative_decrease`` below
        that of the first quarter (averaging smooths out batch-to-batch noise)."""
        losses = [x for x in self.losses if x is not None]
        if len(losses) < 2:
            raise ValueError("need at least two recorded losses")
        window = max(1, len(losses) // 4)
        first = sum(losses[:window]) / window
        last = sum(losses[-window:]) / window
        target = first * (1 - min_relative_decrease) if first > 0 else first
        if not (math.isfinite(last) and last < target):
            self._fail(f"Loss did not decrease by {min_relative_decrease:.0%}:", [
                f"mean of the first {window} step(s) {first:.4g} -> last {window} step(s) {last:.4g}",
            ], ["check the loss sign, the learning rate, and that the optimizer sees the parameters "
                "(assert_optimizer_coverage)."])
        return self

    def assert_healthy(
        self,
        *,
        max_grad_norm: float | None = None,
        min_relative_decrease: float | None = None,
        every: int = 1,
        max_skipped: int = 0,
    ) -> TrainingRun:
        """Optimizer coverage, finite values and no skipped updates; plus a gradient-norm
        bound and a loss decrease when given."""
        if self.coverage_problems:
            self._fail("Optimizer coverage:", self.coverage_problems, [_COVERAGE_HINT])
        self.assert_finite()
        self.assert_no_skipped_updates(every=every, max_skipped=max_skipped)
        if max_grad_norm is not None:
            self.assert_grad_norm_below(max_grad_norm)
        if min_relative_decrease is not None:
            self.assert_loss_decreases(min_relative_decrease)
        return self


_COVERAGE_HINT = ("build the optimizers from the final model (after deepcopy, re-creation or loading), "
                  "give each parameter to exactly one optimizer, and freeze parameters on purpose "
                  "(requires_grad_(False)) or pass ignore=.")


def _train(
    model,
    inputs: Any,
    loss_fn: LossFn,
    *,
    steps: int,
    lr: float,
    optimizer: Callable | None,
    goal: Callable[[float], float] | None = None,
) -> TrainingRun:
    """Train a copy; with ``goal(initial_loss) -> threshold``, stop once the loss reaches it."""
    trainee = copy.deepcopy(model).train()
    inputs = to_device(inputs, model_device(trainee))
    params = [p for p in trainee.parameters() if p.requires_grad]
    if not params:
        raise ValueError("the model has no trainable parameters")
    opt = optimizer(params) if optimizer is not None else torch.optim.Adam(params, lr=lr)
    run = TrainingRun()
    threshold = None
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = _loss(trainee, inputs, loss_fn)
        run.losses.append(float(loss.detach()))
        if not math.isfinite(run.losses[-1]):
            return run
        if goal is not None:
            threshold = goal(run.losses[0]) if threshold is None else threshold
            if run.losses[-1] <= threshold:
                return run
        loss.backward()
        opt.step()
    with torch.no_grad():
        run.losses.append(float(_loss(trainee, inputs, loss_fn)))
    return run


def assert_loss_decreases(
    model,
    inputs: Any,
    loss_fn: LossFn,
    *,
    steps: int = 20,
    min_relative_decrease: float = 0.1,
    lr: float = 1e-3,
    optimizer: Callable | None = None,
) -> TrainingRun:
    """Assert a few optimizer steps on one batch lower the loss by ``min_relative_decrease``.

    A fast smoke test for "the training loop is wired up": wrong sign, frozen
    parameters, a learning rate of zero or a broken optimizer all fail it.
    ``optimizer`` is ``fn(params) -> torch.optim.Optimizer`` (default: Adam with ``lr``).
    Trains a copy in training mode. Returns the :class:`TrainingRun`.
    """
    run = _train(model, inputs, loss_fn, steps=steps, lr=lr, optimizer=optimizer)
    target = run.initial * (1 - min_relative_decrease) if run.initial > 0 else run.initial
    if not (math.isfinite(run.final) and run.final < target):
        raise AssertionError(
            f"Loss did not decrease by {min_relative_decrease:.0%} in {steps} steps "
            f"({run.initial:.4g} -> {run.final:.4g}).\n  {run.summary()}\n"
            "Hint: check the loss sign, that parameters require grad and reach the "
            "optimizer, and the learning rate."
        )
    return run


def assert_overfits_one_batch(
    model,
    inputs: Any,
    loss_fn: LossFn,
    *,
    steps: int = 200,
    max_final_ratio: float = 0.05,
    target_loss: float | None = None,
    lr: float = 1e-2,
    optimizer: Callable | None = None,
) -> TrainingRun:
    """Assert the model can memorize a single batch (the classic first debugging step).

    Passes once the loss drops to ``target_loss`` or, by default, to
    ``max_final_ratio`` x the initial loss. A model that can't overfit one small batch
    has a bug (or no capacity): wrong labels alignment, a too-strong regularizer, a
    disconnected layer... Trains a copy in training mode, stopping early on success.
    Returns the :class:`TrainingRun`.
    """
    def goal_for(initial: float) -> float:
        return target_loss if target_loss is not None else initial * max_final_ratio

    run = _train(model, inputs, loss_fn, steps=steps, lr=lr, optimizer=optimizer, goal=goal_for)
    goal = goal_for(run.initial)
    if not run.final <= goal:
        raise AssertionError(
            f"Could not overfit one batch in {steps} steps: loss {run.initial:.4g} -> "
            f"{run.final:.4g}, goal <= {goal:.4g}.\n  {run.summary()}\n"
            "Hint: a model that cannot memorize one batch usually has a bug: misaligned "
            "targets, dropout/weight decay too strong, a disconnected layer "
            "(assert_all_params_receive_grad), or too low a learning rate."
        )
    return run
