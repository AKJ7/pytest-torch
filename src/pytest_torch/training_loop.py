"""Training loops: run your real training step for a few steps and check what it does.

A *setup* is what your training code builds: model, optimizers, schedulers, a GradScaler
and any other stateful objects. A *step* is your training step, ``step(setup, batch)``,
which does the forward pass, backward pass and optimizer and scheduler steps, and returns
the loss (a tensor, a float, or a dict of named losses)::

    def make_setup():
        model = Net()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return TrainingSetup(model, optimizer, schedulers=torch.optim.lr_scheduler.StepLR(optimizer, 10))

    def train_step(setup, batch):
        x, y = batch
        loss = F.cross_entropy(setup.model(x), y)
        loss.backward()
        setup.optimizer.step()
        setup.optimizer.zero_grad()
        setup.scheduler.step()
        return loss

    run_training(make_setup, batches, train_step, steps=20).assert_healthy()

Everything here builds fresh setups with ``make_setup`` and seeds before each run, so
runs are reproducible and comparable.
"""
from __future__ import annotations

import copy
import functools
import inspect
import itertools
import math
import random
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

from . import tolerances
from .assertions import format_problems, leaves_diff
from .training import _COVERAGE_HINT, StepRecord, TrainingRun, _ignored
from .utils import _is_dataclass_instance, find_batch_size, flatten, model_device, seed_everything, to_device, tree_map

StepFn = Callable[["TrainingSetup", Any], Any]


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
def _named(obj: Any, prefix: str) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    if isinstance(obj, (list, tuple)):
        if len(obj) == 1:
            return {prefix: obj[0]}
        return {f"{prefix}[{i}]": o for i, o in enumerate(obj)}
    return {prefix: obj}


@dataclass
class TrainingSetup:
    """Everything a training step uses.

    ``optimizers`` and ``schedulers`` take one object, a list, or a dict of named ones
    (stored as dicts). ``state`` holds other objects with ``state_dict()`` and
    ``load_state_dict()`` that a checkpoint must carry, such as an EMA model.
    ``step_index`` is the index of the current step (set by the runner).
    """

    model: Any
    optimizers: Any
    schedulers: Any = None
    scaler: Any = None
    state: dict[str, Any] = field(default_factory=dict)
    step_index: int = 0

    def __post_init__(self) -> None:
        self.optimizers = _named(self.optimizers, "optimizer")
        self.schedulers = _named(self.schedulers, "scheduler")
        if not self.optimizers:
            raise ValueError("a TrainingSetup needs at least one optimizer")

    @property
    def optimizer(self) -> Any:
        """The only optimizer."""
        return self._only(self.optimizers, "optimizer")

    @property
    def scheduler(self) -> Any:
        """The only scheduler."""
        return self._only(self.schedulers, "scheduler")

    @staticmethod
    def _only(objs: dict[str, Any], kind: str) -> Any:
        if len(objs) != 1:
            raise AttributeError(f"{len(objs)} {kind}s: use setup.{kind}s[name], names: {list(objs)}")
        return next(iter(objs.values()))


def _build(make_setup: Callable, device: Any) -> TrainingSetup:
    params = [
        p for p in inspect.signature(make_setup).parameters.values()
        if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    built = make_setup(device) if params else make_setup()
    if isinstance(built, tuple):
        built = TrainingSetup(*built)
    if not isinstance(built, TrainingSetup):
        raise TypeError(
            "make_setup must return a TrainingSetup or a (model, optimizer[, scheduler]) tuple, "
            f"got {type(built).__name__}"
        )
    built.model.to(device)  # keeps parameter identity: the optimizers stay valid
    built.model.train()
    return built


def _batch_source(data: Any, steps: int) -> Callable[[int], Any]:
    """``batch_for(step)``: a function of the step, a list of batches (cycled), an
    iterable such as a DataLoader (its first ``steps`` batches, cycled), or one batch."""
    if callable(data) and not isinstance(data, torch.nn.Module):
        return data
    if isinstance(data, list):
        if not data:
            raise ValueError("data is an empty list")
        return lambda i: data[i % len(data)]
    if isinstance(data, (torch.Tensor, tuple, Mapping)) or _is_dataclass_instance(data):
        return lambda _: data
    if hasattr(data, "__iter__"):
        batches = list(itertools.islice(iter(data), steps))
        if not batches:
            raise ValueError("data yielded no batches")
        return lambda i: batches[i % len(batches)]
    return lambda _: data


def standard_step(
    loss_fn: Callable[[Any, Any], torch.Tensor],
    *,
    accumulation_steps: int = 1,
    max_grad_norm: float | None = None,
) -> StepFn:
    """A conventional training step, for when you don't have your own.

    ``loss_fn(model, batch)`` returns the loss. Gradients accumulate over
    ``accumulation_steps`` steps (the loss is divided accordingly); then every optimizer
    steps (through ``setup.scaler`` when there is one, with gradients clipped to
    ``max_grad_norm`` after unscaling), gradients are reset and every scheduler steps.
    """
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1")

    def step(setup: TrainingSetup, batch: Any) -> torch.Tensor:
        loss = loss_fn(setup.model, batch)
        scaled = loss / accumulation_steps
        scaler = setup.scaler
        (scaler.scale(scaled) if scaler is not None else scaled).backward()
        if (setup.step_index + 1) % accumulation_steps == 0:
            for optimizer in setup.optimizers.values():
                if max_grad_norm is not None:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    params = [p for g in optimizer.param_groups for p in g["params"]]
                    torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                if scaler is not None:
                    scaler.step(optimizer)
                else:
                    optimizer.step()
            if scaler is not None:
                scaler.update()
            for optimizer in setup.optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            for scheduler in setup.schedulers.values():
                scheduler.step()
        return loss.detach()

    return step


def _resolve_step(step: StepFn | None, loss_fn: Callable | None) -> StepFn:
    if (step is None) == (loss_fn is None):
        raise TypeError("pass either step=(setup, batch) -> loss or loss_fn=(model, batch) -> loss")
    return step if step is not None else standard_step(loss_fn)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Optimizer coverage
# --------------------------------------------------------------------------- #
def optimizer_coverage_problems(
    model: Any, optimizers: Any, *, ignore: Sequence[str] = (), allow_frozen: bool = True
) -> list[str]:
    """Problems found by :func:`assert_optimizer_coverage` (empty when there are none)."""
    optimizers = _named(optimizers, "optimizer")
    names: dict[int, str] = {}
    for name, param in model.named_parameters():
        names.setdefault(id(param), name)  # tied weights: the first name
    params = {id(p): p for _, p in model.named_parameters()}
    places: dict[int, list[str]] = defaultdict(list)
    foreign: dict[str, list[tuple]] = defaultdict(list)
    for opt_name, optimizer in optimizers.items():
        groups = optimizer.param_groups
        for index, group in enumerate(groups):
            where = f"{opt_name!r} group {index}" if len(groups) > 1 else repr(opt_name)
            for param in group["params"]:
                places[id(param)].append(where)
                if id(param) not in names:
                    foreign[where].append(tuple(param.shape))
    problems = []
    for pid, name in names.items():
        if _ignored(name, ignore):
            continue
        param = params[pid]
        if param.requires_grad and pid not in places:
            problems.append(f"{name}: trainable but in no optimizer, so never updated")
        elif len(places[pid]) > 1:
            problems.append(f"{name}: in {', '.join(places[pid])}: updated {len(places[pid])} times per step")
        elif not allow_frozen and not param.requires_grad and pid in places:
            problems.append(f"{name}: frozen (requires_grad=False) but in {places[pid][0]}")
    for where, shapes in foreign.items():
        problems.append(
            f"{where}: {len(shapes)} tensor(s) that aren't parameters of the model (e.g. shape "
            f"{shapes[0]}): the optimizer was built for another copy of the model"
        )
    return problems


def assert_optimizer_coverage(
    model: Any, optimizers: Any, *, ignore: Sequence[str] = (), allow_frozen: bool = True
) -> None:
    """Assert every trainable parameter is in exactly one optimizer.

    Fails on trainable parameters that no optimizer updates, parameters updated by
    several optimizers or parameter groups, and optimizers holding tensors that aren't
    the model's parameters (built before a ``deepcopy``, re-creation or reload of the
    model). With ``allow_frozen=False``, frozen parameters inside an optimizer fail too.
    ``optimizers`` is one optimizer, a list or a dict; ``ignore`` takes fnmatch patterns
    of parameter names.
    """
    problems = optimizer_coverage_problems(model, optimizers, ignore=ignore, allow_frozen=allow_frozen)
    if problems:
        raise AssertionError(
            format_problems(f"Optimizer coverage: {len(problems)} problem(s):", problems)
            + f"\nHint: {_COVERAGE_HINT}"
        )


# --------------------------------------------------------------------------- #
# Telemetry
# --------------------------------------------------------------------------- #
def _losses(result: Any) -> dict[str, float]:
    def value(v: Any) -> float:
        if isinstance(v, torch.Tensor):
            if v.numel() != 1:
                raise TypeError(f"a loss must be a scalar, got a tensor of shape {tuple(v.shape)}")
            return float(v.detach())
        return float(v)

    if result is None:
        return {}
    if isinstance(result, Mapping):
        return {str(k): value(v) for k, v in result.items()}
    return {"loss": value(result)}


def _clip_functions() -> list[tuple[Any, str]]:
    import torch.nn.utils.clip_grad as clip_grad

    return [(torch.nn.utils, "clip_grad_norm_"), (torch.nn.utils, "clip_grad_value_"),
            (clip_grad, "clip_grad_norm_"), (clip_grad, "clip_grad_value_")]


class _Recorder:
    """Wrappers around optimizer steps (and a GradScaler and clipping) that fill StepRecords."""

    def __init__(self, setup: TrainingSetup, *, track_updates: bool = True, capture_grads: bool = False,
                 track_amp: bool = False) -> None:
        self.setup = setup
        self.track_updates = track_updates
        self.capture_grads = capture_grads
        self.names: dict[int, str] = {}
        for name, param in setup.model.named_parameters():
            self.names.setdefault(id(param), name)
        self.record = StepRecord(step=0)
        self.captured: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)  # step -> grads
        self._before: dict[str, list[torch.Tensor]] = {}
        self._undo: list[Callable[[], None]] = []
        scaler = setup.scaler
        self.amp = track_amp and scaler is not None and _scaler_enabled(scaler)
        self._inside_scaler_step = 0
        self._unscaled: set[int] = set()
        self._calls: dict[str, int] = defaultdict(int)

    def _name(self, param: torch.Tensor) -> str:
        return self.names.get(id(param), f"<tensor of shape {tuple(param.shape)} outside the model>")

    # hooks
    def _pre(self, opt_name: str, optimizer: Any, _args: Any, _kwargs: Any) -> None:
        rec = self.record
        params = [p for g in optimizer.param_groups for p in g["params"]]
        grads = [(p, p.grad) for p in params if p.grad is not None]
        norms = [g.detach().float().norm() for _, g in grads]
        rec.grad_norm[opt_name] = float(torch.stack(norms).norm()) if norms else 0.0
        rec.nonfinite_grads += [self._name(p) for p, g in grads if not bool(torch.isfinite(g).all())]
        rec.lrs[opt_name] = [float(g["lr"]) for g in optimizer.param_groups]
        if self.track_updates:
            self._before[opt_name] = [p.detach().clone() for p in params]
        if self.capture_grads:
            self.captured[rec.step].update({self._name(p): g.detach().clone() for p, g in grads})
        if self.amp and not self._inside_scaler_step:
            rec.amp_problems.append(
                f"{opt_name!r}.step() called directly instead of scaler.step({opt_name}): it applies "
                "scaled gradients and skips the inf/NaN check"
            )

    def _post(self, opt_name: str, optimizer: Any, _args: Any, _kwargs: Any) -> None:
        rec = self.record
        rec.optimizer_steps[opt_name] = rec.optimizer_steps.get(opt_name, 0) + 1
        before = self._before.pop(opt_name, None)
        if before is not None:
            params = [p for g in optimizer.param_groups for p in g["params"]]
            total = sum(float((p.detach().float() - b.float()).pow(2).sum()) for p, b in zip(params, before))
            rec.update_norm[opt_name] = math.sqrt(total)

    # GradScaler and clipping wrappers
    def _wrap_scaler(self, scaler: Any) -> None:
        original = {m: getattr(scaler, m) for m in ("scale", "unscale_", "step", "update")}

        def scale(outputs: Any) -> Any:
            self._calls["scale"] += 1
            return original["scale"](outputs)

        def unscale_(optimizer: Any) -> Any:
            self._calls["unscale_"] += 1
            self._unscaled.add(id(optimizer))
            return original["unscale_"](optimizer)

        def step(optimizer: Any, *args: Any, **kwargs: Any) -> Any:
            self._calls["step"] += 1
            self._inside_scaler_step += 1
            try:
                return original["step"](optimizer, *args, **kwargs)
            finally:
                self._inside_scaler_step -= 1

        def update(*args: Any, **kwargs: Any) -> Any:
            self._calls["update"] += 1
            self._unscaled.clear()
            return original["update"](*args, **kwargs)

        for name, fn in (("scale", scale), ("unscale_", unscale_), ("step", step), ("update", update)):
            setattr(scaler, name, fn)
            self._undo.append(functools.partial(delattr, scaler, name))

    def _wrap_clipping(self) -> None:
        owners = {
            name: {id(p) for g in opt.param_groups for p in g["params"]}
            for name, opt in self.setup.optimizers.items()
        }
        for module, attr in _clip_functions():
            original = getattr(module, attr)

            def clip(parameters: Any, *args: Any, _original: Any = original, _attr: str = attr, **kwargs: Any) -> Any:
                params = [parameters] if isinstance(parameters, torch.Tensor) else list(parameters)
                ids = {id(p) for p in params}
                for name, opt in self.setup.optimizers.items():
                    if owners[name] & ids and id(opt) not in self._unscaled:
                        self.record.amp_problems.append(
                            f"{_attr}() before scaler.unscale_() for optimizer {name!r}: it clips the "
                            "scaled gradients, so the threshold is off by the loss scale"
                        )
                return _original(params, *args, **kwargs)

            setattr(module, attr, clip)
            self._undo.append(functools.partial(setattr, module, attr, original))

    def _wrap_step(self, name: str, optimizer: Any) -> None:
        # Wraps the instance's step rather than registering step hooks: on torch < 2.3, once
        # torch.compile has run, optimizers created afterwards never call their hooks.
        original = optimizer.step  # may be an LR scheduler's wrapper: its markers are copied
        had_attribute = "step" in vars(optimizer)

        @functools.wraps(original)
        def step(*args: Any, **kwargs: Any) -> Any:
            self._pre(name, optimizer, args, kwargs)
            result = original(*args, **kwargs)
            self._post(name, optimizer, args, kwargs)
            return result

        optimizer.step = step
        if had_attribute:
            self._undo.append(functools.partial(setattr, optimizer, "step", original))
        else:
            self._undo.append(functools.partial(delattr, optimizer, "step"))

    def __enter__(self) -> _Recorder:
        for name, optimizer in self.setup.optimizers.items():
            self._wrap_step(name, optimizer)
        if self.amp:
            self._wrap_scaler(self.setup.scaler)
            self._wrap_clipping()
        return self

    def __exit__(self, *exc: Any) -> None:
        for undo in reversed(self._undo):
            undo()
        self._undo.clear()

    # steps
    def begin(self, step: int) -> None:
        self.record = StepRecord(step=step, optimizer_steps=dict.fromkeys(self.setup.optimizers, 0))
        self._calls.clear()

    def end(self, result: Any) -> StepRecord:
        rec = self.record
        rec.losses = _losses(result)
        for name, optimizer in self.setup.optimizers.items():
            rec.lrs.setdefault(name, [float(g["lr"]) for g in optimizer.param_groups])
        scaler = self.setup.scaler
        if scaler is not None and _scaler_enabled(scaler):
            rec.loss_scale = float(scaler.get_scale())
        if self.amp and (self._calls["step"] or self._calls["unscale_"]) and not self._calls["scale"]:
            rec.amp_problems.append("scaler.step() without scaler.scale(loss): the backward pass used "
                                    "the unscaled loss, so the update is divided by the loss scale")
        if self.amp and self._calls["step"]:
            if not self._calls["update"]:
                rec.amp_problems.append("scaler.step() without scaler.update(): the loss scale never adapts "
                                        "and an overflow is skipped forever")
        return rec


def _scaler_enabled(scaler: Any) -> bool:
    is_enabled = getattr(scaler, "is_enabled", None)
    return bool(is_enabled()) if callable(is_enabled) else True


def _run_steps(setup: TrainingSetup, recorder: _Recorder, step: StepFn, batch_for: Callable[[int], Any],
               indices: range, run: TrainingRun) -> None:
    device = model_device(setup.model)
    for index in indices:
        setup.step_index = index
        recorder.begin(index)
        result = step(setup, to_device(batch_for(index), device))
        record = recorder.end(result)
        run.records.append(record)
        if record.loss is not None:
            run.losses.append(record.loss)


def run_training(
    make_setup: Callable[..., Any],
    data: Any,
    step: StepFn | None = None,
    *,
    loss_fn: Callable[[Any, Any], torch.Tensor] | None = None,
    steps: int = 20,
    device: Any = "cpu",
    seed: int = 0,
    track_updates: bool = True,
) -> TrainingRun:
    """Run ``steps`` training steps and record telemetry for each.

    ``make_setup()`` (or ``make_setup(device)``) returns a :class:`TrainingSetup` or a
    ``(model, optimizer[, scheduler])`` tuple; the model is moved to ``device`` and put
    in training mode. ``data`` is one batch, a list of batches (cycled), an iterable such
    as a DataLoader, or a function ``step_index -> batch``. ``step(setup, batch)`` is your
    training step; or pass ``loss_fn(model, batch)`` to use :func:`standard_step`.

    Returns a :class:`TrainingRun` with ``losses``, per-step ``records`` (losses,
    gradient norm, update size and learning rates per optimizer, optimizer step counts,
    non-finite gradients, GradScaler scale), the trained ``setup`` and the optimizer
    coverage problems, and assertions such as ``run.assert_healthy()``. With
    ``track_updates=False`` parameters aren't copied before each optimizer step (no
    ``update_norm``), which saves memory on big models.
    """
    step = _resolve_step(step, loss_fn)
    seed_everything(seed)
    setup = _build(make_setup, device)
    run = TrainingRun(setup=setup, coverage_problems=optimizer_coverage_problems(setup.model, setup.optimizers))
    with _Recorder(setup, track_updates=track_updates) as recorder:
        _run_steps(setup, recorder, step, _batch_source(data, steps), range(steps), run)
    return run


# --------------------------------------------------------------------------- #
# Checkpoint and resume
# --------------------------------------------------------------------------- #
def _get_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    for kind in ("cuda", "xpu"):
        backend = getattr(torch, kind, None)
        if backend is not None and backend.is_available() and hasattr(backend, "get_rng_state_all"):
            state[kind] = backend.get_rng_state_all()
    return state


def _set_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])
    for kind in ("cuda", "xpu"):
        if kind in state:
            getattr(torch, kind).set_rng_state_all(state[kind])


def training_state(setup: TrainingSetup) -> dict[str, Any]:
    """Everything needed to resume training exactly: model, optimizers, schedulers,
    scaler, ``setup.state`` and the random number generators."""
    return {
        "model": setup.model.state_dict(),
        "optimizers": {n: o.state_dict() for n, o in setup.optimizers.items()},
        "schedulers": {n: s.state_dict() for n, s in setup.schedulers.items()},
        "scaler": setup.scaler.state_dict() if setup.scaler is not None else None,
        "state": {n: o.state_dict() for n, o in setup.state.items()},
        "rng": _get_rng_state(),
    }


def save_training_state(setup: TrainingSetup, path: str | Path) -> None:
    """Save :func:`training_state` to ``path``: a reference checkpoint function."""
    torch.save(training_state(setup), path)


def load_training_state(setup: TrainingSetup, path: str | Path) -> None:
    """Restore a checkpoint written by :func:`save_training_state` into ``setup``."""
    state = torch.load(path, map_location="cpu", weights_only=False)  # our own file: RNG states aren't tensors
    setup.model.load_state_dict(state["model"])
    for name, optimizer in setup.optimizers.items():
        optimizer.load_state_dict(state["optimizers"][name])
    for name, scheduler in setup.schedulers.items():
        scheduler.load_state_dict(state["schedulers"][name])
    if setup.scaler is not None and state["scaler"] is not None:
        setup.scaler.load_state_dict(state["scaler"])
    for name, obj in setup.state.items():
        obj.load_state_dict(state["state"][name])
    _set_rng_state(state["rng"])


def _snapshot(setup: TrainingSetup) -> dict[str, Any]:
    return {
        "model": {k: v.detach().clone() for k, v in setup.model.state_dict().items()},
        "optimizers": {n: copy.deepcopy(o.state_dict()) for n, o in setup.optimizers.items()},
        "lrs": {n: [float(g["lr"]) for g in o.param_groups] for n, o in setup.optimizers.items()},
        "schedulers": {n: copy.deepcopy(s.state_dict()) for n, s in setup.schedulers.items()},
        "scaler": copy.deepcopy(setup.scaler.state_dict()) if setup.scaler is not None else None,
        "state": {n: copy.deepcopy(o.state_dict()) for n, o in setup.state.items()},
        "rng": _get_rng_state(),
    }


def _diff(loaded: Any, saved: Any, path: str) -> list[str]:
    return leaves_diff(dict(flatten(loaded, path)), dict(flatten(saved, path)), rtol=0.0, atol=0.0,
                       check_device=False)


def _first(problems: list[str]) -> str:
    """The path of the first problem from leaves_diff."""
    if not problems:
        return ""
    for prefix, what in (("missing in actual: ", "missing"), ("unexpected in actual: ", "unexpected")):
        if problems[0].startswith(prefix):
            return f"{problems[0][len(prefix):].split(', ')[0]} {what}"
    return problems[0].split(": ")[0]


def _rng_differs(a: Any, b: Any) -> bool:
    if isinstance(a, torch.Tensor):
        return not torch.equal(a, b)
    if isinstance(a, (list, tuple)):
        return len(a) != len(b) or any(_rng_differs(x, y) for x, y in zip(a, b))
    try:
        import numpy as np

        if isinstance(a, np.ndarray):
            return not np.array_equal(a, b)
    except ImportError:
        pass
    return bool(a != b)


def _state_differences(loaded: dict[str, Any], saved: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(definite problems, RNG differences) between the state after loading and when saving."""
    problems = []
    model = _diff(loaded["model"], saved["model"], "model")
    if model:
        problems.append(f"model: {len(model)} tensor(s) differ from the saved weights (e.g. {_first(model)}): "
                        "weights not saved or not restored")
    for name in saved["optimizers"]:
        state = _diff(loaded["optimizers"][name]["state"], saved["optimizers"][name]["state"], "state")
        if state:
            problems.append(f"optimizer {name!r}: state differs (e.g. {_first(state)}): optimizer state not "
                            "restored, so momentum, Adam moments and step counts restart")
        for group, (now, then) in enumerate(zip(loaded["lrs"][name], saved["lrs"][name])):
            if not math.isclose(now, then, rel_tol=1e-12, abs_tol=0.0):
                problems.append(f"optimizer {name!r} group {group}: lr {now:.6g} after loading, {then:.6g} when saved")
    for name in saved["schedulers"]:
        diff = _diff(loaded["schedulers"][name], saved["schedulers"][name], "scheduler")
        if diff:
            problems.append(f"scheduler {name!r}: state differs ({diff[0]}): scheduler state not restored, "
                            "so the learning-rate schedule restarts")
    if saved["scaler"] is not None:
        diff = _diff(loaded["scaler"], saved["scaler"], "scaler")
        if diff:
            problems.append(f"GradScaler: state differs ({diff[0]}): scaler state not restored, so the loss "
                            "scale restarts")
    for name in saved["state"]:
        diff = _diff(loaded["state"][name], saved["state"][name], name)
        if diff:
            problems.append(f"{name}: state differs (e.g. {_first(diff)}): not saved or not restored")
    rng = [kind for kind in saved["rng"] if kind not in loaded["rng"] or _rng_differs(loaded["rng"][kind], saved["rng"][kind])]
    rng_problems = [f"random number generators ({', '.join(rng)}) not restored: this matters when training "
                    "draws random numbers (dropout, augmentation, sampling)"] if rng else []
    return problems, rng_problems


@dataclass
class ResumeReport:
    """Result of :func:`assert_resume_equivalent`."""

    resume_at: int
    continuous: TrainingRun
    resumed: TrainingRun
    state_differences: list[str]  # state after loading vs when saving (RNG included)
    diverged_at: int | None  # first step whose loss differs, if any


def assert_resume_equivalent(
    make_setup: Callable[..., Any],
    data: Any,
    step: StepFn | None = None,
    *,
    loss_fn: Callable[[Any, Any], torch.Tensor] | None = None,
    steps: int = 8,
    resume_at: int | None = None,
    save: Callable[[TrainingSetup, Path], Any] | None = None,
    load: Callable[[TrainingSetup, Path], Any] | None = None,
    device: Any = "cpu",
    seed: int = 0,
    rtol: float | None = None,
    atol: float | None = None,
) -> ResumeReport:
    """Assert that stopping, checkpointing and resuming training changes nothing.

    Trains ``steps`` steps without interruption, saving a checkpoint with ``save(setup,
    path)`` after ``resume_at`` steps (default: half). Then builds a fresh setup, restores
    the checkpoint with ``load(setup, path)`` and trains the remaining steps. Pass your
    real checkpoint functions; the defaults (:func:`save_training_state`,
    :func:`load_training_state`) save everything.

    Fails when the resumed losses or final weights differ from the uninterrupted run, and
    when the state right after loading differs from the state when saving: model weights,
    optimizer state and learning rates, scheduler, GradScaler, ``setup.state``. Random
    number generators that weren't restored are reported as the likely cause when the
    runs diverge (and ignored when they don't, e.g. without dropout).
    """
    step = _resolve_step(step, loss_fn)
    resume_at = steps // 2 if resume_at is None else resume_at
    if not 0 < resume_at < steps:
        raise ValueError(f"resume_at must be between 1 and steps - 1 ({steps - 1}), got {resume_at}")
    save = save or save_training_state
    load = load or load_training_state
    batch_for = _batch_source(data, steps)
    rtol, atol = tolerances.for_check("training_resume", device, rtol, atol)

    with tempfile.TemporaryDirectory(prefix="pytest-torch-resume-") as tmp:
        path = Path(tmp) / "checkpoint.pt"
        seed_everything(seed)
        setup = _build(make_setup, device)
        continuous = TrainingRun(setup=setup)
        with _Recorder(setup) as recorder:
            _run_steps(setup, recorder, step, batch_for, range(resume_at), continuous)
            save(setup, path)
            saved = _snapshot(setup)
            _run_steps(setup, recorder, step, batch_for, range(resume_at, steps), continuous)

        seed_everything(seed + 1)  # a fresh process: nothing carries over but the checkpoint
        fresh = _build(make_setup, device)
        load(fresh, path)
        loaded = _snapshot(fresh)
        resumed = TrainingRun(setup=fresh)
        with _Recorder(fresh) as recorder:
            _run_steps(fresh, recorder, step, batch_for, range(resume_at, steps), resumed)

    definite, rng = _state_differences(loaded, saved)
    expected = continuous.records[resume_at:]
    diverged_at, divergence = None, []
    for want, got in zip(expected, resumed.records):
        a, b = got.loss, want.loss
        if a is not None and b is not None and not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
            diverged_at = got.step
            divergence.append(f"first divergence at step {got.step}: loss {a:.6g} after resuming, "
                              f"{b:.6g} without interruption")
            break
    weights = leaves_diff(dict(flatten(dict(fresh.model.state_dict()), "model")),
                          dict(flatten(dict(setup.model.state_dict()), "model")),
                          rtol=rtol, atol=atol, check_device=False)
    if weights:
        divergence.append(f"final weights: {len(weights)} tensor(s) differ (e.g. {weights[0]})")
    report = ResumeReport(resume_at, continuous, resumed, definite + rng, diverged_at)
    if definite or divergence:
        causes = definite + (rng if divergence else [])
        text = format_problems(
            f"Resuming from a checkpoint saved after step {resume_at} does not reproduce the "
            "uninterrupted run:", divergence or ["losses and weights match so far, but the restored state differs"],
        )
        if causes:
            text += "\n" + format_problems("State after loading vs when saving:", causes)
        else:
            text += ("\nThe state after loading matches the saved one, so the difference comes from outside "
                     "the checkpoint: a stateful data pipeline, plain Python attributes (step counters, EMA "
                     "weights: add them to setup.state), or nondeterministic kernels (--torch-deterministic).")
        raise AssertionError(text + "\nHint: save and restore model, optimizer, scheduler, GradScaler and RNG "
                                    "state together (see save_training_state).")
    return report


# --------------------------------------------------------------------------- #
# Gradient accumulation
# --------------------------------------------------------------------------- #
def split_batch(batch: Any, parts: int) -> list[Any]:
    """Split every tensor whose first dimension is the batch size into ``parts`` equal
    micro-batches; other values are passed to each micro-batch unchanged."""
    size = find_batch_size(batch)
    if size is None or size % parts:
        raise ValueError(f"cannot split a batch of size {size} into {parts} equal micro-batches; pass split=")
    n = size // parts

    def part(i: int) -> Any:
        return tree_map(lambda t: t[i * n:(i + 1) * n] if t.dim() >= 1 and t.shape[0] == size else t, batch)

    return [part(i) for i in range(parts)]


def _flat(grads: Mapping[str, torch.Tensor], keys: list[str]) -> torch.Tensor:
    return torch.cat([grads[k].detach().double().reshape(-1).cpu() for k in keys])


def _no_dropout(model: Any) -> list[Any]:
    """Keep dropout modules inactive in training mode (forward pre-hooks, so train() can't undo it)."""
    handles = []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.p > 0:
            handles.append(module.register_forward_pre_hook(lambda m, _args: setattr(m, "training", False)))
    return handles


def _accumulation_hints(model: Any) -> list[str]:
    hints = []
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        hints.append("BatchNorm computes statistics per micro-batch, so accumulation can't match a full "
                     "batch exactly: compare with a looser rtol/atol, or test with GroupNorm/LayerNorm")
    hints.append("a loss averaged over elements whose count differs between micro-batches (masked tokens) "
                 "needs normalizing by the total count of the full batch")
    return hints


def assert_accumulation_equivalent(
    make_setup: Callable[..., Any],
    data: Any,
    make_step: Callable[[int], StepFn] | None = None,
    *,
    loss_fn: Callable[[Any, Any], torch.Tensor] | None = None,
    accumulation_steps: int = 4,
    cycles: int = 2,
    split: Callable[[Any, int], list[Any]] | None = None,
    disable_dropout: bool = True,
    device: Any = "cpu",
    seed: int = 0,
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    """Assert gradient accumulation over micro-batches matches one step on the full batch.

    ``make_step(accumulation_steps)`` returns your training step configured for that many
    accumulation steps (it can read ``setup.step_index``); or pass ``loss_fn(model,
    batch)`` to test :func:`standard_step`. For each of ``cycles`` batches, one step with
    ``make_step(1)`` on the full batch is compared with ``accumulation_steps`` steps of
    ``make_step(accumulation_steps)`` on its micro-batches (``split(batch, n)``, default
    :func:`split_batch`), from identical fresh setups.

    Fails on optimizer steps before the last micro-batch or missing after it, and on
    gradients or weights that differ, saying when gradients are ``accumulation_steps``
    times too large or too small (loss not divided, or divided twice) and when they carry
    over from the previous cycle (no ``zero_grad()`` after the step). Dropout is disabled
    in both runs by default, since it draws different masks for a split batch.
    """
    if (make_step is None) == (loss_fn is None):
        raise TypeError("pass either make_step=(accumulation_steps) -> step or loss_fn=(model, batch) -> loss")
    if make_step is None:
        make_step = lambda k: standard_step(loss_fn, accumulation_steps=k)  # type: ignore[arg-type]  # noqa: E731
    if accumulation_steps < 2:
        raise ValueError("accumulation_steps must be >= 2")
    k = accumulation_steps
    split = split or split_batch
    batch_for = _batch_source(data, cycles)
    rtol, atol = tolerances.for_check("grad_accumulation", device, rtol, atol)

    def run(n: int) -> tuple[TrainingSetup, _Recorder, list[list[dict[str, int]]], list[dict[str, torch.Tensor]]]:
        seed_everything(seed)
        setup = _build(make_setup, device)
        handles = _no_dropout(setup.model) if disable_dropout else []
        train_step = make_step(n)
        timeline: list[list[dict[str, int]]] = []
        weights = []
        try:
            with _Recorder(setup, track_updates=False, capture_grads=True) as recorder:
                for cycle in range(cycles):
                    batch = to_device(batch_for(cycle), model_device(setup.model))
                    counts = []
                    if n == 1:  # the reference starts from clean gradients, whatever the step does
                        setup.model.zero_grad(set_to_none=True)
                        for optimizer in setup.optimizers.values():
                            optimizer.zero_grad(set_to_none=True)
                    for j, micro in enumerate(split(batch, n) if n > 1 else [batch]):
                        setup.step_index = cycle * n + j
                        recorder.begin(setup.step_index)
                        train_step(setup, micro)
                        counts.append(dict(recorder.end(None).optimizer_steps))
                    timeline.append(counts)
                    weights.append({k_: v.detach().clone() for k_, v in setup.model.state_dict().items()})
        finally:
            for handle in handles:
                handle.remove()
        return setup, recorder, timeline, weights

    _, ref_rec, ref_timeline, ref_weights = run(1)
    accumulated, acc_rec, acc_timeline, acc_weights = run(k)

    problems = []
    for cycle in range(cycles):
        expected = ref_timeline[cycle][0]
        for j, counts in enumerate(acc_timeline[cycle]):
            for name, count in counts.items():
                want = expected[name] if j == k - 1 else 0
                if count != want:
                    what = ("premature optimizer step" if j < k - 1 else
                            "missing optimizer step" if count < want else "extra optimizer step")
                    problems.append(f"cycle {cycle + 1}, micro-batch {j + 1} of {k}: {name!r} stepped {count} "
                                    f"time(s), expected {want} ({what})")
    if problems:
        raise AssertionError(
            format_problems(f"Gradient accumulation over {k} micro-batches steps at the wrong time:", problems)
            + "\nHint: step, zero the gradients and step the scheduler only when "
              "(step_index + 1) % accumulation_steps == 0."
        )

    previous: dict[str, torch.Tensor] | None = None
    for cycle in range(cycles):
        ref_grads = ref_rec.captured.get(cycle, {})
        acc_grads = acc_rec.captured.get(cycle * k + k - 1, {})
        grad_problems = leaves_diff(acc_grads, ref_grads, rtol=rtol, atol=atol, check_device=False)
        if grad_problems:
            causes = []
            keys = [key for key in ref_grads if key in acc_grads]
            if keys:
                a, r = _flat(acc_grads, keys), _flat(ref_grads, keys)
                denom = float(r.dot(r))
                cosine = float(a.dot(r)) / (float(a.norm() * r.norm()) or 1.0)
                if denom > 0 and cosine > 0.999:
                    ratio = float(a.dot(r)) / denom
                    if math.isclose(ratio, k, rel_tol=0.02):
                        causes.append(f"gradients are {k}x the full-batch gradients: the loss isn't divided "
                                      "by accumulation_steps")
                    elif math.isclose(ratio, 1 / k, rel_tol=0.02):
                        causes.append(f"gradients are 1/{k} of the full-batch gradients: the loss is divided "
                                      "by accumulation_steps twice")
                    elif not math.isclose(ratio, 1.0, rel_tol=0.02):
                        causes.append(f"gradients are {ratio:.3g}x the full-batch gradients: check how the loss "
                                      "is scaled per micro-batch")
                if previous is not None and not causes:
                    carried = {key: ref_grads[key] + previous[key] for key in keys if key in previous}
                    if carried and not leaves_diff({key: acc_grads[key] for key in carried}, carried,
                                                   rtol=max(rtol, 1e-4), atol=max(atol, 1e-6), check_device=False):
                        causes.append("gradients from the previous cycle are still there: zero_grad() isn't "
                                      "called after the optimizer step")
            text = format_problems(
                f"Gradient accumulation over {k} micro-batches differs from one full-batch step "
                f"(cycle {cycle + 1} of {cycles}):", grad_problems)
            if causes:
                text += "\n" + format_problems("Likely cause:", causes)
            else:
                text += "".join(f"\nHint: {h}." for h in _accumulation_hints(accumulated.model))
            raise AssertionError(text)
        weight_problems = leaves_diff(acc_weights[cycle], ref_weights[cycle], rtol=rtol, atol=atol,
                                      check_device=False)
        if weight_problems:
            raise AssertionError(
                format_problems(f"Weights differ after accumulation cycle {cycle + 1} although the gradients "
                                "match:", weight_problems)
                + "\nHint: the optimizer or scheduler sees something per micro-batch: gradient clipping or "
                  "scheduler steps inside the accumulation loop, or parameters stepped twice."
            )
        previous = acc_grads


# --------------------------------------------------------------------------- #
# Mixed precision
# --------------------------------------------------------------------------- #
def assert_mixed_precision_training(
    make_setup: Callable[..., Any],
    data: Any,
    step: StepFn | None = None,
    *,
    loss_fn: Callable[[Any, Any], torch.Tensor] | None = None,
    steps: int = 20,
    max_skipped_fraction: float = 0.25,
    device: Any = "cpu",
    seed: int = 0,
) -> TrainingRun:
    """Run a mixed-precision training step and check the GradScaler is used correctly.

    Fails when:

    - ``optimizer.step()`` is called directly instead of ``scaler.step(optimizer)``;
    - ``scaler.step()`` runs without ``scaler.scale(loss)`` or without ``scaler.update()``;
    - gradients are clipped (``torch.nn.utils.clip_grad_norm_``/``clip_grad_value_``)
      before ``scaler.unscale_(optimizer)``;
    - non-finite gradients reach an optimizer step, or the loss itself isn't finite;
    - more than ``max_skipped_fraction`` of the steps are skipped after overflows;
    - parameters aren't finite after training.

    Without a GradScaler (``setup.scaler``, e.g. bfloat16 autocast), only the last three
    apply. Returns the :class:`TrainingRun`.
    """
    step = _resolve_step(step, loss_fn)
    seed_everything(seed)
    setup = _build(make_setup, device)
    run = TrainingRun(setup=setup, coverage_problems=optimizer_coverage_problems(setup.model, setup.optimizers))
    crash: Exception | None = None
    with _Recorder(setup, track_amp=True) as recorder:
        try:
            _run_steps(setup, recorder, step, _batch_source(data, steps), range(steps), run)
        except Exception as exc:  # torch raises on some misuses: report them with what was recorded
            crash = exc
            run.records.append(recorder.end(None))

    problems: list[str] = []
    steps_of: dict[str, list[int]] = defaultdict(list)
    for r in run.records:
        for problem in dict.fromkeys(r.amp_problems):
            steps_of[problem].append(r.step)
    for problem, at in steps_of.items():
        problems.append(f"{problem} (step{'s' if len(at) > 1 else ''} {', '.join(map(str, at[:5]))}"
                        f"{' ...' if len(at) > 5 else ''})")
    bad_loss = [r.step for r in run.records if r.loss is not None and not math.isfinite(r.loss)]
    if bad_loss:
        problems.append(f"the loss is not finite at step(s) {', '.join(map(str, bad_loss[:5]))}: the forward pass "
                        "overflows in low precision; compute the loss in float32 (outside autocast) or use bfloat16")
    applied = [r.step for r in run.records if r.nonfinite_grads]
    if applied:
        problems.append(f"non-finite gradients reached optimizer.step() at step(s) {', '.join(map(str, applied[:5]))}")
    completed = run.records[:-1] if crash is not None else run.records
    if recorder.amp and completed:
        skipped = [r.step for r in completed if r.skipped]
        if len(skipped) > max_skipped_fraction * len(completed):
            scales = ", ".join(_fmt_scale(r.loss_scale) for r in completed[:8])
            problems.append(f"{len(skipped)} of {len(completed)} steps skipped after inf/NaN gradients "
                            f"(loss scale: {scales}{' ...' if len(completed) > 8 else ''})")
    for name, param in setup.model.named_parameters():
        if not bool(torch.isfinite(param.detach()).all()):
            problems.append(f"{name}: parameter is not finite after training")
            break
    if crash is not None:
        message = str(crash).strip().splitlines()[0] if str(crash).strip() else ""
        problems.append(f"step {run.records[-1].step} raised {type(crash).__name__}: {message}")
    if problems:
        try:
            run._fail(f"Mixed-precision training: {len(problems)} problem(s):", problems, [
                "the canonical order is scaler.scale(loss).backward(); scaler.unscale_(opt); clip; "
                "scaler.step(opt); scaler.update(); opt.zero_grad().",
            ], [s for at in steps_of.values() for s in at[:1]] + bad_loss[:1] + applied[:1])
        except AssertionError as error:
            raise error from crash
    return run


def _fmt_scale(scale: float | None) -> str:
    return "-" if scale is None else f"{scale:g}"


# --------------------------------------------------------------------------- #
# Fixture object
# --------------------------------------------------------------------------- #
class TrainingRunner:
    """The ``training_runner`` fixture: the functions of this module, bound to the test's
    ``torch_device`` and ``torch_seed``."""

    def __init__(self, device: Any = "cpu", seed: int = 0) -> None:
        self.device = device
        self.seed = seed

    def _bind(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs.setdefault("device", self.device)
        kwargs.setdefault("seed", self.seed)
        return kwargs

    def run(self, make_setup: Callable[..., Any], data: Any, step: StepFn | None = None, **kwargs: Any) -> TrainingRun:
        """:func:`run_training`."""
        return run_training(make_setup, data, step, **self._bind(kwargs))

    def assert_resume_equivalent(self, make_setup: Callable[..., Any], data: Any, step: StepFn | None = None,
                                 **kwargs: Any) -> ResumeReport:
        """:func:`assert_resume_equivalent`."""
        return assert_resume_equivalent(make_setup, data, step, **self._bind(kwargs))

    def assert_accumulation_equivalent(self, make_setup: Callable[..., Any], data: Any,
                                       make_step: Callable[[int], StepFn] | None = None, **kwargs: Any) -> None:
        """:func:`assert_accumulation_equivalent`."""
        assert_accumulation_equivalent(make_setup, data, make_step, **self._bind(kwargs))

    def assert_mixed_precision_training(self, make_setup: Callable[..., Any], data: Any,
                                        step: StepFn | None = None, **kwargs: Any) -> TrainingRun:
        """:func:`assert_mixed_precision_training`."""
        return assert_mixed_precision_training(make_setup, data, step, **self._bind(kwargs))
