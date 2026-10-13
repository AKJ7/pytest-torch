"""Failure reproduction bundles: record what a failing test saw, replay it anywhere.

    pytest --torch-repro-dir=repro                 # every failing test writes repro/<test>.pt
    pytest --torch-repro=repro/<test>.pt           # rerun just that test, same seed and inputs
    pytest --torch-repro=repro/<test>.pt --torch-device=cpu   # ...on another device

A bundle holds the test id and parameters, seed, device, every tensor ``make_tensor``
produced (replayed bit for bit, so platform RNG differences don't matter), the values
passed to :func:`record_repro`, the active tolerances and the environment. Model weights
are only stored up to ``--torch-repro-max-weights-mb`` (default 0: off), since models
built under ``torch_seed`` are reproducible from the seed.

Torch-free at import: the plugin imports this module at startup.
"""
from __future__ import annotations

import os
import platform
import re
import sys
import warnings
from pathlib import Path
from typing import Any

FORMAT = 1

_state: dict[str, Any] = {
    "dir": None,  # record mode: where bundles go
    "max_weights_mb": 0.0,
    "replay": None,  # replay mode: the loaded bundle
    "current": None,  # _Recording of the running test
}


class _Recording:
    def __init__(self, nodeid: str, params: dict[str, str], replay: dict | None):
        self.nodeid = nodeid
        self.params = params
        self.seed: int | None = None
        self.device: str | None = None
        self.inputs: list[Any] = []  # make_tensor outputs, in call order
        self.recorded: dict[str, Any] = {}
        self.state_dicts: dict[str, Any] = {}
        self.skipped_weights: dict[str, float] = {}
        self.replay = replay
        self.replay_index = 0


# --------------------------------------------------------------------------- #
# configuration (plugin)
# --------------------------------------------------------------------------- #
def configure(directory: str | None, max_weights_mb: float, replay_path: str | None) -> None:
    _state["dir"] = directory
    _state["max_weights_mb"] = float(max_weights_mb)
    _state["replay"] = load_repro(replay_path) if replay_path else None


def replay_bundle() -> dict | None:
    return _state["replay"]


def recording_enabled() -> bool:
    return _state["dir"] is not None or _state["replay"] is not None


def start_test(nodeid: str, params: dict[str, str], replaying: bool) -> None:
    replay = _state["replay"] if replaying else None
    _state["current"] = _Recording(nodeid, params, replay) if recording_enabled() else None


def finish_test() -> None:
    _state["current"] = None


def current() -> _Recording | None:
    return _state["current"]


# --------------------------------------------------------------------------- #
# hooks for fixtures
# --------------------------------------------------------------------------- #
def note_seed(seed: int) -> None:
    rec = current()
    if rec is not None:
        rec.seed = int(seed)


def note_device(device: str) -> None:
    rec = current()
    if rec is not None:
        rec.device = str(device)


def make_tensor_hook(shape: tuple, dtype: Any, device: Any, generate):
    """``make_tensor`` calls this: replay a recorded tensor, or generate and record one."""
    rec = current()
    if rec is None:
        return generate()
    if rec.replay is not None:
        recorded = rec.replay.get("inputs", [])
        if rec.replay_index < len(recorded):
            tensor = recorded[rec.replay_index]
            rec.replay_index += 1
            if tuple(tensor.shape) == tuple(shape) and tensor.dtype == dtype:
                return tensor.to(device)
            warnings.warn(
                f"repro replay: make_tensor call {rec.replay_index} asked for {tuple(shape)} {dtype}, "
                f"the bundle has {tuple(tensor.shape)} {tensor.dtype}; the test changed, "
                "generating fresh values from here on",
                stacklevel=3,
            )
            rec.replay_index = len(recorded)
        return generate()
    tensor = generate()
    rec.inputs.append(tensor.detach().cpu().clone())
    return tensor


def record_repro(name: str, value: Any) -> Any:
    """Store ``value`` in the failure bundle of the running test and return it.

    When replaying a bundle, return the recorded value instead (for ``nn.Module``s: the
    module, with recorded weights loaded into it if the bundle has them). Values may be
    tensors or nested dicts/lists/tuples of tensors and plain Python values. Outside
    ``--torch-repro-dir`` / ``--torch-repro`` runs this just returns ``value``.

        batch = record_repro("batch", next(iter(loader)))
        model = record_repro("model", model)
    """
    rec = current()
    if rec is None:
        return value
    import torch

    from .utils import to_device, tree_map

    if isinstance(value, torch.nn.Module):
        if rec.replay is not None:
            state = rec.replay.get("state_dicts", {}).get(name)
            if state is not None:
                value.load_state_dict(state)
            return value
        state = value.state_dict()
        size_mb = sum(t.numel() * t.element_size() for t in state.values() if isinstance(t, torch.Tensor)) / 2**20
        if size_mb <= _state["max_weights_mb"]:
            rec.state_dicts[name] = {k: v.detach().cpu().clone() for k, v in state.items()}
        else:
            rec.skipped_weights[name] = round(size_mb, 2)
        return value
    if rec.replay is not None and name in rec.replay.get("recorded", {}):
        from .utils import flatten

        tensors = [leaf for _, leaf in flatten(value) if isinstance(leaf, torch.Tensor)]
        return to_device(rec.replay["recorded"][name], tensors[0].device if tensors else "cpu")
    rec.recorded[name] = tree_map(lambda t: t.detach().cpu().clone(), value)
    return value


# --------------------------------------------------------------------------- #
# writing and reading bundles
# --------------------------------------------------------------------------- #
def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")[:150] or "test"


def _environment(device: str | None) -> dict[str, Any]:
    import importlib.metadata

    import torch

    env = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "env": {k: os.environ[k] for k in ("CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES") if k in os.environ},
    }
    try:
        env["pytest_torch"] = importlib.metadata.version("pytest-torch")
    except importlib.metadata.PackageNotFoundError:
        pass
    if device is not None:
        from .devices import device_name

        try:
            env["device_name"] = device_name(device)
        except Exception:
            pass
    if torch.version.cuda:
        env["cuda"] = str(torch.version.cuda)
    return env


def write_bundle(failure: str) -> Path | None:
    """Write the running test's bundle (record mode only); return its path."""
    rec = current()
    if rec is None or _state["dir"] is None:
        return None
    import torch

    from . import tolerances

    bundle = {
        "format": FORMAT,
        "nodeid": rec.nodeid,
        "params": rec.params,
        "seed": rec.seed,
        "device": rec.device,
        "inputs": rec.inputs,
        "recorded": rec.recorded,
        "state_dicts": rec.state_dicts,
        "weights_not_saved_mb": rec.skipped_weights,
        "tolerances": {"config": dict(tolerances._config), "overrides": list(tolerances._overrides)},
        "environment": _environment(rec.device),
        "failure": failure[-4000:],
        "argv": [str(a) for a in sys.argv[1:]],
    }
    directory = Path(_state["dir"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_slug(rec.nodeid)}.pt"
    torch.save(bundle, path)
    return path


def load_repro(path: str | os.PathLike) -> dict:
    """Load a bundle (e.g. to inspect ``bundle["inputs"]`` in a notebook)."""
    import torch

    bundle = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(bundle, dict) or bundle.get("format") != FORMAT:
        raise ValueError(f"{path} is not a pytest-torch repro bundle (format {FORMAT})")
    return bundle


def describe(bundle: dict) -> list[str]:
    """Header lines for a replay, including environment differences."""
    lines = [f"replaying {bundle['nodeid']} (seed {bundle['seed']}, recorded on {bundle['device']})"]
    try:
        here = _environment(None)
    except Exception:
        return lines
    recorded = bundle.get("environment", {})
    for key in ("torch", "python", "platform"):
        if key in recorded and recorded[key] != here.get(key):
            lines.append(f"  differs: {key} {recorded[key]!r} recorded, {here.get(key)!r} here")
    if bundle.get("weights_not_saved_mb"):
        lines.append(
            "  weights not in the bundle (rebuilt from the seed): "
            + ", ".join(f"{k} ({v} MB)" for k, v in bundle["weights_not_saved_mb"].items())
        )
    return lines
