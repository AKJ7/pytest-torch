"""Checkpoints: saving and reloading preserves predictions; old checkpoints still load.

    assert_checkpoint_roundtrip(make_model, inputs)          # save -> fresh model -> load -> same outputs
    assert_checkpoint_loads(make_model, "ckpts/v1.pt")       # yesterday's checkpoint, today's code
"""
from __future__ import annotations

import io
import os
import re
from collections.abc import Mapping
from typing import Any, Callable

import torch

from . import tolerances
from .assertions import assert_outputs_close, format_problems
from .utils import model_device, run_inference, to_device


def _shared_groups(model: torch.nn.Module) -> set[frozenset[str]]:
    """Groups of parameter/buffer names that share storage (tied weights)."""
    by_ptr: dict[int, set[str]] = {}
    for name, tensor in [*model.named_parameters(remove_duplicate=False), *model.named_buffers(remove_duplicate=False)]:
        if tensor.numel():
            by_ptr.setdefault(tensor.data_ptr(), set()).add(name)
    return {frozenset(names) for names in by_ptr.values() if len(names) > 1}


def _plain_tensor_attributes(model: torch.nn.Module) -> list[str]:
    """Tensors stored as plain attributes: they are not part of the state_dict."""
    found = []
    for module_name, module in model.named_modules():
        for attr, value in vars(module).items():
            if isinstance(value, torch.Tensor) and not attr.startswith("_"):
                found.append(f"{module_name}.{attr}" if module_name else attr)
    return found


def _non_persistent_buffers(model: torch.nn.Module) -> list[str]:
    saved = set(model.state_dict())
    return [name for name, _ in model.named_buffers() if name not in saved]


def _save_load(state: Mapping[str, torch.Tensor], fmt: str) -> dict[str, torch.Tensor]:
    if fmt == "torch":
        buffer = io.BytesIO()
        torch.save(dict(state), buffer)
        buffer.seek(0)
        return torch.load(buffer, map_location="cpu", weights_only=True)
    if fmt == "safetensors":
        try:
            from safetensors.torch import load, save
        except ImportError as exc:
            raise ImportError("format='safetensors' needs: pip install safetensors") from exc
        try:
            return load(save({k: v.detach().cpu().contiguous() for k, v in state.items()}))
        except RuntimeError as exc:  # shared tensors are refused
            raise AssertionError(
                f"safetensors refused the state_dict: {exc}\n"
                "Hint: tied weights must be saved once; use safetensors.torch.save_model / "
                "load_model, which deduplicate them."
            ) from None
    raise ValueError("format must be 'torch' or 'safetensors'")


def assert_checkpoint_roundtrip(
    make_model: Callable[[], torch.nn.Module],
    inputs: Any,
    *,
    model: torch.nn.Module | None = None,
    format: str = "torch",
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    """Save ``model``'s state_dict, load it into a fresh ``make_model()`` and compare outputs.

    The fresh model is built under a different seed, so anything not restored from the
    checkpoint shows up as a difference. ``model`` defaults to ``make_model()``; both are
    put in eval mode on the device of ``model``. ``format`` is ``"torch"`` (``torch.save``)
    or ``"safetensors"``. On failure the message lists the usual culprits: tensors kept as
    plain attributes, non-persistent buffers, and tied weights that came apart.
    """
    rtol, atol = tolerances.for_check("checkpoint", None, rtol, atol)
    original = (model if model is not None else make_model()).eval()
    device = model_device(original)
    inputs = to_device(inputs, device)
    expected = run_inference(original, inputs)

    restored_state = _save_load(original.state_dict(), format)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(torch.initial_seed() + 7919)  # different init than the original
        fresh = make_model()
    fresh = fresh.to(device).eval()
    result = fresh.load_state_dict(restored_state, strict=False)

    problems = []
    if result.missing_keys:
        problems.append("missing after reload: " + ", ".join(result.missing_keys))
    if result.unexpected_keys:
        problems.append("unexpected after reload: " + ", ".join(result.unexpected_keys))
    tied_before, tied_after = _shared_groups(original), _shared_groups(fresh)
    for group in sorted(tied_before - tied_after, key=sorted):
        problems.append(f"tied weights came apart: {', '.join(sorted(group))}")
    try:
        assert_outputs_close(run_inference(fresh, inputs), expected, rtol=rtol, atol=atol)
    except AssertionError as exc:
        problems.append(f"outputs differ after reload:\n{exc}")
        suspects = [f"plain tensor attribute (not saved): {n}" for n in _plain_tensor_attributes(original)]
        suspects += [f"non-persistent buffer (not saved): {n}" for n in _non_persistent_buffers(original)]
        if suspects:
            problems.append("likely culprits:\n" + "\n".join(suspects))
    if problems:
        raise AssertionError(format_problems(f"Checkpoint round-trip ({format}) failed:", problems))


_SIZE_MISMATCH = re.compile(
    r"size mismatch for (?P<name>\S+): copying a param with shape torch\.Size\(\[(?P<ckpt>[^\]]*)\]\) "
    r"from checkpoint, the shape in current model is torch\.Size\(\[(?P<model>[^\]]*)\]\)"
)


def _shape_message(line: str) -> str:
    """torch's 'size mismatch for w: copying a param with shape ...' -> 'w: checkpoint (4, 4), model (5, 4)'."""
    match = _SIZE_MISMATCH.search(line)
    if match is None:
        return line.strip()

    def shape(text: str) -> str:
        dims = [d.strip() for d in text.split(",") if d.strip()]
        return "(" + ", ".join(dims) + ("," if len(dims) == 1 else "") + ")"

    return f"{match['name']}: checkpoint {shape(match['ckpt'])}, model {shape(match['model'])}"


_STATE_KEYS = ("state_dict", "model_state_dict", "model", "module", "net")


def _extract_state(checkpoint: Any, key: str | None) -> dict[str, torch.Tensor]:
    if key is not None:
        checkpoint = checkpoint[key]
    elif isinstance(checkpoint, Mapping) and not all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        for candidate in _STATE_KEYS:
            value = checkpoint.get(candidate)
            if isinstance(value, Mapping) and value and all(isinstance(v, torch.Tensor) for v in value.values()):
                checkpoint = value
                break
        else:
            raise ValueError(
                f"could not find the state_dict in the checkpoint (top-level keys: {list(checkpoint)}); pass key="
            )
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"expected a state_dict mapping, got {type(checkpoint).__name__}")
    return dict(checkpoint)


def assert_checkpoint_loads(
    make_model: Callable[[], torch.nn.Module],
    checkpoint: str | os.PathLike | Mapping[str, Any],
    *,
    key: str | None = None,
    strip_prefix: str | None = None,
    strict: bool = True,
    weights_only: bool = True,
) -> torch.nn.Module:
    """Load an existing checkpoint into ``make_model()`` and report every incompatibility.

    ``checkpoint`` is a path or an already loaded object; the state_dict is found at
    ``key`` or under a common name (``state_dict``, ``model_state_dict``, ``model``, ...).
    ``strip_prefix`` removes e.g. ``"module."`` (DataParallel/DDP). With ``strict=False``
    only shape mismatches fail. Returns the model with the weights loaded.
    """
    where = "" if isinstance(checkpoint, Mapping) else f"{os.fspath(checkpoint)} "
    if not isinstance(checkpoint, Mapping):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=weights_only)
    state = _extract_state(checkpoint, key)
    if strip_prefix:
        state = {k[len(strip_prefix):] if k.startswith(strip_prefix) else k: v for k, v in state.items()}

    model = make_model()
    expected = model.state_dict()
    # Load through the model itself, so its compatibility code (_load_from_state_dict,
    # load_state_dict pre-hooks renaming old keys) runs exactly as in production.
    problems = []
    try:
        result = model.load_state_dict(state, strict=False)
    except RuntimeError as exc:  # shape mismatches: torch lists all of them
        problems += [
            "shape mismatch: " + _shape_message(line)
            for line in str(exc).splitlines()
            if line.strip().startswith("size mismatch")
        ] or [f"load_state_dict failed: {exc}"]
        missing, unexpected = [], []
    else:
        missing, unexpected = list(result.missing_keys), list(result.unexpected_keys)
    if strict:
        problems += [f"missing in checkpoint: {k}" for k in missing]
        problems += [f"unexpected in checkpoint: {k}" for k in unexpected]
    if problems:
        hint = ""
        common = os.path.commonprefix(unexpected) if unexpected else ""
        if common.endswith(".") and all(k[len(common):] in expected for k in unexpected):
            hint = f"\nHint: every unexpected key starts with {common!r}: pass strip_prefix={common!r}."
        raise AssertionError(
            format_problems(f"Checkpoint {where}does not fit the model:", problems, limit=20) + hint
        )
    return model
