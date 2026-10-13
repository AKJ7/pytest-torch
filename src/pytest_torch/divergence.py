"""Find the first layer where two models (or two runs of one model) diverge.

    report = localize_divergence(model_cpu, model_gpu, example_inputs)
    print(report)

Forward hooks record every module's inputs and outputs in both models; layers are
matched by name (override with ``module_map``) and compared in execution order.
Limitations: layers fused or skipped by ``torch.compile`` / TorchScript don't fire
hooks, and later layers inherit earlier errors, so always start at the first row.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from . import tolerances
from .utils import call_model, cast_floating, flatten, model_device, model_dtype, to_device

_INF = float("inf")


@dataclass
class LayerDiff:
    name: str
    module_type: str
    call_index: int
    max_abs: float
    max_rel: float
    n_bad: int
    numel: int
    diverged: bool
    inputs_diverged: bool = False
    note: str = ""


@dataclass
class DivergenceReport:
    layers: list[LayerDiff]
    rtol: float
    atol: float

    @property
    def diverged(self) -> list[LayerDiff]:
        return [layer for layer in self.layers if layer.diverged]

    @property
    def first(self) -> LayerDiff | None:
        return next((layer for layer in self.layers if layer.diverged), None)

    def format(self, max_rows: int = 8) -> str:
        tol = f"rtol={self.rtol:g}, atol={self.atol:g}"
        div = self.diverged
        if not div:
            worst = max((layer.max_abs for layer in self.layers), default=0.0)
            return (
                f"No divergence: all {len(self.layers)} layers within tolerance "
                f"({tol}); largest abs error {worst:.2e}"
            )
        first = div[0]
        lines = [
            f"Divergence: {len(div)} of {len(self.layers)} layers exceed tolerance "
            f"({tol}). First: '{first.name}' ({first.module_type})",
            f"  {'layer':<32} {'type':<16} {'max abs':>9} {'max rel':>9}  inputs",
        ]
        for layer in div[:max_rows]:
            lines.append(
                f"  {layer.name:<32} {layer.module_type:<16} {layer.max_abs:>9.2e} "
                f"{layer.max_rel:>9.2e}  {'DIFF' if layer.inputs_diverged else 'ok'}"
                + (f"  [{layer.note}]" if layer.note else "")
            )
        if len(div) > max_rows:
            lines.append(f"  ... and {len(div) - max_rows} more")
        if first.inputs_diverged:
            lines.append(
                "The first diverging layer already receives different inputs: the error "
                "entered between layers (functional ops, residual adds, glue code)."
            )
        else:
            lines.append(
                "The first diverging layer receives matching inputs: the error originates "
                "inside it. Later rows mostly inherit that error."
            )
        return "\n".join(lines)

    __str__ = format

    def raise_if_diverged(self) -> None:
        if self.first is not None:
            raise AssertionError(self.format())


# --------------------------------------------------------------------------- #
# recording
# --------------------------------------------------------------------------- #
def _tensors(obj: Any, prefix: str) -> list[tuple[str, torch.Tensor]]:
    return [
        (path, leaf.detach().cpu())
        for path, leaf in flatten(obj, prefix)
        if isinstance(leaf, torch.Tensor)
    ]


def _collect(model, inputs, leaf_only: bool):
    records: dict[tuple[str, int], tuple[str, list, list]] = {}  # execution order
    counts: dict[str, int] = defaultdict(int)
    handles = []

    def make_hook(name: str):
        def hook(module, args, output):
            index = counts[name]
            counts[name] += 1
            outs = _tensors(output, "out")
            if outs:
                records[(name, index)] = (
                    type(module).__name__,
                    _tensors(args, "in"),
                    outs,
                )

        return hook

    for name, module in model.named_modules():
        if leaf_only and next(module.children(), None) is not None:
            continue
        handles.append(module.register_forward_hook(make_hook(name or "<root>")))
    try:
        with torch.inference_mode():
            call_model(model, inputs)
    finally:
        for handle in handles:
            handle.remove()
    return records


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #
@dataclass
class _Cmp:
    diverged: bool
    max_abs: float = 0.0
    max_rel: float = 0.0
    n_bad: int = 0
    numel: int = 0
    note: str = ""


def _compare(a_list, b_list, rtol: float, atol: float) -> _Cmp:
    da, db = dict(a_list), dict(b_list)
    if set(da) != set(db):
        return _Cmp(True, _INF, _INF, note="tensor structure differs")
    result = _Cmp(False)
    tiny = torch.finfo(torch.float64).tiny
    for path, a in da.items():
        b = db[path]
        if a.shape != b.shape:
            return _Cmp(True, _INF, _INF, note=f"shape {tuple(a.shape)} vs {tuple(b.shape)}")
        if a.numel() == 0:
            continue
        a64, b64 = a.to(torch.float64), b.to(torch.float64)
        finite = torch.isfinite(a64) & torch.isfinite(b64)
        structural = (
            (torch.isnan(a64) != torch.isnan(b64))
            | (torch.isinf(a64) != torch.isinf(b64))
            | (torch.isinf(a64) & torch.isinf(b64) & (a64 != b64))
        )
        diff = (a64 - b64).abs()
        bad = (finite & (diff > atol + rtol * b64.abs())) | structural
        result.numel += a.numel()
        n_bad = int(bad.sum())
        result.n_bad += n_bad
        if structural.any():
            result.max_abs = result.max_rel = _INF
            result.note = "NaN/Inf mismatch"
        elif finite.any():
            result.max_abs = max(result.max_abs, float(diff[finite].max()))
            rel = diff[finite] / b64[finite].abs().clamp_min(tiny)
            result.max_rel = max(result.max_rel, float(rel.max()))
        if n_bad:
            result.diverged = True
    return result


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def localize_divergence(
    model_a,
    model_b,
    inputs: Any,
    *,
    inputs_b: Any = None,
    rtol: float | None = None,
    atol: float | None = None,
    module_map: Mapping[str, str] | None = None,
    leaf_only: bool = True,
) -> DivergenceReport:
    """Compare per-layer activations of ``model_a`` (reference) and ``model_b``.

    ``inputs`` are moved to each model's device; floating inputs for ``model_b`` are
    cast to its parameter dtype (pass ``inputs_b`` to control this yourself).
    ``module_map`` maps names in ``model_a`` to names in ``model_b`` when they differ.
    ``leaf_only=False`` also records container modules.
    """
    rtol, atol = tolerances.for_check("divergence", model_device(model_b), rtol, atol)
    module_map = dict(module_map or {})
    inputs_a = to_device(inputs, model_device(model_a))
    b_inputs = to_device(inputs if inputs_b is None else inputs_b, model_device(model_b))
    dtype_b = model_dtype(model_b)
    if dtype_b is not None:
        b_inputs = cast_floating(b_inputs, dtype_b)

    rec_a = _collect(model_a, inputs_a, leaf_only)
    rec_b = _collect(model_b, b_inputs, leaf_only)

    layers: list[LayerDiff] = []
    for (name, index), (type_name, ins_a, outs_a) in rec_a.items():
        key_b = (module_map.get(name, name), index)
        if key_b not in rec_b:
            layers.append(
                LayerDiff(name, type_name, index, _INF, _INF, 0, 0, True,
                          note="no matching layer in model_b")
            )
            continue
        _, ins_b, outs_b = rec_b[key_b]
        out_cmp = _compare(outs_a, outs_b, rtol, atol)
        in_cmp = _compare(ins_a, ins_b, rtol, atol) if ins_a else _Cmp(False)
        layers.append(
            LayerDiff(
                name=name,
                module_type=type_name,
                call_index=index,
                max_abs=out_cmp.max_abs,
                max_rel=out_cmp.max_rel,
                n_bad=out_cmp.n_bad,
                numel=out_cmp.numel,
                diverged=out_cmp.diverged,
                inputs_diverged=in_cmp.diverged,
                note=out_cmp.note,
            )
        )
    return DivergenceReport(layers, rtol, atol)
