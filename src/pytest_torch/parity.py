"""Parity checks: eager vs deployment targets, models vs references, quantized vs fp32."""
from __future__ import annotations

import copy
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

import torch

from .assertions import assert_outputs_close, format_problems, leaves_diff
from .utils import flatten, model_device, run_inference, to_device, tree_map


def _to_cpu(obj: Any) -> Any:
    return tree_map(lambda t: t.detach().cpu(), obj)


def _positional_args(inputs: Any) -> tuple:
    if isinstance(inputs, Mapping):
        raise ValueError(
            "TorchScript/ONNX helpers need positional inputs: pass a tensor or a tuple "
            "(wrap kwargs-only models in a small nn.Module)"
        )
    if isinstance(inputs, (list, tuple)):
        return tuple(inputs)
    return (inputs,)


def _assert_close(actual: Any, expected: Any, what: str, *, rtol, atol) -> None:
    try:
        assert_outputs_close(
            _to_cpu(actual), _to_cpu(expected), rtol=rtol, atol=atol, check_device=False
        )
    except AssertionError as exc:
        raise AssertionError(f"{what}\n{exc}") from None


# --------------------------------------------------------------------------- #
# Deployment targets
# --------------------------------------------------------------------------- #
def assert_compile_parity(
    model,
    inputs: Any,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    runs: int = 2,
    **compile_kwargs,
) -> None:
    """Assert ``torch.compile(model)`` matches eager mode.

    The compiled model runs ``runs`` times (the second call exercises the cached
    graph and guards). Extra kwargs go to ``torch.compile``, e.g. ``backend="eager"``
    to test graph capture without needing a C++ compiler.
    """
    inputs = to_device(inputs, model_device(model))
    reference = run_inference(model, inputs)
    compiled = torch.compile(model, **compile_kwargs)
    for i in range(runs):
        out = run_inference(compiled, inputs)
        _assert_close(
            out,
            reference,
            f"torch.compile output (run {i + 1}/{runs}) differs from eager:",
            rtol=rtol,
            atol=atol,
        )


def assert_torchscript_parity(
    model,
    inputs: Any,
    *,
    mode: str = "trace",
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    """Assert the TorchScript version (``mode`` = "trace" or "script") matches eager.

    ``inputs`` must be a tensor or a tuple of tensors (no kwargs). Note that tracing
    only records the control-flow path taken for these inputs.
    """
    if mode not in ("trace", "script"):
        raise ValueError("mode must be 'trace' or 'script'")
    args = _positional_args(to_device(inputs, model_device(model)))
    reference = run_inference(model, args)
    with torch.no_grad():
        if mode == "trace":
            scripted = torch.jit.trace(model, args, strict=False, check_trace=False)
        else:
            scripted = torch.jit.script(model)
    out = run_inference(scripted, args)
    _assert_close(
        out, reference, f"TorchScript ({mode}) output differs from eager:", rtol=rtol, atol=atol
    )


def assert_onnx_parity(
    model,
    inputs: Any,
    *,
    rtol: float = 1e-3,
    atol: float = 1e-4,
    export_kwargs: Mapping[str, Any] | None = None,
    providers: tuple[str, ...] = ("CPUExecutionProvider",),
) -> None:
    """Export to ONNX, run with ONNX Runtime, and compare against PyTorch (on CPU).

    ``inputs`` must be a tensor or a flat tuple of tensors. The ONNX exporter's
    defaults differ between torch versions: pin them via ``export_kwargs`` (e.g.
    ``{"dynamo": False, "opset_version": 17}``). Needs ``pip install pytest-torch[onnx]``.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "assert_onnx_parity needs onnxruntime: pip install 'pytest-torch[onnx]'"
        ) from exc

    cpu_model = copy.deepcopy(model).cpu()
    args = _positional_args(to_device(inputs, "cpu"))
    if not all(isinstance(a, torch.Tensor) for a in args):
        raise ValueError("assert_onnx_parity needs a tensor or a flat tuple of tensors")

    reference = run_inference(cpu_model, args)
    ref_leaves = [(p, t) for p, t in flatten(reference) if isinstance(t, torch.Tensor)]
    if not ref_leaves:
        raise ValueError("the model produced no tensor outputs")

    kwargs: dict[str, Any] = {
        "input_names": [f"input_{i}" for i in range(len(args))],
        "output_names": [f"output_{i}" for i in range(len(ref_leaves))],
    }
    kwargs.update(export_kwargs or {})

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.onnx")
        with torch.no_grad():
            torch.onnx.export(cpu_model, args, path, **kwargs)
        session = ort.InferenceSession(path, providers=list(providers))
        onnx_inputs = session.get_inputs()
        if len(onnx_inputs) != len(args):
            raise AssertionError(
                f"ONNX graph has {len(onnx_inputs)} inputs but {len(args)} were given "
                "(the exporter may have pruned unused inputs)"
            )
        feeds = {spec.name: arg.numpy() for spec, arg in zip(onnx_inputs, args)}
        outputs = session.run(None, feeds)

    if len(outputs) != len(ref_leaves):
        raise AssertionError(
            f"ONNX model has {len(outputs)} outputs, the PyTorch model has {len(ref_leaves)}"
        )
    actual = {path: torch.from_numpy(o) for (path, _), o in zip(ref_leaves, outputs)}
    expected = {path: t.detach() for path, t in ref_leaves}
    problems = leaves_diff(
        actual, expected, rtol=rtol, atol=atol, check_dtype=False, check_device=False
    )
    if problems:
        raise AssertionError(
            format_problems("ONNX Runtime output differs from PyTorch:", problems)
        )


# --------------------------------------------------------------------------- #
# Model vs. model: accuracy budgets (quantization, pruning, distillation, ports)
# --------------------------------------------------------------------------- #
@dataclass
class AgreementMetrics:
    max_abs_error: float
    mean_abs_error: float
    cosine_similarity: float  # mean over the last-dim vectors
    top1_agreement: float | None  # None when the last dim has size 1

    def __str__(self) -> str:
        top1 = "n/a" if self.top1_agreement is None else f"{self.top1_agreement:.4f}"
        return (
            f"max_abs_error={self.max_abs_error:.3e}, mean_abs_error={self.mean_abs_error:.3e}, "
            f"cosine_similarity={self.cosine_similarity:.6f}, top1_agreement={top1}"
        )


def _select_tensor(output: Any, select: Callable | str | None) -> torch.Tensor:
    if callable(select):
        picked = select(output)
    elif isinstance(select, str):
        leaves = dict(flatten(output))
        if select not in leaves:
            raise ValueError(f"unknown output path {select!r}; available: {list(leaves)}")
        picked = leaves[select]
    elif select is None:
        picked = next(
            (
                leaf
                for _, leaf in flatten(output)
                if isinstance(leaf, torch.Tensor) and leaf.is_floating_point()
            ),
            None,
        )
    else:
        raise TypeError("select must be a callable, an output path string, or None")
    if not isinstance(picked, torch.Tensor):
        raise ValueError("could not pick a tensor from the model output; pass select=")
    return picked


def _as_f64(t: torch.Tensor) -> torch.Tensor:
    t = t.detach()
    if t.is_quantized:
        t = t.dequantize()
    return t.to("cpu", torch.float64)


def compare_outputs(reference: Any, candidate: Any, *, select=None) -> AgreementMetrics:
    """Agreement metrics between two model outputs (default: first floating tensor)."""
    ref = _as_f64(_select_tensor(reference, select))
    cand = _as_f64(_select_tensor(candidate, select))
    if ref.shape != cand.shape:
        raise AssertionError(f"output shapes differ: {tuple(ref.shape)} != {tuple(cand.shape)}")
    if ref.dim() == 0:
        ref, cand = ref.reshape(1), cand.reshape(1)

    diff = (ref - cand).abs()
    a = ref.reshape(-1, ref.shape[-1])
    b = cand.reshape(-1, cand.shape[-1])
    na, nb = a.norm(dim=-1), b.norm(dim=-1)
    denom = na * nb
    cos = torch.where(
        denom > 0,
        (a * b).sum(-1) / denom.clamp_min(1e-300),
        (na == nb).to(a.dtype),  # two zero vectors count as identical
    )
    top1 = None
    if a.shape[-1] > 1:
        top1 = float((a.argmax(-1) == b.argmax(-1)).double().mean())
    return AgreementMetrics(
        max_abs_error=float(diff.max()) if diff.numel() else 0.0,
        mean_abs_error=float(diff.mean()) if diff.numel() else 0.0,
        cosine_similarity=float(cos.mean()),
        top1_agreement=top1,
    )


def assert_models_agree(
    reference,
    candidate,
    inputs: Any,
    *,
    min_top1: float | None = None,
    min_cosine: float | None = None,
    max_abs: float | None = None,
    select=None,
    inputs_candidate: Any = None,
) -> AgreementMetrics:
    """Assert ``candidate`` stays within an accuracy budget of ``reference``.

    Typical use: fp32 model vs. its quantized/pruned/distilled version. Budgets:
    ``min_top1`` (fraction of identical argmax predictions), ``min_cosine`` (mean cosine
    similarity of output vectors) and ``max_abs`` (max absolute error). Set at least
    one. Returns the measured :class:`AgreementMetrics` so you can log them.
    """
    if min_top1 is None and min_cosine is None and max_abs is None:
        raise ValueError("set at least one budget: min_top1, min_cosine or max_abs")
    ref_out = run_inference(reference, to_device(inputs, model_device(reference)))
    cand_in = inputs if inputs_candidate is None else inputs_candidate
    cand_out = run_inference(candidate, to_device(cand_in, model_device(candidate)))
    metrics = compare_outputs(ref_out, cand_out, select=select)

    failures = []
    if max_abs is not None and not (metrics.max_abs_error <= max_abs):
        failures.append(f"max_abs_error {metrics.max_abs_error:.3e} > budget {max_abs:.3e}")
    if min_cosine is not None and not (metrics.cosine_similarity >= min_cosine):
        failures.append(
            f"cosine_similarity {metrics.cosine_similarity:.6f} < budget {min_cosine:.6f}"
        )
    if min_top1 is not None:
        if metrics.top1_agreement is None:
            raise ValueError("min_top1 is undefined for outputs with a last dim of size 1")
        if not (metrics.top1_agreement >= min_top1):
            failures.append(f"top1_agreement {metrics.top1_agreement:.4f} < budget {min_top1:.4f}")
    if failures:
        raise AssertionError(
            "Candidate model is outside its accuracy budget:\n  "
            + "\n  ".join(failures)
            + f"\n  metrics: {metrics}"
        )
    return metrics


# --------------------------------------------------------------------------- #
# Reference implementations (e.g. your port vs. the official Hugging Face model)
# --------------------------------------------------------------------------- #
def assert_matches_reference(
    model,
    reference: Any,
    inputs: Any,
    *,
    select: Callable | None = None,
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    """Assert ``model(inputs)`` matches a reference implementation.

    ``reference`` is either a model/callable (called with the same inputs) or a
    precomputed output (pytree). ``select`` extracts the part to compare, e.g.
    ``lambda out: out.logits``; it is applied to ``model``'s output and to the output
    of a callable reference, but not to a precomputed reference.
    """
    actual = run_inference(model, to_device(inputs, model_device(model)))
    if callable(reference):
        expected = run_inference(reference, to_device(inputs, model_device(reference)))
        if select is not None:
            expected = select(expected)
    else:
        expected = reference
    if select is not None:
        actual = select(actual)
    _assert_close(actual, expected, "Output differs from the reference:", rtol=rtol, atol=atol)


def _default_generate(model, inputs, max_new_tokens, kwargs):
    kwargs = {"do_sample": False, "max_new_tokens": max_new_tokens, **kwargs}
    with torch.inference_mode():
        if isinstance(inputs, Mapping):
            return model.generate(**inputs, **kwargs)
        return model.generate(inputs, **kwargs)


def _prompt_len(inputs: Any) -> int | None:
    if isinstance(inputs, Mapping) and isinstance(inputs.get("input_ids"), torch.Tensor):
        return int(inputs["input_ids"].shape[-1])
    if isinstance(inputs, torch.Tensor):
        return int(inputs.shape[-1])
    return None


def assert_generation_equal(
    model,
    reference,
    inputs: Any,
    *,
    max_new_tokens: int = 16,
    generate: Callable | None = None,
    **generate_kwargs,
) -> None:
    """Assert greedy decoding gives identical token sequences for ``model`` and ``reference``.

    ``inputs`` is an ``input_ids`` tensor or a dict of kwargs for ``generate``. Extra
    kwargs (e.g. ``pad_token_id=0``) go to ``model.generate``. For non-Hugging-Face
    models pass ``generate=fn(model, inputs, max_new_tokens, kwargs) -> token tensor``.
    """
    gen = generate or _default_generate

    def run(m):
        out = gen(m, to_device(inputs, model_device(m)), max_new_tokens, generate_kwargs)
        out = getattr(out, "sequences", out)
        return out.detach().cpu()

    a, b = run(model), run(reference)
    if a.shape == b.shape and torch.equal(a, b):
        return
    if a.shape[0] != b.shape[0]:
        raise AssertionError(f"Generated batch sizes differ: {a.shape[0]} != {b.shape[0]}")

    prompt = _prompt_len(inputs)
    common = min(a.shape[1], b.shape[1])
    lines = []
    for row in range(a.shape[0]):
        mismatch = (a[row, :common] != b[row, :common]).nonzero()
        if len(mismatch):
            pos = int(mismatch[0])
            where = ""
            if prompt is not None:
                where = f" (generated token {pos - prompt})" if pos >= prompt else " (inside the prompt)"
            lines.append(
                f"row {row}: first difference at position {pos}{where}: "
                f"{int(a[row, pos])} != {int(b[row, pos])}"
            )
    if not lines:
        lines.append(
            f"sequence lengths differ: {a.shape[1]} != {b.shape[1]} (common prefix identical)"
        )
    raise AssertionError("Generated sequences differ:\n  " + "\n  ".join(lines))
