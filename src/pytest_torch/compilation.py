"""torch.compile health checks: graph breaks and recompilations."""
from __future__ import annotations

from typing import Any

import torch

from .utils import model_device, run_inference, to_device


def _reset() -> None:
    reset = getattr(getattr(torch, "compiler", None), "reset", None)
    if reset is None:
        import torch._dynamo as dynamo

        reset = dynamo.reset
    reset()


def _head(text: str, lines: int = 12) -> str:
    return "\n".join(text.strip().splitlines()[:lines])


def assert_no_graph_breaks(model, inputs: Any, *, backend: str = "eager") -> None:
    """Assert the model compiles as ONE graph (``fullgraph=True``).

    Uses the cheap ``eager`` backend by default: graph breaks are decided while tracing,
    so no C++ compiler is needed. On failure the message carries the reason and the
    offending source line reported by torch.
    """
    inputs = to_device(inputs, model_device(model))
    _reset()
    try:
        compiled = torch.compile(model, fullgraph=True, backend=backend)
        try:
            run_inference(compiled, inputs)
        except Exception as exc:
            raise AssertionError(
                "torch.compile(fullgraph=True) hit a graph break (or failed to trace):\n"
                + _head(str(exc))
            ) from exc
    finally:
        _reset()


def assert_no_recompiles(
    model, inputs: Any, *more_inputs: Any, max_recompiles: int = 0
) -> int:
    """Assert repeated calls (and any ``more_inputs``) don't trigger recompilation.

    Compiles with a counting backend, runs ``inputs`` once, then runs ``inputs`` again plus
    every item of ``more_inputs`` and counts how many extra graphs were compiled. A changed
    tensor shape normally costs ONE recompilation (dynamic shapes kick in), so use
    ``max_recompiles=1`` when passing different batch sizes. Returns the recompile count.
    """
    device = model_device(model)
    first = to_device(inputs, device)
    batches = [first] + [to_device(extra, device) for extra in more_inputs]
    calls: list[int] = []

    def counting_backend(gm, example_inputs):
        calls.append(len(calls) + 1)
        return gm.forward

    _reset()
    try:
        compiled = torch.compile(model, backend=counting_backend)
        run_inference(compiled, first)
        initial = len(calls)
        for batch in batches:
            run_inference(compiled, batch)
        recompiles = len(calls) - initial
    finally:
        _reset()

    if recompiles > max_recompiles:
        note = (
            f"\n  The first call already compiled {initial} graphs (graph breaks?); "
            "see assert_no_graph_breaks."
            if initial > 1
            else ""
        )
        raise AssertionError(
            f"{recompiles} recompilation(s) after the first compile (allowed: {max_recompiles})."
            + note
            + "\n  Common causes: changing shapes (pad, or allow max_recompiles=1 for dynamic "
            "shapes), Python state mutated in forward(), Python scalars as arguments, changing "
            "dtype/device/requires_grad. Run with TORCH_LOGS=recompiles for the exact guard."
        )
    return recompiles
