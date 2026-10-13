"""Invariance and metamorphic checks: padding, masks, permutations, custom transforms."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import torch

from .assertions import format_problems, leaves_diff
from .utils import flatten, model_device, run_inference, to_device, tree_map


def _check(actual: dict, expected: dict, header: str, *, rtol, atol) -> None:
    problems = leaves_diff(actual, expected, rtol=rtol, atol=atol)
    if problems:
        raise AssertionError(format_problems(f"{header}: {len(problems)} problem(s)", problems))


# --------------------------------------------------------------------------- #
# Generic metamorphic relations
# --------------------------------------------------------------------------- #
def assert_invariant(
    model,
    inputs: Any,
    transform: Callable[[Any], Any],
    *,
    name: str = "the transform",
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    """Assert ``model(transform(x)) == model(x)`` (output is invariant to ``transform``)."""
    inputs = to_device(inputs, model_device(model))
    base = run_inference(model, inputs)
    other = run_inference(model, transform(inputs))
    _check(
        dict(flatten(other)),
        dict(flatten(base)),
        f"Output not invariant under {name}",
        rtol=rtol,
        atol=atol,
    )


def assert_equivariant(
    model,
    inputs: Any,
    input_transform: Callable[[Any], Any],
    output_transform: Callable[[Any], Any],
    *,
    name: str = "the transform",
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    """Assert ``model(input_transform(x)) == output_transform(model(x))``."""
    inputs = to_device(inputs, model_device(model))
    expected = output_transform(run_inference(model, inputs))
    actual = run_inference(model, input_transform(inputs))
    _check(
        dict(flatten(actual)),
        dict(flatten(expected)),
        f"Output not equivariant under {name}",
        rtol=rtol,
        atol=atol,
    )


# --------------------------------------------------------------------------- #
# Permutations
# --------------------------------------------------------------------------- #
def _axis_size(inputs: Any, dim: int) -> int:
    if dim < 0:
        raise ValueError("use a non-negative dim")
    for _, leaf in flatten(inputs, "inputs"):
        if isinstance(leaf, torch.Tensor) and leaf.dim() > dim:
            return int(leaf.shape[dim])
    raise ValueError(f"no input tensor has dimension {dim}")


def _permutation(n: int, seed: int) -> torch.Tensor:
    if n < 2:
        raise ValueError("need at least 2 elements along the permuted dim")
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    if torch.equal(perm, torch.arange(n)):  # never test with the identity
        perm = torch.arange(n).flip(0)
    return perm


def _permute(obj: Any, perm: torch.Tensor, dim: int, n: int) -> Any:
    def fn(t: torch.Tensor) -> torch.Tensor:
        if t.dim() > dim and t.shape[dim] == n:
            return t.index_select(dim, perm.to(t.device))
        return t

    return tree_map(fn, obj)


def assert_permutation_invariant(
    model, inputs: Any, *, dim: int = 1, seed: int = 0, rtol=None, atol=None
) -> None:
    """Assert the output doesn't change when ``dim`` of the inputs is permuted (set models).

    Every tensor whose size along ``dim`` equals that of the first input tensor is
    permuted consistently.
    """
    inputs = to_device(inputs, model_device(model))
    n = _axis_size(inputs, dim)
    perm = _permutation(n, seed)
    assert_invariant(
        model,
        inputs,
        lambda x: _permute(x, perm, dim, n),
        name=f"a permutation of dim {dim}",
        rtol=rtol,
        atol=atol,
    )


def assert_permutation_equivariant(
    model, inputs: Any, *, dim: int = 1, seed: int = 0, rtol=None, atol=None
) -> None:
    """Assert permuting ``dim`` of the inputs permutes the same dim of the outputs.

    Output tensors are permuted only if their size along ``dim`` equals the permuted
    size, so pooled outputs must be invariant. A hidden size that coincidentally equals
    the sequence length would be misinterpreted.
    """
    inputs = to_device(inputs, model_device(model))
    n = _axis_size(inputs, dim)
    perm = _permutation(n, seed)
    assert_equivariant(
        model,
        inputs,
        lambda x: _permute(x, perm, dim, n),
        lambda out: _permute(out, perm, dim, n),
        name=f"a permutation of dim {dim}",
        rtol=rtol,
        atol=atol,
    )


# --------------------------------------------------------------------------- #
# Padding and attention masks (Hugging Face style: input_ids + attention_mask)
# --------------------------------------------------------------------------- #
def _require_mapping(inputs: Any, *keys: str) -> None:
    if not isinstance(inputs, Mapping) or any(k not in inputs for k in keys):
        raise ValueError(
            f"inputs must be a dict (kwargs for the model) containing {list(keys)}"
        )


def _pad(inputs, length, extra, side, seq_dim, pad_value, mask_key) -> dict:
    padded = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor) and value.dim() > seq_dim and value.shape[seq_dim] == length:
            shape = list(value.shape)
            shape[seq_dim] = extra
            filler = value.new_full(shape, 0 if key == mask_key else pad_value)
            parts = [value, filler] if side == "right" else [filler, value]
            padded[key] = torch.cat(parts, dim=seq_dim)
        else:
            padded[key] = value
    return padded


def assert_padding_invariant(
    model,
    inputs: Mapping[str, Any],
    *,
    extra: int = 4,
    side: str = "right",
    pad_value: int = 0,
    ids_key: str = "input_ids",
    mask_key: str = "attention_mask",
    seq_dim: int = 1,
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    """Assert extra padding doesn't change the outputs for the real tokens.

    Every input tensor whose ``seq_dim`` size equals the sequence length gets ``extra``
    positions appended (``side="right"``) or prepended (``"left"``): ``pad_value`` for
    ids/other tensors, ``0`` for the attention mask. Outputs that are sequence-aligned
    are compared on the original positions only; pooled outputs are compared as is.
    Models with absolute position ids can legitimately differ under left padding unless
    you pass ``position_ids`` explicitly; prefer ``side="right"`` for those.
    """
    _require_mapping(inputs, ids_key)
    if side not in ("right", "left"):
        raise ValueError("side must be 'right' or 'left'")
    inputs = to_device(inputs, model_device(model))
    length = int(inputs[ids_key].shape[seq_dim])
    padded = _pad(inputs, length, extra, side, seq_dim, pad_value, mask_key)

    expected = dict(flatten(run_inference(model, inputs)))
    start = 0 if side == "right" else extra
    actual = {}
    for path, leaf in flatten(run_inference(model, padded)):
        ref = expected.get(path)
        if (
            isinstance(leaf, torch.Tensor)
            and isinstance(ref, torch.Tensor)
            and leaf.dim() > seq_dim
            and ref.dim() > seq_dim
            and leaf.shape[seq_dim] == length + extra
            and ref.shape[seq_dim] == length
        ):
            leaf = leaf.narrow(seq_dim, start, length)
        actual[path] = leaf
    _check(
        actual,
        expected,
        f"Padding invariance violated (+{extra} {side})",
        rtol=rtol,
        atol=atol,
    )


def assert_mask_invariant(
    model,
    inputs: Mapping[str, Any],
    *,
    ids_key: str = "input_ids",
    mask_key: str = "attention_mask",
    vocab_size: int | None = None,
    seed: int = 0,
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    """Assert that the content of masked-out positions doesn't affect the real tokens.

    Replaces every masked id with a different random id and compares the outputs at the
    unmasked positions (sequence-aligned outputs, batch dim 0 and sequence dim 1) and
    all pooled outputs. Catches attention-mask bugs, e.g. padding leaking into pooling.
    """
    _require_mapping(inputs, ids_key, mask_key)
    inputs = to_device(inputs, model_device(model))
    ids = inputs[ids_key]
    keep = inputs[mask_key].bool()
    masked = ~keep
    if not masked.any():
        raise ValueError("the attention mask has no padding (all ones); nothing to check")
    high = int(ids.max()) + 1 if vocab_size is None else int(vocab_size)
    if high < 2:
        raise ValueError("need a vocabulary of at least 2 ids")

    generator = torch.Generator().manual_seed(seed)
    delta = torch.randint(1, high, tuple(ids.shape), generator=generator).to(ids.device)
    scrambled = torch.where(masked, (ids + delta) % high, ids).to(ids.dtype)
    changed = {**inputs, ids_key: scrambled}

    def select(leaf):
        if isinstance(leaf, torch.Tensor) and leaf.dim() >= 2 and leaf.shape[:2] == keep.shape:
            return leaf[keep]
        return leaf

    expected = {p: select(l) for p, l in flatten(run_inference(model, inputs))}
    actual = {p: select(l) for p, l in flatten(run_inference(model, changed))}
    _check(actual, expected, "Mask invariance violated (masked tokens leak)", rtol=rtol, atol=atol)
