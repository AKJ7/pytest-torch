"""Stateful and streaming models: chunked processing, resets and request isolation.

    report = assert_streaming_equivalent(
        full_runner=model.forward,
        streaming_runner=model.forward_chunk,
        inputs=sequence,
        chunk_sizes=[1, 7, 32],
        reset=model.reset_cache,
    )

Three checks, reported separately:

- **chunked**: for every chunk size, feeding the sequence chunk by chunk and concatenating
  the outputs agrees with one full pass (under the chosen equivalence);
- **reset**: after ``reset()`` (or a fresh ``init_state()``), a second streaming pass
  reproduces the first exactly;
- **isolation**: one request does not contaminate another. With internal state: request A,
  reset, request B must equal B alone. With explicit state: A and B interleaved chunk by
  chunk, each with its own state, must each equal their isolated run.

The expected equivalence is user-defined, because some models intentionally behave
differently when chunked (look-ahead, chunk-local attention, downsampling): see
:class:`StreamingEquivalence` and the ``equivalence`` argument.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Union

import torch

from . import tolerances
from .assertions import format_problems, leaves_diff
from .utils import _is_dataclass_instance, _is_namedtuple, call_model, flatten, tree_map


@dataclasses.dataclass(frozen=True)
class StreamingEquivalence:
    """How streamed outputs must relate to the full pass.

    ``rtol``/``atol``: tolerances (default: the ``streaming`` tolerance, configurable in
    ``[tool.pytest_torch.tolerance]``). ``skip_first``/``skip_last``: positions to ignore
    at the start/end of *each chunk*, e.g. ``skip_last=2`` for a model that looks two
    steps ahead and so can't finish a chunk's last outputs without the next chunk.
    ``skip_warmup``: positions to ignore at the start of the whole sequence.
    """

    rtol: float | None = None
    atol: float | None = None
    skip_first: int = 0
    skip_last: int = 0
    skip_warmup: int = 0


EquivalenceFn = Callable[[Any, Any, int], Any]
Equivalence = Union[None, StreamingEquivalence, EquivalenceFn]


@dataclasses.dataclass
class StreamingCheck:
    name: str  # "chunked[7]", "reset", "isolation"
    passed: bool
    message: str = ""


@dataclasses.dataclass
class StreamingReport:
    checks: list[StreamingCheck]

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[StreamingCheck]:
        return [c for c in self.checks if not c.passed]

    def format(self) -> str:
        lines = [f"Streaming: {sum(c.passed for c in self.checks)} passed, {len(self.failures)} failed"]
        for check in self.checks:
            lines.append(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name}")
            if not check.passed:
                lines += ["         " + line for line in check.message.splitlines()]
        return "\n".join(lines)

    __str__ = format


# --------------------------------------------------------------------------- #
# splitting inputs and combining outputs
# --------------------------------------------------------------------------- #
def _sequence_length(inputs: Any, seq_dim: int) -> int:
    """The longest size along ``seq_dim``: side inputs (embeddings, flags) are shorter."""
    sizes = [int(leaf.shape[seq_dim]) for _, leaf in flatten(inputs, "inputs")
             if isinstance(leaf, torch.Tensor) and leaf.dim() > seq_dim]
    if not sizes:
        raise ValueError(f"no input tensor has a dimension {seq_dim} to stream over")
    return max(sizes)


def _chunks(inputs: Any, chunk_size: int, seq_dim: int) -> list[tuple[Any, int]]:
    """``(chunk, chunk_length)`` pairs: slices of every input tensor whose size along
    ``seq_dim`` is the sequence length; other inputs (speaker embeddings, flags, ...) go to
    every chunk unchanged."""
    length = _sequence_length(inputs, seq_dim)

    def piece(start: int):
        def cut(t: torch.Tensor) -> torch.Tensor:
            if t.dim() > seq_dim and t.shape[seq_dim] == length:
                return t.narrow(seq_dim, start, min(chunk_size, length - start))
            return t

        return tree_map(cut, inputs)

    return [(piece(start), min(chunk_size, length - start)) for start in range(0, length, chunk_size)]


def _combine(outputs: list[Any], lengths: list[int], seq_dim: int) -> Any:
    """Concatenate per-chunk outputs along ``seq_dim`` where their length matches the chunk;
    keep the last chunk's value otherwise (summaries, final states)."""
    first = outputs[0]
    if isinstance(first, torch.Tensor):
        per_step = all(
            isinstance(o, torch.Tensor) and o.dim() > seq_dim and o.shape[seq_dim] == n
            for o, n in zip(outputs, lengths)
        )
        return torch.cat(outputs, dim=seq_dim) if per_step else outputs[-1]
    if isinstance(first, Mapping):
        combined = {k: _combine([o[k] for o in outputs], lengths, seq_dim) for k in first}
        try:
            return type(first)(combined)
        except Exception:
            return combined
    if _is_namedtuple(first):
        return type(first)(*[_combine([o[i] for o in outputs], lengths, seq_dim) for i in range(len(first))])
    if isinstance(first, (list, tuple)):
        return type(first)(_combine([o[i] for o in outputs], lengths, seq_dim) for i in range(len(first)))
    if _is_dataclass_instance(first):
        fields = {f.name: _combine([getattr(o, f.name) for o in outputs], lengths, seq_dim)
                  for f in dataclasses.fields(first)}
        return dataclasses.replace(first, **fields)
    return outputs[-1]


def _detached(obj: Any) -> Any:
    return tree_map(lambda t: t.detach().clone(), obj)


# --------------------------------------------------------------------------- #
# comparison under an equivalence policy
# --------------------------------------------------------------------------- #
def _keep_mask(length: int, chunk_size: int, policy: StreamingEquivalence) -> torch.Tensor:
    keep = torch.ones(length, dtype=torch.bool)
    keep[: policy.skip_warmup] = False
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        keep[start : start + policy.skip_first] = False
        if policy.skip_last:
            keep[max(start, end - policy.skip_last) : end] = False
    return keep


def _policy_compare(full: Any, streamed: Any, chunk_size: int, length: int, seq_dim: int,
                    policy: StreamingEquivalence, device: Any) -> list[str]:
    rtol, atol = tolerances.for_check("streaming", device, policy.rtol, policy.atol)
    keep = _keep_mask(length, chunk_size, policy)
    if not bool(keep.any()):
        raise ValueError(f"chunk size {chunk_size}: the equivalence policy skips every position")

    def select(t: torch.Tensor) -> torch.Tensor:
        if t.dim() > seq_dim and t.shape[seq_dim] == length:
            return t.index_select(seq_dim, keep.nonzero().squeeze(1).to(t.device))
        return t

    return leaves_diff(dict(flatten(tree_map(select, streamed))), dict(flatten(tree_map(select, full))),
                       rtol=rtol, atol=atol, check_device=False)


def _compare(full: Any, streamed: Any, chunk_size: int, length: int, seq_dim: int,
             equivalence: Any, device: Any) -> list[str]:
    if isinstance(equivalence, Mapping):  # per chunk size
        equivalence = equivalence.get(chunk_size, equivalence.get("default"))
    if equivalence is None:
        equivalence = StreamingEquivalence()
    if isinstance(equivalence, StreamingEquivalence):
        return _policy_compare(full, streamed, chunk_size, length, seq_dim, equivalence, device)
    if callable(equivalence):
        try:
            verdict = equivalence(full, streamed, chunk_size)
        except AssertionError as exc:
            return [str(exc) or "custom equivalence failed"]
        return ["custom equivalence returned False"] if verdict is False else []
    raise TypeError("equivalence must be None, a StreamingEquivalence, a callable, or a dict of those")


# --------------------------------------------------------------------------- #
# runners
# --------------------------------------------------------------------------- #
class _Stream:
    def __init__(self, streaming_runner, reset, init_state, seq_dim, combine):
        if reset is None and init_state is None:
            raise ValueError(
                "pass reset= (models keeping their own state) or init_state= (runners taking and "
                "returning a state): the checks need a known starting state"
            )
        self.runner, self.reset, self.init_state = streaming_runner, reset, init_state
        self.seq_dim, self.combine = seq_dim, combine

    def start(self):
        if self.reset is not None:
            self.reset()
        return self.init_state() if self.init_state is not None else None

    def step(self, chunk, state):
        if self.init_state is None:
            return call_model(self.runner, chunk), None
        out = self.runner(chunk, state)
        if not (isinstance(out, tuple) and len(out) == 2):
            raise TypeError("with init_state=, streaming_runner(chunk, state) must return (output, new_state)")
        return out

    def finish(self, outputs, chunks):
        if self.combine is not None:
            return self.combine(outputs)
        return _combine(outputs, [n for _, n in chunks], self.seq_dim)

    def run(self, inputs, chunk_size):
        chunks = _chunks(inputs, chunk_size, self.seq_dim)
        state, outputs = self.start(), []
        for chunk, _ in chunks:
            out, state = self.step(chunk, state)
            outputs.append(_detached(out))
        return self.finish(outputs, chunks)


def _device_of(obj: Any):
    tensors = [leaf for _, leaf in flatten(obj) if isinstance(leaf, torch.Tensor)]
    return tensors[0].device if tensors else None


def _other_request(inputs: Any, seq_dim: int) -> Any:
    """A different request of the same length: every streamed input reversed in time."""
    length = _sequence_length(inputs, seq_dim)
    return tree_map(lambda t: t.flip(seq_dim) if t.dim() > seq_dim and t.shape[seq_dim] == length else t, inputs)


def assert_streaming_equivalent(
    full_runner: Callable,
    streaming_runner: Callable,
    inputs: Any,
    *,
    chunk_sizes: Sequence[int] = (1, 7, 32),
    reset: Callable[[], Any] | None = None,
    init_state: Callable[[], Any] | None = None,
    equivalence: Equivalence | Mapping[Any, Equivalence] = None,
    seq_dim: int = 1,
    combine: Callable[[list], Any] | None = None,
    other_inputs: Any = None,
    checks: Sequence[str] = ("chunked", "reset", "isolation"),
    raise_on_failure: bool = True,
) -> StreamingReport:
    """Check a stateful model's streaming path against its full path. See the module docs.

    ``full_runner(inputs)`` and ``streaming_runner(chunk)`` are called like a model (a
    tensor, a tuple of positional arguments, or a dict of keyword arguments). With
    ``init_state``, the streaming runner is called as ``streaming_runner(chunk, state)``
    and returns ``(output, new_state)``. Inputs are split along ``seq_dim``; tensors of
    another length (embeddings, flags) go to every chunk whole; the sequence length is the
    longest size along ``seq_dim``.

    ``equivalence``: ``None`` (outputs agree within the ``streaming`` tolerance), a
    :class:`StreamingEquivalence` (tolerances, positions to skip), a callable
    ``(full_output, streamed_output, chunk_size)`` that raises ``AssertionError`` or returns
    ``False`` on a mismatch, or a dict of those keyed by chunk size (``"default"`` for the
    rest). ``other_inputs`` is the second request for the isolation check (default: the
    inputs reversed in time). Returns a :class:`StreamingReport`.
    """
    unknown = set(checks) - {"chunked", "reset", "isolation"}
    if unknown:
        raise ValueError(f"unknown check(s) {sorted(unknown)}; choose from chunked, reset, isolation")
    if not chunk_sizes or any(int(c) < 1 for c in chunk_sizes):
        raise ValueError("chunk_sizes must be positive integers")
    stream = _Stream(streaming_runner, reset, init_state, seq_dim, combine)
    length = _sequence_length(inputs, seq_dim)
    device = _device_of(inputs)
    results: list[StreamingCheck] = []

    with torch.inference_mode():
        if "chunked" in checks:
            if reset is not None:
                reset()
            full = _detached(call_model(full_runner, inputs))
            for size in chunk_sizes:
                streamed = stream.run(inputs, int(size))
                problems = _compare(full, streamed, int(size), length, seq_dim, equivalence, device)
                results.append(StreamingCheck(
                    f"chunked[{size}]", not problems,
                    format_problems(f"chunk size {size}: streamed output differs from the full pass", problems)
                    if problems else "",
                ))

        size = int(min(chunk_sizes))
        if "reset" in checks:
            first = stream.run(inputs, size)
            second = stream.run(inputs, size)
            problems = leaves_diff(dict(flatten(second)), dict(flatten(first)), rtol=0.0, atol=0.0, check_device=False)
            results.append(StreamingCheck(
                "reset", not problems,
                format_problems("after a reset, streaming the same input again gave different outputs "
                                "(state survives the reset)", problems) if problems else "",
            ))

        if "isolation" in checks:
            other = other_inputs if other_inputs is not None else _other_request(inputs, seq_dim)
            alone = stream.run(inputs, size)
            if init_state is None:
                stream.run(other, size)  # request B, then reset, then request A
                after_other = stream.run(inputs, size)
                problems = leaves_diff(dict(flatten(after_other)), dict(flatten(alone)),
                                       rtol=0.0, atol=0.0, check_device=False)
                header = "a request processed after another one (with a reset between) differs from running it alone"
            else:
                other_alone = stream.run(other, size)
                mixed_a, mixed_b = _interleave(stream, inputs, other, size)
                problems = [f"request A: {p}" for p in leaves_diff(
                    dict(flatten(mixed_a)), dict(flatten(alone)), rtol=0.0, atol=0.0, check_device=False)]
                problems += [f"request B: {p}" for p in leaves_diff(
                    dict(flatten(mixed_b)), dict(flatten(other_alone)), rtol=0.0, atol=0.0, check_device=False)]
                header = ("two requests interleaved chunk by chunk, each with its own state, differ from "
                          "running them alone (state is shared somewhere)")
            results.append(StreamingCheck("isolation", not problems,
                                          format_problems(header, problems) if problems else ""))

    report = StreamingReport(results)
    if raise_on_failure and not report.ok:
        raise AssertionError(report.format())
    return report


def _interleave(stream: _Stream, a: Any, b: Any, size: int):
    chunks_a, chunks_b = _chunks(a, size, stream.seq_dim), _chunks(b, size, stream.seq_dim)
    state_a, state_b = stream.init_state(), stream.init_state()
    if stream.reset is not None:
        stream.reset()
    outs_a, outs_b = [], []
    for i in range(max(len(chunks_a), len(chunks_b))):
        if i < len(chunks_a):
            out, state_a = stream.step(chunks_a[i][0], state_a)
            outs_a.append(_detached(out))
        if i < len(chunks_b):
            out, state_b = stream.step(chunks_b[i][0], state_b)
            outs_b.append(_detached(out))
    return stream.finish(outs_a, chunks_a), stream.finish(outs_b, chunks_b)
