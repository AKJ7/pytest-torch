import pytest

torch = pytest.importorskip("torch")
from torch import nn

from pytest_torch import StreamingEquivalence, assert_streaming_equivalent


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class CausalConvStream(nn.Module):
    """Causal 1-D conv over (batch, time, channels) that streams by caching left context."""

    def __init__(self, channels=4, kernel=3, forget_cache=False, leaky_reset=False):
        super().__init__()
        torch.manual_seed(0)
        self.conv = nn.Conv1d(channels, channels, kernel)
        self.kernel = kernel
        self.forget_cache, self.leaky_reset = forget_cache, leaky_reset
        self.cache = None

    def _conv(self, x, left):
        x = torch.cat([left, x], dim=1).transpose(1, 2)
        return self.conv(x).transpose(1, 2)

    def forward(self, x):
        return self._conv(x, x.new_zeros(x.shape[0], self.kernel - 1, x.shape[2]))

    def forward_chunk(self, x):
        if self.cache is None or self.forget_cache:
            self.cache = x.new_zeros(x.shape[0], self.kernel - 1, x.shape[2])
        out = self._conv(x, self.cache)
        self.cache = torch.cat([self.cache, x], dim=1)[:, -(self.kernel - 1):]
        return out

    def reset_cache(self):
        if not self.leaky_reset:  # the bug: a reset that keeps the previous request's context
            self.cache = None


class GRUStream(nn.Module):
    """Internal hidden state; dict inputs with a per-request side input; dict outputs."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.gru = nn.GRU(4, 6, batch_first=True)
        self.condition = nn.Linear(2, 6)
        self.hidden = None

    def forward(self, frames, speaker):
        out, h = self.gru(frames, self.condition(speaker).unsqueeze(0))
        return {"frames": out, "summary": h[-1]}

    def forward_chunk(self, frames, speaker):
        h0 = self.hidden if self.hidden is not None else self.condition(speaker).unsqueeze(0)
        out, self.hidden = self.gru(frames, h0)
        return {"frames": out, "summary": self.hidden[-1]}

    def reset(self):
        self.hidden = None


class LookAhead(nn.Module):
    """Uses one future frame: chunked outputs can't be final at each chunk's last position."""

    def __init__(self):
        super().__init__()
        self.cache = None

    def forward(self, x):
        nxt = torch.cat([x[:, 1:], torch.zeros_like(x[:, :1])], dim=1)
        return x + 0.5 * nxt

    def forward_chunk(self, x):
        return self.forward(x)  # no look-ahead buffering: the boundary position differs

    def reset(self):
        pass


@pytest.fixture
def sequence():
    return torch.randn(2, 20, 4, generator=torch.Generator().manual_seed(0))


# --------------------------------------------------------------------------- #
# Internal state
# --------------------------------------------------------------------------- #
def test_correct_streaming_model_passes_every_check(sequence):
    model = CausalConvStream().eval()
    report = assert_streaming_equivalent(
        model.forward, model.forward_chunk, sequence, chunk_sizes=[1, 7, 32], reset=model.reset_cache
    )
    assert [c.name for c in report.checks] == ["chunked[1]", "chunked[7]", "chunked[32]", "reset", "isolation"]
    assert report.ok


def test_forgotten_cache_fails_chunked_but_not_one_big_chunk(sequence):
    model = CausalConvStream(forget_cache=True).eval()
    report = assert_streaming_equivalent(model.forward, model.forward_chunk, sequence,
                                         chunk_sizes=[1, 7, 32], reset=model.reset_cache, raise_on_failure=False)
    failed = {c.name for c in report.failures}
    assert failed == {"chunked[1]", "chunked[7]"}  # 32 >= length: a single chunk is the full pass
    assert "chunk size 7: streamed output differs from the full pass" in report.format()


def test_leaky_reset_fails_reset_and_isolation(sequence):
    model = CausalConvStream(leaky_reset=True).eval()
    with pytest.raises(AssertionError) as info:
        assert_streaming_equivalent(model.forward, model.forward_chunk, sequence,
                                    chunk_sizes=[5], reset=model.reset_cache, checks=["reset", "isolation"])
    message = str(info.value)
    assert "[FAIL] reset" in message and "state survives the reset" in message
    assert "[FAIL] isolation" in message


def test_dict_inputs_side_inputs_and_summaries(sequence):
    model = GRUStream().eval()
    inputs = {"frames": sequence, "speaker": torch.randn(2, 2)}  # speaker goes to every chunk whole
    assert_streaming_equivalent(model.forward, model.forward_chunk, inputs, chunk_sizes=[1, 6], reset=model.reset)
    # The side input listed first must not be mistaken for the sequence.
    reordered = {"speaker": inputs["speaker"], "frames": sequence}
    assert_streaming_equivalent(model.forward, model.forward_chunk, reordered, chunk_sizes=[6], reset=model.reset)


# --------------------------------------------------------------------------- #
# Explicit state
# --------------------------------------------------------------------------- #
class Cell(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.cell = nn.GRUCell(4, 6)
        self.calls = 0  # used by the leaky variant below

    def full(self, x):
        h, outs = x.new_zeros(x.shape[0], 6), []
        for t in range(x.shape[1]):
            h = self.cell(x[:, t], h)
            outs.append(h)
        return torch.stack(outs, 1)

    def step(self, chunk, state):
        outs = []
        for t in range(chunk.shape[1]):
            state = self.cell(chunk[:, t], state)
            outs.append(state)
        return torch.stack(outs, 1), state


def test_explicit_state_runner(sequence):
    cell = Cell().eval()
    assert_streaming_equivalent(cell.full, cell.step, sequence, chunk_sizes=[1, 3, 20],
                                init_state=lambda: sequence.new_zeros(2, 6))


def test_state_shared_between_requests_is_caught_by_interleaving(sequence):
    cell = Cell().eval()
    shared = {"last": None}

    def leaky_step(chunk, state):
        # The bug: a module-level "last state" used instead of the request's own state.
        state = shared["last"] if shared["last"] is not None else state
        out, state = cell.step(chunk, state)
        shared["last"] = state
        return out, state

    def init():
        shared["last"] = None  # looks like a reset... but requests still share `shared`
        return sequence.new_zeros(2, 6)

    report = assert_streaming_equivalent(cell.full, leaky_step, sequence, chunk_sizes=[5],
                                         init_state=init, raise_on_failure=False)
    assert {c.name for c in report.failures} == {"isolation"}
    assert "request A" in report.format() and "state is shared somewhere" in report.format()


def test_explicit_runner_must_return_output_and_state(sequence):
    cell = Cell().eval()
    with pytest.raises(TypeError, match=r"must return \(output, new_state\)"):
        assert_streaming_equivalent(cell.full, lambda c, s: c, sequence, chunk_sizes=[5],
                                    init_state=lambda: None)


# --------------------------------------------------------------------------- #
# User-defined equivalence
# --------------------------------------------------------------------------- #
def test_look_ahead_model_fails_by_default_and_passes_with_its_policy(sequence):
    model = LookAhead()
    with pytest.raises(AssertionError, match=r"chunked\[5\]"):
        assert_streaming_equivalent(model.forward, model.forward_chunk, sequence, chunk_sizes=[5],
                                    reset=model.reset, checks=["chunked"])
    # Intentional: the last position of every chunk waits for the next chunk, so skip it.
    # With one-frame chunks there is nothing to look at: the output is the input itself.
    assert_streaming_equivalent(
        model.forward, model.forward_chunk, sequence, chunk_sizes=[1, 5, 7], reset=model.reset,
        equivalence={
            1: lambda full, streamed, n: torch.allclose(streamed, 1.5 * sequence) is False
            and torch.allclose(streamed, sequence),
            "default": StreamingEquivalence(skip_last=1),
        },
    )


def test_skip_options_and_tolerances(sequence):
    model = CausalConvStream().eval()
    for policy in (StreamingEquivalence(skip_warmup=3), StreamingEquivalence(skip_first=1, atol=1e-3),
                   StreamingEquivalence(rtol=0, atol=0)):
        assert_streaming_equivalent(model.forward, model.forward_chunk, sequence, chunk_sizes=[4],
                                    reset=model.reset_cache, equivalence=policy, checks=["chunked"])
    with pytest.raises(ValueError, match="skips every position"):
        assert_streaming_equivalent(model.forward, model.forward_chunk, sequence, chunk_sizes=[2],
                                    reset=model.reset_cache, equivalence=StreamingEquivalence(skip_last=2))


def test_custom_callable_equivalence(sequence):
    model = CausalConvStream().eval()

    def same_final_frame(full, streamed, chunk_size):
        return torch.allclose(full[:, -1], streamed[:, -1], atol=1e-5)

    assert_streaming_equivalent(model.forward, model.forward_chunk, sequence, chunk_sizes=[3],
                                reset=model.reset_cache, equivalence=same_final_frame)
    broken = CausalConvStream(forget_cache=True).eval()
    with pytest.raises(AssertionError, match="custom equivalence returned False"):
        assert_streaming_equivalent(broken.forward, broken.forward_chunk, sequence, chunk_sizes=[1],
                                    reset=broken.reset_cache, equivalence=lambda f, s, n: torch.equal(f, s),
                                    checks=["chunked"])


def test_custom_combine(sequence):
    model = CausalConvStream().eval()
    seen = []

    def combine(outputs):
        seen.append(len(outputs))
        return torch.cat(outputs, dim=1)

    assert_streaming_equivalent(model.forward, model.forward_chunk, sequence, chunk_sizes=[7],
                                reset=model.reset_cache, combine=combine, checks=["chunked"])
    assert seen == [3]  # 20 frames in chunks of 7


# --------------------------------------------------------------------------- #
# Usage errors
# --------------------------------------------------------------------------- #
def test_usage_errors(sequence):
    model = CausalConvStream()
    with pytest.raises(ValueError, match=r"reset= .* or init_state="):
        assert_streaming_equivalent(model.forward, model.forward_chunk, sequence)
    with pytest.raises(ValueError, match="positive"):
        assert_streaming_equivalent(model.forward, model.forward_chunk, sequence, chunk_sizes=[0], reset=model.reset_cache)
    with pytest.raises(ValueError, match="unknown check"):
        assert_streaming_equivalent(model.forward, model.forward_chunk, sequence, reset=model.reset_cache, checks=["speed"])
    with pytest.raises(ValueError, match="no input tensor has a dimension 3"):
        assert_streaming_equivalent(model.forward, model.forward_chunk, sequence, reset=model.reset_cache, seq_dim=3)
