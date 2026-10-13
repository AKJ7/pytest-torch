"""Stateful and streaming models: chunked processing, resets and request isolation.

Run: pytest examples/test_streaming.py -v
"""
import pytest
import torch
from torch import nn

from pytest_torch import StreamingEquivalence, assert_streaming_equivalent


class StreamingDenoiser(nn.Module):
    """A causal conv front end plus a GRU, usable on whole clips or on live audio frames.

    `forward` handles a whole clip; `forward_chunk` keeps the conv's left context and the
    GRU's hidden state between calls, until `reset_cache()` starts a new stream.
    """

    def __init__(self, features=8, hidden=16, kernel=4):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(features, hidden, kernel)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, features)
        self.context = None  # last kernel-1 input frames
        self.hidden = None   # GRU state

    def reset_cache(self):
        self.context = None  # last kernel-1 input frames
        self.hidden = None   # GRU state
        # Try commenting out `self.hidden = None`: the chunked, reset and isolation checks
        # fail, because every new stream starts from the previous stream's GRU state.

    def _frontend(self, frames, left):
        x = torch.cat([left, frames], dim=1).transpose(1, 2)
        return torch.relu(self.conv(x)).transpose(1, 2)

    def forward(self, frames):
        left = frames.new_zeros(frames.shape[0], self.kernel - 1, frames.shape[2])
        h, _ = self.gru(self._frontend(frames, left))
        return frames - self.out(h)  # estimated clean frames

    def forward_chunk(self, frames):
        if self.context is None:
            self.context = frames.new_zeros(frames.shape[0], self.kernel - 1, frames.shape[2])
        # Try passing zeros instead of `self.context` (forgetting the conv's left context):
        # every chunk size but the full length fails, at each chunk's first frames.
        h, self.hidden = self.gru(self._frontend(frames, self.context), self.hidden)
        self.context = torch.cat([self.context, frames], dim=1)[:, -(self.kernel - 1):]
        return frames - self.out(h)


@pytest.fixture
def denoiser(torch_seed, torch_device):
    return StreamingDenoiser().to(torch_device).eval()


def test_streaming_matches_offline_processing(denoiser, make_tensor):
    clip = make_tensor(2, 50, 8)  # (batch, frames, features)
    report = assert_streaming_equivalent(
        full_runner=denoiser.forward,
        streaming_runner=denoiser.forward_chunk,
        inputs=clip,
        chunk_sizes=[1, 7, 32],  # frame by frame, odd sizes, and larger than a typical buffer
        reset=denoiser.reset_cache,
    )
    print(report)  # with `pytest -s`: [PASS] chunked[1] ... [PASS] reset, [PASS] isolation


# --------------------------------------------------------------------------- #
# Explicit state: a pure step function, with the state passed in and returned
# --------------------------------------------------------------------------- #
def make_step(model):
    def step(frames, state):
        context, hidden = state
        h, hidden = model.gru(model._frontend(frames, context), hidden)
        context = torch.cat([context, frames], dim=1)[:, -(model.kernel - 1):]
        return frames - model.out(h), (context, hidden)

    return step


def test_explicit_state_keeps_concurrent_requests_apart(denoiser, make_tensor):
    clip = make_tensor(2, 30, 8)
    # With init_state, isolation runs two requests interleaved chunk by chunk, each with its
    # own state: what a server handling several streams at once does.
    assert_streaming_equivalent(
        denoiser.forward,
        make_step(denoiser),
        clip,
        chunk_sizes=[1, 5],
        init_state=lambda: (clip.new_zeros(2, denoiser.kernel - 1, 8), None),
        other_inputs=make_tensor(2, 30, 8),  # a genuinely different second request
    )


# --------------------------------------------------------------------------- #
# Models that intentionally differ when chunked
# --------------------------------------------------------------------------- #
class LookAheadSmoother(nn.Module):
    """Averages each frame with the next two: it needs two frames of look-ahead."""

    def forward(self, frames):
        padded = torch.cat([frames, frames.new_zeros(frames.shape[0], 2, frames.shape[2])], dim=1)
        return (padded[:, :-2] + padded[:, 1:-1] + padded[:, 2:]) / 3

    def forward_chunk(self, frames):
        return self.forward(frames)  # each chunk's last 2 frames can't see the next chunk

    def reset(self):
        pass


def test_look_ahead_model_with_a_declared_policy(make_tensor):
    model = LookAheadSmoother()
    clip = make_tensor(2, 40, 8)
    assert_streaming_equivalent(
        model.forward,
        model.forward_chunk,
        clip,
        chunk_sizes=[5, 8, 40],
        reset=model.reset,
        # The deviation is by design: only each chunk's last 2 frames may differ.
        # Try skip_last=1: the second-to-last frame of every chunk then fails.
        equivalence=StreamingEquivalence(skip_last=2),
    )


def test_per_chunk_size_expectations(make_tensor):
    model = LookAheadSmoother()
    clip = make_tensor(1, 12, 8)

    def no_future_at_all(full, streamed, chunk_size):
        # One frame per chunk: no look-ahead is possible, each output is frame / 3.
        assert torch.allclose(streamed, clip / 3, atol=1e-6), "1-frame chunks must see no future"

    assert_streaming_equivalent(
        model.forward, model.forward_chunk, clip, chunk_sizes=[1, 4], reset=model.reset,
        equivalence={1: no_future_at_all, "default": StreamingEquivalence(skip_last=2)},
    )
