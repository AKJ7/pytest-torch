"""A small GPT with a KV cache: correctness of attention, caching, generation and training.

Run: pytest examples/integration/language_model -v
"""
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from pytest_torch import (
    ModelSuite,
    TensorSpec,
    assert_generation_equal,
    assert_invariant,
    assert_outputs_close,
    assert_streaming_equivalent,
    record_repro,
)

from .model import CORPUS, VOCAB, TinyGPT, decode, encode

TORCH_VERSION = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
# torch.export can't return None (the empty `past` slot) before torch 2.4.
CAN_EXPORT = TORCH_VERSION >= (2, 4)


# --------------------------------------------------------------------------- #
# The generated suite
# --------------------------------------------------------------------------- #
def next_token_loss(output, input_ids):
    return F.cross_entropy(output.logits[:, :-1].flatten(0, 1), input_ids[:, 1:].flatten())


class TestTinyGPT(ModelSuite):
    # checkpoint: the tied lm_head/embedding weights must survive save and reload
    checks = ("contract", "checkpoint", "output_spec", "training") + (("export",) if CAN_EXPORT else ())
    dynamic_batch = True
    output_spec = {"logits": TensorSpec(("B", "T", len(VOCAB))), "past": None}
    check_options = {
        "inference_contract": {"atol": 1e-5, "rtol": 1e-4},
        "overfits_one_batch": {"steps": 300, "lr": 3e-3},
    }

    def make_model(self):
        return TinyGPT()

    def make_inputs(self, make_tensor):
        return make_tensor(4, 24, dtype=torch.long, low=0, high=len(VOCAB))

    def loss(self, output, inputs):
        return next_token_loss(output, inputs)


# --------------------------------------------------------------------------- #
# Attention: a position must never see the future
# --------------------------------------------------------------------------- #
class PrefixLogits(nn.Module):
    """Logits of the first `k` positions only."""

    def __init__(self, lm, k):
        super().__init__()
        self.lm, self.k = lm, k

    def forward(self, input_ids):
        return self.lm(input_ids).logits[:, : self.k]


@pytest.fixture
def lm(torch_seed, torch_device):
    return TinyGPT().to(torch_device).eval()


@pytest.mark.parametrize("k", [1, 7, 15])
def test_future_tokens_do_not_change_past_logits(lm, make_tensor, k):
    ids = make_tensor(3, 16, dtype=torch.long, low=0, high=len(VOCAB))

    def rewrite_future(t):
        t = t.clone()
        t[:, k:] = (t[:, k:] + 1) % len(VOCAB)
        return t

    # Try deleting the `masked_fill(future, ...)` line in model.py: attention reads ahead
    # and this test fails for every k < 16.
    assert_invariant(PrefixLogits(lm, k), ids, rewrite_future, name=f"changing tokens after position {k}",
                     atol=1e-5, rtol=1e-4)


# --------------------------------------------------------------------------- #
# The KV cache: an optimization that must not change a single token
# --------------------------------------------------------------------------- #
class WithoutCache(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def generate(self, input_ids, max_new_tokens, **kwargs):
        return self.lm.generate(input_ids, max_new_tokens, use_cache=False)


def test_cached_logits_match_full_recompute(lm, make_tensor):
    ids = make_tensor(2, 12, dtype=torch.long, low=0, high=len(VOCAB))
    with torch.inference_mode():
        full = lm(ids).logits
        out = lm(ids[:, :5], use_cache=True)
        steps = [out.logits]
        for t in range(5, 12):
            out = lm(ids[:, t : t + 1], past=out.past, use_cache=True)
            steps.append(out.logits)
    assert_outputs_close(torch.cat(steps, dim=1), full, atol=1e-5, rtol=1e-4)


def test_chunked_prefill_through_the_cache_matches_one_pass(lm, make_tensor):
    # The KV cache is an explicit streaming state: feeding the prompt in chunks through the
    # cache must give the same logits as one full pass (chunked prefill).
    ids = make_tensor(2, 20, dtype=torch.long, low=0, high=len(VOCAB))

    def prefill(chunk, past):
        out = lm(chunk, past=past, use_cache=True)
        return out.logits, out.past

    assert_streaming_equivalent(
        lambda input_ids: lm(input_ids).logits,
        prefill,
        ids,
        chunk_sizes=[1, 3, 8],
        init_state=lambda: None,  # an empty cache
    )


def test_cached_generation_matches_uncached(trained_lm):
    prompts = torch.stack([encode("pack my box"), encode("the quick b"), encode("how vexingl")])
    # Try `start = 0` in TinyGPT.forward (forgetting the cache offset for positions):
    # the message names the first wrong token of every row.
    assert_generation_equal(trained_lm, WithoutCache(trained_lm), prompts, max_new_tokens=24)


# --------------------------------------------------------------------------- #
# Quality of the trained model
# --------------------------------------------------------------------------- #
def test_trained_model_continues_the_corpus(trained_lm):
    prompt = record_repro("prompt", encode("pack my box").unsqueeze(0))  # in failure bundles
    text = decode(trained_lm.generate(prompt, max_new_tokens=10)[0])
    assert text.startswith("pack my box with five"), text


def windows(ids, length=32, batch=8):
    starts = range(0, len(ids) - length, length // 2)
    pairs = [(ids[s : s + length], ids[s + 1 : s + length + 1]) for s in starts]
    return [(torch.stack([x for x, _ in pairs[i : i + batch]]), torch.stack([y for _, y in pairs[i : i + batch]]))
            for i in range(0, len(pairs), batch)]


def test_corpus_loss_stays_low(trained_lm, corpus_ids, metric_baseline):
    def loss(output, batch):
        return F.cross_entropy(output.logits.flatten(0, 1), batch[1].flatten())

    metric_baseline(trained_lm, windows(corpus_ids), {"loss": loss},
                    higher_is_better=False, tolerance=0.05, maximum={"loss": 0.5})


def test_corpus_is_in_vocabulary():
    assert decode(encode(CORPUS)) == CORPUS
