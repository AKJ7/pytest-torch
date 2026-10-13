"""Porting or optimizing a model: prove the new version matches the old one.

Run: pytest examples/test_reimplementation.py -v
"""
import pytest
import torch
from torch import nn

from pytest_torch import assert_generation_equal, assert_matches_reference, localize_divergence


# --------------------------------------------------------------------------- #
# A hand-written attention block vs. torch's nn.MultiheadAttention
# --------------------------------------------------------------------------- #
class MyAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    @classmethod
    def from_torch(cls, ref: nn.MultiheadAttention):
        mine = cls(ref.embed_dim, ref.num_heads)
        with torch.no_grad():
            mine.qkv.weight.copy_(ref.in_proj_weight)
            mine.qkv.bias.copy_(ref.in_proj_bias)
            mine.out.weight.copy_(ref.out_proj.weight)
            mine.out.bias.copy_(ref.out_proj.bias)
        return mine

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        # Try dropping the scale (`/ q.shape[-1] ** 0.5`): the reference test fails.
        attn = (q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5).softmax(-1)
        return self.out((attn @ v).transpose(1, 2).reshape(b, n, d))


class TorchAttention(nn.Module):
    def __init__(self, dim=32, heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x):
        return self.mha(x, x, x, need_weights=False)[0]


def test_attention_port_matches_torch(torch_seed, torch_device, make_tensor):
    reference = TorchAttention().to(torch_device).eval()
    mine = MyAttention.from_torch(reference.mha).to(torch_device).eval()
    assert_matches_reference(mine, reference, make_tensor(2, 7, 32), atol=1e-5)


# --------------------------------------------------------------------------- #
# Inference optimization: fold BatchNorm into the preceding Linear
# --------------------------------------------------------------------------- #
def build(fused: bool):
    norm = nn.Identity() if fused else nn.BatchNorm1d(32)
    return nn.Sequential(nn.Linear(16, 32), norm, nn.ReLU(), nn.Linear(32, 4))


def fold_batchnorm(model):
    lin, bn = model[0], model[1]
    fused = build(fused=True)
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    with torch.no_grad():
        fused[0].weight.copy_(lin.weight * scale[:, None])
        # Try forgetting `- bn.running_mean`: the test fails with the worst element.
        fused[0].bias.copy_((lin.bias - bn.running_mean) * scale + bn.bias)
        fused[3].load_state_dict(model[3].state_dict())
    return fused.to(lin.weight.device).eval()


def test_batchnorm_folding_preserves_outputs(torch_seed, torch_device, make_tensor):
    original = build(fused=False)
    with torch.no_grad():  # non-trivial running statistics
        original[1].running_mean.uniform_(-1, 1)
        original[1].running_var.uniform_(0.5, 2)
    original = original.to(torch_device).eval()
    assert_matches_reference(fold_batchnorm(original), original, make_tensor(8, 16), atol=1e-5)


# --------------------------------------------------------------------------- #
# A refactor that renames layers: if outputs drift, which layer is to blame?
# --------------------------------------------------------------------------- #
class Refactored(nn.Module):
    """Same network as `build(fused=False)`, with named submodules instead of indices."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(16, 32)
        self.norm = nn.BatchNorm1d(32)
        self.act = nn.ReLU()
        self.head = nn.Linear(32, 4)

    def forward(self, x):
        # Try `self.act(self.norm(self.encoder(x)) * 2)`: the report flags layer '2' (the
        # ReLU, 'act' here) with inputs=DIFF, i.e. the bug sits between two layers.
        return self.head(self.act(self.norm(self.encoder(x))))


def test_refactor_matches_layer_by_layer(torch_seed, torch_device, make_tensor):
    original = build(fused=False).to(torch_device).eval()
    refactored = Refactored().to(torch_device).eval()
    names = {"0": "encoder", "1": "norm", "2": "act", "3": "head"}
    refactored.load_state_dict({
        key.replace(f"{old}.", f"{new}.", 1): value
        for key, value in original.state_dict().items()
        for old, new in names.items()
        if key.startswith(f"{old}.")
    })
    report = localize_divergence(original, refactored, make_tensor(8, 16), module_map=names)
    report.raise_if_diverged()  # on failure: a table of layers, first culprit named


# --------------------------------------------------------------------------- #
# Greedy decoding: a KV-cache-style incremental decoder vs. full recomputation
# --------------------------------------------------------------------------- #
class TinyLM(nn.Module):
    def __init__(self, vocab=40, dim=24):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.rnn = nn.GRU(dim, dim, batch_first=True)
        self.head = nn.Linear(dim, vocab)

    def forward(self, ids, state=None):
        out, state = self.rnn(self.emb(ids), state)
        return self.head(out), state


class RecomputeDecoder(nn.Module):
    """Slow and obviously correct: rerun the whole prefix for every new token."""

    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def generate(self, ids, max_new_tokens, do_sample=False):
        for _ in range(max_new_tokens):
            logits, _ = self.lm(ids)
            ids = torch.cat([ids, logits[:, -1:].argmax(-1)], dim=1)
        return ids


class IncrementalDecoder(RecomputeDecoder):
    """Fast: feed only the newest token and carry the recurrent state (a "KV cache")."""

    def generate(self, ids, max_new_tokens, do_sample=False):
        logits, state = self.lm(ids)
        for _ in range(max_new_tokens):
            nxt = logits[:, -1:].argmax(-1)
            ids = torch.cat([ids, nxt], dim=1)
            # Try `self.lm(nxt)` (dropping the state): the test reports the first wrong token.
            logits, state = self.lm(nxt, state)
        return ids


@pytest.mark.parametrize("prompt_len", [1, 5])
def test_incremental_decoding_matches_recompute(torch_seed, torch_device, make_tensor, prompt_len):
    lm = TinyLM().to(torch_device).eval()
    prompt = make_tensor(3, prompt_len, dtype=torch.long, low=0, high=40)
    # Anything with an HF-style .generate() works; otherwise pass generate=fn.
    assert_generation_equal(IncrementalDecoder(lm), RecomputeDecoder(lm), prompt, max_new_tokens=12)
