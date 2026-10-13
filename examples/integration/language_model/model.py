"""Project code: a tiny GPT with a KV cache and greedy generation, plus a character tokenizer."""
from __future__ import annotations

import math
from typing import NamedTuple, Optional

import torch
from torch import nn
from torch.nn import functional as F

CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "pack my box with five dozen liquor jugs. "
    "how vexingly quick daft zebras jump. "
)
VOCAB = sorted(set(CORPUS))


def encode(text: str) -> torch.Tensor:
    return torch.tensor([VOCAB.index(c) for c in text], dtype=torch.long)


def decode(ids: torch.Tensor) -> str:
    return "".join(VOCAB[i] for i in ids.tolist())


class LMOutput(NamedTuple):
    logits: torch.Tensor
    # one (keys, values) pair per layer, each (batch, heads, length, head_dim); None without cache
    past: Optional[tuple]


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, past=None):
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        # Queries sit at the end of the (cached + new) keys: mask keys from their future.
        offset = k.shape[2] - t
        future = torch.ones(t, k.shape[2], dtype=torch.bool, device=x.device).triu(offset + 1)
        scores = scores.masked_fill(future, float("-inf"))
        y = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, t, d)
        return self.out(y), (k, v)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, past=None):
        attn, present = self.attn(self.ln1(x), past)
        x = x + attn
        return x + self.mlp(self.ln2(x)), present


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int = len(VOCAB), dim: int = 48, heads: int = 4, layers: int = 2,
                 max_len: int = 128):
        super().__init__()
        self.tokens = nn.Embedding(vocab_size, dim)
        self.positions = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.tokens.weight  # weight tying

    def forward(self, input_ids: torch.Tensor, past: Optional[tuple] = None, use_cache: bool = False) -> LMOutput:
        start = 0 if past is None else past[0][0].shape[2]
        positions = torch.arange(start, start + input_ids.shape[1], device=input_ids.device)
        x = self.tokens(input_ids) + self.positions(positions)
        presents = []
        for i, block in enumerate(self.blocks):
            x, present = block(x, None if past is None else past[i])
            presents.append(present)
        logits = self.lm_head(self.ln(x))
        return LMOutput(logits, tuple(presents) if use_cache else None)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, use_cache: bool = True, **_) -> torch.Tensor:
        """Greedy decoding; with use_cache=False every step recomputes the whole prefix."""
        ids, past = input_ids, None
        step_input = ids
        for _ in range(max_new_tokens):
            if use_cache:
                out = self(step_input, past=past, use_cache=True)
                past = out.past
            else:
                out = self(ids)
            next_id = out.logits[:, -1:].argmax(-1)
            ids = torch.cat([ids, next_id], dim=1)
            step_input = next_id
        return ids
