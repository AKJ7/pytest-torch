"""Project code: a tokenizer, a transformer text classifier and a synthetic sentiment dataset."""
from __future__ import annotations

import random
from typing import NamedTuple

import torch
from torch import nn

PAD, UNK = "<pad>", "<unk>"

POSITIVE = ["great", "excellent", "love", "wonderful", "superb", "delightful", "fantastic"]
NEGATIVE = ["awful", "terrible", "hate", "boring", "dreadful", "poor", "disappointing"]
NEUTRAL = ["the", "film", "was", "a", "plot", "actor", "scene", "really", "quite", "story", "it"]
LABELS = ("negative", "neutral", "positive")


class Tokenizer:
    """Whitespace tokenizer with a fixed vocabulary, padding and attention masks."""

    def __init__(self, words: list[str]):
        self.vocab = {PAD: 0, UNK: 1, **{w: i + 2 for i, w in enumerate(sorted(set(words)))}}
        self.pad_id = 0

    def __len__(self) -> int:
        return len(self.vocab)

    def __call__(self, texts: list[str], max_len: int = 16) -> dict[str, torch.Tensor]:
        ids = torch.full((len(texts), max_len), self.pad_id, dtype=torch.long)
        mask = torch.zeros(len(texts), max_len, dtype=torch.long)
        for row, text in enumerate(texts):
            tokens = [self.vocab.get(w, 1) for w in text.lower().split()][:max_len]
            ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            mask[row, : len(tokens)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def make_dataset(n: int, seed: int = 0) -> list[tuple[str, int]]:
    """Sentences of filler words plus sentiment words; the label is the majority sentiment."""
    rng = random.Random(seed)
    data = []
    for _ in range(n):
        label = rng.randrange(3)
        words = rng.choices(NEUTRAL, k=rng.randint(2, 8))
        if label != 1:
            pool = POSITIVE if label == 2 else NEGATIVE
            words += rng.choices(pool, k=rng.randint(1, 3))
        rng.shuffle(words)
        data.append((" ".join(words), label))
    return data


class ClassifierOutput(NamedTuple):
    # A NamedTuple, not a dataclass: torch.export handles NamedTuples out of the box, while a
    # dataclass output must first be registered with torch.export.register_dataclass.
    logits: torch.Tensor
    pooled: torch.Tensor


class TextClassifier(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 32, heads: int = 4, layers: int = 2, classes: int = 3,
                 max_len: int = 64):
        super().__init__()
        self.tokens = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.positions = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=2 * dim, dropout=0.1, batch_first=True)
        # No nested-tensor fast path: its padded positions differ from the regular path.
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(dim, classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> ClassifierOutput:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        h = self.tokens(input_ids) + self.positions(positions)
        h = self.encoder(h, src_key_padding_mask=attention_mask == 0)
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1)  # mean over real tokens only
        return ClassifierOutput(self.head(pooled), pooled)
