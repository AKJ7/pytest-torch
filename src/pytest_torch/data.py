"""Realistic synthetic batches: padded token sequences, images, seeded data loaders.

Everything random goes through ``make_tensor``, so batches are reproducible from the
test's seed and recorded in failure reproduction bundles.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

PADDING = ("right", "left", None)


class _DictDataset(Dataset):
    def __init__(self, columns: Mapping[str, torch.Tensor]):
        lengths = {len(v) for v in columns.values()}
        if len(lengths) != 1:
            raise ValueError(f"columns have different lengths: { {k: len(v) for k, v in columns.items()} }")
        self.columns = dict(columns)
        self.length = lengths.pop()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {k: v[index] for k, v in self.columns.items()}


class BatchFactory:
    """The ``make_batch`` fixture: ``tokens``, ``images`` and ``loader``."""

    def __init__(self, make_tensor: Callable[..., torch.Tensor], seed: int = 0, device: Any = "cpu"):
        self.make_tensor = make_tensor
        self.seed = seed
        self.device = device

    def tokens(
        self,
        batch_size: int,
        seq_len: int,
        *,
        vocab_size: int = 1000,
        padding: str | None = "right",
        min_length: int | None = None,
        pad_token_id: int = 0,
        labels: int | str | None = None,
        token_type_ids: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Token ids and attention mask for sequences of different lengths.

        Lengths are drawn between ``min_length`` (default half of ``seq_len``) and
        ``seq_len``; the first row always has the full length and, with more than one row,
        the last row has ``min_length``, so a batch always mixes padded and unpadded rows.
        ``padding`` is ``"right"``, ``"left"`` or ``None`` (every row full length). Real
        tokens never equal ``pad_token_id``; padding positions hold it and are 0 in
        ``attention_mask``. ``labels`` adds ``"labels"``: an int for that many classes per
        row, or ``"lm"`` for the ids with padding set to -100 (the ignore index).
        """
        if padding not in PADDING:
            raise ValueError(f"padding must be one of {PADDING}, got {padding!r}")
        if not 0 <= pad_token_id < vocab_size or vocab_size < 2:
            raise ValueError(f"need 0 <= pad_token_id < vocab_size and vocab_size >= 2, got "
                             f"{pad_token_id} and {vocab_size}")
        shortest = max(1, seq_len // 2) if min_length is None else min_length
        if not 1 <= shortest <= seq_len:
            raise ValueError(f"min_length must be between 1 and seq_len ({seq_len}), got {shortest}")
        long = torch.long
        if padding is None or shortest == seq_len:
            lengths = torch.full((batch_size,), seq_len, dtype=long)
        else:
            lengths = self.make_tensor(batch_size, dtype=long, low=shortest, high=seq_len + 1).cpu()
            lengths[0] = seq_len
            if batch_size > 1:
                lengths[-1] = shortest
        positions = torch.arange(seq_len)
        if padding == "left":
            mask = positions[None, :] >= (seq_len - lengths)[:, None]
        else:
            mask = positions[None, :] < lengths[:, None]
        mask = mask.to(self.device)
        ids = self.make_tensor(batch_size, seq_len, dtype=long, low=0, high=vocab_size - 1)
        ids = ids + (ids >= pad_token_id).long()  # skip the pad id: [0, vocab) without it
        ids = ids.masked_fill(~mask, pad_token_id)
        batch = {"input_ids": ids, "attention_mask": mask.long()}
        if token_type_ids:
            batch["token_type_ids"] = torch.zeros_like(ids)
        if labels == "lm":
            batch["labels"] = ids.masked_fill(~mask, -100)
        elif isinstance(labels, int):
            batch["labels"] = self.make_tensor(batch_size, dtype=long, low=0, high=labels)
        elif labels is not None:
            raise ValueError(f"labels must be a number of classes or 'lm', got {labels!r}")
        return batch

    def images(
        self,
        batch_size: int,
        channels: int = 3,
        height: int = 224,
        width: int | None = None,
        *,
        uint8: bool = False,
        layout: str = "nchw",
        dtype: torch.dtype | None = None,
        labels: int | None = None,
    ) -> Any:
        """A batch of images: normalized floats (about N(0, 1), like ImageNet-normalized
        pixels) or, with ``uint8=True``, raw pixels in [0, 255]. ``layout`` is ``"nchw"`` or
        ``"nhwc"`` (raw images as decoded). With ``labels`` (a number of classes), returns
        ``(images, labels)``."""
        width = height if width is None else width
        if layout not in ("nchw", "nhwc"):
            raise ValueError(f"layout must be 'nchw' or 'nhwc', got {layout!r}")
        shape = (batch_size, channels, height, width) if layout == "nchw" else (batch_size, height, width, channels)
        if uint8:
            images = self.make_tensor(*shape, dtype=torch.uint8, low=0, high=256)
        else:
            images = self.make_tensor(*shape, dtype=dtype)
        if labels is None:
            return images
        return images, self.make_tensor(batch_size, dtype=torch.long, low=0, high=labels)

    def loader(self, data: Any, batch_size: int = 8, *, shuffle: bool = False, **kwargs: Any) -> DataLoader:
        """A ``DataLoader`` whose shuffling is seeded by the test's seed (single-process by
        default). ``data`` is a ``Dataset``, a tensor, a tuple of tensors (rows of a
        ``TensorDataset``) or a dict of equally long tensors (each item a dict)."""
        if isinstance(data, torch.Tensor):
            dataset: Any = TensorDataset(data)
        elif isinstance(data, tuple) and all(isinstance(t, torch.Tensor) for t in data):
            dataset = TensorDataset(*data)
        elif isinstance(data, Mapping):
            dataset = _DictDataset(data)
        else:
            dataset = data
        kwargs.setdefault("num_workers", 0)
        kwargs.setdefault("generator", torch.Generator().manual_seed(self.seed))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
