"""Checkpoints: reloading preserves predictions, and old checkpoints still load.

Run: pytest examples/test_checkpoints.py -v
"""
import torch
from torch import nn

from pytest_torch import assert_checkpoint_loads, assert_checkpoint_roundtrip


class LanguageModel(nn.Module):
    def __init__(self, vocab=100, dim=32):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.block = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.lm_head.weight = self.emb.weight  # tied weights: must survive a reload
        # A random projection that is part of the model, so it must be saved with it.
        # Try `self.projection = torch.randn(dim, dim) / dim**0.5` (a plain attribute, not a
        # buffer): the round-trip fails and names `projection` as the culprit.
        self.register_buffer("projection", torch.randn(dim, dim) / dim**0.5)

    def forward(self, ids):
        return self.lm_head(self.block(self.emb(ids)) @ self.projection)


def test_save_and_reload_preserves_predictions(torch_seed, torch_device, make_tensor):
    ids = make_tensor(2, 10, dtype=torch.long, low=0, high=100)
    assert_checkpoint_roundtrip(lambda: LanguageModel().to(torch_device), ids)


class LanguageModelV2(LanguageModel):
    """Today's code: the block was renamed, a checkpoint saved by v1 must still load."""

    def __init__(self):
        super().__init__()
        self.encoder = self.block  # new name...
        del self.block  # ...old name gone

    def forward(self, ids):
        return self.lm_head(self.encoder(self.emb(ids)) @ self.projection)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Backward compatibility shim: map v1 names to v2 names.
        for key in [k for k in state_dict if k.startswith(prefix + "block.")]:
            state_dict[prefix + "encoder." + key[len(prefix + "block."):]] = state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


def test_v1_checkpoint_loads_into_v2_code(torch_seed, tmp_path):
    # In a real project the v1 checkpoint is a small file committed next to the tests.
    v1 = LanguageModel()
    path = tmp_path / "lm_v1.pt"
    torch.save({"step": 1000, "model_state_dict": v1.state_dict()}, path)

    # Try deleting `_load_from_state_dict` above: the test lists every missing `encoder.*`
    # key and every unexpected `block.*` key instead of failing on the first one.
    v2 = assert_checkpoint_loads(lambda: LanguageModelV2(), path)
    ids = torch.randint(0, 100, (2, 10))
    with torch.inference_mode():
        assert torch.equal(v1.eval()(ids), v2.eval()(ids))
