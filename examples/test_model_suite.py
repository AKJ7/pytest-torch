"""A full test suite from one class: describe the model, pick the checks.

Run: pytest examples/test_model_suite.py -v
"""
import torch
from torch import nn
from torch.nn import functional as F

from pytest_torch import ModelSuite


class TextClassifier(nn.Module):
    def __init__(self, vocab=100, dim=32, classes=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.encoder = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.head = nn.Linear(dim, classes)

    def forward(self, input_ids, attention_mask):
        h = self.encoder(self.emb(input_ids))
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1)
        return {"logits": self.head(pooled), "hidden": h}


class TestTextClassifier(ModelSuite):
    # contract: eval mode, finite, deterministic, batch invariance, devices, no mutation
    # compile / export: torch.compile and torch.export match eager
    # autocast: bf16 mixed precision stays close
    # training: finite grads, every parameter learns, loss goes down, one batch overfits
    checks = ("contract", "compile", "export", "autocast", "training")
    dynamic_batch = True  # the exported program must also accept other batch sizes
    check_options = {
        "inference_contract": {"dtypes": (torch.bfloat16,)},
        "overfits_one_batch": {"steps": 300},
    }

    def make_model(self):
        return TextClassifier()

    def make_inputs(self, make_tensor):
        ids = make_tensor(8, 12, dtype=torch.long, low=1, high=100)
        mask = torch.ones_like(ids)
        mask[4:, 8:] = 0  # some padded rows
        return {"input_ids": ids, "attention_mask": mask}

    def loss(self, output, inputs):
        labels = inputs["input_ids"][:, 0] % 4  # any fixed labels will do for sanity checks
        return F.cross_entropy(output["logits"], labels)

    # Generated tests are ordinary methods: add your own next to them.
    def test_logits_shape(self, torch_seed, torch_device, make_tensor):
        model = self.make_model().to(torch_device).eval()
        with torch.inference_mode():
            assert model(**self.make_inputs(make_tensor))["logits"].shape == (8, 4)
