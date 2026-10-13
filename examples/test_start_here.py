"""The three starting paths from the README, runnable.

Run: pytest examples/test_start_here.py -v
"""
import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from pytest_torch import (
    ModelSuite,
    Probabilities,
    TensorSpec,
    assert_checkpoint_loads,
    assert_matches_reference,
    assert_models_agree,
    assert_output_spec,
    check_inference_contract,
    localize_divergence,
)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 3))

    def forward(self, x):
        logits = self.body(x)
        return {"logits": logits, "probs": logits.softmax(-1)}


@pytest.fixture
def model(torch_seed, torch_device):
    return Classifier().to(torch_device).eval()


# --------------------------------------------------------------------------- #
# 1. Check one prediction
# --------------------------------------------------------------------------- #
SPEC = {"logits": TensorSpec(("B", 3)), "probs": Probabilities(("B", 3))}


def test_one_prediction(model, make_tensor):
    x = make_tensor(8, 16)
    check_inference_contract(model, x)  # eval mode, finite, deterministic, batch-invariant, devices
    with torch.inference_mode():
        assert_output_spec(model(x), SPEC, B=8)  # shapes, dtypes, ranges, probabilities


# --------------------------------------------------------------------------- #
# 2. Check a model change
# --------------------------------------------------------------------------- #
class RefactoredClassifier(nn.Module):
    """The same network, rewritten with named layers."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(16, 32)
        self.act = nn.ReLU()
        self.head = nn.Linear(32, 3)

    def forward(self, x):
        logits = self.head(self.act(self.encoder(x)))
        return {"logits": logits, "probs": logits.softmax(-1)}


@pytest.fixture
def refactored(model, torch_device):
    new = RefactoredClassifier().to(torch_device).eval()
    new.encoder.load_state_dict(model.body[0].state_dict())
    new.head.load_state_dict(model.body[2].state_dict())
    return new


def test_refactor_matches_the_original(model, refactored, make_tensor):
    x = make_tensor(8, 16)
    assert_matches_reference(refactored, model, x)  # same outputs, path by path
    # ...and layer by layer: on failure this names the first layer that disagrees.
    localize_divergence(model, refactored, x, module_map={"body.0": "encoder", "body.1": "act",
                                                          "body.2": "head"}).raise_if_diverged()


def test_bf16_model_stays_close(model, make_tensor):
    x = make_tensor(256, 16)
    bf16 = copy.deepcopy(model).to(torch.bfloat16)
    assert_models_agree(model, bf16, x, inputs_candidate=x.to(torch.bfloat16), min_top1=0.95)


# --------------------------------------------------------------------------- #
# 3. Check a release
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def trained():
    torch.manual_seed(0)
    model = Classifier()
    x = torch.randn(512, 16, generator=torch.Generator().manual_seed(0))
    y = x[:, :3].argmax(-1)  # a learnable rule
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(200):
        optimizer.zero_grad()
        F.cross_entropy(model(x)["logits"], y).backward()
        optimizer.step()
    return model.eval()


@pytest.fixture
def val_loader():
    x = torch.randn(256, 16, generator=torch.Generator().manual_seed(1))
    y = x[:, :3].argmax(-1)
    return [(x[i : i + 64], y[i : i + 64]) for i in range(0, 256, 64)]


def accuracy(output, batch):
    return (output["logits"].argmax(-1) == batch[1]).float()  # per sample: paired comparisons


def test_release_accuracy(trained, val_loader, metric_baseline):
    metric_baseline(trained, val_loader, {"accuracy": accuracy}, minimum={"accuracy": 0.9},
                    context={"dataset": "synthetic", "split": "validation"})


def test_last_release_checkpoint_loads(trained, tmp_path):
    path = tmp_path / "classifier_v1.pt"  # in a real project: a file committed with the tests
    torch.save(trained.state_dict(), path)
    assert_checkpoint_loads(Classifier, path)


def test_speed(model, make_tensor, perf_baseline):
    perf_baseline(model, make_tensor(32, 16))  # skipped until recorded with --torch-update-perf


# --------------------------------------------------------------------------- #
# Or generate the tests
# --------------------------------------------------------------------------- #
class TestClassifier(ModelSuite):
    checks = ("contract", "compile", "export", "output_spec", "checkpoint", "training")
    output_spec = {"logits": TensorSpec(("B", 3))}

    def make_model(self):
        return Classifier()

    def make_inputs(self, make_tensor):
        return make_tensor(8, 16)

    def loss(self, output, inputs):  # only needed for "training"
        return output["logits"].logsumexp(-1).mean()
