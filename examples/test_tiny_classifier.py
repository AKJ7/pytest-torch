"""Example usage of pytest-torch. Run: pytest examples/ -v"""
import pytest
import torch
from torch import nn

from pytest_torch import (
    assert_batch_invariant,
    assert_finite,
    assert_outputs_close,
    check_inference_contract,
    localize_divergence,
)


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Linear(32, 4)
        )

    def forward(self, x):
        return {"logits": self.net(x), "probs": self.net(x).softmax(-1)}


@pytest.fixture
def model(torch_seed, torch_device):
    return TinyClassifier().to(torch_device).eval()


def test_inference_contract(model, torch_device):
    # One call: eval mode, finite outputs, determinism, batch invariance,
    # cross-device parity and no state mutation. Raises with hints on failure.
    check_inference_contract(model, torch.randn(8, 16, device=torch_device))


def test_output_structure(model, torch_device, inference_mode):
    out = model(torch.randn(8, 16, device=torch_device))
    assert out["logits"].shape == (8, 4)
    assert_finite(out, "output")


def test_batch_invariance(model, torch_device):
    assert_batch_invariant(model, torch.randn(8, 16, device=torch_device))


@pytest.mark.torch_golden("tiny_classifier_v1")
def test_matches_golden(model, torch_device, inference_mode, golden):
    golden.check(model(torch.randn(8, 16, device=torch_device)), atol=1e-4)


def test_cpu_gpu_agree_layer_by_layer(model):
    import copy

    cpu_model = copy.deepcopy(model).cpu()
    report = localize_divergence(cpu_model, model, torch.randn(8, 16))
    report.raise_if_diverged()  # message names the first diverging layer


@pytest.mark.torch_max_latency(ms=50)
def test_latency(model, torch_device, latency):
    latency(model, torch.randn(32, 16, device=torch_device))
