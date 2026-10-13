"""Example usage of pytest-torch. Run: pytest examples/ -v"""
import pytest
import torch
from torch import nn

from pytest_torch import assert_batch_invariant, assert_finite


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Linear(32, 4)
        )

    def forward(self, x):
        return self.net(x)


@pytest.fixture
def model(torch_seed, torch_device):
    return TinyClassifier().to(torch_device).eval()


def test_output_shape_and_finite(model, torch_device, inference_mode):
    y = model(torch.randn(8, 16, device=torch_device))
    assert y.shape == (8, 4)
    assert_finite(y, "logits")


def test_batch_invariance(model, torch_device):
    assert_batch_invariant(model, torch.randn(8, 16, device=torch_device))


@pytest.mark.torch_golden("tiny_classifier_v1")
def test_matches_golden(model, torch_device, inference_mode, golden):
    x = torch.randn(8, 16, device=torch_device)
    golden.check(model(x), atol=1e-4)


@pytest.mark.torch_max_latency(ms=50)
def test_latency(model, torch_device, latency):
    latency(model, torch.randn(32, 16, device=torch_device))
