"""Metamorphic tests: relations that must hold without knowing the right answer.

Run: pytest examples/test_metamorphic.py -v
"""
import pytest
import torch
from torch import nn

from pytest_torch import (
    assert_equivariant,
    assert_invariant,
    assert_permutation_equivariant,
    assert_permutation_invariant,
)


class DeepSets(nn.Module):
    """Point-cloud classifier: encode each point, sum-pool, classify the set."""

    def __init__(self):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 32))
        self.rho = nn.Linear(32, 5)

    def forward(self, points):  # (batch, n_points, 3)
        # Try `self.phi(points).flatten(1)[:, :32]`: the permutation test below fails.
        return self.rho(self.phi(points).sum(dim=1))


class TokenTagger(nn.Module):
    """Per-token tagger without positional information: tags follow their tokens."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(100, 16)
        self.head = nn.Linear(16, 7)

    def forward(self, input_ids):
        return self.head(self.emb(input_ids))


class GlobalPoolCNN(nn.Module):
    """1x1 convolutions + spatial mean: flipping or rotating the image cannot matter.
    (With kernel_size=3 the flip test below fails: learned kernels are not symmetric.)"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=1)
        self.head = nn.Linear(8, 2)

    def forward(self, images):
        return self.head(self.conv(images).relu().mean(dim=(2, 3)))


@pytest.fixture
def points(make_tensor):
    return make_tensor(4, 10, 3)


def test_point_cloud_classifier_ignores_point_order(torch_seed, torch_device, points):
    model = DeepSets().to(torch_device).eval()
    assert_permutation_invariant(model, points, dim=1)


def test_point_cloud_classifier_ignores_translation_after_centering(torch_seed, torch_device, points):
    # Wrap the model with its preprocessing to state the property of the whole pipeline.
    model = DeepSets().to(torch_device).eval()

    def centered(x):
        return model(x - x.mean(dim=1, keepdim=True))

    assert_invariant(centered, points, lambda x: x + 5.0, name="translation", atol=1e-4)


def test_tags_move_with_their_tokens(torch_seed, torch_device, make_tensor):
    model = TokenTagger().to(torch_device).eval()
    assert_permutation_equivariant(model, make_tensor(2, 9, dtype=torch.long, low=0, high=100))


def test_spatial_pooling_ignores_pixel_layout(torch_seed, torch_device, make_tensor):
    model = GlobalPoolCNN().to(torch_device).eval()
    images = make_tensor(2, 3, 8, 8)
    assert_invariant(model, images, lambda x: x.flip(-1), name="horizontal flip")
    assert_invariant(model, images, lambda x: x.rot90(1, dims=(2, 3)), name="90-degree rotation")


def test_bias_free_relu_network_is_positively_homogeneous(torch_seed, torch_device, make_tensor):
    # f(a * x) == a * f(x) for a > 0: a cheap check that catches a stray bias or normalization.
    model = nn.Sequential(
        nn.Linear(16, 32, bias=False), nn.ReLU(), nn.Linear(32, 4, bias=False)
    ).to(torch_device).eval()
    assert_equivariant(
        model, make_tensor(8, 16), lambda x: 3.0 * x, lambda y: 3.0 * y, name="scaling by 3"
    )
