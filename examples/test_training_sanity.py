"""Training sanity checks: catch a broken training setup in seconds, not after a day of GPU time.

Run: pytest examples/test_training_sanity.py -v
"""
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from pytest_torch import (
    assert_all_params_receive_grad,
    assert_gradcheck,
    assert_gradients_finite,
    assert_loss_decreases,
    assert_overfits_one_batch,
)


class TwoTower(nn.Module):
    """Query/document encoders scored by cosine similarity."""

    def __init__(self):
        super().__init__()
        self.query = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
        self.doc = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
        self.temperature = nn.Parameter(torch.tensor(10.0))

    def forward(self, q, d):
        # Try `self.query(d)` for the documents: test_every_parameter_learns names `doc.*`.
        sim = F.normalize(self.query(q), dim=-1) @ F.normalize(self.doc(d), dim=-1).T
        return sim * self.temperature


@pytest.fixture
def model(torch_seed, torch_device):
    return TwoTower().to(torch_device)


@pytest.fixture
def batch(make_tensor):
    return make_tensor(8, 16), make_tensor(8, 16)


def contrastive(scores):
    # Matching pairs sit on the diagonal.
    return F.cross_entropy(scores, torch.arange(scores.shape[0], device=scores.device))


def test_gradients_are_finite(model, batch):
    assert_gradients_finite(model, batch, contrastive)


def test_every_parameter_learns(model, batch):
    assert_all_params_receive_grad(model, batch, contrastive)


def test_loss_goes_down(model, batch):
    # Try `lr=0.0` (a classic scheduler or config bug): the test fails and shows the loss curve.
    assert_loss_decreases(model, batch, contrastive, lr=1e-2)


def test_can_memorize_one_batch(model, batch):
    run = assert_overfits_one_batch(model, batch, contrastive, steps=300)
    print(run.summary())  # with `pytest -s`: the loss curve


class ClampedSquare(torch.autograd.Function):
    """x**2 clamped at 4, with a hand-written backward."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x * x).clamp(max=4.0)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        # Try dropping the mask: gradients leak where the output is clamped.
        return 2 * x * grad * (x * x < 4.0)


def test_custom_backward_matches_finite_differences():
    x = torch.tensor([[0.5, -1.0, 3.0], [1.5, -2.5, 0.1]])  # some values clamp, some don't
    assert_gradcheck(ClampedSquare.apply, x)
