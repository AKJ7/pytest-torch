import pytest

torch = pytest.importorskip("torch")
from torch import nn

from pytest_torch import (
    assert_equivariant,
    assert_invariant,
    assert_mask_invariant,
    assert_padding_invariant,
    assert_permutation_equivariant,
    assert_permutation_invariant,
)


class Encoder(nn.Module):
    """Per-token projection + pooled vector; ``respect_mask=False`` is the classic bug."""

    def __init__(self, respect_mask=True):
        super().__init__()
        torch.manual_seed(0)
        self.emb = nn.Embedding(50, 8)
        self.proj = nn.Linear(8, 8)
        self.respect_mask = respect_mask

    def forward(self, input_ids, attention_mask):
        h = self.proj(self.emb(input_ids))
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        if self.respect_mask:
            pooled = (h * m).sum(1) / m.sum(1).clamp_min(1)
        else:
            pooled = h.mean(1)  # padding leaks into the pooled vector
        return {"hidden": h, "pooled": pooled}


class DeepSets(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.phi = nn.Linear(4, 8)
        self.rho = nn.Linear(8, 2)

    def forward(self, x):
        return self.rho(self.phi(x).sum(dim=1))


class OrderDependent(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.phi = nn.Linear(4, 8)

    def forward(self, x):
        w = torch.arange(1, x.shape[1] + 1, dtype=x.dtype, device=x.device).view(1, -1, 1)
        return (self.phi(x) * w).sum(dim=1)


class Cumulative(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(4, 8)

    def forward(self, x):
        return torch.cumsum(self.lin(x), dim=1)


@pytest.fixture
def batch():
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(1, 50, (3, 6), generator=g)
    mask = torch.ones_like(ids)
    mask[1, 4:] = 0
    mask[2, 3:] = 0
    return {"input_ids": ids, "attention_mask": mask}


# --------------------------------------------------------------------------- #
# padding / masks
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("side", ["right", "left"])
def test_padding_invariant_model_passes(batch, side):
    assert_padding_invariant(Encoder().eval(), batch, side=side)


def test_padding_leak_is_detected(batch):
    with pytest.raises(AssertionError, match="Padding invariance violated"):
        assert_padding_invariant(Encoder(respect_mask=False).eval(), batch)


def test_mask_invariant_model_passes(batch):
    assert_mask_invariant(Encoder().eval(), batch)


def test_mask_leak_is_detected(batch):
    with pytest.raises(AssertionError, match="Mask invariance violated"):
        assert_mask_invariant(Encoder(respect_mask=False).eval(), batch)


def test_mask_check_needs_actual_padding(batch):
    batch["attention_mask"] = torch.ones_like(batch["attention_mask"])
    with pytest.raises(ValueError, match="no padding"):
        assert_mask_invariant(Encoder().eval(), batch)


def test_padding_check_needs_dict_inputs():
    with pytest.raises(ValueError, match="dict"):
        assert_padding_invariant(Encoder().eval(), torch.zeros(2, 3, dtype=torch.long))


# --------------------------------------------------------------------------- #
# permutations and generic metamorphic relations
# --------------------------------------------------------------------------- #
def test_set_model_is_permutation_invariant():
    assert_permutation_invariant(DeepSets().eval(), torch.randn(3, 5, 4))


def test_order_dependence_is_detected():
    with pytest.raises(AssertionError, match="not invariant"):
        assert_permutation_invariant(OrderDependent().eval(), torch.randn(3, 5, 4))


def test_per_token_model_is_permutation_equivariant():
    assert_permutation_equivariant(nn.Linear(4, 8).eval(), torch.randn(3, 5, 4))


def test_cumulative_model_is_not_equivariant():
    with pytest.raises(AssertionError, match="not equivariant"):
        assert_permutation_equivariant(Cumulative().eval(), torch.randn(3, 5, 4))


def test_encoder_is_equivariant_with_invariant_pooling(batch):
    assert_permutation_equivariant(Encoder().eval(), batch)


def test_generic_invariance_and_equivariance():
    x = torch.randn(4, 3)

    class Abs(nn.Module):
        def forward(self, t):
            return t.abs()

    assert_invariant(Abs(), x, lambda t: -t, name="sign flip")
    with pytest.raises(AssertionError, match="sign flip"):
        assert_invariant(nn.Identity(), x, lambda t: -t, name="sign flip")
    assert_equivariant(nn.Identity(), x, lambda t: t * 2, lambda out: out * 2)
