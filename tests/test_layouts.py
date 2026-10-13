import pytest

torch = pytest.importorskip("torch")
from torch import nn

from pytest_torch import assert_layout_invariant
from pytest_torch.invariance import noncontiguous


@pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4), (4, 1), (2, 3, 8, 8)])
def test_noncontiguous_keeps_values(shape):
    x = torch.randn(*shape)
    y = noncontiguous(x)
    assert not y.is_contiguous()
    assert torch.equal(x, y)


def test_conv_net_is_layout_invariant():
    model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(1)).eval()
    assert_layout_invariant(model, torch.randn(2, 3, 8, 8))


class ViewBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(12, 2)

    def forward(self, x):
        return self.lin(x.view(x.shape[0], -1))


def test_view_on_noncontiguous_input_is_caught():
    with pytest.raises(AssertionError, match=r"(?s)fails on noncontiguous inputs.*use \.reshape\(\)"):
        assert_layout_invariant(ViewBug().eval(), torch.randn(2, 3, 4))


class LayoutDependent(nn.Module):
    def forward(self, x):
        return x.contiguous().reshape(-1)[: x.numel() // 2] if x.is_contiguous() else x.reshape(-1)[:1]


def test_layout_dependent_output_is_caught():
    with pytest.raises(AssertionError, match="Output changes with noncontiguous inputs"):
        assert_layout_invariant(LayoutDependent(), torch.randn(2, 4))


def test_dict_inputs_and_channels_last_skipped_without_4d():
    class Kw(nn.Module):
        def forward(self, input_ids, attention_mask):
            return (input_ids * attention_mask).sum(-1)

    ids = torch.randint(0, 9, (3, 5))
    assert_layout_invariant(Kw(), {"input_ids": ids, "attention_mask": torch.ones_like(ids)})


def test_unknown_layout():
    with pytest.raises(ValueError, match="unknown layout"):
        assert_layout_invariant(nn.Identity(), torch.randn(2, 3), layouts=("sparse",))
