import copy

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from pytest_torch import localize_divergence


def make():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).eval()


def test_identical_models_do_not_diverge():
    a = make()
    report = localize_divergence(a, copy.deepcopy(a), torch.randn(4, 8))
    assert report.first is None
    assert "No divergence" in report.format()
    report.raise_if_diverged()  # no-op


def test_first_diverging_layer_is_reported():
    a = make()
    b = copy.deepcopy(a)
    with torch.no_grad():
        b[2].weight.add_(0.5)
    report = localize_divergence(a, b, torch.randn(4, 8))
    assert report.first is not None and report.first.name == "2"
    assert not report.first.inputs_diverged
    assert [layer.name for layer in report.layers if not layer.diverged] == ["0", "1"]
    assert "originates inside" in report.format()
    with pytest.raises(AssertionError, match="First: '2'"):
        report.raise_if_diverged()


def test_module_map_for_renamed_layers():
    a = make()
    b = nn.ModuleDict({"fc1": copy.deepcopy(a[0]), "act": copy.deepcopy(a[1]),
                       "fc2": copy.deepcopy(a[2])})

    class Wrapper(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = layers

        def forward(self, x):
            return self.layers["fc2"](self.layers["act"](self.layers["fc1"](x)))

    report = localize_divergence(
        a,
        Wrapper(b),
        torch.randn(4, 8),
        module_map={"0": "layers.fc1", "1": "layers.act", "2": "layers.fc2"},
    )
    assert report.first is None
