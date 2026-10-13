"""Regression cases for comparison edge cases that once slipped through."""
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from pytest_torch import (
    assert_outputs_close,
    assert_tensors_close,
    localize_divergence,
)


# --------------------------------------------------------------------------- #
# Complex tensors: the imaginary part counts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_complex_differing_only_in_imaginary_part(dtype):
    a = torch.tensor([1 + 1j, 2 + 2j], dtype=dtype)
    b = torch.tensor([1 + 1j, 2 + 5j], dtype=dtype)
    with pytest.raises(AssertionError, match=r"(?s)1/2 elements differ.*index \(1,\).*actual=\(2\+2j\), expected=\(2\+5j\)"):
        assert_tensors_close(a, b)


def test_complex_within_tolerance_and_conjugate_views():
    a = torch.tensor([1 + 1j, 2 - 3j])
    assert_tensors_close(a, a + 1e-7j)
    assert_tensors_close(a.conj(), torch.tensor([1 - 1j, 2 + 3j]))  # lazy conj view is resolved
    with pytest.raises(AssertionError):
        assert_tensors_close(a.conj(), a)


def test_complex_nan_in_either_part():
    a = torch.tensor([complex(1, float("nan"))])
    assert_tensors_close(a, a.clone())
    with pytest.raises(AssertionError, match="NaN"):
        assert_tensors_close(a, torch.tensor([1 + 1j]))


# --------------------------------------------------------------------------- #
# Integers: compared exactly, even beyond float64 precision
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("big", [2**53, 2**62, -(2**60)])
def test_adjacent_integers_above_2_to_the_53(big):
    a = torch.tensor([7, big], dtype=torch.int64)
    b = torch.tensor([7, big + 1], dtype=torch.int64)
    with pytest.raises(AssertionError, match=rf"(?s)1/2 elements differ.*actual={big}, expected={big + 1}"):
        assert_tensors_close(a, b)
    assert_tensors_close(a, a.clone())


def test_integers_are_exact_by_default_but_honor_explicit_tolerance():
    a = torch.tensor([10_000_000])
    b = torch.tensor([10_000_010])  # within the float32 default rtol (1.3e-6 * 1e7 = 13)
    with pytest.raises(AssertionError, match="exact"):
        assert_tensors_close(a, b)
    assert_tensors_close(a, b, atol=10, rtol=0)  # an explicit tolerance still applies


def test_bool_tensors():
    assert_tensors_close(torch.tensor([True, False]), torch.tensor([True, False]))
    with pytest.raises(AssertionError, match="1/2 elements differ"):
        assert_tensors_close(torch.tensor([True, False]), torch.tensor([True, True]))


# --------------------------------------------------------------------------- #
# NaN: an explicit equal_nan policy
# --------------------------------------------------------------------------- #
NAN = float("nan")


def test_matching_nans_are_equal_by_default():
    t = torch.tensor([1.0, NAN, 3.0])
    assert_tensors_close(t, t.clone())
    assert_outputs_close({"x": t, "s": NAN}, {"x": t.clone(), "s": NAN})


def test_equal_nan_false_rejects_matching_nans():
    t = torch.tensor([1.0, NAN, 3.0])
    with pytest.raises(AssertionError, match=r"1 NaN in actual and 1 in expected \(equal_nan=False\)"):
        assert_tensors_close(t, t.clone(), equal_nan=False)
    with pytest.raises(AssertionError, match="equal_nan=False"):
        assert_outputs_close({"s": NAN}, {"s": NAN}, equal_nan=False)
    assert_tensors_close(torch.ones(2), torch.ones(2), equal_nan=False)  # no NaN: no effect


def test_nan_positions_must_match_whatever_the_policy():
    a, b = torch.tensor([NAN, 1.0]), torch.tensor([1.0, NAN])
    for equal_nan in (True, False):
        with pytest.raises(AssertionError, match="NaN"):
            assert_tensors_close(a, b, equal_nan=equal_nan)


# --------------------------------------------------------------------------- #
# Output structure: empty containers and changed container kinds
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("actual", "expected", "message"),
    [
        ([], {}, r"output: empty sequence vs empty mapping"),
        ({"a": []}, {"a": {}}, r"output\['a'\]: empty sequence vs empty mapping"),
        ({"a": {}}, {}, r"output\['a'\]: only in actual \(empty mapping\)"),
        ({}, {"a": []}, r"output\['a'\]: only in expected \(empty sequence\)"),
        ([torch.ones(1)], [torch.ones(1), []], r"output\[1\]: only in expected \(empty sequence\)"),
        ({"a": torch.ones(1)}, {"a": [torch.ones(1)]}, r"output\['a'\]: tensor vs sequence"),
        ({"a": [torch.ones(1)]}, {"a": {"0": torch.ones(1)}}, r"output\['a'\]: sequence vs mapping"),
    ],
)
def test_structure_changes_are_reported(actual, expected, message):
    with pytest.raises(AssertionError, match=message):
        assert_outputs_close(actual, expected)


def test_same_structure_passes_and_tuple_vs_list_is_not_a_change():
    out = {"logits": torch.ones(2), "aux": (torch.zeros(1), []), "extra": {}}
    assert_outputs_close(out, {"logits": torch.ones(2), "aux": [torch.zeros(1), []], "extra": {}})
    assert_outputs_close([], [])
    assert_outputs_close({}, {})


def test_structure_and_value_problems_are_reported_together():
    with pytest.raises(AssertionError) as info:
        assert_outputs_close({"a": torch.ones(2), "b": []}, {"a": torch.zeros(2), "b": {}})
    message = str(info.value)
    assert "output['b']: empty sequence vs empty mapping" in message
    assert "output['a']: 2/2 elements differ" in message


# --------------------------------------------------------------------------- #
# Layer-by-layer divergence: in-place ops must not rewrite captured activations
# --------------------------------------------------------------------------- #
class InPlaceAfterLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        h = self.lin(x)
        h.mul_(2)  # mutates `lin`'s output after its forward hook captured it
        return self.head(h)


class OutOfPlaceAfterLayer(InPlaceAfterLayer):
    def forward(self, x):
        return self.head(self.lin(x) * 2)


def test_in_place_op_after_a_layer_is_not_a_false_divergence():
    torch.manual_seed(0)
    a = InPlaceAfterLayer().eval()
    b = OutOfPlaceAfterLayer().eval()
    b.load_state_dict(a.state_dict())
    report = localize_divergence(a, b, torch.randn(3, 4))
    assert not report.diverged, report.format()


def test_in_place_module_does_not_hide_a_divergence_before_it():
    # Both linears output only negative values, differing by 100; the in-place ReLU then
    # zeroes both. Aliased captures would record the zeros and see no divergence at all.
    torch.manual_seed(0)
    a = nn.Sequential(nn.Linear(4, 4), nn.ReLU(inplace=True)).eval()
    b = nn.Sequential(nn.Linear(4, 4), nn.ReLU(inplace=True)).eval()
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        a[0].bias.fill_(-100.0)
        b[0].bias.fill_(-200.0)
    report = localize_divergence(a, b, torch.rand(3, 4))
    first, relu = report.layers
    assert first.diverged and not first.inputs_diverged, report.format()  # starts inside '0'
    assert relu.inputs_diverged and not relu.diverged, report.format()  # ReLU saw the real inputs
    assert report.first.name == "0"
