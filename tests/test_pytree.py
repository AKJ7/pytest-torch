from collections import OrderedDict, namedtuple
from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from pytest_torch import assert_finite, assert_outputs_close
from pytest_torch.utils import flatten, tree_map

Out = namedtuple("Out", ["logits", "hidden"])


@dataclass
class Result:
    scores: torch.Tensor
    label: str


def make_output():
    return {
        "a": torch.zeros(2),
        "b": [torch.ones(2), Out(torch.zeros(1), 3)],
        "res": Result(torch.arange(3.0), "cat"),
    }


def test_flatten_paths():
    paths = [p for p, _ in flatten(make_output())]
    assert paths == [
        "output['a']",
        "output['b'][0]",
        "output['b'][1].logits",
        "output['b'][1].hidden",
        "output['res'].scores",
        "output['res'].label",
    ]


def test_tree_map_preserves_structure():
    out = tree_map(lambda t: t + 1, make_output())
    assert isinstance(out["b"][1], Out)
    assert isinstance(out["res"], Result)
    assert torch.equal(out["a"], torch.ones(2))
    assert out["res"].label == "cat"


def test_nested_outputs_close():
    assert_outputs_close(make_output(), make_output())


def test_mismatch_reports_path():
    actual = make_output()
    actual["b"][0] = actual["b"][0] + 1.0
    with pytest.raises(AssertionError, match=r"output\['b'\]\[0\]"):
        assert_outputs_close(actual, make_output())


def test_non_tensor_leaf_mismatch_reported():
    actual = make_output()
    actual["res"] = Result(torch.arange(3.0), "dog")
    with pytest.raises(AssertionError, match=r"output\['res'\]\.label"):
        assert_outputs_close(actual, make_output())


def test_missing_and_extra_keys():
    with pytest.raises(AssertionError, match="missing in actual"):
        assert_outputs_close({"a": torch.zeros(1)}, {"a": torch.zeros(1), "b": torch.zeros(1)})
    with pytest.raises(AssertionError, match="unexpected in actual"):
        assert_outputs_close({"a": torch.zeros(1), "b": torch.zeros(1)}, {"a": torch.zeros(1)})


def test_ordered_dict_like_model_output():
    out = OrderedDict(logits=torch.zeros(2), hidden_states=(torch.ones(2),))
    assert_outputs_close(out, OrderedDict(out))


def test_assert_finite_reports_path():
    bad = {"ok": torch.zeros(2), "bad": [torch.tensor([1.0, float("inf")])]}
    with pytest.raises(AssertionError, match=r"output\['bad'\]\[0\] has 0 NaN and 1 Inf"):
        assert_finite(bad, "output")
