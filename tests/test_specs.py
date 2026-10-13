import dataclasses

import pytest

torch = pytest.importorskip("torch")

from pytest_torch import Probabilities, TensorSpec, assert_output_spec


@pytest.fixture
def output():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(8, 10, generator=g)
    return {
        "logits": logits,
        "probs": logits.softmax(-1),
        "log_probs": logits.log_softmax(-1),
        "hidden": torch.randn(8, 5, 64, generator=g),
        "aux": [torch.zeros(8), torch.ones(8, dtype=torch.long)],
    }


SPEC = {
    "logits": TensorSpec(("B", 10), dtype=torch.float32),
    "probs": Probabilities(("B", 10)),
    "log_probs": Probabilities(log=True),
    "hidden": TensorSpec(("B", "T", 64)),
    "aux": [TensorSpec(("B",), min=0, max=0), TensorSpec(("B",), dtype=(torch.int32, torch.int64))],
}


def test_matching_output_returns_bound_dims(output):
    assert assert_output_spec(output, SPEC) == {"B": 8, "T": 5}
    assert assert_output_spec(output, SPEC, B=8, T=5) == {"B": 8, "T": 5}


def test_every_problem_is_reported_with_its_path(output):
    output["logits"] = output["logits"].double()[:, :9]
    output["probs"] = output["probs"] * 2
    output["hidden"] = torch.full((4, 5, 64), float("nan"))
    del output["log_probs"]
    with pytest.raises(AssertionError) as info:
        assert_output_spec(output, SPEC, B=8)
    message = str(info.value)
    for expected in [
        "output['logits']: shape (8, 9) vs spec ('B', 10): dim 1 is 9, expected 10",
        "output['logits']: dtype torch.float64, expected torch.float32",
        "output['probs']: probabilities outside [0, 1]",
        "output['probs']: sum along dim -1 is not 1",
        "output['hidden']: shape (4, 5, 64) vs spec ('B', 'T', 64): dim 0: B=4, but B=8 from argument B=8",
        "output['hidden']: 1280 NaN and 0 Inf values",
        "output: missing key 'log_probs'",
    ]:
        assert expected in message, message


def test_symbol_conflicts_name_where_the_symbol_was_bound(output):
    output["hidden"] = torch.zeros(7, 5, 64)
    with pytest.raises(AssertionError, match=r"B=7, but B=8 from output\['logits'\] dim 0"):
        assert_output_spec(output, SPEC)


def test_ellipsis_none_and_skipped_entries():
    out = {"x": torch.zeros(2, 3, 4, 5), "y": torch.zeros(3), "z": "ignored"}
    assert_output_spec(out, {"x": TensorSpec((..., "C")), "y": TensorSpec((None,)), "w": None}, C=5)
    with pytest.raises(AssertionError, match="fewer than 2 dims"):
        assert_output_spec(torch.zeros(3), TensorSpec(("A", ..., "C")))
    with pytest.raises(ValueError, match="at most one"):
        assert_output_spec(torch.zeros(3), TensorSpec((..., ...)))


def test_ranges_and_finiteness():
    t = torch.tensor([0.0, 0.5, float("inf")])
    with pytest.raises(AssertionError, match="0 NaN and 1 Inf"):
        assert_output_spec(t, TensorSpec())
    assert_output_spec(t, TensorSpec(finite=False))
    with pytest.raises(AssertionError, match=r"above max 0\.4 \(largest 0\.5\)"):
        assert_output_spec(t[:2], TensorSpec(max=0.4))


def test_probabilities_along_another_dim():
    probs = torch.rand(3, 4).softmax(0)
    assert_output_spec(probs, Probabilities(dim=0))
    with pytest.raises(AssertionError, match="sum along dim -1"):
        assert_output_spec(probs, Probabilities())


def test_dataclass_and_tuple_outputs():
    @dataclasses.dataclass
    class Out:
        logits: torch.Tensor
        extra: tuple

    out = Out(torch.zeros(2, 3), (torch.zeros(2),))
    assert_output_spec(out, {"logits": TensorSpec(("B", 3)), "extra": [TensorSpec(("B",))]})
    with pytest.raises(AssertionError, match="Out has no attribute 'missing'"):
        assert_output_spec(out, {"missing": TensorSpec()})
    with pytest.raises(AssertionError, match="expected 2 items, got 1"):
        assert_output_spec(out, {"extra": [TensorSpec(), TensorSpec()]})


def test_non_tensor_where_a_tensor_is_expected():
    with pytest.raises(AssertionError, match="expected a tensor, got str"):
        assert_output_spec({"x": "nope"}, {"x": TensorSpec()})
