import copy
import inspect

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from pytest_torch import (
    assert_compile_parity,
    assert_matches_reference,
    assert_models_agree,
    assert_onnx_parity,
    assert_torchscript_parity,
    compare_outputs,
)


def make():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).eval()


class TwoHead(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.a = nn.Linear(4, 3)
        self.b = nn.Linear(4, 2)

    def forward(self, x):
        return {"logits": self.a(x), "aux": self.b(x)}


class Noisy(nn.Module):
    def forward(self, x):
        return x + torch.rand_like(x)


compile_unsupported = not hasattr(torch, "compile") or not getattr(
    getattr(torch, "compiler", None), "is_dynamo_supported", lambda: False
)()


# --------------------------------------------------------------------------- #
# deployment targets
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(compile_unsupported, reason="torch.compile (dynamo) not supported here")
def test_compile_parity_passes_with_eager_backend():
    assert_compile_parity(make(), torch.randn(4, 8), backend="eager")


@pytest.mark.skipif(compile_unsupported, reason="torch.compile (dynamo) not supported here")
def test_compile_parity_detects_differences():
    with pytest.raises(AssertionError, match="torch.compile"):
        assert_compile_parity(Noisy(), torch.randn(4, 3), backend="eager")


def test_torchscript_trace_parity_with_dict_outputs():
    assert_torchscript_parity(TwoHead().eval(), torch.randn(4, 4))


def test_torchscript_rejects_kwargs_inputs():
    with pytest.raises(ValueError, match="positional"):
        assert_torchscript_parity(make(), {"x": torch.randn(4, 8)})


def test_onnx_parity():
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    legacy = "dynamo" in inspect.signature(torch.onnx.export).parameters
    assert_onnx_parity(
        make(), torch.randn(4, 8), export_kwargs={"dynamo": False} if legacy else None
    )


# --------------------------------------------------------------------------- #
# accuracy budgets
# --------------------------------------------------------------------------- #
def test_identical_models_agree():
    model = make()
    metrics = assert_models_agree(
        model,
        copy.deepcopy(model),
        torch.randn(16, 8),
        min_top1=1.0,
        min_cosine=0.999999,
        max_abs=1e-9,
    )
    assert metrics.top1_agreement == 1.0


def test_perturbed_model_exceeds_budget():
    model = make()
    broken = copy.deepcopy(model)
    with torch.no_grad():
        broken[2].weight.mul_(-1.0)
    with pytest.raises(AssertionError, match="accuracy budget"):
        assert_models_agree(model, broken, torch.randn(16, 8), max_abs=1e-3)


def test_at_least_one_budget_is_required():
    model = make()
    with pytest.raises(ValueError, match="at least one budget"):
        assert_models_agree(model, model, torch.randn(4, 8))


def test_select_output_by_path_and_callable():
    model = TwoHead().eval()
    x = torch.randn(5, 4)
    with torch.no_grad():
        out = model(x)
    by_path = compare_outputs(out, out, select="output['aux']")
    by_callable = compare_outputs(out, out, select=lambda o: o["logits"])
    assert by_path.max_abs_error == 0.0 and by_callable.max_abs_error == 0.0
    with pytest.raises(ValueError, match="unknown output path"):
        compare_outputs(out, out, select="output['nope']")


# --------------------------------------------------------------------------- #
# reference implementations
# --------------------------------------------------------------------------- #
def test_matches_callable_and_precomputed_reference():
    model = make()
    x = torch.randn(4, 8)
    assert_matches_reference(model, copy.deepcopy(model), x)
    with torch.no_grad():
        expected = model(x)
    assert_matches_reference(model, expected, x)


def test_mismatch_with_reference_is_reported():
    model = make()
    x = torch.randn(4, 8)
    with torch.no_grad():
        expected = model(x) + 1.0
    with pytest.raises(AssertionError, match="differs from the reference"):
        assert_matches_reference(model, expected, x)
