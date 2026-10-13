import copy
import inspect

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from pytest_torch import (
    assert_aoti_parity,
    assert_autocast_parity,
    assert_compile_parity,
    assert_export_parity,
    assert_matches_reference,
    assert_models_agree,
    assert_onnx_parity,
    assert_torchscript_parity,
    compare_outputs,
    is_compile_supported,
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


compile_unsupported = not is_compile_supported()


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


needs_export = pytest.mark.skipif(
    compile_unsupported or not hasattr(torch, "export"), reason="torch.export not supported here"
)


class KwargsModel(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = nn.Embedding(20, 4)

    def forward(self, input_ids, attention_mask):
        h = self.emb(input_ids) * attention_mask.unsqueeze(-1)
        return {"pooled": h.sum(1), "hidden": h}


class DataDependent(nn.Module):
    def forward(self, x):
        return x * 2 if x.sum().item() > 0 else x


@needs_export
def test_export_parity_static_shapes():
    program = assert_export_parity(make(), torch.randn(4, 8))
    assert hasattr(program, "module")


@needs_export
def test_export_static_shapes_reject_other_batch_sizes():
    with pytest.raises(AssertionError, match="dynamic_batch=True"):
        assert_export_parity(make(), torch.randn(4, 8), torch.randn(6, 8))


@needs_export
def test_export_dynamic_batch_with_kwargs_and_dict_outputs():
    def batch(n):
        g = torch.Generator().manual_seed(n)
        return {
            "input_ids": torch.randint(0, 20, (n, 5), generator=g),
            "attention_mask": torch.ones(n, 5, dtype=torch.long),
        }

    assert_export_parity(KwargsModel().eval(), batch(4), batch(7), dynamic_batch=True)


@needs_export
def test_export_failure_is_an_assertion_with_a_hint():
    with pytest.raises(AssertionError, match=r"(?s)torch.export failed.*Hint"):
        assert_export_parity(DataDependent(), torch.randn(3, 2), strict=True)


def test_dynamic_batch_needs_batch_of_two():
    with pytest.raises(ValueError, match="batch size >= 2"):
        assert_export_parity(make(), torch.randn(1, 8), dynamic_batch=True)


@needs_export
@pytest.mark.slow
def test_aoti_parity_with_dynamic_batch():
    # Compiles C++ (about 20 s); skips without torch >= 2.6 or a compiler.
    assert_aoti_parity(make(), torch.randn(4, 8), torch.randn(9, 8), dynamic_batch=True)


def test_autocast_parity_bf16():
    assert_autocast_parity(make(), torch.randn(16, 8))


def test_autocast_drift_is_reported():
    class LargeOffset(nn.Module):
        """Adds and removes a large offset: exact in fp32, lossy in bf16 (8-bit mantissa)."""

        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x):
            return (self.lin(x) + 1000.0) - 1000.0

    with pytest.raises(AssertionError, match=r"autocast\(cpu, torch.bfloat16\)"):
        assert_autocast_parity(LargeOffset().eval(), torch.randn(16, 8))


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
