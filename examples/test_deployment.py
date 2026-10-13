"""Does the model you ship behave like the model you tested?

Covers torch.compile, torch.export, AOTInductor, TorchScript, ONNX, a pruned model,
bf16 and autocast.
Run: pytest examples/test_deployment.py -v   (ONNX needs: pip install "pytest-torch[onnx]")
"""
import copy

import pytest
import torch
from torch import nn

from pytest_torch import (
    assert_aoti_parity,
    assert_autocast_parity,
    assert_compile_parity,
    assert_export_parity,
    assert_models_agree,
    assert_onnx_parity,
    assert_torchscript_parity,
    check_inference_contract,
    is_compile_supported,
)

needs_dynamo = pytest.mark.skipif(not is_compile_supported(), reason="torch.compile/export not supported here")


class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(32, 128), nn.GELU(), nn.LayerNorm(128), nn.Linear(128, 10))

    def forward(self, x):
        return self.body(x)


@pytest.fixture
def model(torch_seed, torch_device):
    return MLPClassifier().to(torch_device).eval()


@needs_dynamo
def test_compiled_matches_eager(model, make_tensor):
    # backend="eager" checks dynamo tracing without needing a C++ toolchain;
    # use the default ("inductor") on machines that have one.
    assert_compile_parity(model, make_tensor(16, 32), backend="eager")


@needs_dynamo
def test_exported_program_matches_eager_for_any_batch_size(model, make_tensor):
    # One exported program, several batch sizes: dim 0 is exported as dynamic.
    assert_export_parity(model, make_tensor(16, 32), make_tensor(3, 32), dynamic_batch=True)


@needs_dynamo
@pytest.mark.slow
def test_aot_compiled_package_matches_eager(model, make_tensor):
    # Compiles C++ ahead of time (~20 s): the artifact a C++ server or edge device would load.
    assert_aoti_parity(model, make_tensor(16, 32), make_tensor(3, 32), dynamic_batch=True)


@pytest.mark.parametrize("mode", ["trace", "script"])
def test_torchscript_matches_eager(model, make_tensor, mode):
    assert_torchscript_parity(model, make_tensor(16, 32), mode=mode)


def test_onnx_matches_eager(model, make_tensor):
    pytest.importorskip("onnxruntime")
    # Pin the exporter: its defaults differ between torch versions.
    assert_onnx_parity(model.cpu(), make_tensor(16, 32).cpu(), export_kwargs={"dynamo": False, "opset_version": 17})


def magnitude_prune(model, sparsity):
    """Zero the smallest `sparsity` fraction of every Linear weight (a compressed model)."""
    pruned = copy.deepcopy(model)
    with torch.no_grad():
        for module in pruned.modules():
            if isinstance(module, nn.Linear):
                w = module.weight
                k = int(sparsity * w.numel())
                threshold = w.abs().flatten().kthvalue(k).values
                w.masked_fill_(w.abs() <= threshold, 0.0)
    return pruned


def test_pruned_model_stays_within_budget(model, make_tensor):
    # Try sparsity=0.9: top-1 agreement drops below the budget and the test says by how much.
    pruned = magnitude_prune(model, sparsity=0.1)
    metrics = assert_models_agree(model, pruned, make_tensor(256, 32), min_top1=0.9, min_cosine=0.99)
    print(metrics)  # shown with `pytest -s`: log it to track compression drift over time


def test_mixed_precision_stays_close(model, make_tensor):
    # How mixed precision really runs: fp32 weights, bf16 matmuls under torch.autocast.
    assert_autocast_parity(model, make_tensor(16, 32), dtype=torch.bfloat16)


def test_bf16_agrees_with_fp32(model, make_tensor):
    # Only the precision check: fp32 vs bf16 within dtype_rtol / dtype_atol.
    check_inference_contract(model, make_tensor(16, 32), checks=["dtype_parity"], dtypes=(torch.bfloat16,))
