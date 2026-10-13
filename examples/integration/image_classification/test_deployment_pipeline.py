"""An image classifier and its deployment pipeline, tested before every release.

Run: pytest examples/integration/image_classification -v        (add -m "not slow" to skip AOTInductor)
Regenerate the golden file: pytest examples/integration/image_classification -k golden --torch-update-golden --torch-device=cpu
"""
import copy

import pytest
import torch

from pytest_torch import (
    ModelSuite,
    Probabilities,
    TensorSpec,
    assert_aoti_parity,
    assert_batch_invariant,
    assert_export_parity,
    assert_layout_invariant,
    assert_models_agree,
    assert_output_spec,
    is_compile_supported,
)

from .model import CLASSES, InferencePipeline, ResNet, make_images

TORCH_VERSION = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
# torch.export with a dynamic batch dim adds a spurious `batch < 16` guard on torch < 2.4.
CAN_EXPORT = is_compile_supported() and TORCH_VERSION >= (2, 4)
needs_export = pytest.mark.skipif(not CAN_EXPORT, reason="dynamic-batch torch.export needs torch >= 2.4")

PIPELINE_SPEC = {
    "logits": TensorSpec(("B", len(CLASSES)), dtype=(torch.float32, torch.bfloat16)),
    "probs": Probabilities(("B", len(CLASSES)), atol=1e-2),  # loose enough for bf16
}


# --------------------------------------------------------------------------- #
# The network itself: a generated suite
# --------------------------------------------------------------------------- #
class TestResNet(ModelSuite):
    checks = ("contract", "compile", "layouts", "autocast", "checkpoint", "output_spec") + (
        ("export",) if CAN_EXPORT else ()
    )
    dynamic_batch = True
    output_spec = TensorSpec(("B", len(CLASSES)))
    check_options = {"inference_contract": {"atol": 1e-5, "rtol": 1e-4}}  # conv reductions

    def make_model(self):
        return ResNet()

    def make_inputs(self, make_tensor):
        return make_tensor(8, 3, 32, 32)


# --------------------------------------------------------------------------- #
# The deployment artifact: raw uint8 images in, probabilities out
# --------------------------------------------------------------------------- #
@pytest.fixture
def images(torch_device):
    return make_images(16, seed=2)[0].to(torch_device)


def test_pipeline_output_contract(pipeline, images):
    with torch.inference_mode():
        assert_output_spec(pipeline(images), PIPELINE_SPEC, B=16)


def test_pipeline_accepts_any_memory_layout(pipeline, images):
    # Decoders and data loaders hand over non-contiguous or channels_last batches.
    # Try `.view(n, h * w, c)` in InferencePipeline.forward: this test fails.
    assert_layout_invariant(pipeline, images)


@needs_export
def test_exported_pipeline_serves_any_batch_size(pipeline, images):
    program = assert_export_parity(pipeline, images, images[:3], dynamic_batch=True, atol=1e-5, rtol=1e-4)
    assert program is not None  # save it with torch.export.save(program, "pipeline.pt2")


@needs_export
@pytest.mark.slow
def test_aot_compiled_pipeline_matches_eager(pipeline, images):
    # The .pt2 package a C++ inference server loads (compiles C++: ~20 s).
    assert_aoti_parity(pipeline.cpu(), images.cpu(), images[:3].cpu(), dynamic_batch=True)


def test_bf16_deployment_keeps_predictions(trained_resnet, torch_device):
    fp32 = InferencePipeline(trained_resnet).to(torch_device).eval()
    bf16 = copy.deepcopy(fp32).to(torch.bfloat16)
    images, _ = make_images(128, seed=3)
    metrics = assert_models_agree(fp32, bf16, images, select="output['probs']", min_top1=0.97, max_abs=0.05)
    print(metrics)  # with `pytest -s`: log how far bf16 drifts


@pytest.mark.torch_dtypes(torch.float32, torch.bfloat16)
def test_every_precision_is_batch_invariant(trained_resnet, torch_device, torch_dtype, images):
    pipeline = InferencePipeline(copy.deepcopy(trained_resnet)).to(torch_device, torch_dtype).eval()
    # BatchNorm in eval mode uses running statistics: no sample may affect another.
    assert_batch_invariant(pipeline, images)


# --------------------------------------------------------------------------- #
# Release gates: accuracy, speed, unchanged outputs
# --------------------------------------------------------------------------- #
def accuracy(output, batch):
    return (output["probs"].argmax(-1) == batch[1]).float()  # per sample: paired comparisons


def test_validation_accuracy(pipeline, val_loader, metric_baseline):
    # The pipeline does its own preprocessing, so its version is part of the context.
    metric_baseline(pipeline, val_loader, {"accuracy": accuracy}, minimum={"accuracy": 0.95},
                    context={"dataset": "synthetic-colors", "split": "validation", "preprocessing": "pipeline-v1"})


@pytest.mark.torch_max_latency(ms=250)  # generous: shared CI runners are slow and noisy
def test_latency_budget(pipeline, images, latency):
    latency(pipeline, images, warmup=2, iters=5)


@pytest.mark.torch_golden("resnet_pipeline_v1")
def test_outputs_unchanged_by_refactors(torch_seed, torch_device, golden):
    # A freshly seeded network: weights are reproducible across machines and torch versions,
    # so any change in this golden output comes from a change in the code.
    pipeline = InferencePipeline(ResNet()).to(torch_device).eval()
    with torch.inference_mode():
        golden.check(pipeline(make_images(4, seed=4)[0].to(torch_device)), atol=1e-4)
