"""The tests ``pytest --torch-model=SPEC`` runs, once per model and device.

This module is collected by the plugin when ``--torch-model`` is given; it isn't meant to
be imported. Select checks with ``--torch-model-checks``.
"""
from __future__ import annotations

import copy

import pytest
import torch

from . import _session
from .assertions import format_problems
from .utils import flatten, is_compile_supported, to_device

#: check name -> tests; the defaults run when --torch-model-checks isn't given
CHECKS = {
    "contract": "test_inference_contract",
    "invariance": "test_invariance",
    "checkpoint": "test_checkpoint_roundtrip",
    "gradients": "test_gradients_finite",
    "export": "test_export_parity",
    "compile": "test_compile_parity",
}
DEFAULT_CHECKS = ("contract", "invariance", "checkpoint", "gradients")
# Models from model zoos are big and compared in float32: allow rounding noise.
TOLERANCE = {"rtol": 1e-4, "atol": 1e-4}

_subjects: dict[tuple, object] = {}


def pytest_generate_tests(metafunc):
    if "torch_model_spec" in metafunc.fixturenames:
        specs = list(_session.model_specs or [])
        metafunc.parametrize("torch_model_spec", specs, ids=specs, scope="module")


def _resolve(spec: str):
    from .zero_code import SpecError, resolve

    key = (spec, _session.model_pretrained, _session.model_inputs)
    if key not in _subjects:
        try:
            _subjects[key] = resolve(spec, pretrained=_session.model_pretrained, inputs=_session.model_inputs)
        except SpecError as exc:
            _subjects[key] = exc
    return _subjects[key]


@pytest.fixture
def subject(torch_model_spec):
    resolved = _resolve(torch_model_spec)
    if isinstance(resolved, Exception):
        pytest.skip(f"{torch_model_spec} could not be built (see test_builds)")
    return resolved


@pytest.fixture
def model(subject, torch_seed, torch_device):
    return subject.build().to(torch_device)


@pytest.fixture
def inputs(subject, torch_generator, torch_device):
    return to_device(subject.inputs.make(torch_generator), torch_device)


def _loss(output):
    terms = [leaf.float().mean() for _, leaf in flatten(output)
             if isinstance(leaf, torch.Tensor) and leaf.is_floating_point() and leaf.requires_grad]
    if not terms:
        raise AssertionError("the model returned no floating-point output that depends on its parameters")
    return torch.stack(terms).sum()


# --------------------------------------------------------------------------- #
# the tests
# --------------------------------------------------------------------------- #
def test_builds(torch_model_spec, torch_seed, torch_device, torch_generator):
    """The model builds and runs on the inferred inputs; says what to do if not."""
    from .utils import call_model

    subject = _resolve(torch_model_spec)
    if isinstance(subject, Exception):
        pytest.fail(f"{torch_model_spec}: {subject}", pytrace=False)
    model = subject.build().to(torch_device)
    with torch.inference_mode():
        call_model(model, to_device(subject.inputs.make(torch_generator), torch_device))
    print(f"{subject.spec}: inputs {subject.inputs.description}")


def test_inference_contract(model, inputs):
    from .contract import check_inference_contract

    check_inference_contract(model, inputs, **TOLERANCE)


def test_invariance(subject, model, inputs):
    """Text models: padding and masked tokens don't leak. Every model: memory layout."""
    from .invariance import assert_layout_invariant, assert_mask_invariant, assert_padding_invariant

    problems = []
    checks = [lambda: assert_layout_invariant(model, inputs, **TOLERANCE)]
    if subject.inputs.kind == "text":
        checks.insert(0, lambda: assert_padding_invariant(model, inputs, **TOLERANCE))
        checks.insert(1, lambda: assert_mask_invariant(model, inputs, vocab_size=subject.inputs.vocab_size, **TOLERANCE))
    for check in checks:
        try:
            check()
        except AssertionError as exc:
            problems.append(str(exc))
    if problems:
        raise AssertionError(format_problems(f"{subject.spec}: invariance violated", problems))


def test_checkpoint_roundtrip(subject, model, inputs, torch_device):
    from .checkpoint import assert_checkpoint_roundtrip

    assert_checkpoint_roundtrip(lambda: subject.build().to(torch_device), inputs, model=model.eval(), **TOLERANCE)


def test_gradients_finite(model, inputs):
    from .training import assert_gradients_finite

    assert_gradients_finite(copy.deepcopy(model).train(), inputs, _loss)


def test_export_parity(model, inputs):
    from .parity import assert_export_parity

    if not is_compile_supported() or not hasattr(torch, "export"):
        pytest.skip("torch.export is not supported here")
    assert_export_parity(model, inputs, **TOLERANCE)


def test_compile_parity(model, inputs):
    from .parity import assert_compile_parity

    if not is_compile_supported():
        pytest.skip("torch.compile is not supported here")
    assert_compile_parity(model, inputs, backend="eager", **TOLERANCE)
