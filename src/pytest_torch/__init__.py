"""pytest-torch: helpers and fixtures for testing PyTorch inference.

The package root must stay import-light: pytest imports ``pytest_torch.plugin``
at startup in *every* project where the plugin is installed, so ``torch`` must
not be imported here. Public helpers are exposed lazily.
"""
import importlib

__version__ = "0.2.0"

_EXPORTS = {
    # assertions
    "assert_tensors_close": "assertions",
    "assert_outputs_close": "assertions",
    "assert_finite": "assertions",
    "assert_batch_invariant": "assertions",
    # one-call contract check
    "check_inference_contract": "contract",
    "ContractReport": "contract",
    "CheckResult": "contract",
    # layer divergence
    "localize_divergence": "divergence",
    "DivergenceReport": "divergence",
    "LayerDiff": "divergence",
    # invariance / metamorphic
    "assert_padding_invariant": "invariance",
    "assert_mask_invariant": "invariance",
    "assert_invariant": "invariance",
    "assert_equivariant": "invariance",
    "assert_permutation_invariant": "invariance",
    "assert_permutation_equivariant": "invariance",
    # parity: deployment targets, references, accuracy budgets
    "assert_compile_parity": "parity",
    "assert_torchscript_parity": "parity",
    "assert_onnx_parity": "parity",
    "assert_models_agree": "parity",
    "compare_outputs": "parity",
    "AgreementMetrics": "parity",
    "assert_matches_reference": "parity",
    "assert_generation_equal": "parity",
    # model zoo + seeding helpers
    "ModelLoader": "integrations",
    "seed_everything": "utils",
    "deterministic_algorithms": "utils",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'pytest_torch' has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)
