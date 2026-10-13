"""pytest-torch: helpers and fixtures for testing PyTorch inference.

The package root must stay import-light: pytest imports ``pytest_torch.plugin``
at startup in *every* project where the plugin is installed, so ``torch`` must
not be imported here. Public helpers are exposed lazily.
"""
import importlib

__version__ = "0.1.0"

_EXPORTS = {
    "assert_tensors_close": "assertions",
    "assert_outputs_close": "assertions",
    "assert_finite": "assertions",
    "assert_batch_invariant": "assertions",
    "check_inference_contract": "contract",
    "ContractReport": "contract",
    "CheckResult": "contract",
    "localize_divergence": "divergence",
    "DivergenceReport": "divergence",
    "LayerDiff": "divergence",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'pytest_torch' has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)
