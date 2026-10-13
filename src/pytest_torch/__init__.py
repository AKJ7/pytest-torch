"""pytest-torch: helpers and fixtures for testing PyTorch inference.

The package root must stay import-light: pytest imports ``pytest_torch.plugin``
at startup in *every* project where the plugin is installed, so ``torch`` must
not be imported here. Assertion helpers are exposed lazily.
"""

__version__ = "0.1.0"
__all__ = ["assert_tensors_close", "assert_finite", "assert_batch_invariant"]


def __getattr__(name):
    if name in __all__:
        from . import assertions

        return getattr(assertions, name)
    raise AttributeError(f"module 'pytest_torch' has no attribute {name!r}")
