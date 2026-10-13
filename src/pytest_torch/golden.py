"""Golden-file (snapshot) support for tensor outputs."""
from __future__ import annotations

from pathlib import Path

import pytest


class Golden:
    def __init__(self, path: Path, update: bool):
        self.path = path
        self.update = update

    def check(self, actual, *, rtol=None, atol=None) -> None:
        import torch

        from pytest_torch.assertions import assert_tensors_close

        actual = actual.detach().cpu()

        if self.update:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"tensor": actual, "torch_version": torch.__version__}, self.path
            )
            return

        if not self.path.exists():
            # Fail loudly: silently creating goldens would let CI pass vacuously.
            pytest.fail(
                f"Golden file {self.path} does not exist. "
                "Run with --torch-update-golden to create it.",
                pytrace=False,
            )

        ref = torch.load(self.path, map_location="cpu", weights_only=True)
        try:
            assert_tensors_close(
                actual, ref["tensor"], rtol=rtol, atol=atol, check_device=False
            )
        except AssertionError as exc:
            raise AssertionError(
                f"{exc}\n  (golden saved with torch {ref.get('torch_version', '?')}, "
                f"running torch {torch.__version__})"
            ) from None
