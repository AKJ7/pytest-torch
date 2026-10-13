"""Golden-file (snapshot) support for tensor and pytree outputs."""
from __future__ import annotations

from pathlib import Path

import pytest

_STORABLE = (int, float, bool, str, type(None))


class Golden:
    def __init__(self, path: Path, update: bool):
        self.path = path
        self.update = update

    def check(self, actual, *, rtol=None, atol=None) -> None:
        """Compare ``actual`` (tensor or nested dict/tuple/dataclass) to the golden file."""
        import torch

        from .assertions import format_problems, leaves_diff
        from .utils import flatten

        leaves = {}
        for path, leaf in flatten(actual):
            if isinstance(leaf, torch.Tensor):
                leaves[path] = leaf.detach().cpu()
            elif isinstance(leaf, _STORABLE):
                leaves[path] = leaf
            else:
                raise TypeError(
                    f"golden: unsupported leaf type {type(leaf).__name__} at {path}"
                )

        if self.update:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"leaves": leaves, "torch_version": torch.__version__}, self.path)
            return

        if not self.path.exists():
            # Fail loudly: silently creating goldens would let CI pass vacuously.
            pytest.fail(
                f"Golden file {self.path} does not exist. "
                "Run with --torch-update-golden to create it.",
                pytrace=False,
            )

        ref = torch.load(self.path, map_location="cpu", weights_only=True)
        problems = leaves_diff(
            leaves, ref["leaves"], rtol=rtol, atol=atol, check_device=False
        )
        if problems:
            raise AssertionError(
                format_problems(
                    f"Golden mismatch: {len(problems)} problem(s)", problems
                )
                + f"\n  (golden saved with torch {ref.get('torch_version', '?')}, "
                f"running torch {torch.__version__})"
            )
