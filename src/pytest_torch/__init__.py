"""pytest-torch: helpers and fixtures for testing PyTorch inference.

The package root must stay import-light: pytest imports ``pytest_torch.plugin``
at startup in *every* project where the plugin is installed, so ``torch`` must
not be imported here. Public helpers are therefore exposed lazily (PEP 562) and
mirrored in a ``TYPE_CHECKING`` block so IDEs and type checkers still see them.

Not exported here (import them directly): ``pytest_torch.strategies`` needs the
optional ``hypothesis`` dependency.
"""
from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

try:  # set at release time from the git tag (see .github/workflows/release.yml)
    __version__ = version("pytest-torch")
except PackageNotFoundError:  # running from a source tree that isn't installed
    __version__ = "0+unknown"

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
    "assert_export_parity": "parity",
    "assert_aoti_parity": "parity",
    "assert_autocast_parity": "parity",
    # training sanity
    "assert_gradients_finite": "training",
    "assert_all_params_receive_grad": "training",
    "assert_gradcheck": "training",
    "assert_loss_decreases": "training",
    "assert_overfits_one_batch": "training",
    "TrainingRun": "training",
    # generated test suites
    "ModelSuite": "suite",
    # performance
    "measure_latency": "perf",
    "measure_peak_memory": "perf",
    "check_perf": "perf",
    "PerfStore": "perf",
    "PerfResult": "perf",
    # torch.compile health, host-device syncs
    "assert_no_graph_breaks": "compilation",
    "assert_no_recompiles": "compilation",
    "forbid_host_sync": "sync",
    "assert_no_host_sync": "sync",
    # nondeterminism
    "measure_nondeterminism": "nondeterminism",
    "diagnose_nondeterminism": "nondeterminism",
    "assert_deterministic": "nondeterminism",
    "NondeterminismReport": "nondeterminism",
    "VarianceResult": "nondeterminism",
    # failure output
    "save_mismatch_heatmap": "report",
    "github_annotation": "report",
    # xdist helpers
    "FileLock": "parallel",
    # model zoo + seeding helpers
    "ModelLoader": "integrations",
    "seed_everything": "utils",
    "deterministic_algorithms": "utils",
    "is_compile_supported": "utils",
}
__all__ = list(_EXPORTS)

if TYPE_CHECKING:  # static analysis only: nothing here runs at import time
    from .assertions import (
        assert_tensors_close,
        assert_outputs_close,
        assert_finite,
        assert_batch_invariant,
    )
    from .contract import check_inference_contract, ContractReport, CheckResult
    from .divergence import localize_divergence, DivergenceReport, LayerDiff
    from .invariance import (
        assert_padding_invariant,
        assert_mask_invariant,
        assert_invariant,
        assert_equivariant,
        assert_permutation_invariant,
        assert_permutation_equivariant,
    )
    from .parity import (
        assert_compile_parity,
        assert_torchscript_parity,
        assert_onnx_parity,
        assert_models_agree,
        compare_outputs,
        AgreementMetrics,
        assert_matches_reference,
        assert_generation_equal,
        assert_export_parity,
        assert_aoti_parity,
        assert_autocast_parity,
    )
    from .training import (
        assert_gradients_finite,
        assert_all_params_receive_grad,
        assert_gradcheck,
        assert_loss_decreases,
        assert_overfits_one_batch,
        TrainingRun,
    )
    from .suite import ModelSuite
    from .perf import measure_latency, measure_peak_memory, check_perf, PerfStore, PerfResult
    from .compilation import assert_no_graph_breaks, assert_no_recompiles
    from .sync import forbid_host_sync, assert_no_host_sync
    from .nondeterminism import (
        measure_nondeterminism,
        diagnose_nondeterminism,
        assert_deterministic,
        NondeterminismReport,
        VarianceResult,
    )
    from .report import save_mismatch_heatmap, github_annotation
    from .parallel import FileLock
    from .integrations import ModelLoader
    from .utils import seed_everything, deterministic_algorithms, is_compile_supported


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'pytest_torch' has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value  # cache: later lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_EXPORTS])