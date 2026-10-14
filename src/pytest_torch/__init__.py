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
    "assert_frozen": "training",
    "assert_loss_decreases": "training",
    "assert_overfits_one_batch": "training",
    "TrainingRun": "training",
    # generated test suites
    "ModelSuite": "suite",
    # performance
    "measure_latency": "perf",
    "measure_peak_memory": "perf",
    "check_perf": "perf",
    "compare_perf": "perf",
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
    "available_devices": "devices",
    "tolerance_overrides": "tolerances",
    # output specs
    "TensorSpec": "specs",
    "Probabilities": "specs",
    "assert_output_spec": "specs",
    # failure reproduction
    "record_repro": "repro",
    "load_repro": "repro",
    # checkpoints
    "assert_checkpoint_roundtrip": "checkpoint",
    "assert_checkpoint_loads": "checkpoint",
    # layouts
    "assert_layout_invariant": "invariance",
    # stateful / streaming models
    "assert_streaming_equivalent": "stateful",
    "StreamingEquivalence": "stateful",
    "StreamingReport": "stateful",
    # benchmarks
    "BenchmarkSuite": "benchmark",
    "BenchmarkResult": "benchmark",
    # dataset evaluation
    "evaluate": "evaluation",
    "evaluate_detailed": "evaluation",
    "paired_difference": "evaluation",
    "MetricResult": "evaluation",
}
__all__ = list(_EXPORTS)

if TYPE_CHECKING:  # static analysis only: nothing here runs at import time
    from .assertions import (
        assert_batch_invariant,
        assert_finite,
        assert_outputs_close,
        assert_tensors_close,
    )
    from .benchmark import BenchmarkResult, BenchmarkSuite
    from .checkpoint import assert_checkpoint_loads, assert_checkpoint_roundtrip
    from .compilation import assert_no_graph_breaks, assert_no_recompiles
    from .contract import CheckResult, ContractReport, check_inference_contract
    from .devices import available_devices
    from .divergence import DivergenceReport, LayerDiff, localize_divergence
    from .evaluation import MetricResult, evaluate, evaluate_detailed, paired_difference
    from .integrations import ModelLoader
    from .invariance import (
        assert_equivariant,
        assert_invariant,
        assert_layout_invariant,
        assert_mask_invariant,
        assert_padding_invariant,
        assert_permutation_equivariant,
        assert_permutation_invariant,
    )
    from .nondeterminism import (
        NondeterminismReport,
        VarianceResult,
        assert_deterministic,
        diagnose_nondeterminism,
        measure_nondeterminism,
    )
    from .parallel import FileLock
    from .parity import (
        AgreementMetrics,
        assert_aoti_parity,
        assert_autocast_parity,
        assert_compile_parity,
        assert_export_parity,
        assert_generation_equal,
        assert_matches_reference,
        assert_models_agree,
        assert_onnx_parity,
        assert_torchscript_parity,
        compare_outputs,
    )
    from .perf import PerfResult, PerfStore, check_perf, compare_perf, measure_latency, measure_peak_memory
    from .report import github_annotation, save_mismatch_heatmap
    from .repro import load_repro, record_repro
    from .specs import Probabilities, TensorSpec, assert_output_spec
    from .stateful import StreamingEquivalence, StreamingReport, assert_streaming_equivalent
    from .suite import ModelSuite
    from .sync import assert_no_host_sync, forbid_host_sync
    from .tolerances import tolerance_overrides
    from .training import (
        TrainingRun,
        assert_all_params_receive_grad,
        assert_frozen,
        assert_gradcheck,
        assert_gradients_finite,
        assert_loss_decreases,
        assert_overfits_one_batch,
    )
    from .utils import deterministic_algorithms, is_compile_supported, seed_everything


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'pytest_torch' has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value  # cache: later lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_EXPORTS])
