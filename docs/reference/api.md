# API

Everything below is importable from `pytest_torch` directly, e.g.
`from pytest_torch import assert_export_parity`.

## Generated suites

::: pytest_torch.suite.ModelSuite

## Assertions

::: pytest_torch.assertions
    options:
      show_root_heading: false
      members: [assert_tensors_close, assert_outputs_close, assert_finite, assert_batch_invariant]

## Inference contract

::: pytest_torch.contract
    options:
      show_root_heading: false
      members: [check_inference_contract, ContractReport, CheckResult]

## Layer divergence

::: pytest_torch.divergence
    options:
      show_root_heading: false
      members: [localize_divergence, DivergenceReport, LayerDiff]

## Invariance

::: pytest_torch.invariance
    options:
      show_root_heading: false

## Parity

::: pytest_torch.parity
    options:
      show_root_heading: false

## Training

::: pytest_torch.training
    options:
      show_root_heading: false

## Performance

::: pytest_torch.perf
    options:
      show_root_heading: false
      members: [measure_latency, measure_peak_memory, check_perf, PerfStore, PerfResult]

## Compilation and host syncs

::: pytest_torch.compilation
    options:
      show_root_heading: false

::: pytest_torch.sync
    options:
      show_root_heading: false

## Nondeterminism

::: pytest_torch.nondeterminism
    options:
      show_root_heading: false

## Hypothesis strategies

::: pytest_torch.strategies
    options:
      show_root_heading: false

## Utilities

::: pytest_torch.utils
    options:
      show_root_heading: false
      members: [seed_everything, deterministic_algorithms, is_compile_supported]

::: pytest_torch.integrations.ModelLoader
