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

## Output specifications

::: pytest_torch.specs
    options:
      show_root_heading: false
      members: [assert_output_spec, TensorSpec, Probabilities]

## Checkpoints

::: pytest_torch.checkpoint
    options:
      show_root_heading: false
      members: [assert_checkpoint_roundtrip, assert_checkpoint_loads]

## Stateful and streaming models

::: pytest_torch.stateful
    options:
      show_root_heading: false
      members: [assert_streaming_equivalent, StreamingEquivalence, StreamingReport]

## Dataset evaluation

::: pytest_torch.evaluation
    options:
      show_root_heading: false
      members: [evaluate, evaluate_detailed, Evaluation, paired_difference, PairedDifference, check_metrics, comparability_problems, model_fingerprint, MetricResult]

## Failure reproduction

::: pytest_torch.repro
    options:
      show_root_heading: false
      members: [record_repro, load_repro]

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

## Invariance and layouts

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

## Training loops

::: pytest_torch.training_loop
    options:
      show_root_heading: false
      members: [TrainingSetup, TrainingRunner, run_training, standard_step, assert_optimizer_coverage, assert_resume_equivalent, ResumeReport, training_state, save_training_state, load_training_state, assert_accumulation_equivalent, split_batch, assert_mixed_precision_training]

## Performance

::: pytest_torch.perf
    options:
      show_root_heading: false
      members: [measure_latency, measure_peak_memory, check_perf, compare_perf, PerfStore, PerfResult]

## Benchmark suites

::: pytest_torch.benchmark
    options:
      show_root_heading: false
      members: [BenchmarkSuite, BenchmarkResult]

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

::: pytest_torch.tolerances
    options:
      show_root_heading: false
      members: [tolerance_overrides]

::: pytest_torch.devices
    options:
      show_root_heading: false
      members: [available_devices]

::: pytest_torch.integrations.ModelLoader
