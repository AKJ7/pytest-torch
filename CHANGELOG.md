# Changelog

## Unreleased

First public release. `pytest-torch` is a pytest plugin for testing PyTorch models. It
covers inference correctness, invariance, parity with deployment targets, training sanity,
performance and determinism, and its failure messages say where a mismatch is and how large
it is. It requires Python 3.9+, pytest 7+ and PyTorch 2.0+. The plugin registers itself
when installed, and importing it never imports `torch`.

### A whole test suite from one class: `ModelSuite`

Subclass `ModelSuite`, describe how to build the model and its inputs, and pick the checks.
pytest collects one test per check, on every available device, with seeded weights and
inputs.

```python
from pytest_torch import ModelSuite

class TestClassifier(ModelSuite):
    checks = ("contract", "compile", "export", "training")

    def make_model(self):
        return Classifier()

    def make_inputs(self, make_tensor):
        return make_tensor(8, 16)

    def loss(self, output, inputs):          # only needed for "training"
        return F.cross_entropy(output, labels)
```

- Available checks: `contract`, `compile`, `export` (these three are the default),
  `torchscript`, `onnx`, `aoti`, `autocast`, `golden` and `training`.
- `check_options` passes arguments to the assertion behind each test, for example
  `{"overfits_one_batch": {"steps": 500}}`.
- With `dynamic_batch = True`, the export test checks that one exported program accepts
  other batch sizes.
- Checks you don't list are not generated at all, so there are no skipped placeholder tests.
- Generated tests are ordinary methods: you can override them or add your own.
- Configuration mistakes (an unknown check, an unknown option, `training` without a
  `loss()` method) fail at collection with a message saying what to fix.

### Fixtures and reproducibility

| Fixture | What it gives you |
|---|---|
| `torch_device` | Runs the test on every available device (CPU, CUDA, Intel XPU, Apple MPS); unavailable ones are skipped |
| `torch_seed` | Seeds `random`, `numpy` and `torch`; the seed is printed when the test fails |
| `torch_generator` | A seeded CPU generator, so the same seed gives identical values on every device |
| `make_tensor` | Reproducible tensors (normal, uniform or integer), drawn on the CPU and moved to the device |
| `inference_mode` | Runs the test body under `torch.inference_mode()` |
| `golden` | Snapshot tests of tensors and nested outputs |
| `torch_dtype` | Runs the test once per dtype (default float32; `--torch-dtype` or `@pytest.mark.torch_dtypes`) |
| `latency`, `perf_baseline` | Latency budgets and performance regression checks |
| `metric_baseline` | Dataset metrics compared with a stored baseline and thresholds |
| `model_loader` | Session-cached Hugging Face, timm and torchvision models |

- **Device selection:** `--torch-device` pins one or more devices.
- **Seeding:** the seed comes from `@pytest.mark.torch_seed(n)` if set, otherwise from
  `--torch-seed-mode=per-test` (a stable seed derived from each test's id), otherwise from
  `--torch-seed`. `--torch-autoseed` seeds every test, and `--torch-deterministic` turns on
  `torch.use_deterministic_algorithms` inside `torch_seed`.
- **Golden files:** `pytest --torch-update-golden` writes them. They store nested outputs
  by path and record the torch version, which is shown when a comparison fails. A missing
  golden file fails the test, so CI can't pass without comparing anything.

### Project-wide tolerances

Set default tolerances once in `pyproject.toml` instead of repeating `rtol=`/`atol=` in
every test:

```toml
[tool.pytest_torch.tolerance]
bfloat16 = { rtol = 2e-2, atol = 1e-2 }          # any comparison of bf16 tensors
"cuda.float32" = { rtol = 1e-4, atol = 1e-5 }    # only when testing on CUDA (TF32)
compile_parity = { rtol = 1e-3, atol = 1e-3 }    # default of assert_compile_parity
```

- **Keys:** a dtype, a check (`compile_parity`, `export_parity`, `device_parity`,
  `autocast_parity`, `gradcheck` and others), or either prefixed with a device. An entry
  may set only `rtol` or only `atol`.
- **One test, class or module:** `@pytest.mark.torch_tolerance(bfloat16=(5e-2, 1e-2))`.
  For a block of code, use `with tolerance_overrides(...)`.
- **Precedence:** an argument passed to the assertion wins, then the marker, then
  `pyproject.toml`, then the built-in default.
- **Mistakes** in the config stop pytest at startup with a message saying what is wrong,
  and failure messages always print the tolerance that was used.

### Assertions with readable failures

- **Comparisons:** `assert_tensors_close` uses default tolerances suited to each dtype.
  `assert_outputs_close` compares nested outputs (dicts, tuples, dataclasses, Hugging Face
  `ModelOutput`s) and reports the path of each mismatch, such as `output['logits']`.
- **`assert_finite`** finds NaN or Inf anywhere in a nested output.
- **`assert_batch_invariant`** checks that each sample's output doesn't depend on the rest
  of the batch.
- Failures report how many elements differ, where the worst one is, and the absolute and
  relative error.
- Model inputs can be a tensor, a tuple of positional arguments or a dict of keyword
  arguments, and they are moved to the model's device automatically.

### Inference contract

`check_inference_contract(model, inputs)` runs the checks every inference model should
pass, and gives a likely cause for each failure:

- **`eval_mode`:** no module was left in training mode.
- **`finite_outputs`:** no NaN or Inf in the outputs.
- **`deterministic`:** two runs give identical outputs.
- **`batch_invariance`:** a sample's output doesn't depend on the rest of the batch.
- **`device_parity`:** results match on the other available devices, or only on the other
  devices pinned with `--torch-device`.
- **`dtype_parity`:** results stay close at lower precision. It only runs if you ask for it,
  for example with `dtypes=(torch.bfloat16,)`.
- **`no_state_mutation`:** inference didn't change any parameter or buffer.

You can select or skip checks and tune tolerances. With `raise_on_failure=False` you get a
report back instead of an exception.

### Find the first diverging layer

`localize_divergence(model_a, model_b, inputs)` compares two models layer by layer (for
example CPU against GPU, fp32 against bf16, or a port against the original) and names the
first layer that disagrees. It also tells you whether the error started inside that layer
or came in from the code between layers. `module_map` pairs up layers whose names differ.

### Invariance and metamorphic tests

These check properties that must hold even when you don't know the right output:

- **`assert_padding_invariant`:** adding padding (left or right) doesn't change the
  outputs for real tokens.
- **`assert_mask_invariant`:** changing the tokens under the attention mask doesn't change
  the outputs.
- **`assert_permutation_invariant`:** reordering the elements of a set doesn't change the
  output.
- **`assert_permutation_equivariant`:** reordering the input reorders the output the same
  way.
- **`assert_invariant` and `assert_equivariant`:** check any relation you define yourself.

Outputs aligned with the sequence are compared only on real tokens.

### Parity with deployment targets and references

- **`torch.compile`:** `assert_compile_parity` compares compiled and eager outputs over
  repeated runs.
- **`torch.export`:** `assert_export_parity` compares the exported program with eager
  mode. Pass extra inputs plus `dynamic_batch=True`, or your own `dynamic_shapes`, to check
  that one program handles several batch sizes. If export fails, the assertion carries
  torch's reason and a hint.
- **AOTInductor:** `assert_aoti_parity` compiles the model ahead of time into a `.pt2`
  package, loads it and compares it with eager mode. This is the artifact a C++ server or
  edge runtime loads. It needs torch 2.6+ and a C++ compiler; the test skips otherwise.
- **Mixed precision:** `assert_autocast_parity` compares a run under `torch.autocast`
  with full precision.
- **TorchScript and ONNX:** `assert_torchscript_parity` (trace or script) and
  `assert_onnx_parity`, which runs the model in ONNX Runtime.
- **Accuracy budgets:** `assert_models_agree` checks a quantized, pruned or distilled
  model against the original on max error, cosine similarity and top-1 agreement, and
  returns the measured numbers so you can log them.
- **Reference implementations:** `assert_matches_reference` compares your model with
  another model or with saved outputs.
- **Text generation:** `assert_generation_equal` checks that greedy decoding produces the
  same tokens and reports the first token that differs in each row.

### Training sanity checks

Catch a broken training setup in seconds. Each helper takes `(model, inputs, loss_fn)`,
where `loss_fn(output)` returns a scalar.

| Function | Catches |
|---|---|
| `assert_gradients_finite` | NaN or Inf in the loss or in any gradient, listed per parameter |
| `assert_all_params_receive_grad` | Unused layers, `.detach()` or `.item()` in `forward`, heads the loss ignores, all-zero gradients |
| `assert_loss_decreases` | A training loop that doesn't learn: zero learning rate, frozen parameters, an optimizer that never sees them |
| `assert_overfits_one_batch` | A model that can't memorize one small batch, usually because of a bug |
| `assert_gradcheck` | A custom `autograd.Function` whose backward disagrees with finite differences |

The checks that run optimizer steps train a copy, so the model you pass in is never
modified. They return a `TrainingRun` with the loss curve.

### Output specifications

`assert_output_spec(output, spec)` checks shapes, dtypes, finite values, value ranges and
probability normalization in one call, reporting every problem with its output path.

```python
spec = {"logits": TensorSpec(("B", "T", 5), dtype=torch.float32), "probs": Probabilities(("B", "T", 5))}
assert_output_spec(model(ids), spec, B=4)
```

- **Shapes** take exact sizes, symbolic dims (`"B"`: the same size everywhere), `None`
  (any size) and `...` (any number of dims).
- **`Probabilities`** checks values in [0, 1] summing to 1 along a dim, or
  log-probabilities with `log=True`.
- **In a `ModelSuite`**, set `output_spec` and add the `output_spec` check.

### Checkpoints

- **`assert_checkpoint_roundtrip`** saves the `state_dict`, reloads it into a fresh model
  built under a different seed, and requires identical outputs. When they differ, it names
  the likely culprits: tensors stored as plain attributes, non-persistent buffers, tied
  weights that came apart. It supports `torch.save` and safetensors.
- **`assert_checkpoint_loads`** loads an existing checkpoint into the current code and lists
  every missing, unexpected and reshaped key at once. It runs the model's own compatibility
  code and suggests `strip_prefix="module."` for DataParallel/DDP checkpoints.
- **In a `ModelSuite`**, add the `checkpoint` check.

### Dataset metrics

- **`metric_baseline`** runs a loader through the model, aggregates metrics and fails when
  one gets worse than its recorded baseline by more than a tolerance. Record baselines with
  `--torch-update-metrics`. Fixed `minimum`/`maximum` thresholds apply on every run.
- **Metrics** are averaged weighted by batch size, or computed by objects with
  `update`/`compute` for metrics that need every prediction (F1, AUC).
- **`evaluate()`** is the same loop without a baseline.

### Failure reproduction bundles

- **Recording:** `--torch-repro-dir=DIR` writes a bundle for every failing test. It holds
  the test id and parameters, seed, device, every `make_tensor` output, values passed to
  `record_repro()`, the active tolerances and the environment.
- **Replaying:** `--torch-repro=BUNDLE` reruns just that test with the same seed and exactly
  the same inputs, on any machine, and lists environment differences. Add `--torch-device`
  to replay a GPU failure on the CPU.
- **Weights** are stored only up to `--torch-repro-max-weights-mb`.

### Device and dtype matrix

- **`torch_dtype`** runs a test once per dtype, for example
  `@pytest.mark.torch_dtypes(torch.float32, torch.bfloat16)`. Dtypes a device can't run at
  all are skipped.
- **`make_tensor`** follows the dtype, and comparison tolerances follow it too.

### Unusual inputs

- **`assert_layout_invariant`** checks that non-contiguous and `channels_last` copies of the
  inputs give the same outputs. It catches `.view()` on non-contiguous tensors and
  layout-specific kernel bugs. In a `ModelSuite`, add the `layouts` check.

### Property-based testing

`pytest_torch.strategies` provides Hypothesis strategies for tensors and shapes. Generated
tensors deliberately include edge cases (all zeros, constants, extreme values, tiny values),
can optionally contain NaN or Inf, and can come in non-contiguous or `channels_last`
layouts. `padded_batches()` draws ragged `input_ids`/`attention_mask` batches covering
length-1, full-length and all-padding-but-one cases. This needs the `hypothesis` extra.

### Performance, compilation and determinism

- **Latency budgets:** `@pytest.mark.torch_max_latency(ms=...)` with the `latency`
  fixture. Measurements include warmup and device synchronization.
- **Regression baselines:** the `perf_baseline` fixture records latency, and peak memory
  on CUDA and XPU, once per machine with `--torch-update-perf`, then compares later runs against
  that record. Baselines are stored per test and per hardware, so numbers from different
  machines never mix.
- **`torch.compile` health:** `assert_no_graph_breaks` and `assert_no_recompiles` check
  that a model compiles into one graph and isn't recompiled on repeated calls.
  `is_compile_supported()` tells you whether compilation works on the current platform.
- **Hidden synchronizations:** `assert_no_host_sync` catches calls that wait for the GPU
  (`.item()`, `.cpu()`, `if tensor:`). CUDA only.
- **Nondeterminism:** `assert_deterministic` checks repeated runs give the same output.
  When they don't, `diagnose_nondeterminism` tells you whether the variation comes from
  random-number use or from nondeterministic GPU kernels.

### Model zoo

The `model_loader` fixture loads Hugging Face models and tokenizers, timm models and
torchvision models. Models come back in eval mode and are cached for the whole session.
These need the `hf`, `timm` and `torchvision` extras.

### Parallel runs and CI

- **pytest-xdist:**
  - `--torch-xdist-gpus` spreads workers across CUDA or Intel XPU GPUs.
  - `--torch-gpu-memory-fraction=auto` splits each GPU's memory between the workers that
    share it.
  - `@pytest.mark.torch_gpu_exclusive` makes heavy tests take turns on each GPU.
- **Failure artifacts:** `--torch-failure-artifacts=DIR` saves a heatmap image of every
  tensor mismatch. This needs the `plots` extra.
- **GitHub annotations:** `--torch-github-annotations` marks failing tests on the pull
  request.

### Packaging and documentation

- **Optional extras:** `hypothesis`, `onnx`, `hf`, `timm`, `torchvision` and `plots`.
- **Type information:** the package ships a `py.typed` file.
- **Licence:** MIT, declared in the package metadata.
- **Documentation:** <https://akj7.github.io/pytest-torch/>, with guides, recipes and an
  API reference.
- **Examples:** runnable examples for every feature are in `examples/`. Most include a
  "Try ..." comment showing a one-line bug that makes the test fail.
