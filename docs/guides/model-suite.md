# Generated test suites

Subclass `ModelSuite`, describe how to build the model and its inputs, pick the checks,
and pytest collects a test per check, on every device.

```python
import torch
from torch.nn import functional as F
from pytest_torch import ModelSuite

class TestTextClassifier(ModelSuite):
    checks = ("contract", "compile", "export", "autocast", "training")
    dynamic_batch = True
    check_options = {
        "inference_contract": {"dtypes": (torch.bfloat16,)},
        "overfits_one_batch": {"steps": 300},
    }

    def make_model(self):
        return TextClassifier()

    def make_inputs(self, make_tensor):
        ids = make_tensor(8, 12, dtype=torch.long, low=1, high=100)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def loss(self, output, inputs):
        return F.cross_entropy(output["logits"], inputs["input_ids"][:, 0] % 4)
```

```
test_models.py::TestTextClassifier::test_inference_contract[cpu] PASSED
test_models.py::TestTextClassifier::test_compile_parity[cpu] PASSED
test_models.py::TestTextClassifier::test_export_parity[cpu] PASSED
test_models.py::TestTextClassifier::test_autocast_parity[cpu] PASSED
test_models.py::TestTextClassifier::test_gradients_finite[cpu] PASSED
...
```

## Checks

| Check | Generated tests | Runs |
|---|---|---|
| `contract` *(default)* | `test_inference_contract` | [`check_inference_contract`](inference.md#inference-contract) |
| `compile` *(default)* | `test_compile_parity` | `assert_compile_parity` (eager backend) |
| `export` *(default)* | `test_export_parity` | `assert_export_parity` |
| `torchscript` | `test_torchscript_parity` | `assert_torchscript_parity` |
| `onnx` | `test_onnx_parity` | `assert_onnx_parity` (skips without onnxruntime) |
| `aoti` | `test_aoti_parity` | `assert_aoti_parity` (compiles C++) |
| `autocast` | `test_autocast_parity` | `assert_autocast_parity` |
| `golden` | `test_golden` | golden file `__golden__/<ClassName>.pt` |
| `output_spec` | `test_output_spec` | [`assert_output_spec`](output-specs.md) with the class's `output_spec`; `B` = batch size |
| `checkpoint` | `test_checkpoint_roundtrip` | [`assert_checkpoint_roundtrip`](checkpoints.md) |
| `layouts` | `test_layout_invariance` | [`assert_layout_invariant`](matrix.md#memory-layouts) |
| `frozen` | `test_frozen` | [`assert_frozen`](training.md) with `frozen` / `frozen_parameters()`; needs `loss()` |
| `training` | `test_gradients_finite`, `test_all_params_receive_grad`, `test_loss_decreases`, `test_overfits_one_batch` | [training sanity](training.md); needs `loss()` |

Checks you don't list don't exist on the class at all (no wall of skips). Compile, export
and AOTInductor tests skip on platforms where dynamo is unsupported.

## Options

`check_options` passes keyword arguments to the assertion behind a test, keyed by the test
name without `test_`:

```python
check_options = {
    "inference_contract": {"skip": ["device_parity"], "dtypes": (torch.float16,)},
    "compile_parity": {"backend": "inductor"},
    "all_params_receive_grad": {"ignore": ["pooler.*"]},
    "overfits_one_batch": {"steps": 500, "lr": 3e-3},
}
```

`dynamic_batch = True` makes `test_export_parity` export with a dynamic batch dimension and
run a smaller batch through the same program (the example batch must be at least 4).

Mistakes such as an unknown check, an unknown option key, or `"training"` without a
`loss()` method fail at collection with a message saying what to fix.

## One suite per config

Projects with many configs (Hydra / habitat-baselines style YAML files, one per experiment)
can run every generated test once per config: each config builds a different model, and
option combinations are where bugs hide.

```python
from pathlib import Path

class TestNavPolicy(ModelSuite):
    configs = sorted(Path("configs").rglob("*.yaml"))
    # Shrink every config so tests stay fast (dotted keys into the loaded config).
    config_overrides = {"habitat_baselines.rl.ddppo.hidden_size": 32}
    checks = ("contract", "output_spec", "checkpoint", "frozen", "training")

    def make_model(self):
        return NavPolicy(self.config)          # the current config, overrides applied

    def make_inputs(self, make_tensor):
        n = self.config["habitat"]["task"]["num_object_categories"]
        return {"rgb": make_tensor(4, 6, 3, 32, 32), "objectgoal": make_tensor(4, 6, dtype=torch.long, high=n)}

    def frozen_parameters(self):               # what `train_encoder: false` freezes
        return [] if self.config["habitat_baselines"]["rl"]["ddppo"]["train_encoder"] else ["visual_encoder.*"]
```

```
test_frozen[transformer_rope_frozen-cpu] PASSED
test_frozen[gru_finetune-cpu] SKIPPED (nothing is frozen in this configuration)
```

- `configs` holds paths (`.yaml`/`.yml` need PyYAML, `.json`) or Python objects; test ids
  use the file name (`config0`, ... for objects). An empty list is an error, so a wrong
  glob can't silently run nothing.
- Override `load_config(self, source)` to compose with Hydra or OmegaConf instead, and
  `config_id(cls, source)` for other ids. A `config_overrides` key that doesn't exist in a
  config fails with the key's name.
- `self.config` is set for generated tests and for your own: request the `model_config`
  fixture in a hand-written test method to run it once per config too.
- With configs, golden files are named `<ClassName>[<config>].pt`, one per config.

## Customizing

Generated tests are ordinary methods: override one to change it, or add your own next to
them. `make_model()` runs after `torch_seed`, so weights are reproducible;
`make_inputs(make_tensor)` gets the seeded, device-aware tensor factory.
