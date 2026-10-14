# Test any model with no code

Point pytest-torch at a model and it builds it, infers example inputs, and runs a suite of
checks on every available device. No test file needed.

```bash
pytest --torch-model=hf:google-bert/bert-base-uncased
pytest --torch-model=timm:resnet50 --torch-model=torchvision:mobilenet_v3_small
pytest --torch-model=mypackage.models:Classifier
pytest --torch-model=path/to/models.py:build_model
```

```
pytest-torch: testing hf:google-bert/bert-base-uncased (random-init); checks: contract, invariance, checkpoint, gradients
torch-model::test_builds[hf:google-bert/bert-base-uncased-cpu] PASSED
torch-model::test_inference_contract[hf:google-bert/bert-base-uncased-cpu] PASSED
torch-model::test_invariance[hf:google-bert/bert-base-uncased-cpu] PASSED
torch-model::test_checkpoint_roundtrip[hf:google-bert/bert-base-uncased-cpu] PASSED
torch-model::test_gradients_finite[hf:google-bert/bert-base-uncased-cpu] PASSED
```

## Model specs

| Spec | Builds | Inputs |
|---|---|---|
| `hf:NAME` or `hf:NAME@AutoModelForX` | `transformers.AutoModel` (or the given class) from the config: random weights, only `config.json` is downloaded. `NAME` can be a local folder | Text models: `input_ids` with a padded `attention_mask` (plus `decoder_input_ids` for encoder-decoders, `use_cache=False`). Vision models: `pixel_values` of the configured size |
| `timm:NAME` | `timm.create_model(NAME)`, random weights | Images of the model's own `input_size` |
| `torchvision:NAME` | `torchvision.models.get_model(NAME)`, classification and segmentation models | 224×224 images (299 for Inception) |
| `package.module:Factory` or `path/to/file.py:Factory` | `Factory()` (a class or function returning an `nn.Module`), or an `nn.Module` object | The model's `example_inputs(batch_size)` if it has one; otherwise inferred from the first layer: `Linear`, `Conv1d/2d/3d`, `Embedding`, a batch-first RNN or transformer layer |

`--torch-model-pretrained` loads pretrained weights instead (downloads them). Specs are
repeatable; every model gets its own tests.

## Checks

| Check | Test | What it runs |
|---|---|---|
| `contract` *(default)* | `test_inference_contract` | [`check_inference_contract`](inference.md#inference-contract): eval mode, finite, deterministic, batch invariance, devices, no state mutation |
| `invariance` *(default)* | `test_invariance` | Text models: padding and mask invariance. Every model: memory-layout invariance |
| `checkpoint` *(default)* | `test_checkpoint_roundtrip` | [`assert_checkpoint_roundtrip`](checkpoints.md) |
| `gradients` *(default)* | `test_gradients_finite` | `assert_gradients_finite` with the sum of the outputs' means as the loss |
| `export` | `test_export_parity` | [`assert_export_parity`](deployment.md#torchexport) |
| `compile` | `test_compile_parity` | `assert_compile_parity` (eager backend) |

Select with `--torch-model-checks=contract,export`. `test_builds` always runs: it builds the
model, runs it on the inferred inputs, and if that fails, says why and how to fix it; the
model's other tests then skip with a pointer to it. Comparisons use `rtol=atol=1e-4`, since
zoo models are large.

## When inputs can't be inferred

```
mymodels:Mystery: can't infer inputs for Mystery. Give the model an `example_inputs(self, batch_size)`
method, or pass --torch-model-inputs (e.g. 4x3x32x32 or input_ids=long:4x16:1000,attention_mask=mask:4x16)
```

Either add a method to the model:

```python
class Mystery(nn.Module):
    def example_inputs(self, batch_size):
        return torch.randn(batch_size, 7, 7)
```

or describe the inputs on the command line: comma-separated tensors, each
`[name=][dtype:]AxBxC[:high]`.

| `--torch-model-inputs=` | Means |
|---|---|
| `4x3x32x32` | one float tensor, normal(0, 1) |
| `long:4x16:1000` | integers in `[0, 1000)` |
| `long:4x8:5,8x3` | two positional arguments |
| `input_ids=long:4x16:1000,attention_mask=mask:4x16` | keyword arguments; `mask` is an attention mask with padded rows, and makes the padding and mask checks run |

## With your own tests

Without file arguments, only the model's tests run. With paths, both run:
`pytest tests/ --torch-model=mypackage.models:Classifier`. Your `pyproject.toml` applies
either way (tolerances, markers, options). For anything beyond these checks (an output
spec, training checks with a real loss, per-config suites), write a
[`ModelSuite`](model-suite.md).
