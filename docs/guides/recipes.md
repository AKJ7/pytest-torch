# Recipes

## Test a Hugging Face model

```python
import torch
from pytest_torch import ModelSuite

class TestBert(ModelSuite):
    checks = ("contract", "export")
    check_options = {"inference_contract": {"skip": ["device_parity"]}}  # big model: one device

    def make_model(self):
        from transformers import AutoConfig, AutoModel
        return AutoModel.from_config(AutoConfig.from_pretrained("bert-base-uncased"))

    def make_inputs(self, make_tensor):
        ids = make_tensor(2, 16, dtype=torch.long, low=1, high=1000)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
```

Use the session-cached `model_loader` fixture for pretrained weights in plain tests:
`model_loader.hf("bert-base-uncased", auto_class="AutoModel")`.

## Prove a padding mask works

```python
def test_padding_does_not_leak(model, batch):
    assert_padding_invariant(model, batch)   # appending pad tokens changes nothing
    assert_mask_invariant(model, batch)      # what sits under the mask changes nothing
```

## Check a quantized or pruned model

```python
metrics = assert_models_agree(fp32_model, int8_model, x, min_top1=0.99, min_cosine=0.995)
print(metrics)   # log it to track drift between releases
```

## Port a model from another codebase

```python
assert_matches_reference(my_port, original, x)        # end to end
localize_divergence(original, my_port, x, module_map={"0": "encoder"}).raise_if_diverged()
```

## Verify a hand-written backward

```python
assert_gradcheck(MyAttentionFunction.apply, (q, k, v))   # small shapes: cost grows with size
```

## Keep slow checks out of the inner loop

AOTInductor compilation and long overfitting runs take tens of seconds. Mark them and run
them in CI only:

```python
@pytest.mark.slow
def test_aoti(model, make_tensor):
    assert_aoti_parity(model, make_tensor(8, 16))
```

```bash
pytest -m "not slow"      # locally
pytest -m slow            # nightly or on one CI job
```

## Skip compile tests where dynamo is unavailable

```python
from pytest_torch import is_compile_supported

needs_dynamo = pytest.mark.skipif(not is_compile_supported(), reason="no torch.compile here")
```

`ModelSuite` does this for you.
