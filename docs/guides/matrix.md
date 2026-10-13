# Devices, dtypes and unusual inputs

## The device and dtype matrix

`torch_device` runs a test on every available device (`cpu`, `cuda`, `xpu`, `mps`).
`torch_dtype` adds a dtype dimension:

```python
@pytest.mark.torch_dtypes(torch.float32, torch.bfloat16)
def test_batch_invariance(torch_device, torch_dtype, make_tensor):
    model = Classifier().to(torch_device, torch_dtype).eval()
    assert_batch_invariant(model, make_tensor(8, 16))   # make_tensor follows torch_dtype
```

```
test_batch_invariance[cpu-float32] PASSED
test_batch_invariance[cpu-bfloat16] PASSED
test_batch_invariance[cuda-float32] PASSED
...
```

- The dtypes come from the closest `@pytest.mark.torch_dtypes(...)`, else `--torch-dtype`
  (repeatable), else float32 only, so the matrix stays opt-in.
- Combinations a device can't run at all (e.g. float64 on MPS) are skipped with the reason.
  A single missing kernel (float16 LayerNorm on CPU in older torch) is a real failure.
- `make_tensor`'s default float dtype follows `torch_dtype`; an explicit `dtype=` wins.
- Comparison tolerances follow the dtype of the tensors compared, and are configurable per
  dtype and device in [`pyproject.toml`](tolerances.md).

## Memory layouts

```python
from pytest_torch import assert_layout_invariant

assert_layout_invariant(model, images)   # contiguous == non-contiguous == channels_last
```

Inputs with identical values but a different memory layout must give the same outputs.
Non-contiguous copies are stored like transposed tensors, so `.view()` on them fails just
as in real pipelines; channels_last applies to 4-D inputs. In a `ModelSuite`, add
`"layouts"` to `checks`. `noncontiguous(t)` is available from `pytest_torch.invariance`.

## Hypothesis: edge-case inputs

```python
from pytest_torch.strategies import padded_batches, tensors

@given(batch=padded_batches(vocab_size=30, max_len=12, side="right"))
def test_padding_never_leaks(batch): ...

@given(x=tensors(shape=(2, 3, 16, 16), noncontiguous=True))
def test_any_layout(x): ...
```

- `padded_batches(vocab_size, batch_size, max_len, min_len, pad_id, side)` draws
  `{"input_ids", "attention_mask"}` with ragged lengths, deliberately covering every
  sequence at `min_len`, one full sequence among short ones, no padding at all, and mixes.
- `tensors(..., noncontiguous=True)` also draws the memory layout.
