# Inference correctness

## Assertions

```python
from pytest_torch import (
    assert_tensors_close, assert_outputs_close, assert_finite, assert_batch_invariant,
)

assert_tensors_close(actual, expected, rtol=1e-4)      # dtype-aware default tolerances
assert_outputs_close(out, reference_out)               # nested outputs, reports the path of each mismatch
assert_finite(out, "output")                           # NaN/Inf anywhere in the structure
assert_batch_invariant(model, batch)                   # model(batch)[i] == model(batch[i:i+1])[0]
```

Failures say how many elements differ, where the worst one is, and the absolute and
relative error:

```
AssertionError: Mask invariance violated (masked tokens leak): 1 problem(s)
  output['pooled']: 11/24 elements differ beyond tolerance (rtol=1.3e-06, atol=1e-05)
    max abs error: 2.1e-01 at index (1, 5) (actual=..., expected=...)
```

## Inference contract

One call runs the checks every inference model should satisfy and explains failures:

```python
report = check_inference_contract(model, x, raise_on_failure=False)
print(report)
```

```
Inference contract: 4 passed, 1 failed, 2 skipped
  [PASS] eval_mode
  [PASS] finite_outputs
  [PASS] deterministic
  [FAIL] batch_invariance
         per-sample results differ from the batched run
         hint: Output depends on batch composition: check batch norm in training mode, ...
  [SKIP] device_parity (no other device available)
  [SKIP] dtype_parity (no dtypes requested, e.g. dtypes=(torch.bfloat16,))
  [PASS] no_state_mutation
```

Select checks with `checks=[...]` / `skip=[...]`; tune tolerances with `rtol`,
`device_rtol`, `dtype_rtol`, ...

## Find the first diverging layer

```python
from pytest_torch import localize_divergence

report = localize_divergence(model_cpu, model_gpu, x)
report.raise_if_diverged()
```

```
Divergence: 3 of 12 layers exceed tolerance (rtol=0.0001, atol=0.0001). First: 'encoder.1' (LayerNorm)
  layer       type        max abs   max rel  inputs
  encoder.1   LayerNorm  3.2e-02   1.1e-01  ok
The first diverging layer receives matching inputs: the error originates inside it.
```

`inputs=ok` means the error started *inside* that layer; `inputs=DIFF` means it entered
between layers (a functional op or residual add). Use `module_map` when layer names differ
between the two models.

## Invariance and metamorphic tests

```python
from pytest_torch import (
    assert_padding_invariant, assert_mask_invariant,
    assert_permutation_invariant, assert_permutation_equivariant,
    assert_invariant, assert_equivariant,
)

assert_padding_invariant(model, batch, extra=4, side="right")  # padding must not change real tokens
assert_mask_invariant(model, batch)                            # masked ids must not leak
assert_permutation_invariant(set_model, points, dim=1)         # order of a set must not matter
assert_permutation_equivariant(per_token_model, tokens)        # permuted in -> permuted out
assert_invariant(model, x, lambda t: -t, name="sign flip")     # any custom relation
```

`batch` is a dict like `{"input_ids": ..., "attention_mask": ...}`. Sequence-aligned
outputs are compared only on real tokens; pooled outputs are compared as is.

## Golden files

```python
@pytest.mark.torch_golden("classifier_v1")
def test_matches_golden(model, make_tensor, golden):
    golden.check(model(make_tensor(8, 16)), atol=1e-4)
```

```bash
pytest --torch-update-golden --torch-device=cpu    # (re)generate a portable reference on CPU
```

A missing golden file **fails**: CI must never pass vacuously.

## Hypothesis strategies

```python
from hypothesis import given, settings
from pytest_torch.strategies import tensors

MODEL = build_model().eval()

@settings(max_examples=50, deadline=None)
@given(x=tensors(shape=(4, 16)))              # zeros, constants, extremes, tiny values, ...
def test_finite_on_edge_cases(x):
    assert torch.isfinite(MODEL(x)).all()
```

Build models at module scope: Hypothesis rejects function-scoped fixtures in `@given`
tests by default.
