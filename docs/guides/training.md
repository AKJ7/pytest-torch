# Training sanity

Catch a broken training setup in seconds instead of after a day of GPU time. Every helper
takes `(model, inputs, loss_fn)`, where `loss_fn(output)` returns a scalar tensor: close
over your targets.

```python
from torch.nn import functional as F
from pytest_torch import (
    assert_gradients_finite, assert_all_params_receive_grad,
    assert_loss_decreases, assert_overfits_one_batch, assert_gradcheck,
)

def loss_fn(output):
    return F.cross_entropy(output, labels)

assert_gradients_finite(model, x, loss_fn)
assert_all_params_receive_grad(model, x, loss_fn, ignore=["pooler.*"])
assert_loss_decreases(model, x, loss_fn, steps=20)
assert_overfits_one_batch(model, x, loss_fn, steps=200)
assert_gradcheck(MyFunction.apply, small_x)
```

## What each check catches

**`assert_gradients_finite`**: NaN or Inf in the loss or in any parameter's gradient,
listed per parameter. Typical causes: `log(0)`, `sqrt(0)`, normalizing a zero vector, fp16
overflow.

**`assert_all_params_receive_grad`**: parameters that are not connected to the loss
(layers defined but never used, `.detach()` or `.item()` in `forward`, a head the loss
ignores) and gradients that are zero everywhere. Freeze intentionally unused parameters or
pass `ignore=` patterns.

```
AssertionError: 4 trainable parameter(s) receive no useful gradient:
  doc.0.weight: no gradient (not connected to the loss)
  ...
```

**`assert_loss_decreases`**: a few optimizer steps on one batch must lower the loss by
`min_relative_decrease` (default 10%). Fails on a zero learning rate, frozen parameters or
an optimizer that never sees the parameters, and prints the loss curve.

**`assert_overfits_one_batch`**: the classic first debugging step. A model that cannot
memorize one small batch has a bug: misaligned targets, too much regularization, a
disconnected layer. Stops early once the loss reaches `max_final_ratio` (default 5%) of its
initial value, or `target_loss`.

**`assert_gradcheck`**: compares autograd gradients with finite differences, in float64.
Use it for custom `autograd.Function`s and hand-written backward passes. On failure it
shows the mismatching jacobian entries:

```
Analytical (autograd) and numerical (finite-difference) gradients differ:
  jacobian[output_0][input_0]: 2/36 elements differ beyond tolerance (rtol=0.001, atol=1e-05)
    max abs error: 6.000e+00 at index (0, 2, 0, 2) (actual=6, expected=0)
```

## Notes

- `assert_loss_decreases` and `assert_overfits_one_batch` train a deep copy in training
  mode, so the model you pass is never modified. They return a `TrainingRun` with the loss
  curve (`run.losses`, `run.summary()`).
- Pass `optimizer=lambda params: torch.optim.SGD(params, lr=0.1)` to use your optimizer.
- The gradient checks overwrite the model's `.grad` fields.
