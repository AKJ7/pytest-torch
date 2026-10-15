# Training loops

[Training sanity](training.md) checks a model with a standard loop. This page tests **your**
training loop: your training step with its custom losses, several optimizers, gradient
accumulation, clipping and mixed precision, and your checkpoint code. Each check runs a few
steps from fresh, seeded setups and explains what went wrong.

| Check | Catches |
|---|---|
| [`assert_optimizer_coverage`](#optimizer-coverage) | Trainable parameters left out of every optimizer or given to several; an optimizer built for another copy of the model |
| [`run_training`](#short-runs-with-telemetry) → `run.assert_healthy()` | Non-finite losses or gradients, exploding gradients, skipped updates, unexpected learning rates, a loss that doesn't decrease |
| [`assert_resume_equivalent`](#checkpoint-and-resume) | Optimizer, scheduler, GradScaler or RNG state missing from a checkpoint |
| [`assert_accumulation_equivalent`](#gradient-accumulation) | A loss not divided by the accumulation steps, gradients not reset, optimizer steps at the wrong micro-batch |
| [`assert_mixed_precision_training`](#mixed-precision) | `optimizer.step()` bypassing the GradScaler, a missing `scaler.update()`, clipping before unscaling, overflows |

## Setup and step

Two functions describe your training code. `make_setup()` builds everything a step needs
and returns a `TrainingSetup` (or a `(model, optimizer[, scheduler])` tuple). It may take
the device as its only argument.

```python
from pytest_torch import TrainingSetup


def make_setup():
    model = Classifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / 4))
    return TrainingSetup(model, optimizer, schedulers=warmup)
```

`TrainingSetup(model, optimizers, schedulers=None, scaler=None, state={})` takes one
optimizer or scheduler, a list or a dict of named ones. `state` holds other objects a
checkpoint must carry, such as an EMA model (anything with `state_dict()` and
`load_state_dict()`). In the step, `setup.optimizer` and `setup.scheduler` are the only
optimizer and scheduler; with several, use `setup.optimizers["name"]`.

The step is your training step, `step(setup, batch)`. It does the forward and backward
passes and the optimizer and scheduler steps, and returns the loss: a tensor, a float, or a
dict of named losses (the `"loss"` entry, or else the first, is the main one).
`setup.step_index` is the index of the current step.

```python
def train_step(setup, batch):
    x, y = batch
    loss = F.cross_entropy(setup.model(x), y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(setup.model.parameters(), 1.0)
    setup.optimizer.step()
    setup.optimizer.zero_grad()
    setup.scheduler.step()
    return loss
```

No step function of your own? Pass `loss_fn=lambda model, batch: ...` instead, and
`standard_step` does the rest: accumulation, clipping, a GradScaler when the setup has one.

`data` is one batch, a list of batches (cycled), an iterable such as a `DataLoader` (its
first batches), or a function from the step index to a batch. Batches are moved to the
model's device.

## The `training_runner` fixture

Every function on this page is also a method of the `training_runner` fixture, which runs it
on the test's `torch_device` with the test's `torch_seed`:

```python
def test_training(training_runner):
    run = training_runner.run(make_setup, batches, train_step, steps=20)
    run.assert_healthy(max_grad_norm=50.0, min_relative_decrease=0.3)
```

Outside pytest, call the functions with `device=` and `seed=`.

## Optimizer coverage

```python
from pytest_torch import assert_optimizer_coverage

setup = make_setup()
assert_optimizer_coverage(setup.model, setup.optimizers)
```

```
Optimizer coverage: 5 problem(s):
  encoder.0.weight: trainable but in no optimizer, so never updated
  ...
  'optimizer': 4 tensor(s) that aren't parameters of the model (e.g. shape (16, 8)): the optimizer was built for another copy of the model
```

It fails on trainable parameters that no optimizer updates, parameters in several optimizers
or parameter groups (updated twice per step), and optimizer tensors that aren't the model's
parameters: an optimizer built before a `deepcopy`, re-creation or reload of the model.
Frozen parameters inside an optimizer are allowed unless `allow_frozen=False`; `ignore=`
takes fnmatch patterns of parameter names. Runs from `run_training` check coverage before
training too, and `run.assert_healthy()` reports the problems.

## Short runs with telemetry

```python
run = run_training(make_setup, batches, train_step, steps=20)
run.assert_healthy(max_grad_norm=50.0, min_relative_decrease=0.3)
run.assert_lr(lambda step: 3e-3 * min(1.0, (step + 1) / 4))
print(run.table())
```

Each optimizer's `step` is wrapped for the duration of the run, so every step is recorded
without changing your code:

```
  step  loss    grad norm  update    lr               opt steps
  0     1.433   1.212      0.03406   0.00075,0.00075  1
  1     1.398   1.087      0.06118   0.0015,0.0015    1
  2     1.302   1.154      0.08533   0.00225,0.00225  1
  ...
```

| `run.records[i]` (`StepRecord`) | Meaning |
|---|---|
| `losses`, `loss` | What the step returned; `loss` is the main one |
| `grad_norm[opt]` | Total L2 norm of the gradients when `opt.step()` ran |
| `update_norm[opt]` | L2 norm of the parameter change made by the step |
| `lrs[opt]` | Learning rate of each parameter group, as used by the step |
| `optimizer_steps[opt]`, `skipped` | How many times each optimizer stepped; `skipped` when none did |
| `nonfinite_grads` | Parameters with NaN or Inf gradients at a step |
| `loss_scale` | The GradScaler's scale after the step |

| Assertion | Fails when |
|---|---|
| `assert_finite()` | A loss, a gradient applied by a step, or a final parameter isn't finite |
| `assert_no_skipped_updates(every=1, max_skipped=0)` | A step (every `every`-th with accumulation) doesn't step any optimizer, or an optimizer steps without changing the parameters (learning rate 0, gradients `None`) |
| `assert_grad_norm_below(max_norm)` | A gradient norm exceeds `max_norm` |
| `assert_lr(expected, optimizer=None, group=0)` | A learning rate differs from `expected(step)`, or from `expected[i]` for a list |
| `assert_loss_decreases(min_relative_decrease=0.1)` | The mean loss of the last quarter of the steps isn't that much below the first quarter's |
| `assert_healthy(max_grad_norm=None, min_relative_decrease=None)` | Coverage, `assert_finite`, `assert_no_skipped_updates`, and the bounds you pass |

Failures include the telemetry table around the failing steps. Every assertion returns the
run, so they chain. `track_updates=False` skips copying the parameters before each step
(no `update_norm`), for models too big to copy.

## Checkpoint and resume

```python
from pytest_torch import assert_resume_equivalent

assert_resume_equivalent(make_setup, batches, train_step, steps=8,
                         save=save_checkpoint, load=load_checkpoint)
```

This trains 8 steps without interruption, saving a checkpoint after 4 (`resume_at`) with
your `save(setup, path)`. A fresh setup, seeded differently, then restores it with your
`load(setup, path)` and trains the last 4 steps. The losses and final weights must match the
uninterrupted run. To explain a difference, the state right after loading is compared with
the state when saving:

```
Resuming from a checkpoint saved after step 4 does not reproduce the uninterrupted run:
  first divergence at step 5: loss 1.43516 after resuming, 1.40454 without interruption
  final weights: 6 tensor(s) differ (...)
State after loading vs when saving:
  optimizer 'optimizer': state differs (e.g. state[0]['step'] missing): optimizer state not restored, so momentum, Adam moments and step counts restart
  optimizer 'optimizer' group 0: lr 0.00075 after loading, 0.003 when saved
```

Missing model weights, optimizer state, learning rates, scheduler, GradScaler or
`setup.state` fail even when the losses still match. Random number generators that weren't
restored are reported when the runs diverge; without randomness in training they don't
matter and don't fail. When everything in the checkpoint matches but the runs still
diverge, the message points at what lives outside it: a stateful data pipeline, plain
Python attributes, nondeterministic kernels.

`save_training_state(setup, path)` and `load_training_state(setup, path)` save and restore
everything, RNG states included; they're the defaults and a reference implementation.
`training_state(setup)` returns the dict they save. The function returns a `ResumeReport`
(`continuous`, `resumed`, `state_differences`, `diverged_at`). The comparison tolerance is
the `training_resume` check's (`rtol=1e-5`, `atol=1e-6`), configurable in
[Tolerances](tolerances.md).

## Gradient accumulation

```python
from pytest_torch import assert_accumulation_equivalent

assert_accumulation_equivalent(
    make_setup, batches, lambda k: Trainer(accumulation_steps=k).step, accumulation_steps=4,
)
```

`make_step(k)` returns your step configured for `k` accumulation steps (or pass `loss_fn=`
to test `standard_step`). From identical fresh setups, one step with `make_step(1)` on a
full batch is compared with 4 steps of `make_step(4)` on its micro-batches, for 2 batches
(`cycles`). The reference starts every step from zeroed gradients. The check fails when:

- an optimizer steps before the last micro-batch, or doesn't step after it;
- the accumulated gradients differ from the full-batch gradients. It says when they are
  `k` times too large (the loss isn't divided by the accumulation steps), `1/k` (divided
  twice), or carry over from the previous cycle (no `zero_grad()` after the step);
- the weights after a cycle differ although the gradients match (clipping or scheduler
  steps per micro-batch).

```
Gradient accumulation over 4 micro-batches differs from one full-batch step (cycle 1 of 2):
  head.weight: 45/48 elements differ beyond tolerance (rtol=0.0001, atol=1e-06)
  ...
Likely cause:
  gradients are 4x the full-batch gradients: the loss isn't divided by accumulation_steps
```

Batches are split along the first dimension (`split_batch`; pass `split=` for other
layouts), so the batch size must be divisible by `accumulation_steps`. Dropout draws
different masks for a split batch, so dropout modules are disabled in both runs
(`disable_dropout=False` keeps them). BatchNorm can't match exactly, since it normalizes each
micro-batch with that micro-batch's statistics: the failure says so. Raise the tolerance
(`grad_accumulation` check, default `rtol=1e-4`, `atol=1e-6`) or test with
GroupNorm/LayerNorm. A clipping threshold that is reached also breaks the equivalence: test
with a high threshold, as the example below does.

## Mixed precision

```python
from pytest_torch import assert_mixed_precision_training

def amp_step(setup, batch):
    x, y = batch
    with torch.autocast("cuda", dtype=torch.float16):
        loss = F.cross_entropy(setup.model(x), y)
    setup.scaler.scale(loss).backward()
    setup.scaler.unscale_(setup.optimizer)
    torch.nn.utils.clip_grad_norm_(setup.model.parameters(), 1.0)
    setup.scaler.step(setup.optimizer)
    setup.scaler.update()
    setup.optimizer.zero_grad()
    return loss

run = assert_mixed_precision_training(make_amp_setup, batches, amp_step, steps=20, device="cuda")
```

With a GradScaler in `setup.scaler`, this fails when:

- `optimizer.step()` is called directly instead of `scaler.step(optimizer)`, applying
  scaled gradients without the inf/NaN check;
- `scaler.step()` runs without `scaler.scale(loss)` or without `scaler.update()`;
- `torch.nn.utils.clip_grad_norm_` or `clip_grad_value_` runs before
  `scaler.unscale_(optimizer)`, so it clips the scaled gradients;
- more than `max_skipped_fraction` (default 25%) of the steps are skipped after overflows.

With or without a scaler (bfloat16 needs none), it also fails on a non-finite loss (the
forward pass overflowing), non-finite gradients reaching an optimizer step, and non-finite
weights after training. When torch itself raises on a misuse, the failure lists that error
next to what was recorded before it.

```
Mixed-precision training: 1 problem(s):
  clip_grad_norm_() before scaler.unscale_() for optimizer 'optimizer': it clips the scaled gradients, so the threshold is off by the loss scale (steps 0, 1, 2, 3)
Telemetry:
  step  loss    grad norm  update     lr   opt steps  scale
  0     0.9113  0.003906   0.0003906  0.1  1          256
  ...
```

Clipping is detected when called as `torch.nn.utils.clip_grad_norm_(...)` (or `nn.utils.`);
a function imported with `from torch.nn.utils import clip_grad_norm_` before the check runs
isn't seen. A GradScaler for the CPU (`torch.amp.GradScaler("cpu")`) needs torch 2.3+.

## Example

[`examples/test_training_loops.py`](https://github.com/AKJ7/pytest-torch/blob/main/examples/test_training_loops.py)
tests a classifier's training code, with parameter groups, warmup, accumulation, clipping
and checkpoints, using each check on this page. Its "Try" comment lists one-line bugs that
make a test fail.

Distributed training (synchronization, consistent updates across ranks) isn't covered yet.
