# Checkpoints

## Round-trip: saving and reloading preserves predictions

```python
from pytest_torch import assert_checkpoint_roundtrip

def test_roundtrip(torch_seed, torch_device, make_tensor):
    ids = make_tensor(2, 10, dtype=torch.long, low=0, high=100)
    assert_checkpoint_roundtrip(lambda: LanguageModel().to(torch_device), ids)
```

The model's `state_dict` is saved (`format="torch"` or `"safetensors"`), loaded into a
fresh `make_model()` built under a *different* seed, and the outputs must be identical.
Anything not restored from the checkpoint shows up, and the failure names the usual
culprits:

```
Checkpoint round-trip (torch) failed:
  outputs differ after reload: ...
  likely culprits:
    plain tensor attribute (not saved): projection
```

It also catches tied weights that come apart, non-persistent buffers changed after
construction, and (with safetensors) shared tensors that can't be saved as is. Pass
`model=` to check an existing instance, e.g. after fine-tuning.

## Compatibility: old checkpoints still load

```python
from pytest_torch import assert_checkpoint_loads

def test_v1_checkpoint_still_loads():
    model = assert_checkpoint_loads(LanguageModel, "tests/data/lm_v1.pt")
```

Commit a small checkpoint from each released version and keep this test: renaming or
reshaping a layer then fails with *every* incompatibility listed, instead of the first:

```
Checkpoint tests/data/lm_v1.pt does not fit the model:
  missing in checkpoint: encoder.0.weight
  unexpected in checkpoint: block.0.weight
  shape mismatch: head.weight: checkpoint (10, 32), model (12, 32)
```

- The state_dict is found under a common key (`state_dict`, `model_state_dict`, `model`,
  ...), or pass `key=`.
- `strip_prefix="module."` handles DataParallel/DDP checkpoints (the error suggests it).
- The model's own compatibility code runs (`_load_from_state_dict`, load pre-hooks), so a
  shim that renames old keys makes the test pass.
- `strict=False` only fails on shape mismatches. `weights_only=False` loads trusted
  checkpoints that contain pickled objects.

In a `ModelSuite`, add `"checkpoint"` to `checks` for the round-trip.
