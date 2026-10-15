# Tolerances

Every comparison that doesn't pass `rtol`/`atol` itself uses a default. Set those defaults
once for the whole project instead of repeating them in every test.

## In `pyproject.toml`

```toml
[tool.pytest_torch.tolerance]
bfloat16 = { rtol = 2e-2, atol = 1e-2 }          # every comparison of bf16 tensors
"cuda.float32" = { rtol = 1e-4, atol = 1e-5 }    # float32, only when testing on CUDA (TF32)
compile_parity = { rtol = 1e-3, atol = 1e-3 }    # default of assert_compile_parity
device_parity = { atol = 1e-3 }                  # the contract's cross-device check; rtol stays built-in
```

A key is one of:

| Key | Applies to |
|---|---|
| a dtype: `float64`, `float32`, `float16`, `bfloat16`, `complex64`, `complex128`, `float8_*` | any comparison of tensors of that dtype: `assert_tensors_close`, `assert_outputs_close`, golden files, invariance and batch checks, the contract's same-device checks, `assert_matches_reference` |
| a check: `compile_parity`, `export_parity`, `aoti_parity`, `torchscript_parity`, `onnx_parity`, `autocast_parity`, `device_parity`, `dtype_parity`, `divergence`, `gradcheck`, `checkpoint`, `layout_invariance`, `streaming`, `training_resume`, `grad_accumulation` | the default of that assertion (or contract check) |
| `<device>.<dtype or check>`, device one of `cpu`, `cuda`, `xpu`, `mps` | the same, only when the test runs on that device |

An entry may set only `rtol` or only `atol`; the other value comes from the next layer.
Mistakes (an unknown key or device, a negative or non-numeric value) stop pytest at startup
with a message saying what is wrong.

## For one test, class or module

```python
@pytest.mark.torch_tolerance(bfloat16=(5e-2, 1e-2), compile_parity={"atol": 1e-3})
def test_low_precision(model, make_tensor):
    ...

@pytest.mark.torch_tolerance(**{"cuda.float32": (1e-3, 1e-4)})   # device keys need **
class TestOnAmpere:
    ...

pytestmark = pytest.mark.torch_tolerance(float16=(1e-2, 1e-3))   # the whole module
```

Values are `(rtol, atol)` or `{"rtol": ..., "atol": ...}`. Markers on the module, the class
and the test all apply; the closest one wins for the keys they share.

Outside pytest (or for one block), use the context manager:

```python
from pytest_torch import tolerance_overrides

with tolerance_overrides(bfloat16=(5e-2, 1e-2)):
    assert_outputs_close(out_bf16, reference)
```

## Resolution order

For each of `rtol` and `atol` separately:

1. the argument passed to the assertion;
2. `@pytest.mark.torch_tolerance` (closest marker first) or `tolerance_overrides`;
3. `[tool.pytest_torch.tolerance]` in `pyproject.toml`;
4. the built-in default.

Within steps 2 and 3, a `<device>.` entry beats the generic one. The device is the test's
`torch_device` when it has one, even if outputs were moved to the CPU before comparing;
otherwise it is the device of the tensors or model being compared.

Failure messages always print the tolerance that was used, e.g.
`(rtol=0.02, atol=0.01)`, so you can tell which layer applied.

## Built-in defaults

| Dtype | rtol | atol |
|---|---|---|
| `float64`, `complex128` | 1e-7 | 1e-7 |
| `float32`, `complex64` (and other dtypes) | 1.3e-6 | 1e-5 |
| `float16` | 1e-3 | 1e-5 |
| `bfloat16` | 1.6e-2 | 1e-5 |

| Check | rtol | atol |
|---|---|---|
| `compile_parity`, `aoti_parity`, `device_parity`, `divergence` | 1e-4 | 1e-4 |
| `export_parity`, `torchscript_parity` | 1e-5 | 1e-5 |
| `onnx_parity` | 1e-3 | 1e-4 |
| `autocast_parity`, `dtype_parity` | 2e-2 | 2e-2 |
| `gradcheck` | 1e-3 | 1e-5 |
