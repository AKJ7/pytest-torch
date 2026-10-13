# Deployment parity

Does the model you ship behave like the model you tested? Every helper runs the eager model
and the deployment artifact on the same inputs and reports mismatches path by path.

## `torch.compile`

`assert_compile_parity` uses torch.compile's default backend (inductor) unless you pass
`backend=`; `backend="eager"` checks graph capture without needing a C++ compiler.
`assert_no_graph_breaks`, `assert_no_recompiles` and `ModelSuite`'s compile check use the
`eager` backend by default.

```python
from pytest_torch import assert_compile_parity, assert_no_graph_breaks, assert_no_recompiles

assert_compile_parity(model, x, backend="eager")   # checks dynamo capture; no C++ toolchain needed
assert_compile_parity(model, x)                    # full inductor compilation
assert_no_graph_breaks(model, x)                   # one graph, with torch's reason if not
assert_no_recompiles(model, x, x_bigger, max_recompiles=1)
```

## `torch.export`

```python
from pytest_torch import assert_export_parity

program = assert_export_parity(model, x)                                # static shapes
assert_export_parity(model, x, x_small, x_large, dynamic_batch=True)    # one program, any batch size
assert_export_parity(model, x, dynamic_shapes={"x": {0: Dim("batch")}}) # torch.export's own spec
```

`dynamic_batch=True` marks dim 0 of every batch-aligned input as dynamic. Extra positional
inputs must run through the *same* program; without dynamic shapes they fail with a hint.
If export itself fails (data-dependent control flow, `.item()`, unsupported Python), you
get an `AssertionError` with torch's explanation. The `ExportedProgram` is returned so you
can save it or lower it further.

## AOTInductor

```python
from pytest_torch import assert_aoti_parity

assert_aoti_parity(model, x, x_small, dynamic_batch=True)
```

Exports, compiles ahead of time into a `.pt2` package, loads it back and compares: the
artifact a C++ server or edge runtime would load. Needs torch 2.6+ and a C++ compiler
(the test skips otherwise); compilation takes tens of seconds, so keep these tests few, or
mark them `slow`.

## Mixed precision

```python
from pytest_torch import assert_autocast_parity, check_inference_contract

assert_autocast_parity(model, x, dtype=torch.bfloat16)        # fp32 weights, bf16 matmuls (torch.autocast)
check_inference_contract(model, x, checks=["dtype_parity"], dtypes=(torch.bfloat16,))  # whole model cast
```

`torch.autocast` is how mixed precision usually runs; casting the whole model is a stricter
test of a pure low-precision deployment.

## TorchScript and ONNX

```python
from pytest_torch import assert_torchscript_parity, assert_onnx_parity

assert_torchscript_parity(model, x, mode="trace")   # or mode="script"; TorchScript is deprecated upstream
assert_onnx_parity(model, x, export_kwargs={"dynamo": True})   # pin the exporter explicitly
```

Both accept only positional tensor inputs. The ONNX exporter's defaults differ between
torch versions, so pin them with `export_kwargs`. Needs `pip install "pytest-torch[onnx]"`.

## Accuracy budgets and references

```python
from pytest_torch import assert_models_agree, assert_matches_reference, assert_generation_equal

metrics = assert_models_agree(fp32, quantized, x, min_top1=0.99, min_cosine=0.999)
assert_matches_reference(my_bert, hf_bert, batch, select=lambda o: o.last_hidden_state)
assert_generation_equal(my_gpt, hf_gpt, input_ids, max_new_tokens=16, pad_token_id=0)
```

`assert_models_agree` returns the measured metrics so you can log them;
`assert_generation_equal` reports the first diverging token per row.
