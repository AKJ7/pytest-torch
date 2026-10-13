# Output specifications

Describe what a model's outputs must look like once, and check it everywhere: shapes,
dtypes, finite values, value ranges and probability normalization.

```python
from pytest_torch import Probabilities, TensorSpec, assert_output_spec

SPEC = {
    "logits": TensorSpec(("B", "T", 5), dtype=torch.float32),
    "probs": Probabilities(("B", "T", 5)),               # in [0, 1], sums to 1 along dim -1
    "confidence": TensorSpec(("B",), min=0.0, max=1.0),
}

def test_outputs(model, make_tensor):
    ids = make_tensor(4, 12, dtype=torch.long, low=0, high=50)
    dims = assert_output_spec(model(ids), SPEC, B=4)    # returns {"B": 4, "T": 12}
```

## Shapes

| Entry | Means |
|---|---|
| `10` | exactly 10 |
| `"B"` | a symbolic dim: every use must have the same size |
| `None` | any size |
| `...` | any number of dims (at most one per shape) |

Symbols are bound by keyword arguments (`B=4`) or by their first occurrence, and a
conflict names where the symbol was bound: `B=7, but B=8 from output['logits'] dim 0`.

## Specs

- **`TensorSpec(shape, dtype, finite=True, min=None, max=None)`**: `dtype` may be a tuple of
  allowed dtypes; `finite=False` allows NaN/Inf.
- **`Probabilities(shape, dim=-1, atol=1e-4, log=False)`**: values in `[0, 1]` summing to 1
  along `dim`; with `log=True`, log-probabilities whose `logsumexp` is 0.

The spec mirrors the output: dicts, lists/tuples, dataclasses, namedtuples and Hugging Face
`ModelOutput`s. Output keys missing from the spec are ignored; `None` in the spec skips an
entry. Every problem is reported, each with its path:

```
Output does not match its spec: 3 problem(s)
  output['logits']: shape (8, 9) vs spec ('B', 10): dim 1 is 9, expected 10
  output['probs']: sum along dim -1 is not 1: 8/8 rows off, worst 2 at (7,) (atol=0.0001)
  output['hidden']: 1280 NaN and 0 Inf values
```

In a `ModelSuite`, set `output_spec = {...}` and add `"output_spec"` to `checks`; `"B"` is
bound to the batch size of `make_inputs()`. Specs also pair well with Hypothesis: check
the spec on every generated input.
