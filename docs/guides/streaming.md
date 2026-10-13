# Stateful and streaming models

Models that keep state between calls (streaming audio, RNNs, causal convolutions with a
cache, transformers with a KV cache) have a second code path: the chunked one. It has to
agree with the full pass, a reset has to really reset, and one request must never leak
into another.

```python
from pytest_torch import assert_streaming_equivalent

def test_streaming(denoiser, make_tensor):
    assert_streaming_equivalent(
        full_runner=denoiser.forward,
        streaming_runner=denoiser.forward_chunk,
        inputs=make_tensor(2, 50, 8),          # (batch, time, features)
        chunk_sizes=[1, 7, 32],
        reset=denoiser.reset_cache,
    )
```

```
Streaming: 3 passed, 2 failed
  [PASS] chunked[1]
  [FAIL] chunked[7]
         chunk size 7: streamed output differs from the full pass
           output: 12/400 elements differ beyond tolerance (rtol=1e-05, atol=1e-05)
  ...
  [FAIL] reset
         after a reset, streaming the same input again gave different outputs (state survives the reset)
```
*(abridged, numbers illustrative)*

## The checks

| Check | Passes when |
|---|---|
| `chunked[n]` | Feeding the sequence in chunks of `n` and concatenating the outputs agrees with the full pass, under the chosen equivalence |
| `reset` | After a reset (or a fresh `init_state()`), streaming the same input again gives exactly the same outputs |
| `isolation` | A request isn't contaminated by another one (see below) |

Select with `checks=["chunked", "reset", "isolation"]`; `raise_on_failure=False` returns the
`StreamingReport` instead of raising.

## Two kinds of state

**Internal state**: the model keeps its cache and has a reset method. Pass `reset=`. The
streaming runner is called like a model: `streaming_runner(chunk)` (or with `*args` /
`**kwargs` for tuple and dict inputs). Isolation runs another request, resets, then
checks that the original request gives the same result as on its own.

**Explicit state**: a pure step function, `streaming_runner(chunk, state) -> (output,
new_state)`, and `init_state=` returning an empty state. Isolation is stronger here: two
requests are interleaved chunk by chunk, each with its own state, like a server handling
several streams at once, and each must equal its isolated run. That catches state shared
through a module attribute or a global.

A KV cache is explicit state, so chunked prefill can be tested directly:

```python
def prefill(chunk, past):
    out = lm(chunk, past=past, use_cache=True)
    return out.logits, out.past

assert_streaming_equivalent(lambda ids: lm(ids).logits, prefill, ids,
                            chunk_sizes=[1, 3, 8], init_state=lambda: None)
```

## Inputs and outputs

- Inputs are split along `seq_dim` (default 1, batch-first). Tensors of another length,
  such as a speaker embedding, go to every chunk whole; the sequence length is the longest
  size along `seq_dim`.
- Chunk outputs whose length matches the chunk are concatenated along `seq_dim`; other
  outputs (a pooled summary, a final state) are taken from the last chunk. Pass `combine=`
  to combine the list of chunk outputs yourself.
- The second request for the isolation check is the inputs reversed in time, or
  `other_inputs=`.

## When chunking intentionally changes the output

Some models behave differently when chunked by design: look-ahead, chunk-local attention,
warm-up. Declare the expected equivalence with `equivalence=`:

```python
from pytest_torch import StreamingEquivalence

# each chunk's last 2 frames wait for future context
equivalence=StreamingEquivalence(skip_last=2)

# also: skip_first=, skip_warmup= (start of the whole sequence), rtol=, atol=
```

Or pass a callable `(full_output, streamed_output, chunk_size)` that raises
`AssertionError` (or returns `False`) on a mismatch, or a dict keyed by chunk size, with
`"default"` for the others:

```python
equivalence={
    1: no_future_at_all,                        # 1-frame chunks: a different, known behavior
    "default": StreamingEquivalence(skip_last=2),
}
```

The default tolerance is the `streaming` check's (`1e-5`), configurable in
[`pyproject.toml`](tolerances.md). The reset and isolation checks compare exactly: the same
input and the same starting state must give bit-identical outputs.
