# Fixtures

| Fixture | What it gives you |
|---|---|
| `torch_device` | Parametrizes the test over the available devices (skips unavailable ones) |
| `torch_seed` | Seeds `random`, `numpy` and `torch`; yields the seed, printed when the test fails |
| `torch_generator` | A seeded **CPU** `Generator`: the same seed gives identical values on every device |
| `make_tensor` | Reproducible tensors, generated on the CPU and moved to the device |
| `inference_mode` | Runs the test under `torch.inference_mode()` |
| `golden` | Snapshot testing of tensors and nested outputs |
| `latency` | Median latency with warmup and device sync, with an optional budget marker |
| `perf_baseline` | Latency/memory regression checks against a stored baseline |
| `model_loader` | Session-cached Hugging Face / timm / torchvision loading |

## `make_tensor`

```python
x = make_tensor(4, 16)                                   # normal(0, 1), float32
ids = make_tensor(4, 8, dtype=torch.long, low=1, high=50)
u = make_tensor(4, 16, kind="uniform", low=0, high=1)
```

## Seeds

Priority: `@pytest.mark.torch_seed(n)` > `--torch-seed-mode=per-test` > `--torch-seed`.
