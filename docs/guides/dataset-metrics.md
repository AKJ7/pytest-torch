# Dataset metrics

Run a loader through a model, aggregate metrics, and fail when they regress.

```python
def accuracy(output, batch):
    return (output.argmax(-1) == batch[1]).float().mean()

def test_accuracy(model, val_loader, metric_baseline):
    metric_baseline(model, val_loader, {"accuracy": accuracy}, minimum={"accuracy": 0.9})
```

```bash
pytest --torch-update-metrics     # record baselines (.torch-metric-baselines.json); commit it
pytest                            # later runs compare against them
```

## Comparing

- **Baseline:** a metric fails when it gets worse than the recorded value by more than
  `tolerance` (absolute; default 0.01; a number or a dict per metric).
  `higher_is_better` (default `True`, or a dict) sets the direction. Improvements never fail.
- **Thresholds:** `minimum={...}` / `maximum={...}` are checked on every run. A missing
  baseline skips the test unless thresholds are given (they still apply), or
  `--torch-metrics-strict` is set (it fails).
- Baselines are stored per test (not per device), so CPU and GPU runs share them;
  `--torch-metrics-file` changes the file.

## Metrics

- `fn(output, batch) -> scalar`: averaged over the dataset, **weighted by batch size**, so
  an uneven last batch doesn't skew it (accuracy, mean loss, ...).
- An object with `update(output, batch)` and `compute()`: for metrics that need every
  prediction (F1, AUC, ...). Wrap a torchmetrics metric in a few lines.

The loop runs in eval and inference mode on the model's device and restores training mode
afterwards. `inputs=` extracts model inputs from a batch (default: `batch[0]` for tuples,
the dict itself as keyword arguments); `max_batches=` caps the work in CI.

`evaluate(model, loader, metrics)` is the same loop without the baseline, for your own
assertions.
