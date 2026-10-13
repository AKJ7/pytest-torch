# Dataset metrics

Run a loader through a model, aggregate metrics, and fail when they regress, but only
against a baseline that measured the same thing, and only when the change is more than
noise.

```python
def accuracy(output, batch):
    return (output.argmax(-1) == batch[1]).float()   # one value per sample: paired comparisons

def test_accuracy(model, val_loader, metric_baseline):
    metric_baseline(
        model, val_loader, {"accuracy": accuracy},
        context={"dataset": "sst2", "split": "validation", "preprocessing": "v3",
                 "checkpoint": "run-42"},
        minimum={"accuracy": 0.9},
    )
```

```bash
pytest --torch-update-metrics     # record baselines; commit .torch-metric-baselines.json
                                  # and .torch-metric-baselines/ (per-sample values)
pytest                            # later runs compare against them
```

## What a baseline records

| Identifier | Recorded from | When it differs from the baseline |
|---|---|---|
| Dataset, split, preprocessing version, anything else | `context={...}` | **fails**: a different evaluation population isn't comparable |
| The data itself | automatic fingerprint of every batch tensor (inputs and labels), in order | **fails**: catches changed data or preprocessing nobody labelled, e.g. a tokenizer upgrade |
| Sample count and selection | automatic: sample count, `max_batches` | **fails**: a partial evaluation isn't the baseline's |
| Metric configuration | automatic: kind (per-sample, per-batch, update/compute) and the metric's `config` attribute | **fails**: macro vs micro averaging, thresholds, ignored labels |
| Checkpoint | `context["checkpoint"]` and an automatic fingerprint of the weights | **recorded only**: comparing a new checkpoint with the old one's baseline is the point |

Batch size doesn't matter: the same samples in the same order give the same fingerprint
however they are batched. A mismatch fails with every difference listed:

```
Metric regression in 'tests/test_eval.py::test_accuracy':
  the baseline measured something else, so it can't be compared:
    context 'split': baseline 'validation', now 'test'
  baseline: run-41 / ce7d6dfeb4f3d3a9, torch 2.6.0, recorded 2026-10-13; now: run-42 / 9a1b..., torch 2.6.0
  (re-record with --torch-update-metrics if the change is intended)
```

Give `update`/`compute` metrics a `config` attribute so it is recorded:

```python
class MacroF1:
    config = {"average": "macro", "ignore_index": -100}
    def update(self, output, batch): ...
    def compute(self): ...
```

## Paired comparisons with confidence intervals

A metric that returns **one value per sample** (a 1-D tensor the size of the batch) is
stored per sample with the baseline. Later runs are compared *paired*, sample by sample,
with a seeded bootstrap confidence interval of the mean difference. A regression needs
both:

- a mean drop larger than `tolerance`, and
- a confidence interval that excludes zero (the drop is not noise).

```
accuracy regressed from 1 to 0.95: drop +0.05 (95% CI +0.04299 .. +0.057, n=4000, paired); allowed drop 0.01
```

The same 5% drop on 40 samples (2 flipped predictions) passes: its interval,
`+0 .. +0.125`, includes zero. A tiny but real drop within `tolerance` passes too. Options: `confidence=0.95`,
`resamples=2000`. Metrics returning one value per batch, or computed by `update`/`compute`,
use the plain rule: a drop larger than `tolerance` fails.

## Comparing

- `tolerance` (absolute; default 0.01; a number or a dict per metric) and
  `higher_is_better` (default `True`, or a dict) set what counts as worse. Improvements
  never fail.
- `minimum={...}` / `maximum={...}` are checked on every run. A missing baseline skips the
  test unless thresholds are given (they still apply), or `--torch-metrics-strict` is set
  (it fails).
- Baselines are stored per test (not per device), so CPU and GPU runs share them;
  `--torch-metrics-file` changes the file.

## Metrics and the loop

- `fn(output, batch)` returning one value per sample (averaged over samples), or a scalar
  (averaged over batches, weighted by batch size).
- An object with `update(output, batch)` and `compute()`: for metrics that need every
  prediction (F1, AUC, ...).

The loop runs in eval and inference mode on the model's device and restores training mode
afterwards. `inputs=` extracts model inputs from a batch (default: `batch[0]` for tuples,
the dict itself as keyword arguments); `max_batches=` caps the work in CI.

`evaluate(model, loader, metrics)` is the same loop returning the values;
`evaluate_detailed(...)` also returns the per-sample values and the provenance, and
`paired_difference(current, baseline)` is the paired test on its own.
