# Failure reproduction

A test that fails only in CI, only on the A100 runner, only on Tuesdays: record what it
saw and replay it locally.

```bash
pytest --torch-repro-dir=repro          # each failing test writes repro/<test>.pt
```

The failure report ends with:

```
bundle: repro/tests_test_model.py_test_contract_cuda.pt
replay: pytest --torch-repro=repro/tests_test_model.py_test_contract_cuda.pt
```

```bash
pytest --torch-repro=repro/<test>.pt                    # same test, seed and inputs
pytest --torch-repro=repro/<test>.pt --torch-device=cpu # a GPU failure, replayed on the CPU
```

The replay runs only that test (its parametrization included), with the recorded seed,
and `make_tensor` returns exactly the recorded tensors, so platform or version RNG
differences don't matter. The header lists environment differences (torch, Python,
platform) between the recording and the replay.

## What a bundle contains

- test id, parameters, seed and device;
- every tensor `make_tensor` produced, in order;
- values passed to `record_repro(name, value)`;
- model weights passed to `record_repro`, up to `--torch-repro-max-weights-mb` (default 0:
  never; models built under `torch_seed` are reproducible from the seed);
- active tolerances, environment (versions, platform, GPU name, relevant env vars) and the
  failure message.

```python
from pytest_torch import record_repro

def test_scores(torch_seed, make_tensor):
    model = record_repro("model", build_model())      # weights if small enough
    batch = record_repro("batch", next(iter(loader)))  # on replay: the recorded batch
```

`ModelSuite` records its model and inputs automatically. Bundles are plain torch files
(`load_repro(path)` or `torch.load(path, weights_only=True)`) for inspection in a notebook.
Inputs from `make_tensor` are synthetic; anything you pass to `record_repro` is stored as
is, so mind real data before uploading bundles as CI artifacts.

In CI, upload the directory on failure:

```yaml
- run: pytest --torch-repro-dir=repro
- uses: actions/upload-artifact@v4
  if: failure()
  with: { name: repro, path: repro }
```
