# Fixtures

The plugin provides these fixtures in every test module; there is nothing to import.

| Fixture | What it gives you |
|---|---|
| [`torch_device`](#torch_device) | Parametrizes the test over the available devices: `cpu`, `cuda`, `xpu`, `mps` (skips unavailable ones) |
| [`torch_seed`](#torch_seed) | Seeds `random`, `numpy` and `torch`; yields the seed, printed when the test fails |
| [`torch_generator`](#torch_generator) | A seeded **CPU** `Generator`: the same seed gives identical values on every device |
| [`make_tensor`](#make_tensor) | Reproducible tensors, generated on the CPU and moved to the device |
| [`make_batch`](#make_batch) | Realistic batches: padded token sequences with attention masks, images, seeded data loaders |
| [`torch_dtype`](#torch_dtype) | Parametrizes the test over dtypes (default float32; `--torch-dtype`, `@pytest.mark.torch_dtypes`) |
| [`inference_mode`](#inference_mode) | Runs the test under `torch.inference_mode()` |
| [`golden`](#golden) | Snapshot testing of tensors and nested outputs |
| [`latency`](#latency) | Median latency with warmup and device sync, with an optional budget marker |
| [`perf_baseline`](#perf_baseline) | Latency/memory regression checks against a stored baseline |
| [`metric_baseline`](#metric_baseline) | Dataset metrics compared with a stored baseline and thresholds |
| [`model_loader`](#model_loader) | Session-cached Hugging Face / timm / torchvision loading, and tiny random-init versions of Hugging Face architectures |
| [`training_runner`](#training_runner) | Short seeded training runs of your own training step, with telemetry, and resume, accumulation and mixed-precision checks |
| [`model_config`](#model_config) | The current config when a test class has `configs` (e.g. a `ModelSuite` run per config) |

The examples below assume `import pytest`, `import torch` and `from torch import nn`.

## `torch_device`

```python
def test_runs_on_every_device(torch_device):
    model = nn.Linear(16, 4).to(torch_device).eval()
    x = torch.randn(2, 16, device=torch_device)
    assert model(x).device.type == torch.device(torch_device).type
```

The test runs once per available device: `test_runs_on_every_device[cpu]`,
`[cuda]`, `[xpu]`, `[mps]`. Pin devices with `--torch-device=cpu` (repeatable). Under
pytest-xdist, `"cuda"` and `"xpu"` become the worker's own GPU (`cuda:1`, ...) while test ids
stay the same.

## `torch_seed`

```python
def test_reproducible_weights(torch_seed):
    first = nn.Linear(16, 4).weight            # random init, seeded by the fixture
    torch.manual_seed(torch_seed)
    assert torch.equal(first, nn.Linear(16, 4).weight)


@pytest.mark.torch_seed(1234)
def test_fixed_seed(torch_seed):
    assert torch_seed == 1234
```

Seeds `random`, `numpy` (if installed) and `torch` on every device, and yields the seed,
which the report of a failing test prints. Priority: `@pytest.mark.torch_seed(n)` >
`--torch-seed-mode=per-test` (a stable seed derived from the test id) > `--torch-seed`
(default 0). `--torch-deterministic` also turns on `torch.use_deterministic_algorithms`,
and `--torch-autoseed` seeds tests that don't request the fixture.

## `torch_generator`

```python
def test_random_mask(torch_generator, torch_device):
    mask = (torch.rand(4, 16, generator=torch_generator) > 0.5).to(torch_device)
    assert mask.shape == (4, 16)
```

A `torch.Generator` on the **CPU**, seeded with `torch_seed`. Drawing on the CPU and then
moving to the device gives identical values on every device, so cross-device comparisons
are fair. `make_tensor` uses it.

## `make_tensor`

```python
def test_inputs(make_tensor, torch_device):
    x = make_tensor(4, 16)                                     # normal(0, 1), float32, on torch_device
    ids = make_tensor(4, 8, dtype=torch.long, low=1, high=50)  # integers in [1, 50)
    u = make_tensor(4, 16, kind="uniform", low=0, high=1)      # uniform in [0, 1)
    flags = make_tensor(4, dtype=torch.bool)
```

`make_tensor(*shape, dtype=None, kind=None, low=None, high=None)`. `kind` is `"normal"`
(floats, the default), `"uniform"` (default range `[-1, 1)`) or `"randint"` (integers, the
default; `high` defaults to 100, or 2 for bools). The default float dtype is float32, or
`torch_dtype` when the test uses it. Under `--torch-repro-dir`, every tensor is recorded in
the failure bundle, and `--torch-repro` replays them bit for bit.

## `make_batch`

```python
def test_mean_pooling_ignores_padding(make_batch, torch_device):
    embed = nn.Embedding(1000, 16).to(torch_device)
    batch = make_batch.tokens(4, 12, vocab_size=1000, padding="right")
    ids, mask = batch["input_ids"], batch["attention_mask"]
    assert mask[0].all() and not mask[-1].all()       # always a full row and a padded row
    pooled = (embed(ids) * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
    unpadded = ids[-1:, : int(mask[-1].sum())]        # the last row without its padding
    torch.testing.assert_close(pooled[-1:], embed(unpadded).mean(1))


def test_images_and_loader(make_batch):
    images, labels = make_batch.images(8, 3, 32, labels=10)      # (8, 3, 32, 32) floats, labels < 10
    raw = make_batch.images(2, 3, 64, uint8=True, layout="nhwc")  # (2, 64, 64, 3) pixels in [0, 255]
    batches = list(make_batch.loader((images, labels), batch_size=4, shuffle=True))
    assert len(batches) == 2 and raw.dtype == torch.uint8
```

`tokens(batch_size, seq_len, vocab_size=1000, padding="right", min_length=None,
pad_token_id=0, labels=None, token_type_ids=False)` returns `input_ids` and
`attention_mask` (a dict, as Hugging Face models take them). Lengths vary between
`min_length` (default half of `seq_len`) and `seq_len`; the first row is always full and the
last always `min_length` long, so every batch exercises padding. Padding goes `"right"`,
`"left"` or nowhere (`None`); real tokens never equal `pad_token_id`. `labels=10` adds a
class per row; `labels="lm"` adds the ids with padding set to -100.

`images(batch_size, channels=3, height=224, width=None, uint8=False, layout="nchw",
dtype=None, labels=None)` returns normalized floats (about N(0, 1)) or, with `uint8=True`,
raw pixels; `layout="nhwc"` matches decoded images. `loader(data, batch_size=8,
shuffle=False, **kwargs)` wraps a dataset, a tensor, a tuple of tensors or a dict of tensors
in a `DataLoader` whose shuffling follows the test's seed (single-process unless you pass
`num_workers`).

Everything is drawn through `make_tensor`, so a batch follows `torch_seed`, lands on
`torch_device`, and is recorded in [failure reproduction bundles](../guides/repro.md).

## `torch_dtype`

```python
@pytest.mark.torch_dtypes(torch.float32, torch.bfloat16)
def test_every_precision(torch_device, torch_dtype, make_tensor):
    model = nn.Linear(16, 4).to(torch_device, torch_dtype).eval()
    x = make_tensor(8, 16)                     # follows torch_dtype
    assert model(x).dtype == torch_dtype
```

Ids: `test_every_precision[cpu-float32]`, `[cpu-bfloat16]`, ... The dtypes come from the
closest `@pytest.mark.torch_dtypes(...)`, else `--torch-dtype` (repeatable), else float32
only. Combinations a device can't run at all (e.g. float64 on MPS) are skipped. Comparison
tolerances follow the dtype of the tensors compared.

## `inference_mode`

```python
def test_without_autograd(make_tensor, inference_mode):
    model = nn.Linear(16, 4).to(make_tensor(1).device)
    out = model(make_tensor(4, 16))
    assert not out.requires_grad
```

Runs the whole test body under `torch.inference_mode()`: no autograd graph, less memory.
Request it only when nothing in the test needs gradients.

## `golden`

```python
@pytest.mark.torch_golden("classifier_v1")
def test_outputs_unchanged(torch_seed, torch_device, make_tensor, golden):
    model = nn.Linear(16, 4).to(torch_device).eval()
    with torch.inference_mode():
        golden.check({"logits": model(make_tensor(8, 16))}, atol=1e-4)
```

```bash
pytest --torch-update-golden --torch-device=cpu    # write __golden__/classifier_v1.pt
pytest                                              # compare against it
```

`golden.check(output, rtol=None, atol=None)` compares a tensor or nested output with the
file `__golden__/<name>.pt` next to the test; the name comes from
`@pytest.mark.torch_golden(name)`, or the test's name. A missing file fails the test, so CI
never passes without comparing. Golden files store the torch version, shown when a
comparison fails. Commit them.

## `latency`

```python
@pytest.mark.torch_max_latency(ms=50)
def test_fast_enough(torch_device, make_tensor, latency):
    model = nn.Linear(16, 4).to(torch_device).eval()
    median_ms = latency(model, make_tensor(32, 16))
```

`latency(fn, *args, warmup=5, iters=20, **kwargs)` returns the median time of
`fn(*args, **kwargs)` in milliseconds, after warmup and with device synchronization, under
`torch.inference_mode()`. With `@pytest.mark.torch_max_latency(ms=...)` the test fails when
the median exceeds the budget; without it, nothing is enforced.

## `perf_baseline`

```python
def test_no_slowdown(torch_device, make_tensor, perf_baseline):
    model = nn.Linear(16, 4).to(torch_device).eval()
    result = perf_baseline(model, make_tensor(32, 16), latency_tolerance=0.3)
    print(result)   # latency 0.012 ms (baseline 0.011 ms)
```

```bash
pytest --torch-update-perf     # record baselines once per machine
pytest                         # later runs compare against them
```

`perf_baseline(fn, *args, name=None, **options)` measures latency (and peak memory on CUDA
and XPU) and compares with the baseline recorded for this test **and this hardware**, in
`.torch-perf-baselines.json`. A regression is `new > baseline * (1 + tolerance) + slack`;
faster is never a failure. Options: `latency_tolerance` (0.25), `latency_slack_ms` (0.05),
`memory_tolerance` (0.10), `memory_slack_mb` (1.0), `warmup`, `iters`. A missing baseline
skips the test (`--torch-perf-strict` fails it). Wrap functions that need keyword arguments
in a lambda.

## `metric_baseline`

```python
def accuracy(output, batch):
    return (output.argmax(-1) == batch[1]).float()        # one value per sample


def test_accuracy(model, val_loader, metric_baseline):
    result = metric_baseline(model, val_loader, {"accuracy": accuracy},
                             minimum={"accuracy": 0.9},
                             context={"dataset": "toy", "split": "validation"})
```

```bash
pytest --torch-update-metrics  # record; commit .torch-metric-baselines.json and .torch-metric-baselines/
pytest                         # compare
```

Evaluates the model on the loader, then compares with the recorded baseline, but only if it
measured the same thing (`context`, data fingerprint, sample selection, metric
configuration). Per-sample metrics are compared paired, with a bootstrap confidence
interval. `minimum`/`maximum` always apply. Options: `name`, `inputs`, `max_batches`,
`context`, `tolerance`, `higher_is_better`, `confidence`, `resamples`. Returns a
`MetricResult`. See [Dataset metrics](../guides/dataset-metrics.md).

## `model_loader`

```python
def test_resnet(model_loader, torch_device, make_tensor):
    model = model_loader.torchvision("resnet18", device=torch_device)   # random init, no download
    assert model(make_tensor(2, 3, 224, 224)).shape == (2, 1000)


def test_bert(model_loader):
    model = model_loader.hf("bert-base-uncased", auto_class="AutoModel")   # downloads once
    tokenizer = model_loader.hf_tokenizer("bert-base-uncased")
    vit = model_loader.timm("vit_tiny_patch16_224")                        # random init by default


def test_llama_architecture(model_loader, make_batch, torch_device):
    model = model_loader.hf_tiny("Qwen/Qwen2.5-7B", device=torch_device)   # config only: 10M params, mostly vocabulary
    batch = make_batch.tokens(2, 16, vocab_size=model.config.vocab_size)
    assert model(**batch).last_hidden_state.shape == (2, 16, 64)
```

Session-scoped: each model is built once per test session, keyed by every argument, and
comes back in eval mode. Cached instances are shared between tests, so `copy.deepcopy`
before mutating one. `hf(name, auto_class=..., dtype=..., device=..., **kwargs)`,
`timm(name, pretrained=False, ...)` and `torchvision(name, weights=None, ...)` need the
`hf`, `timm` and `torchvision` extras.

`hf_tiny(name, auto_class="AutoModel", layers=2, hidden_size=64, heads=4, vocab_size=None,
seed=0, **overrides)` tests a real architecture in about a second on a CPU: it downloads
only the config (or takes a `PretrainedConfig` or a local folder), shrinks it with
`shrink_config` and builds a randomly initialized model. Layer counts, widths, attention
and key/value heads (grouped-query attention stays grouped), feed-forward sizes, expert
counts and per-layer lists are reduced, in nested configs too; the vocabulary is kept so a
real tokenizer still fits, unless you pass `vocab_size`. Other keyword arguments set config
attributes. The weights depend only on `seed`, so they're the same in every run and test
order. `auto_class` also takes a model class name such as
`"LlavaForConditionalGeneration"`. Checked on 26 architectures, from BERT, GPT-2 and T5 to
Llama, Mixtral, Gemma, Whisper, CLIP, Mamba and Llava.

## `training_runner`

```python
from torch.nn import functional as F
from pytest_torch import TrainingSetup


def make_setup():
    model = nn.Linear(16, 4)
    return TrainingSetup(model, torch.optim.SGD(model.parameters(), lr=0.1))


def train_step(setup, batch):
    x, y = batch
    loss = F.cross_entropy(setup.model(x), y)
    loss.backward()
    setup.optimizer.step()
    setup.optimizer.zero_grad()
    return loss


def test_training(training_runner, make_tensor):
    batch = (make_tensor(8, 16), make_tensor(8, dtype=torch.long, low=0, high=4))
    run = training_runner.run(make_setup, batch, train_step, steps=20)
    run.assert_healthy(min_relative_decrease=0.2)
    training_runner.assert_resume_equivalent(make_setup, batch, train_step, steps=6)
```

A `TrainingRunner` bound to the test's `torch_device` and `torch_seed`. Its methods `run`,
`assert_resume_equivalent`, `assert_accumulation_equivalent` and
`assert_mixed_precision_training` are the functions of the same names, with `device` and
`seed` filled in. Each run builds a fresh setup with `make_setup` and moves the model to the
device. See [Training loops](../guides/training-loops.md).

## `model_config`

```python
class TestNavPolicy(ModelSuite):
    configs = sorted(Path("configs").glob("*.yaml"))
    config_overrides = {"model.hidden_size": 32}

    def make_model(self):
        return NavPolicy(self.config)

    def make_inputs(self, make_tensor):
        return make_tensor(4, 16)

    def test_hidden_size_was_shrunk(self, model_config):     # runs once per config
        assert model_config["model"]["hidden_size"] == 32
        assert self.config is model_config
```

In a test class with a `configs` attribute, every test that requests `model_config` (all
of `ModelSuite`'s generated tests do) runs once per config, with ids like
`test_hidden_size_was_shrunk[transformer_rope]`. The value is the loaded config (YAML or
JSON files are parsed; `load_config` can be overridden) with `config_overrides` applied, and
it is also set as `self.config`. Without `configs`, the fixture is `None`. See
[one suite per config](../guides/model-suite.md#one-suite-per-config).
