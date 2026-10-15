"""Every pytest-torch marker, on a test, on a class and on a module.

Run: pytest examples/test_markers.py -v
Regenerate the golden file: pytest examples/test_markers.py --torch-update-golden --torch-device=cpu

| Marker                                   | Read by                   | Effect                                         |
|------------------------------------------|---------------------------|------------------------------------------------|
| torch_seed(n)                            | torch_seed, make_tensor   | Fixed seed for this test, class or module      |
| torch_golden(name)                       | golden                    | Golden file name, shareable between tests      |
| torch_max_latency(ms=...)                | latency                   | Fails when the median latency is over budget   |
| torch_gpu_exclusive(timeout=...)         | (autouse)                 | One such test at a time per GPU, under xdist   |
| torch_tolerance(**entries)               | every comparison          | Default rtol/atol, as in pyproject.toml        |
| torch_dtypes(*dtypes)                    | torch_dtype, make_tensor  | Runs the test once per dtype                   |
| torch_threads(n)                         | (autouse)                 | CPU threads torch uses during the test         |
| torch_benchmark                          | --torch-benchmark         | A benchmark: deselected unless the flag is set |
| torch_requires(**requirements)           | (collection)              | Skips unless hardware, versions, packages fit  |
| torch_skip_on(*devices, dtype=, reason=) | torch_device, torch_dtype | Skips only the runs on those devices           |
| torch_xfail_on(*devices, dtype=, ...)    | torch_device, torch_dtype | Strict expected failure on those devices       |
| torch_deterministic                      | (autouse)                 | Deterministic algorithms during the test       |
| torch_default_dtype(dtype)               | (autouse)                 | torch's default dtype during the test          |
| torch_matmul_precision(precision)        | (autouse)                 | Float32 matmul precision; "highest": no TF32   |
| torch_no_host_sync                       | (autouse)                 | Host-device syncs raise (tests on CUDA)        |
| torch_detect_anomaly                     | (autouse)                 | Backward NaNs name the operation that made them|
| torch_max_memory(mb=...)                 | (test body)               | Peak memory budget (CUDA, XPU)                 |
| torch_memory_leak(iterations=5)          | (test body)               | Fails when repeated runs keep accumulating     |

Benchmarks have their own example: examples/test_benchmarks.py.

The closest marker wins: a test's own marker overrides its class's, which overrides the
module's (`pytestmark`). For torch_tolerance all levels apply, the closest winning per key.
"""
import pytest
import torch
from torch import nn

from pytest_torch import assert_autocast_parity, assert_tensors_close

# Module level: every test in this file that requests `torch_seed` (directly or through
# `make_tensor`) is seeded with 1234, unless a closer marker says otherwise.
pytestmark = pytest.mark.torch_seed(1234)


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))

    def forward(self, x):
        return {"logits": self.net(x), "probs": self.net(x).softmax(-1)}


@pytest.fixture
def model(torch_seed, torch_device):
    return TinyClassifier().to(torch_device).eval()


# --------------------------------------------------------------------------- #
# torch_seed
# --------------------------------------------------------------------------- #
def test_module_seed_applies(torch_seed):
    assert torch_seed == 1234


@pytest.mark.torch_seed(7)
def test_test_seed_overrides_module_seed(torch_seed):
    assert torch_seed == 7


@pytest.mark.torch_seed(7)
def test_same_seed_gives_same_inputs(make_tensor):
    # Inputs are drawn on the CPU, so seed 7 gives the same values on every device:
    # this is what makes cross-device comparisons and bug reports reproducible.
    expected = torch.randn(3, 4, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    assert torch.equal(make_tensor(3, 4).cpu(), expected.float())


@pytest.mark.torch_seed(42)
class TestClassSeed:
    def test_class_seed_applies(self, torch_seed):
        assert torch_seed == 42

    @pytest.mark.torch_seed(43)
    def test_method_seed_overrides_class_seed(self, torch_seed):
        assert torch_seed == 43


# --------------------------------------------------------------------------- #
# torch_golden
# --------------------------------------------------------------------------- #
# Without the marker the file is named after the test. Naming it explicitly lets several
# tests share one reference (here: eager and inference_mode must both match it), and lets
# you version it ("_v1") when the model changes on purpose.
@pytest.mark.torch_golden("markers_classifier_v1")
def test_eager_matches_golden(model, make_tensor, golden):
    with torch.no_grad():
        golden.check(model(make_tensor(8, 16)), atol=1e-4)


@pytest.mark.torch_golden("markers_classifier_v1")
def test_inference_mode_matches_the_same_golden(model, make_tensor, golden, inference_mode):
    golden.check(model(make_tensor(8, 16)), atol=1e-4)


# --------------------------------------------------------------------------- #
# torch_max_latency
# --------------------------------------------------------------------------- #
@pytest.mark.torch_max_latency(ms=100)  # generous: shared CI runners are noisy
def test_latency_budget(model, make_tensor, latency):
    median_ms = latency(model, make_tensor(32, 16), warmup=3, iters=10)
    print(f"median latency: {median_ms:.3f} ms")  # shown with `pytest -s`


def test_latency_without_budget_only_measures(model, make_tensor, latency):
    # No marker: nothing is enforced, the median is returned for you to log or compare.
    assert latency(model, make_tensor(32, 16), warmup=1, iters=3) > 0


# --------------------------------------------------------------------------- #
# torch_gpu_exclusive
# --------------------------------------------------------------------------- #
# Under `pytest -n 4`, tests carrying this marker wait for each other on the same GPU (a
# file lock shared by all xdist workers), so two memory-hungry tests never collide. Without
# xdist, or on the CPU, it is a harmless no-op. `timeout` is how long to wait for the lock.
@pytest.mark.torch_gpu_exclusive(timeout=600)
def test_big_batch_runs_alone_on_its_gpu(model, make_tensor):
    with torch.inference_mode():
        assert model(make_tensor(4096, 16))["logits"].shape == (4096, 4)


# --------------------------------------------------------------------------- #
# torch_tolerance
# --------------------------------------------------------------------------- #
# Overrides default tolerances for comparisons that don't pass rtol/atol themselves. Keys
# are the same as in [tool.pytest_torch.tolerance] in pyproject.toml: a dtype, a check
# (compile_parity, autocast_parity, device_parity, ...), or either prefixed with a device.
# Values are (rtol, atol) or {"rtol": ..., "atol": ...}; either may be left out.
@pytest.mark.torch_tolerance(bfloat16=(5e-2, 1e-2))
def test_dtype_tolerance():
    # 3% apart: over the built-in bf16 default (rtol=1.6e-2), within the marker's 5e-2.
    expected = torch.linspace(1, 2, 16, dtype=torch.bfloat16)
    assert_tensors_close(expected * 1.03, expected)


class OffsetHead(nn.Module):
    """Adds and removes an offset of 100: exact in fp32, off by up to ~0.25 in bf16."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(16, 4)

    def forward(self, x):
        return (self.lin(x) + 100.0) - 100.0


@pytest.mark.torch_tolerance(autocast_parity={"atol": 0.5})
def test_check_tolerance(torch_seed, make_tensor):
    # Over the built-in autocast_parity default (atol=2e-2), within the marker's 0.5.
    assert_autocast_parity(OffsetHead().eval(), make_tensor(16, 16))
# Device-specific key (needs ** because of the dot): applies only when torch_device is CUDA,
# e.g. to allow for TF32 matmuls. On other devices the built-in float32 default applies.
@pytest.mark.torch_tolerance(**{"cuda.float32": (1e-3, 1e-4)})
def test_device_specific_tolerance(model, make_tensor):
    x = make_tensor(8, 16)
    with torch.inference_mode():
        assert_tensors_close(model(x)["logits"], model(x.clone())["logits"])


@pytest.mark.torch_tolerance(bfloat16=(5e-2, 1e-2))
class TestToleranceLevels:
    def test_class_level_applies(self):
        expected = torch.linspace(1, 2, 16, dtype=torch.bfloat16)
        assert_tensors_close(expected * 1.03, expected)

    @pytest.mark.torch_tolerance(bfloat16={"rtol": 0.0})  # closest wins for rtol; atol stays 1e-2
    def test_method_overrides_one_value(self, make_tensor):
        x = make_tensor(64, kind="uniform", low=-1e-3, high=1e-3)  # tiny values: atol does the work
        assert_tensors_close((x * 1.5).bfloat16(), x.bfloat16())


# --------------------------------------------------------------------------- #
# torch_dtypes
# --------------------------------------------------------------------------- #
# Only tests that request `torch_dtype` are parametrized: [cpu-float32], [cpu-bfloat16]...
# Without the marker they run with --torch-dtype values, or float32 alone.
@pytest.mark.torch_dtypes(torch.float32, torch.bfloat16)
def test_runs_once_per_dtype(torch_dtype, make_tensor):
    assert make_tensor(4).dtype == torch_dtype  # make_tensor's default follows torch_dtype


def test_without_marker_the_default_applies(torch_dtype):
    assert torch_dtype == torch.float32  # unless --torch-dtype says otherwise


# --------------------------------------------------------------------------- #
# torch_threads
# --------------------------------------------------------------------------- #
# Single-threaded for this test only, e.g. for stable timings; restored afterwards.
@pytest.mark.torch_threads(1)
def test_single_threaded():
    assert torch.get_num_threads() == 1


# --------------------------------------------------------------------------- #
# torch_requires
# --------------------------------------------------------------------------- #
# Skips with a readable reason unless every requirement holds: devices (cuda, xpu, mps,
# gpu), versions (torch, python: specifiers like ">=2.4"), compile, gpu_memory_gb,
# cuda_capability and packages. Markers on the module, class and test all apply.
@pytest.mark.torch_requires(torch=">=2.0", packages=["numpy"])
def test_needs_numpy():
    import numpy as np

    assert np.asarray(torch.ones(2)).sum() == 2


@pytest.mark.torch_requires(cuda=True, cuda_capability=(8, 0))  # bfloat16 tensor cores: Ampere+
def test_needs_an_ampere_gpu(make_tensor):
    x = make_tensor(64, 64, dtype=torch.bfloat16).cuda()
    assert (x @ x).dtype == torch.bfloat16


# --------------------------------------------------------------------------- #
# torch_skip_on / torch_xfail_on
# --------------------------------------------------------------------------- #
# Apply only to the runs on the given devices (and dtypes, with dtype=); the others run.
@pytest.mark.torch_skip_on("mps", reason="float64 isn't supported on MPS")
def test_float64(torch_device):
    assert torch.ones(2, dtype=torch.float64, device=torch_device).sum() == 2


# Strict: if a torch release implements the op, the run fails as XPASS, so you remove the
# marker. raises= keeps a different error a real failure.
def fused_scale(x):
    if x.device.type == "cpu":
        raise NotImplementedError("fused_scale has no CPU kernel yet")
    return x * 2


@pytest.mark.torch_xfail_on("cpu", reason="fused_scale is GPU-only", raises=NotImplementedError)
def test_known_gap(torch_device):
    assert fused_scale(torch.ones(2, device=torch_device)).sum() == 4


# --------------------------------------------------------------------------- #
# Modes: set for the test (fixtures included) and restored afterwards
# --------------------------------------------------------------------------- #
@pytest.mark.torch_deterministic
def test_deterministic():
    assert torch.are_deterministic_algorithms_enabled()


@pytest.mark.torch_default_dtype(torch.float64)
def test_float64_by_default():
    assert nn.Linear(2, 2).weight.dtype == torch.float64


@pytest.mark.torch_matmul_precision("highest")  # no TF32: exact float32 comparisons on Ampere+
def test_exact_float32_matmul(make_tensor, torch_device):
    a, b = make_tensor(32, 32).to(torch_device), make_tensor(32, 32).to(torch_device)
    assert_tensors_close((a @ b).cpu(), a.cpu().double().matmul(b.cpu().double()).float(), rtol=1e-5, atol=1e-5)


@pytest.mark.torch_no_host_sync  # .item(), .cpu(), `if tensor:` raise on CUDA; not checked elsewhere
def test_forward_never_syncs(model, make_tensor):
    model(make_tensor(8, 16))


@pytest.mark.torch_detect_anomaly  # a NaN in backward names the forward op that produced it
def test_gradients(model, make_tensor):
    model(make_tensor(8, 16))["logits"].sum().backward()


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #
@pytest.mark.torch_max_memory(mb=64)  # peak memory of the test body; CUDA and XPU only
def test_inference_fits(model, make_tensor):
    with torch.inference_mode():
        model(make_tensor(256, 16))


HISTORY = []


@pytest.mark.torch_memory_leak(iterations=5)  # runs the body 1 + 5 times
def test_no_leak(model, make_tensor):
    loss = model(make_tensor(8, 16))["logits"].sum()
    HISTORY.append(loss.item())  # Try: HISTORY.append(loss): every run keeps a graph alive


# --------------------------------------------------------------------------- #
# Combining markers
# --------------------------------------------------------------------------- #
@pytest.mark.torch_seed(0)
@pytest.mark.torch_max_latency(ms=200)
@pytest.mark.torch_gpu_exclusive
class TestHeavyModel:
    """Class-level markers apply to every method: seeded, budgeted, one at a time per GPU."""

    def test_seed(self, torch_seed):
        assert torch_seed == 0

    def test_large_batch_latency(self, model, make_tensor, latency):
        latency(model, make_tensor(1024, 16), warmup=2, iters=5)
