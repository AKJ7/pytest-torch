"""Device handling, with a fake XPU backend standing in for an Intel GPU."""
import pytest

torch = pytest.importorskip("torch")

from pytest_torch import available_devices, devices, seed_everything
from pytest_torch.contract import _default_parity_devices
from pytest_torch.perf import hardware_id, measure_peak_memory, sync


class FakeXPU:
    """Records calls; mirrors the parts of torch.xpu that pytest-torch uses."""

    def __init__(self, count=2, hide=()):
        self.calls = []
        self.count = count
        self.allocated = 1000
        self.hide = set(hide)  # API an older torch.xpu lacks

    def __getattribute__(self, name):
        if name != "hide" and name in object.__getattribute__(self, "hide"):
            raise AttributeError(name)
        return object.__getattribute__(self, name)

    def _is_in_bad_fork(self):  # torch.manual_seed consults it before seeding XPUs
        return False

    def is_available(self):
        return True

    def device_count(self):
        return self.count

    def synchronize(self, device=None):
        self.calls.append(("synchronize", str(device)))

    def manual_seed_all(self, seed):
        self.calls.append(("manual_seed_all", seed))

    def get_device_name(self, device=None):
        return "Intel(R) Data Center GPU Max 1100"

    def set_device(self, device):
        self.calls.append(("set_device", str(device)))

    def reset_peak_memory_stats(self, device=None):
        self.calls.append(("reset_peak_memory_stats", str(device)))

    def memory_allocated(self, device=None):
        return self.allocated

    def max_memory_allocated(self, device=None):
        return self.allocated + 3 * 2**20

    def set_per_process_memory_fraction(self, fraction, index):
        self.calls.append(("set_per_process_memory_fraction", fraction, index))


@pytest.fixture
def xpu(monkeypatch):
    from pytest_torch import _session

    fake = FakeXPU()
    monkeypatch.setattr(torch, "xpu", fake, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(_session, "selected_devices", None)  # independent of the outer --torch-device
    return fake


@pytest.fixture
def no_xpu(monkeypatch):
    monkeypatch.delattr(torch, "xpu", raising=False)


def test_xpu_is_listed_after_cpu(xpu):
    assert available_devices()[:2] == ["cpu", "xpu"]
    assert devices.is_available("xpu") and devices.is_available("xpu:1")
    assert devices.device_count("xpu") == 2


def test_missing_xpu_backend_is_unavailable(no_xpu):
    assert "xpu" not in available_devices()
    assert not devices.is_available("xpu")
    assert devices.device_count("xpu") == 0


def test_broken_xpu_runtime_counts_as_unavailable(xpu, monkeypatch):
    def boom():
        raise RuntimeError("driver not found")

    monkeypatch.setattr(xpu, "is_available", boom)
    assert not devices.is_available("xpu")


def test_sync_and_seed_reach_xpu(xpu):
    sync("xpu:1")
    seed_everything(5)
    assert ("synchronize", "xpu:1") in xpu.calls
    assert ("manual_seed_all", 5) in xpu.calls


def test_perf_fingerprint_names_the_xpu(xpu):
    assert hardware_id("xpu:0") == "xpu:Intel(R) Data Center GPU Max 1100"


def test_peak_memory_on_xpu(xpu):
    assert measure_peak_memory(lambda: None, device="xpu:0") == pytest.approx(3.0)
    assert ("reset_peak_memory_stats", "xpu:0") in xpu.calls


def test_peak_memory_needs_memory_stats(monkeypatch):
    monkeypatch.setattr(torch, "xpu", FakeXPU(hide={"max_memory_allocated"}), raising=False)
    assert measure_peak_memory(lambda: None, device="xpu:0") is None


def test_contract_compares_cpu_with_xpu(xpu):
    assert _default_parity_devices(torch.device("cpu")) == ["xpu"]
    assert _default_parity_devices(torch.device("xpu")) == ["cpu"]


def test_xdist_workers_are_spread_over_xpus(xpu, monkeypatch, pytestconfig):
    from pytest_torch import plugin

    monkeypatch.setattr(plugin, "_configured_gpus", set())
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw3")
    monkeypatch.setenv("PYTEST_XDIST_WORKER_COUNT", "4")
    monkeypatch.setattr(
        pytestconfig.option, "torch_gpu_memory_fraction", "auto", raising=False
    )
    assert plugin._assign_worker_gpu(pytestconfig, "xpu") == "xpu:1"  # gw3 -> 3 % 2
    assert ("set_device", "xpu:1") in xpu.calls
    fraction = next(c for c in xpu.calls if c[0] == "set_per_process_memory_fraction")
    assert fraction[2] == 1 and 0 < fraction[1] < 0.5  # two workers share each XPU
    assert plugin._assign_worker_gpu(pytestconfig, "xpu:0") == "xpu:0"  # explicit index kept


def test_memory_fraction_unsupported_warns(monkeypatch, pytestconfig):
    from pytest_torch import plugin

    monkeypatch.setattr(torch, "xpu", FakeXPU(hide={"set_per_process_memory_fraction"}), raising=False)
    monkeypatch.setattr(pytestconfig.option, "torch_gpu_memory_fraction", "0.5", raising=False)
    with pytest.warns(UserWarning, match="cannot cap xpu memory"):
        plugin._apply_memory_fraction(pytestconfig, "xpu", 0)


def test_torch_device_option_accepts_xpu(pytester):
    pytester.makepyfile(
        """
        def test_device(torch_device):
            assert torch_device.startswith("xpu")
        """
    )
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "--torch-device=xpu", "-rs")
    if devices.is_available("xpu"):
        result.assert_outcomes(passed=1)
    else:
        result.assert_outcomes(skipped=1)
        result.stdout.fnmatch_lines(["*xpu is not available*"])


def test_unknown_device_type_is_skipped_not_crashed(pytester):
    pytester.makepyfile("def test_device(torch_device):\n    pass\n")
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "--torch-device=nonsense")
    result.assert_outcomes(skipped=1)
