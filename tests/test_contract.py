import pytest

torch = pytest.importorskip("torch")
from torch import nn

from pytest_torch import check_inference_contract


def statuses(report):
    return {r.name: r.status for r in report.results}


def make_ok():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).eval()


class TwoHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 2)
        self.b = nn.Linear(4, 3)

    def forward(self, x):
        return {"logits": self.a(x), "aux": (self.b(x), x.sum(-1))}


class Noisy(nn.Module):
    def forward(self, x):
        return x + torch.rand_like(x)


class BatchCoupled(nn.Module):
    def forward(self, x):
        return x - x.mean(dim=0, keepdim=True)


class Counter(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("calls", torch.zeros(()))

    def forward(self, x):
        self.calls += 1
        return x * 2


def test_clean_model_passes():
    report = check_inference_contract(make_ok(), torch.randn(4, 8), raise_on_failure=False)
    assert report.ok, report.format()
    st = statuses(report)
    for name in ("eval_mode", "finite_outputs", "deterministic", "batch_invariance",
                 "no_state_mutation"):
        assert st[name] == "pass", report.format()
    assert st["dtype_parity"] == "skip"


def test_pytree_outputs_and_kwargs_inputs():
    model = TwoHead().eval()
    report = check_inference_contract(
        model, {"x": torch.randn(4, 4)}, raise_on_failure=False
    )
    assert report.ok, report.format()


def test_nondeterminism_detected_and_raises():
    with pytest.raises(AssertionError, match="deterministic"):
        check_inference_contract(Noisy(), torch.randn(4, 3))


def test_batch_dependence_detected():
    report = check_inference_contract(
        BatchCoupled().eval(), torch.randn(4, 3), raise_on_failure=False
    )
    assert statuses(report)["batch_invariance"] == "fail"
    assert "hint:" in report.format()


def test_state_mutation_detected():
    report = check_inference_contract(
        Counter().eval(), torch.randn(4, 3), raise_on_failure=False
    )
    assert statuses(report)["no_state_mutation"] == "fail"


def test_train_mode_detected():
    model = nn.Sequential(nn.Linear(32, 32), nn.Dropout(0.5))  # still in train mode
    report = check_inference_contract(model, torch.randn(8, 32), raise_on_failure=False)
    st = statuses(report)
    assert st["eval_mode"] == "fail"
    assert st["deterministic"] == "fail"


def test_check_selection_and_unknown_names():
    report = check_inference_contract(
        make_ok(), torch.randn(4, 8), checks=["finite_outputs"], raise_on_failure=False
    )
    assert [r.name for r in report.results] == ["finite_outputs"]
    with pytest.raises(ValueError, match="unknown check"):
        check_inference_contract(make_ok(), torch.randn(4, 8), checks=["nope"])


def test_dtype_parity_opt_in():
    report = check_inference_contract(
        make_ok(),
        torch.randn(4, 8),
        checks=["dtype_parity"],
        dtypes=(torch.bfloat16,),
        dtype_rtol=0.1,
        dtype_atol=0.1,
        raise_on_failure=False,
    )
    assert statuses(report)["dtype_parity"] == "pass", report.format()


# --------------------------------------------------------------------------- #
# device_parity respects --torch-device
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_gpu(monkeypatch):
    """Pretend an "mps" device exists (it is never actually used by these tests)."""
    from pytest_torch import _session, devices

    monkeypatch.setattr(devices, "available_devices", lambda: ["cpu", "mps"])
    monkeypatch.setattr(devices, "is_available", lambda d: str(d).split(":")[0] in ("cpu", "mps"))
    monkeypatch.setattr(_session, "selected_devices", None)
    return _session


def test_parity_uses_every_available_device_when_not_pinned(fake_gpu):
    from pytest_torch.contract import _default_parity_devices

    assert _default_parity_devices(torch.device("cpu")) == ["mps"]
    assert _default_parity_devices(torch.device("mps")) == ["cpu"]


def test_parity_only_uses_pinned_devices(fake_gpu):
    from pytest_torch.contract import _default_parity_devices

    fake_gpu.selected_devices = ["cpu"]
    assert _default_parity_devices(torch.device("cpu")) == []
    fake_gpu.selected_devices = ["cpu", "mps"]
    assert _default_parity_devices(torch.device("cpu")) == ["mps"]
    fake_gpu.selected_devices = ["cpu", "cuda"]  # pinned but unavailable: ignored
    assert _default_parity_devices(torch.device("cpu")) == []


def test_pinned_cpu_skips_device_parity_with_a_clear_reason(fake_gpu):
    fake_gpu.selected_devices = ["cpu"]
    report = check_inference_contract(make_ok(), torch.randn(4, 8), checks=["device_parity"])
    (result,) = report.results
    assert result.status == "skip" and result.message == "no other device selected (--torch-device)"


def test_plugin_records_the_pinned_devices(pytester):
    pytester.makepyfile(
        """
        from pytest_torch import _session

        def test_selection():
            assert _session.selected_devices == ["cpu"]
        """
    )
    pytester.runpytest_subprocess("-p", "no:cacheprovider", "--torch-device=cpu").assert_outcomes(passed=1)


# --------------------------------------------------------------------------- #
# edge cases: aliased outputs, complete state, empty selections
# --------------------------------------------------------------------------- #
class ReusedOutputBuffer(nn.Module):
    """Returns the same tensor object every call, overwritten in place with new values."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 4)
        self.out = torch.zeros(4, 4)  # a plain attribute: reused, not model state
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        self.out.copy_(self.lin(x) * self.calls)  # different values on every call
        return self.out


def test_reused_output_buffer_is_not_mistaken_for_determinism():
    torch.manual_seed(0)
    report = check_inference_contract(ReusedOutputBuffer().eval(), torch.randn(4, 8),
                                      checks=["deterministic"], raise_on_failure=False)
    assert statuses(report) == {"deterministic": "fail"}, report.format()


class NonPersistentCounter(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 4)
        self.register_buffer("seen", torch.zeros(()), persistent=False)  # not in state_dict

    def forward(self, x):
        self.seen += 1
        return self.lin(x)


class LazyCache(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 4)

    def forward(self, x):
        if not hasattr(self, "cache"):
            self.register_buffer("cache", x.mean(0).detach().clone())  # new state during inference
        return self.lin(x)


def test_non_persistent_buffer_mutation_is_detected():
    report = check_inference_contract(NonPersistentCounter().eval(), torch.randn(4, 8),
                                      checks=["no_state_mutation"], raise_on_failure=False)
    assert statuses(report) == {"no_state_mutation": "fail"}
    assert "seen" in report.format()


def test_state_added_during_inference_is_detected():
    report = check_inference_contract(LazyCache().eval(), torch.randn(4, 8),
                                      checks=["no_state_mutation"], raise_on_failure=False)
    assert statuses(report) == {"no_state_mutation": "fail"}
    assert "added during inference: cache" in report.format()


def test_plain_python_attributes_are_outside_the_mutation_check():
    # The boundary: registered parameters and buffers (persistent or not) are model state;
    # plain Python attributes, such as this call counter, are not.
    class Counter(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 4)
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return self.lin(x)

    report = check_inference_contract(Counter().eval(), torch.randn(4, 8), checks=["no_state_mutation"])
    assert statuses(report) == {"no_state_mutation": "pass"}


class CountingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 4)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.lin(x)


@pytest.mark.parametrize("selection", [{"checks": []}, {"skip": list(__import__("pytest_torch.contract", fromlist=["CHECKS"]).CHECKS)}])
def test_an_empty_selection_is_rejected_without_running_the_model(selection):
    model = CountingModel().eval()
    with pytest.raises(ValueError, match="no checks selected"):
        check_inference_contract(model, torch.randn(4, 8), **selection)
    assert model.calls == 0


def test_eval_mode_alone_does_not_run_the_model():
    model = CountingModel().eval()
    report = check_inference_contract(model, torch.randn(4, 8), checks=["eval_mode"])
    assert statuses(report) == {"eval_mode": "pass"} and model.calls == 0
