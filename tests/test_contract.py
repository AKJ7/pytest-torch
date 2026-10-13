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
