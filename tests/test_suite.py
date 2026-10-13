import pytest

pytest.importorskip("torch")

HEADER = """
import torch
from torch import nn
from torch.nn import functional as F
from pytest_torch import ModelSuite


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))

    def forward(self, x):
        return self.body(x)
"""


def run(pytester, body, *args):
    pytester.makepyfile(HEADER + body)
    # subprocess: in-process runs re-import torch modules, which torch.optim does not survive
    return pytester.runpytest_subprocess("-p", "no:cacheprovider", "--torch-device=cpu", "-W", "ignore", *args)


def test_default_checks_generate_three_tests(pytester):
    result = run(
        pytester,
        """
class TestNet(ModelSuite):
    def make_model(self):
        return Net()

    def make_inputs(self, make_tensor):
        return make_tensor(8, 8)
""",
        "-v",
    )
    result.assert_outcomes(passed=3)
    result.stdout.fnmatch_lines(
        [
            "*TestNet::test_inference_contract[[]cpu[]] PASSED*",
            "*TestNet::test_compile_parity[[]cpu[]] PASSED*",
            "*TestNet::test_export_parity[[]cpu[]] PASSED*",
        ]
    )


def test_training_checks_and_options(pytester):
    result = run(
        pytester,
        """
class TestNet(ModelSuite):
    checks = ("contract", "training", "autocast")
    check_options = {"overfits_one_batch": {"steps": 300}}

    def make_model(self):
        return Net()

    def make_inputs(self, make_tensor):
        return make_tensor(8, 8)

    def loss(self, output, inputs):
        return F.cross_entropy(output, torch.arange(8) % 3)
""",
        "-v",
    )
    result.assert_outcomes(passed=6)
    result.stdout.fnmatch_lines(["*test_overfits_one_batch*PASSED*"])
    assert "test_export_parity" not in result.stdout.str()


def test_buggy_model_fails_the_right_tests(pytester):
    result = run(
        pytester,
        """
class Leaky(Net):
    def __init__(self):
        super().__init__()
        self.unused = nn.Linear(8, 8)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        return self.drop(self.body(x))


class TestLeaky(ModelSuite):
    checks = ("contract", "training")

    def make_model(self):
        return Leaky()

    def make_inputs(self, make_tensor):
        return make_tensor(8, 8)

    def loss(self, output, inputs):
        return output.pow(2).mean()

    def test_inference_contract(self, torch_seed, torch_device, make_tensor):
        # Overriding a generated test: here, keep the model in training mode on purpose.
        from pytest_torch import check_inference_contract

        report = check_inference_contract(
            self.make_model().to(torch_device), self.make_inputs(make_tensor), raise_on_failure=False
        )
        assert {r.name for r in report.failures} >= {"eval_mode", "deterministic"}
""",
    )
    result.assert_outcomes(passed=4, failed=1)
    result.stdout.fnmatch_lines(["*test_all_params_receive_grad*", "*unused.weight: no gradient*"])


def test_dynamic_batch_export(pytester):
    result = run(
        pytester,
        """
class TestNet(ModelSuite):
    checks = ("export",)
    dynamic_batch = True

    def make_model(self):
        return Net()

    def make_inputs(self, make_tensor):
        return make_tensor(8, 8)
""",
    )
    result.assert_outcomes(passed=1)


def test_golden_check_uses_the_class_name(pytester):
    body = """
class TestNet(ModelSuite):
    checks = ("golden",)

    def make_model(self):
        return Net()

    def make_inputs(self, make_tensor):
        return make_tensor(8, 8)
"""
    run(pytester, body).assert_outcomes(failed=1)  # missing golden file fails
    run(pytester, body, "--torch-update-golden").assert_outcomes(passed=1)
    assert (pytester.path / "__golden__" / "TestNet.pt").exists()
    run(pytester, body).assert_outcomes(passed=1)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('checks = ("contract", "nope")', "unknown check(s) ['nope']"),
        ('check_options = {"compile": {}}', "unknown check_options key(s) ['compile']"),
        ('checks = ("training",)', "need a loss(self, output, inputs) method"),
    ],
)
def test_configuration_errors_are_reported_at_collection(pytester, body, message):
    result = run(pytester, f"\nclass TestNet(ModelSuite):\n    {body}\n")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines([f"*{message.replace('[', '[[]')}*"])
