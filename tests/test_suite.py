import pytest

torch = pytest.importorskip("torch")
# torch.export arrived in torch 2.1; before that the export check skips.
HAS_EXPORT = getattr(getattr(torch, "export", None), "export", None) is not None

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
    result.assert_outcomes(passed=3 if HAS_EXPORT else 2, skipped=0 if HAS_EXPORT else 1)
    result.stdout.fnmatch_lines(
        [
            "*TestNet::test_inference_contract[[]cpu[]] PASSED*",
            "*TestNet::test_compile_parity[[]cpu[]] PASSED*",
            f"*TestNet::test_export_parity[[]cpu[]] {'PASSED' if HAS_EXPORT else 'SKIPPED'}*",
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
    result.assert_outcomes(passed=1) if HAS_EXPORT else result.assert_outcomes(skipped=1)


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


def test_output_spec_checkpoint_and_layout_checks(pytester):
    result = run(
        pytester,
        """
from pytest_torch import Probabilities, TensorSpec


class Heads(Net):
    def forward(self, x):
        logits = self.body(x)
        return {"logits": logits, "probs": logits.softmax(-1)}


class TestNet(ModelSuite):
    checks = ("output_spec", "checkpoint", "layouts")
    output_spec = {"logits": TensorSpec(("B", 3), dtype=torch.float32), "probs": Probabilities(("B", 3))}

    def make_model(self):
        return Heads()

    def make_inputs(self, make_tensor):
        return make_tensor(8, 8)


class TestWrongSpec(TestNet):
    checks = ("output_spec",)
    output_spec = {"logits": TensorSpec(("B", 4))}
""",
        "-v",
    )
    result.assert_outcomes(passed=3, failed=1)
    result.stdout.fnmatch_lines(["*TestWrongSpec::test_output_spec*", "*dim 1 is 3, expected 4*"])


def test_output_spec_check_needs_a_spec(pytester):
    result = run(pytester, '\nclass TestNet(ModelSuite):\n    checks = ("output_spec",)\n')
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*needs an output_spec attribute*"])


# --------------------------------------------------------------------------- #
# one suite per config
# --------------------------------------------------------------------------- #
CONFIG_SUITE = """
import torch
from torch import nn
from torch.nn import functional as F
from pytest_torch import ModelSuite


class Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = nn.Linear(8, cfg["model"]["hidden"])
        self.head = nn.Linear(cfg["model"]["hidden"], 3)
        if cfg["model"]["freeze_encoder"]:
            self.encoder.requires_grad_(False)

    def forward(self, x):
        return self.head(torch.relu(self.encoder(x)))


class TestNet(ModelSuite):
    configs = {configs}
    config_overrides = {overrides}
    checks = ("contract", "frozen", "golden")

    def make_model(self):
        return Net(self.config)

    def make_inputs(self, make_tensor):
        return make_tensor(4, 8)

    def loss(self, output, inputs):
        return F.cross_entropy(output, torch.arange(4, device=output.device) % 3)

    def frozen_parameters(self):
        return ["encoder.*"] if self.config["model"]["freeze_encoder"] else []

    def test_own(self, model_config):
        assert self.config is model_config and self.config["model"]["hidden"] == {expected_hidden}
"""


def write_configs(pytester):
    (pytester.path / "configs").mkdir()
    (pytester.path / "configs" / "frozen.yaml").write_text("model:\n  hidden: 64\n  freeze_encoder: true\n")
    (pytester.path / "configs" / "finetune.json").write_text('{"model": {"hidden": 64, "freeze_encoder": false}}')


def run_configs(pytester, configs, overrides="{}", expected_hidden=64, *args):
    pytester.makepyfile(test_cfg=CONFIG_SUITE.format(configs=configs, overrides=overrides,
                                                     expected_hidden=expected_hidden))
    return pytester.runpytest_subprocess("-p", "no:cacheprovider", "--torch-device=cpu", "-W", "ignore", "-v", "-rs", *args)


def test_every_generated_test_runs_once_per_config(pytester):
    pytest.importorskip("yaml")
    write_configs(pytester)
    configs = 'sorted(str(p) for p in __import__("pathlib").Path("configs").iterdir())'
    run_configs(pytester, configs, "{}", 64, "--torch-update-golden").assert_outcomes(passed=7, skipped=1)
    result = run_configs(pytester, configs, "{}", 64)
    result.assert_outcomes(passed=7, skipped=1)
    for line in ("*TestNet::test_inference_contract[[]finetune-cpu[]] PASSED*",
                 "*TestNet::test_frozen[[]frozen-cpu[]] PASSED*",
                 "*TestNet::test_own[[]frozen[]] PASSED*"):
        result.stdout.fnmatch_lines([line])
    result.stdout.fnmatch_lines(["*nothing is frozen in this configuration*"])
    goldens = sorted(p.name for p in (pytester.path / "__golden__").iterdir())
    assert goldens == ["TestNet[finetune].pt", "TestNet[frozen].pt"]  # one per config


def test_overrides_and_python_object_configs(pytester):
    configs = '[{"model": {"hidden": 64, "freeze_encoder": True}}, {"model": {"hidden": 32, "freeze_encoder": False}}]'
    result = run_configs(pytester, configs, '{"model.hidden": 16}', 16, "-k", "own")
    result.assert_outcomes(passed=2)
    result.stdout.fnmatch_lines(["*test_own[[]config0[]] PASSED*", "*test_own[[]config1[]] PASSED*"])


def test_bad_override_path_and_empty_configs_are_reported(pytester):
    configs = '[{"model": {"hidden": 64, "freeze_encoder": True}}]'
    result = run_configs(pytester, configs, '{"modle.hidden": 16}', 16, "-k", "own")
    result.stdout.fnmatch_lines(["*config_overrides: 'modle.hidden': the config has no 'modle' there*"])
    result = run_configs(pytester, "[]")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*configs is empty (check the path or glob that builds it)*"])
