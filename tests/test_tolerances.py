import pytest

torch = pytest.importorskip("torch")

from pytest_torch import tolerance_overrides, tolerances
from pytest_torch.tolerances import for_check, for_dtype


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    """Start every test from built-in defaults, whatever this repo's pyproject.toml says."""
    monkeypatch.setattr(tolerances, "_config", {})
    monkeypatch.setattr(tolerances, "_overrides", [])
    monkeypatch.setattr(tolerances, "_test_device", None)


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def test_parse_accepts_tables_tuples_and_partial_entries():
    parsed = tolerances.parse(
        {"bfloat16": {"rtol": 0.02, "atol": 0.01}, "cuda.float32": (1e-4, 1e-5), "compile_parity": {"atol": 1}},
        "test",
    )
    assert parsed == {
        "bfloat16": {"rtol": 0.02, "atol": 0.01},
        "cuda.float32": {"rtol": 1e-4, "atol": 1e-5},
        "compile_parity": {"atol": 1.0},
    }


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"bfloat61": {"rtol": 1}}, "unknown tolerance key 'bfloat61'"),
        ({"tpu.float32": {"rtol": 1}}, "unknown device 'tpu'"),
        ({"float32": {"rtl": 1}}, "must be {rtol = ..., atol = ...}"),
        ({"float32": {}}, "must be {rtol = ..., atol = ...}"),
        ({"float32": {"rtol": -1}}, "must be a number >= 0"),
        ({"float32": {"rtol": "1e-3"}}, "must be a number >= 0"),
        ({"float32": 0.1}, "must be {rtol = ..., atol = ...}"),
    ],
)
def test_parse_rejects_mistakes(raw, message):
    with pytest.raises(ValueError, match=message.replace("{", r"\{").replace("}", r"\}")):
        tolerances.parse(raw, "test")


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #
def test_builtin_defaults():
    assert for_dtype(torch.bfloat16) == (1.6e-2, 1e-5)
    assert for_dtype(torch.int64) == tolerances.FALLBACK
    assert for_check("compile_parity") == (1e-4, 1e-4)


def test_config_overrides_one_value_and_keeps_the_other():
    tolerances.configure(tolerances.parse({"bfloat16": {"rtol": 0.05}}, "test"))
    assert for_dtype(torch.bfloat16) == (0.05, 1e-5)


def test_device_entry_beats_generic_entry_only_on_that_device():
    tolerances.configure(tolerances.parse({"float32": (1e-5, 1e-5), "cuda.float32": (1e-3, 1e-4)}, "test"))
    assert for_dtype(torch.float32, "cuda:1") == (1e-3, 1e-4)
    assert for_dtype(torch.float32, "cpu") == (1e-5, 1e-5)
    assert for_dtype(torch.float32) == (1e-5, 1e-5)


def test_the_test_device_decides_which_device_entries_apply():
    tolerances.configure(tolerances.parse({"cuda.float32": (1e-3, 1e-4)}, "test"))
    with tolerances.active_test_device("cuda:0"):
        # outputs are often moved to the CPU before comparing: the test's device still counts
        assert for_dtype(torch.float32, "cpu") == (1e-3, 1e-4)
    assert for_dtype(torch.float32, "cpu") == tolerances.DTYPE_DEFAULTS["float32"]


def test_overrides_beat_config_and_explicit_values_beat_everything():
    tolerances.configure(tolerances.parse({"compile_parity": (1e-3, 1e-3)}, "test"))
    with tolerance_overrides(compile_parity={"rtol": 0.1}):
        assert for_check("compile_parity") == (0.1, 1e-3)
        with tolerance_overrides(compile_parity=(0.2, 0.2)):
            assert for_check("compile_parity") == (0.2, 0.2)
        assert for_check("compile_parity", rtol=0.5) == (0.5, 1e-3)
    assert for_check("compile_parity") == (1e-3, 1e-3)


def test_assertions_use_the_resolved_defaults():
    from pytest_torch import assert_tensors_close

    a = torch.ones(4, dtype=torch.bfloat16)
    b = a + 0.0625  # 6% off: beyond the bf16 default
    with pytest.raises(AssertionError, match=r"rtol=0\.016"):
        assert_tensors_close(a, b)
    with tolerance_overrides(bfloat16=(0.1, 0.0)):
        assert_tensors_close(a, b)
    with pytest.raises(AssertionError, match=r"rtol=0\.01, atol=0\)"):  # explicit rtol wins, atol from the override
        with tolerance_overrides(bfloat16=(0.1, 0.0)):
            assert_tensors_close(a, b, rtol=0.01)


# --------------------------------------------------------------------------- #
# pyproject.toml and the marker, end to end
# --------------------------------------------------------------------------- #
TEST_BODY = """
import pytest
import torch
from torch import nn
from pytest_torch import assert_autocast_parity, assert_tensors_close


class LargeOffset(nn.Module):  # lossy under bf16 autocast, exact in fp32
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, x):
        return (self.lin(x) + 1000.0) - 1000.0


def test_bf16(torch_device):
    a = torch.ones(4, dtype=torch.bfloat16)
    assert_tensors_close(a, a + 0.0625)


def test_autocast():
    torch.manual_seed(0)
    assert_autocast_parity(LargeOffset().eval(), torch.randn(16, 8))


@pytest.mark.torch_tolerance(bfloat16=(0.0, 0.0))
def test_marker_tightens(torch_device):
    a = torch.ones(4, dtype=torch.bfloat16)
    assert_tensors_close(a, a + 0.0625)
"""


def run(pytester, pyproject, *args):
    pytester.makefile(".toml", pyproject=pyproject)
    pytester.makepyfile(TEST_BODY)
    return pytester.runpytest_subprocess("-p", "no:cacheprovider", "--torch-device=cpu", "-W", "ignore", *args)


def test_without_config_the_builtin_defaults_apply(pytester):
    run(pytester, "").assert_outcomes(failed=3)


def test_pyproject_config_and_marker(pytester):
    result = run(
        pytester,
        """
[tool.pytest_torch.tolerance]
bfloat16 = { rtol = 0.1 }
autocast_parity = { atol = 5.0 }
""",
    )
    result.assert_outcomes(passed=2, failed=1)
    result.stdout.fnmatch_lines(["*test_marker_tightens*", "*rtol=0, atol=0*"])


def test_device_entries_only_apply_on_their_device(pytester):
    result = run(pytester, '[tool.pytest_torch.tolerance]\n"cuda.bfloat16" = { rtol = 0.1 }\n', "-k", "bf16")
    result.assert_outcomes(failed=1)


def test_invalid_config_is_a_usage_error(pytester):
    result = run(pytester, '[tool.pytest_torch.tolerance]\nbfloat61 = { rtol = 0.1 }\n')
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*unknown tolerance key 'bfloat61'*"])


def test_invalid_marker_fails_the_test(pytester):
    pytester.makepyfile(
        """
import pytest

@pytest.mark.torch_tolerance(floaty=(1, 1))
def test_x():
    pass
"""
    )
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*torch_tolerance: tolerance_overrides: unknown tolerance key 'floaty'*"])
