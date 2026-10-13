import pytest

pytest.importorskip("torch")

BODY = """
import pytest
import torch


def test_default(torch_dtype, make_tensor):
    assert torch_dtype == torch.float32
    assert make_tensor(2).dtype == torch.float32


@pytest.mark.torch_dtypes(torch.float32, "bfloat16")
def test_marker(torch_device, torch_dtype, make_tensor):
    x = make_tensor(3)
    assert x.dtype == torch_dtype                                # follows torch_dtype
    assert make_tensor(3, dtype=torch.long).dtype == torch.long  # explicit dtypes win


def test_untouched(make_tensor):
    assert make_tensor(2).dtype == torch.float32
"""


def run(pytester, *args):
    pytester.makepyfile(test_dt=BODY)
    return pytester.runpytest_subprocess("-p", "no:cacheprovider", "--torch-device=cpu", "-v", *args)


def test_marker_and_default(pytester):
    result = run(pytester)
    result.assert_outcomes(passed=4)
    result.stdout.fnmatch_lines(
        ["*test_default[[]cpu-float32[]] PASSED*", "*test_marker[[]cpu-float32[]] PASSED*", "*test_marker[[]cpu-bfloat16[]] PASSED*"]
    )


def test_command_line_dtypes(pytester):
    result = run(pytester, "--torch-dtype=float16", "--torch-dtype=float64", "-k", "default")
    result.stdout.fnmatch_lines(["*test_default[[]cpu-float16[]]*", "*test_default[[]cpu-float64[]]*"])
    result.assert_outcomes(failed=2)  # test_default asserts float32


def test_unknown_dtype_is_a_usage_error(pytester):
    result = run(pytester, "--torch-dtype=float13")
    result.stdout.fnmatch_lines(["*unsupported torch dtype 'float13'*"])
    assert result.ret != 0


def test_unsupported_combination_is_skipped(pytester, monkeypatch):
    from pytest_torch import plugin

    monkeypatch.setattr(plugin, "_dtype_support", {})
    import torch

    assert plugin._unsupported("cpu", torch.float32) is None
    assert plugin._unsupported("nonsense", torch.float32) is not None
