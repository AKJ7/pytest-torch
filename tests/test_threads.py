import re

import pytest

torch = pytest.importorskip("torch")

PROBE = """
import torch
import pytest


def test_default():
    print("THREADS", torch.get_num_threads())


@pytest.mark.torch_threads(1)
def test_single_threaded():
    assert torch.get_num_threads() == 1
"""


def threads(result):
    return int(re.search(r"THREADS (\d+)", result.stdout.str()).group(1))


@pytest.fixture
def probe(pytester, monkeypatch):
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):  # CI runners may set them
        monkeypatch.delenv(variable, raising=False)
    pytester.makepyfile(test_probe=PROBE)

    def run(*args):
        result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-s", *args)
        result.assert_outcomes(passed=2)
        return result

    return run


def test_explicit_thread_count(probe):
    result = probe("--torch-num-threads=2")
    assert threads(result) == 2
    result.stdout.fnmatch_lines(["pytest-torch: 2 CPU thread(s) per process (--torch-num-threads)"])


def test_auto_divides_the_cores_between_xdist_workers(probe, monkeypatch):
    from pytest_torch.plugin import _cpu_count

    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw1")  # what a worker of `pytest -n 4` sees
    monkeypatch.setenv("PYTEST_XDIST_WORKER_COUNT", "4")
    assert threads(probe()) == max(1, _cpu_count() // 4)


@pytest.mark.parametrize("variable", ["OMP_NUM_THREADS", "MKL_NUM_THREADS"])
def test_auto_respects_the_users_thread_variables(probe, monkeypatch, variable):
    # The user's own setting wins: auto changes nothing, whatever torch makes of the
    # variables (it prefers MKL_NUM_THREADS over OMP_NUM_THREADS, for one).
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw1")
    monkeypatch.setenv("PYTEST_XDIST_WORKER_COUNT", "4")
    monkeypatch.setenv(variable, "3")
    auto = probe()
    assert threads(auto) == threads(probe("--torch-num-threads=none"))
    auto.stdout.no_fnmatch_line("*CPU thread(s) per process*")


def test_auto_leaves_a_single_process_alone(probe):
    default = threads(probe("--torch-num-threads=none"))
    assert threads(probe()) == default


def test_marker_restores_the_previous_count(pytester):
    pytester.makepyfile(
        """
        import torch
        import pytest

        @pytest.mark.torch_threads(1)
        def test_one():
            assert torch.get_num_threads() == 1
        """
    )
    before = torch.get_num_threads()
    pytester.runpytest("-p", "no:cacheprovider").assert_outcomes(passed=1)
    assert torch.get_num_threads() == before


def test_invalid_values(pytester):
    pytester.makepyfile("def test_x(): pass")
    result = pytester.runpytest_subprocess("--torch-num-threads=zero")
    assert result.ret != 0
    result.stderr.fnmatch_lines(["*--torch-num-threads must be a positive integer, 'auto' or 'none', got 'zero'*"])
