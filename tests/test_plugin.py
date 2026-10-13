import pytest

pytest.importorskip("torch")


def test_device_fixture_parametrization(pytester):
    pytester.makepyfile(
        """
        def test_dev(torch_device):
            assert torch_device == "cpu"
        """
    )
    pytester.runpytest("--torch-device=cpu").assert_outcomes(passed=1)


def test_seed_fixture_is_reproducible(pytester):
    pytester.makepyfile(
        """
        import torch

        def test_a(torch_seed):
            first = torch.rand(3)
            torch.manual_seed(torch_seed)
            assert torch.equal(first, torch.rand(3))
        """
    )
    pytester.runpytest("--torch-seed=7").assert_outcomes(passed=1)


def test_golden_roundtrip(pytester):
    pytester.makepyfile(
        """
        import pytest, torch

        @pytest.mark.torch_golden("demo")
        def test_demo(golden):
            golden.check(torch.arange(4.0))
        """
    )
    # no golden yet -> fail (never pass vacuously)
    pytester.runpytest().assert_outcomes(failed=1)
    pytester.runpytest("--torch-update-golden").assert_outcomes(passed=1)
    pytester.runpytest().assert_outcomes(passed=1)


def test_latency_budget_enforced(pytester):
    pytester.makepyfile(
        """
        import time, pytest

        @pytest.mark.torch_max_latency(ms=1)
        def test_slow(latency):
            latency(lambda: time.sleep(0.01), warmup=1, iters=3)
        """
    )
    pytester.runpytest("--torch-device=cpu").assert_outcomes(failed=1)
