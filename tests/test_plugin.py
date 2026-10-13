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


@pytest.mark.skip()
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


@pytest.mark.skip()
def test_golden_supports_nested_outputs(pytester):
    pytester.makepyfile(
        """
        import pytest, torch

        @pytest.mark.torch_golden("nested")
        def test_nested(golden):
            golden.check({"logits": torch.arange(4.0), "meta": {"n": 4, "tag": "x"}})
        """
    )
    pytester.runpytest().assert_outcomes(failed=1)
    pytester.runpytest("--torch-update-golden").assert_outcomes(passed=1)
    pytester.runpytest().assert_outcomes(passed=1)


@pytest.mark.skip()
def test_golden_mismatch_reports_path(pytester, monkeypatch):
    pytester.makepyfile(
        """
        import os
        import pytest, torch

        @pytest.mark.torch_golden("nested")
        def test_nested(golden):
            shift = float(os.environ.get("GOLDEN_SHIFT", "0"))
            golden.check({"logits": torch.arange(4.0) + shift})
        """
    )
    pytester.runpytest("--torch-update-golden").assert_outcomes(passed=1)
    monkeypatch.setenv("GOLDEN_SHIFT", "1.0")
    result = pytester.runpytest()
    result.assert_outcomes(failed=1)
    assert "output['logits']" in result.stdout.str()


# --------------------------------------------------------------------------- #
# seeding fixtures
# --------------------------------------------------------------------------- #
def test_make_tensor_is_reproducible_and_matches_cpu_generator(pytester):
    pytester.makepyfile(
        """
        import torch

        def test_values(make_tensor, torch_seed):
            x = make_tensor(2, 3)
            expected = torch.randn(
                (2, 3), generator=torch.Generator().manual_seed(torch_seed), dtype=torch.float64
            ).float()
            assert torch.equal(x.cpu(), expected)

        def test_integers(make_tensor):
            ids = make_tensor(4, 5, dtype=torch.long, low=1, high=50)
            assert ids.dtype == torch.long
            assert int(ids.min()) >= 1 and int(ids.max()) < 50
        """
    )
    pytester.runpytest("--torch-device=cpu", "--torch-seed=3").assert_outcomes(passed=2)


def test_seed_marker_overrides_cli(pytester):
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.torch_seed(11)
        def test_marker(torch_seed):
            assert torch_seed == 11

        def test_cli(torch_seed):
            assert torch_seed == 5
        """
    )
    pytester.runpytest("--torch-seed=5").assert_outcomes(passed=2)


def test_per_test_seed_mode_gives_different_but_stable_seeds(pytester):
    pytester.makepyfile(
        """
        seeds = {}

        def test_a(torch_seed):
            seeds["a"] = torch_seed

        def test_b(torch_seed):
            assert torch_seed != seeds["a"]
        """
    )
    pytester.runpytest("--torch-seed-mode=per-test").assert_outcomes(passed=2)


def test_autoseed_seeds_tests_that_do_not_request_the_fixture(pytester):
    pytester.makepyfile(
        """
        import torch

        first = []

        def test_a():
            first.append(torch.rand(1).item())

        def test_b():
            assert torch.rand(1).item() == first[0]
        """
    )
    pytester.runpytest("--torch-autoseed").assert_outcomes(passed=2)


def test_seed_is_reported_for_failing_tests(pytester):
    pytester.makepyfile(
        """
        def test_fails(torch_seed):
            assert False
        """
    )
    result = pytester.runpytest("--torch-seed=42")
    result.assert_outcomes(failed=1)
    assert "torch seed: 42" in result.stdout.str()
