import importlib
import types

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def no_stale_bytecode(monkeypatch):
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")


def run(pytester, source, *args):
    pytester.makepyfile(test_subject=source)
    return pytester.runpytest("-p", "no:cacheprovider", "--torch-device=cpu", "-rsxXP", *args)


# --------------------------------------------------------------------------- #
# torch_requires
# --------------------------------------------------------------------------- #
def test_requirements_skip_with_a_reason(pytester):
    result = run(pytester, """
        import pytest

        @pytest.mark.torch_requires(torch=">=99")
        def test_future_torch(): pass

        @pytest.mark.torch_requires(python="<3")
        def test_old_python(): pass

        @pytest.mark.torch_requires(packages=["no_such_package_xyz"])
        def test_package(): pass

        @pytest.mark.torch_requires(xpu=True, gpu_memory_gb=10_000)
        def test_hardware(): pass

        @pytest.mark.torch_requires(torch=">=2.0", python=">=3.9", packages="pytest")
        def test_satisfied(): pass
    """)
    result.assert_outcomes(passed=1, skipped=4)
    result.stdout.fnmatch_lines_random([
        "*needs torch >=99 (installed: *)*",
        "*needs Python <3 (this is *)*",
        "*needs the 'no_such_package_xyz' package*",
        "*needs XPU*",
    ])


def test_requirements_stack_and_reject_unknown_keys(pytester):
    result = run(pytester, """
        import pytest

        pytestmark = pytest.mark.torch_requires(torch=">=2.0")

        @pytest.mark.torch_requires(torch="<2.0")
        def test_contradiction(): pass

        @pytest.mark.torch_requires(cuda_version="12")
        def test_typo(): pass
    """)
    assert result.ret != 0
    result.stderr.fnmatch_lines(["*torch_requires: unknown requirement(s) [[]'cuda_version'[]]*"])
    pytester.makepyfile(test_subject="""
        import pytest

        pytestmark = pytest.mark.torch_requires(torch=">=2.0")

        @pytest.mark.torch_requires(torch="<2.0")
        def test_contradiction(): pass
    """)
    result = pytester.runpytest("-p", "no:cacheprovider", "-rs")
    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(["SKIPPED [[]1[]] test_subject.py:*: needs torch <2.0 (installed: *)"])


# --------------------------------------------------------------------------- #
# torch_skip_on / torch_xfail_on
# --------------------------------------------------------------------------- #
def test_device_specific_skips_and_expected_failures(pytester):
    result = run(pytester, """
        import pytest

        @pytest.mark.torch_skip_on("cpu", reason="no kernel")
        def test_skipped(torch_device): pass

        @pytest.mark.torch_skip_on("mps", "cuda")
        def test_runs_here(torch_device): pass

        @pytest.mark.torch_xfail_on("cpu", reason="not implemented", raises=NotImplementedError)
        def test_known_gap(torch_device):
            raise NotImplementedError

        @pytest.mark.torch_xfail_on("cpu")
        def test_fixed_upstream(torch_device):
            pass  # strict: an unexpected pass fails, so the marker gets removed

        @pytest.mark.torch_xfail_on("cpu", raises=NotImplementedError)
        def test_other_error(torch_device):
            raise ValueError("a different bug")
    """)
    result.assert_outcomes(passed=1, skipped=1, xfailed=1, failed=2)
    result.stdout.fnmatch_lines_random(["*no kernel*", "*not implemented*", "*XPASS(strict)*"])


def test_expected_failure_for_one_dtype(pytester):
    result = run(pytester, """
        import pytest

        @pytest.mark.torch_dtypes("float32", "float64")
        @pytest.mark.torch_xfail_on("cpu", dtype="float64", reason="float64 kernel missing")
        def test_dtype(torch_device, torch_dtype):
            import torch
            assert torch_dtype != torch.float64
    """)
    result.assert_outcomes(passed=1, xfailed=1)


def test_device_markers_need_torch_device(pytester):
    result = run(pytester, """
        import pytest

        @pytest.mark.torch_skip_on("cpu")
        def test_no_device(): pass
    """)
    assert result.ret != 0
    result.stderr.fnmatch_lines(["*torch_skip_on applies to tests that use the torch_device fixture*"])


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
def test_mode_markers_apply_during_the_test_and_restore(pytester):
    result = run(pytester, """
        import pytest
        import torch

        @pytest.fixture
        def weights():
            return torch.zeros(2)  # built inside the mode: fixtures see it too

        @pytest.mark.torch_deterministic
        @pytest.mark.torch_default_dtype(torch.float64)
        @pytest.mark.torch_matmul_precision("medium")
        @pytest.mark.torch_detect_anomaly
        def test_modes(weights):
            assert torch.are_deterministic_algorithms_enabled()
            assert weights.dtype == torch.float64
            assert torch.get_float32_matmul_precision() == "medium"
            assert torch.is_anomaly_enabled()

        def test_after():
            assert not torch.are_deterministic_algorithms_enabled()
            assert torch.get_default_dtype() == torch.float32
            assert torch.get_float32_matmul_precision() == "highest"
            assert not torch.is_anomaly_enabled()

        @pytest.mark.torch_no_host_sync
        def test_cpu_is_not_checked(torch_device):
            assert torch.ones(1).item() == 1.0
    """)
    result.assert_outcomes(passed=3)
    result.stdout.fnmatch_lines(["*torch_no_host_sync*", "*not checked: the test doesn't run on CUDA*"])


def test_detect_anomaly_names_the_operation(pytester):
    result = run(pytester, """
        import pytest
        import torch

        @pytest.mark.torch_detect_anomaly
        def test_nan_gradient():
            x = torch.tensor([0.0], requires_grad=True)
            torch.sqrt(x * 0 - 1).sum().backward()
    """, "-W", "ignore")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*SqrtBackward0*"])


# --------------------------------------------------------------------------- #
# memory
# --------------------------------------------------------------------------- #
def test_memory_leak_is_detected_with_its_cause(pytester):
    result = run(pytester, """
        import pytest
        import torch

        HISTORY = []
        CACHE = {}

        @pytest.mark.torch_memory_leak
        def test_keeps_graphs():
            x = torch.ones(4, requires_grad=True)
            loss = (x * 2).sum()
            HISTORY.append(loss)  # should be loss.item()

        @pytest.mark.torch_memory_leak(iterations=3)
        def test_cache_fills_once():
            CACHE.setdefault("table", torch.ones(1000))

        @pytest.mark.torch_memory_leak
        def test_temporaries_only():
            torch.randn(100, 100) @ torch.randn(100, 100)
    """)
    result.assert_outcomes(passed=2, failed=1)
    result.stdout.fnmatch_lines([
        "*Memory leak: every run of the test leaves more behind*",
        "*live tensors after each run: *",
        "*new: () float32 on cpu with an autograd graph (SumBackward0), held by a list of 6 item(s)*",
        "*Hint: tensors that still hold their autograd graph are being kept*",
    ])


def test_leak_check_and_failure_bundles(pytester):
    source = """
        import pytest

        @pytest.mark.torch_memory_leak
        def test_inputs(make_tensor):
            make_tensor(4, 4).sum()
            assert False, "fails after the leak check"
    """
    pytester.makepyfile(test_subject=source)
    args = ("-p", "no:cacheprovider", "--torch-device=cpu")  # subprocesses: two full torch sessions
    result = pytester.runpytest_subprocess(*args, "--torch-repro-dir=bundles")
    result.assert_outcomes(failed=1)
    result.stdout.no_fnmatch_line("*Memory leak*")  # recorded inputs aren't a leak
    bundle = next((pytester.path / "bundles").glob("*"))
    assert len(torch.load(bundle, weights_only=False)["inputs"]) == 1  # one run's inputs
    replayed = pytester.runpytest_subprocess(*args, f"--torch-repro={bundle}", "-W", "error::UserWarning")
    replayed.assert_outcomes(failed=1)
    replayed.stdout.no_fnmatch_line("*Memory leak*")


def test_memory_budget_is_reported_where_it_cannot_be_measured(pytester):
    result = run(pytester, """
        import pytest
        import torch

        @pytest.mark.torch_max_memory(mb=1)
        def test_on_cpu(torch_device):
            torch.ones(10_000_000)
    """)
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*not checked: no allocator statistics on cpu*"])


def test_memory_budget_on_a_gpu(monkeypatch):
    # No GPU here: a fake allocator stands in for torch.cuda. The modules come from
    # sys.modules, which is what markers' own imports see after pytester's in-process runs.
    devices = importlib.import_module("pytest_torch.devices")
    markers = importlib.import_module("pytest_torch.markers")

    state = {"allocated": 100 * 2**20, "peak": 0}
    fake = types.SimpleNamespace(
        memory_allocated=lambda device: state["allocated"],
        max_memory_allocated=lambda device: state["peak"],
        reset_peak_memory_stats=lambda device: state.update(peak=state["allocated"]),
    )
    monkeypatch.setattr(devices, "supports_memory_stats", lambda device: True)
    monkeypatch.setattr(devices, "_gpu_backend", lambda device: fake)
    monkeypatch.setattr(devices, "synchronize", lambda device: None)
    sections = []
    item = types.SimpleNamespace(funcargs={"torch_device": "cuda:0"},
                                 add_report_section=lambda *a: sections.append(a[2]))

    def body(mb):
        return lambda: state.update(peak=state["allocated"] + mb * 2**20)

    markers._run_with_budget(item, body(30), 50.0)
    assert sections[-1] == "peak 30.0 MB on cuda:0 (budget 50 MB)"
    with pytest.raises(pytest.fail.Exception, match=r"peak memory 80.0 MB exceeds the budget of 50 MB on cuda:0 "
                                                     r"\(memory allocated during the test, above the 100.0 MB"):
        markers._run_with_budget(item, body(80), 50.0)


def test_memory_markers_validate_their_arguments(pytester):
    result = run(pytester, """
        import pytest

        @pytest.mark.torch_max_memory(mb=-1)
        def test_bad(): pass
    """)
    assert result.ret != 0
    result.stderr.fnmatch_lines(["*torch_max_memory needs a positive mb=, got -1*"])
