import pytest

torch = pytest.importorskip("torch")

from pytest_torch import load_repro

TEST_FILE = """
import pytest
import torch
from torch import nn
from pytest_torch import record_repro


def test_fails(torch_seed, make_tensor):
    model = record_repro("model", nn.Linear(4, 2))
    x = make_tensor(3, 4)
    stats = record_repro("stats", {"mean": x.mean(), "note": "hello"})
    print("X00", float(x[0, 0]), "W00", float(model.weight[0, 0]), "MEAN", float(stats["mean"]))
    assert False, "boom"


@pytest.mark.parametrize("k", [1, 2])
def test_param(torch_device, make_tensor, k):
    x = make_tensor(2)
    print("PARAM", k, float(x[0]))
    assert k == 1


def test_passes():
    pass
"""


def run(pytester, *args):
    return pytester.runpytest_subprocess("-p", "no:cacheprovider", "--torch-device=cpu", "-s", *args)


def value(result, tag):
    line = next(line for line in result.outlines if tag in line)
    words = line.split()
    return float(words[words.index(tag) + 1])


@pytest.fixture
def recorded(pytester):
    pytester.makepyfile(test_rep=TEST_FILE)
    result = run(pytester, "--torch-repro-dir=bundles", "--torch-repro-max-weights-mb=1")
    result.assert_outcomes(passed=2, failed=2)
    return result


def test_failing_tests_write_bundles(pytester, recorded):
    bundles = sorted(p.name for p in (pytester.path / "bundles").iterdir())
    assert bundles == ["test_rep.py_test_fails_cpu.pt", "test_rep.py_test_param_cpu-2.pt"]
    recorded.stdout.fnmatch_lines(["*replay: pytest --torch-repro=bundles/test_rep.py_test_fails_cpu.pt*"])

    bundle = load_repro(pytester.path / "bundles" / "test_rep.py_test_fails_cpu.pt")
    assert bundle["nodeid"] == "test_rep.py::test_fails[cpu]"
    assert bundle["seed"] == 0 and bundle["device"] == "cpu"
    assert len(bundle["inputs"]) == 1 and tuple(bundle["inputs"][0].shape) == (3, 4)
    assert bundle["recorded"]["stats"]["note"] == "hello"
    assert set(bundle["state_dicts"]["model"]) == {"weight", "bias"}
    assert "boom" in bundle["failure"]
    assert bundle["environment"]["torch"] == str(torch.__version__)


def test_replay_reuses_seed_inputs_and_weights(pytester, recorded):
    path = pytester.path / "bundles" / "test_rep.py_test_fails_cpu.pt"
    # Tamper with the bundle: the replay must use *its* tensors and weights, not regenerate them.
    bundle = load_repro(path)
    bundle["inputs"][0][0, 0] = 42.0
    bundle["state_dicts"]["model"]["weight"][0, 0] = -7.0
    bundle["recorded"]["stats"]["mean"] = torch.tensor(3.5)
    torch.save(bundle, path)

    replay = run(pytester, f"--torch-repro={path}", "--torch-seed=999")
    replay.assert_outcomes(failed=1)
    assert value(replay, "X00") == 42.0
    assert value(replay, "W00") == -7.0
    assert value(replay, "MEAN") == 3.5
    assert "deselected" in replay.outlines[-1]


def test_replay_selects_the_right_parametrization(pytester, recorded):
    path = pytester.path / "bundles" / "test_rep.py_test_param_cpu-2.pt"
    replay = run(pytester, f"--torch-repro={path}")
    replay.assert_outcomes(failed=1)
    assert value(replay, "PARAM") == 2


def test_weights_are_not_stored_by_default(pytester):
    pytester.makepyfile(test_rep=TEST_FILE)
    run(pytester, "--torch-repro-dir=bundles", "-k", "test_fails")
    bundle = load_repro(pytester.path / "bundles" / "test_rep.py_test_fails_cpu.pt")
    assert bundle["state_dicts"] == {}
    assert "model" in bundle["weights_not_saved_mb"]


def test_nothing_is_recorded_without_the_option(pytester):
    pytester.makepyfile(test_rep=TEST_FILE)
    result = run(pytester)
    result.assert_outcomes(passed=2, failed=2)
    assert not (pytester.path / "bundles").exists()
    assert "torch-repro" not in result.stdout.str()


def test_bad_bundle_is_a_usage_error(pytester):
    pytester.makepyfile(test_rep=TEST_FILE)
    (pytester.path / "junk.pt").write_bytes(b"not a bundle")
    result = run(pytester, "--torch-repro=junk.pt")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*--torch-repro: cannot load junk.pt*"])


def test_missing_test_is_a_usage_error(pytester, recorded):
    path = pytester.path / "bundles" / "test_rep.py_test_fails_cpu.pt"
    (pytester.path / "test_rep.py").write_text("def test_other():\n    pass\n")
    result = run(pytester, f"--torch-repro={path}")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*was not collected*"])


def test_replay_collects_the_bundle_file_even_outside_default_patterns(pytester):
    # Recorded with an explicit path to a file that default collection would ignore.
    pytester.makepyfile(checks_rep="def test_fails(make_tensor):\n    assert make_tensor(1).item() > 100\n")
    run(pytester, "checks_rep.py", "--torch-repro-dir=bundles").assert_outcomes(failed=1)
    path = pytester.path / "bundles" / "checks_rep.py_test_fails_cpu.pt"
    run(pytester, f"--torch-repro={path}").assert_outcomes(failed=1)
