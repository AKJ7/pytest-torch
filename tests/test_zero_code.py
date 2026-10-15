import importlib.util

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from pytest_torch.zero_code import SpecError, infer_from_layers, parse_input_spec, resolve

MODELS = '''
import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))

    def forward(self, x):
        return self.net(x)


class LeakyText(nn.Module):
    """Mean-pools over every position, padding included: a real bug."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(50, 8)

    def forward(self, input_ids, attention_mask):
        return self.emb(input_ids).mean(1)

    def example_inputs(self, batch_size):
        ids = torch.randint(1, 50, (batch_size, 12))
        mask = torch.ones_like(ids)
        mask[1, 8:] = 0
        return {"input_ids": ids, "attention_mask": mask}


def make_cnn():
    return nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                         nn.Flatten(), nn.Linear(8, 2))


class NoLayers(nn.Module):
    def forward(self, x):
        return x * 2


class TwoInputs(nn.Module):
    def forward(self, a, b):
        return a.float() @ b
'''


def run(pytester, *args):
    pytester.makepyfile(mymodels=MODELS)
    return pytester.runpytest_subprocess("-p", "no:cacheprovider", "--torch-device=cpu", "-W", "ignore", "-v", "-rs", *args)


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
def test_local_models_are_tested_with_no_test_file(pytester):
    result = run(pytester, "--torch-model=mymodels:Classifier", "--torch-model=mymodels.py:make_cnn")
    result.assert_outcomes(passed=10, deselected=4)  # builds, contract, invariance, checkpoint, gradients
    result.stdout.fnmatch_lines([
        "*pytest-torch: testing mymodels:Classifier, mymodels.py:make_cnn (random-init); checks: contract, invariance, checkpoint, gradients*",
    ])
    result.stdout.fnmatch_lines(["*torch-model::test_builds[[]mymodels:Classifier-cpu[]] PASSED*"])


def test_a_real_bug_is_found(pytester):
    result = run(pytester, "--torch-model=mymodels:LeakyText")
    result.assert_outcomes(passed=4, failed=1, deselected=2)
    result.stdout.fnmatch_lines(["*mymodels:LeakyText: invariance violated*", "*Padding invariance violated*"])


def test_a_model_that_cant_be_built_fails_once_with_the_fix(pytester):
    result = run(pytester, "--torch-model=mymodels:NoLayers")
    result.assert_outcomes(failed=1, skipped=4, deselected=2)
    result.stdout.fnmatch_lines([
        "*can't infer inputs for NoLayers. Give the model an `example_inputs(self, batch_size)` method, or pass --torch-model-inputs*",
        "*could not be built (see test_builds)*",
    ])


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("mymodels:Missing", "'mymodels' has no attribute 'Missing'"),
        ("nosuchpackage.models:Net", "can't import 'nosuchpackage.models'"),
        ("missing_file.py:Net", "no such file: missing_file.py"),
        ("justaname", "expected package.module:Factory or path/to/file.py:Factory"),
    ],
)
def test_bad_specs_explain_themselves(pytester, spec, message):
    result = run(pytester, f"--torch-model={spec}")
    result.assert_outcomes(failed=1, skipped=4, deselected=2)
    result.stdout.fnmatch_lines([f"*{message}*"])


def test_explicit_inputs_and_check_selection(pytester):
    result = run(pytester, "--torch-model=mymodels:TwoInputs", "--torch-model-inputs=long:4x8:5,8x3",
                 "--torch-model-checks=contract,compile")
    result.assert_outcomes(passed=3, deselected=4)
    result.stdout.fnmatch_lines(["*test_compile_parity[[]mymodels:TwoInputs-cpu[]] PASSED*"])


def test_unknown_check_is_a_usage_error(pytester):
    result = run(pytester, "--torch-model=mymodels:Classifier", "--torch-model-checks=contract,speed")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*--torch-model-checks: unknown [[]'speed'[]]*"])


def test_project_tests_run_too_only_when_paths_are_given(pytester):
    pytester.makepyfile(test_project="def test_mine():\n    pass\n")
    alone = run(pytester, "--torch-model=mymodels:Classifier")
    alone.assert_outcomes(passed=5, deselected=2)  # the project's tests are not collected
    both = run(pytester, "--torch-model=mymodels:Classifier", "test_project.py")
    both.assert_outcomes(passed=6, deselected=2)


# --------------------------------------------------------------------------- #
# input inference and --torch-model-inputs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("model", "shape", "dtype"),
    [
        (nn.Sequential(nn.Linear(16, 4)), (4, 16), torch.float32),
        (nn.Sequential(nn.Conv2d(3, 8, 3)), (4, 3, 64, 64), torch.float32),
        (nn.Sequential(nn.Conv1d(2, 8, 3)), (4, 2, 64), torch.float32),
        (nn.Sequential(nn.Embedding(30, 8)), (4, 16), torch.int64),
        (nn.GRU(6, 8, batch_first=True), (4, 16, 6), torch.float32),
        (nn.TransformerEncoderLayer(8, 2, batch_first=True), (4, 16, 8), torch.float32),
    ],
)
def test_inputs_are_inferred_from_the_first_layer(model, shape, dtype):
    sample = infer_from_layers(model).make(torch.Generator().manual_seed(0))
    assert tuple(sample.shape) == shape and sample.dtype == dtype


def test_sequence_first_rnn_is_not_guessed():
    with pytest.raises(SpecError, match="can't infer inputs for GRU"):
        infer_from_layers(nn.GRU(6, 8))


def test_input_spec_syntax():
    g = torch.Generator().manual_seed(0)
    single = parse_input_spec("4x3x32x32").make(g)
    assert single.shape == (4, 3, 32, 32) and single.dtype == torch.float32
    a, b = parse_input_spec("long:4x8:5, float16:2x2").make(g)
    assert a.dtype == torch.int64 and int(a.max()) < 5 and b.dtype == torch.float16
    plan = parse_input_spec("input_ids=long:4x16:1000,attention_mask=mask:4x16")
    named = plan.make(g)
    assert plan.kind == "text" and plan.vocab_size == 1000
    assert named["attention_mask"].sum() < 64  # padded rows
    for bad, message in [("4by16", "can't read '4by16'"), ("mask:4x2x2", r"a mask is \(batch, length\)"),
                         ("a=4x4,4x4", "either all named")]:
        with pytest.raises(SpecError, match=message):
            parse_input_spec(bad)


def test_inputs_that_dont_fit_are_reported_at_resolve_time(tmp_path):
    path = tmp_path / "net.py"
    path.write_text("from torch import nn\n\ndef make():\n    return nn.Linear(16, 4)\n")
    with pytest.raises(SpecError, match=r"inferred inputs .* don't fit the model.*Pass --torch-model-inputs"):
        resolve(f"{path}:make", inputs="4x7")


# --------------------------------------------------------------------------- #
# model zoos (skipped where the package isn't installed; no network needed)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="needs transformers")
def test_hugging_face_model_from_a_local_config(pytester, tmp_path):
    from transformers import BertConfig

    config_dir = tmp_path / "tiny-bert"
    BertConfig(vocab_size=99, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
               intermediate_size=37, max_position_embeddings=64).save_pretrained(config_dir)
    subject = resolve(f"hf:{config_dir}")
    assert subject.inputs.kind == "text" and subject.inputs.vocab_size == 99
    result = run(pytester, f"--torch-model=hf:{config_dir}")
    result.assert_outcomes(passed=5, deselected=2)


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="needs transformers")
def test_hf_tiny_shrinks_a_full_size_architecture(pytester, tmp_path):
    from transformers import LlamaConfig

    config_dir = tmp_path / "llama"
    LlamaConfig(vocab_size=1000).save_pretrained(config_dir)  # 32 layers of width 4096
    model = resolve(f"hf-tiny:{config_dir}").build()
    assert model.config.num_hidden_layers == 2 and model.config.hidden_size == 64
    result = run(pytester, f"--torch-model=hf-tiny:{config_dir}")
    result.assert_outcomes(passed=5, deselected=2)
    with pytest.raises(SpecError, match="drop --torch-model-pretrained"):
        resolve(f"hf-tiny:{config_dir}", pretrained=True)


@pytest.mark.skipif(importlib.util.find_spec("transformers") is not None, reason="transformers is installed")
def test_missing_model_zoo_package_says_how_to_install_it():
    with pytest.raises(SpecError, match=r"transformers is not installed: pip install 'pytest-torch\[hf\]'"):
        resolve("hf:bert-base-uncased")
