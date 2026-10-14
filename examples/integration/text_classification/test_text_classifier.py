"""A transformer text classifier, tested the way a team shipping it would.

Run: pytest examples/integration/text_classification -v
"""
import pytest
import torch
from torch.nn import functional as F

from pytest_torch import (
    ModelSuite,
    Probabilities,
    TensorSpec,
    assert_checkpoint_loads,
    assert_mask_invariant,
    assert_output_spec,
    assert_padding_invariant,
)

from .model import LABELS, NEGATIVE, NEUTRAL, POSITIVE, TextClassifier, Tokenizer

TORCH_VERSION = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
# nn.TransformerEncoder's fused inference kernel can only be exported in non-strict mode,
# which handles it from torch 2.5 on.
CAN_EXPORT = TORCH_VERSION >= (2, 5)

TOKENIZER = Tokenizer(NEGATIVE + NEUTRAL + POSITIVE)
SENTENCES = [
    "the film was great",
    "awful plot and a boring actor",
    "it was a story",
    "really quite wonderful the scene was superb",
    "dreadful",
    "the actor",
    "love it",
    "poor film really poor",
]


# --------------------------------------------------------------------------- #
# The generated suite: one class, a dozen tests on every device
# --------------------------------------------------------------------------- #
class TestTextClassifier(ModelSuite):
    checks = ("contract", "output_spec", "checkpoint", "training") + (("export",) if CAN_EXPORT else ())
    dynamic_batch = True  # one exported program must serve any batch size
    output_spec = {"logits": TensorSpec(("B", 3)), "pooled": TensorSpec(("B", 32))}
    check_options = {
        # Attention softmax reorders reductions between batch sizes: allow float32 rounding.
        "inference_contract": {"atol": 1e-5, "rtol": 1e-4},
        "export_parity": {"strict": False, "atol": 1e-5, "rtol": 1e-4},
        "overfits_one_batch": {"steps": 300, "lr": 3e-3},
    }

    def make_model(self):
        return TextClassifier(len(TOKENIZER))

    def make_inputs(self, make_tensor):
        return TOKENIZER(SENTENCES)  # real tokenized text, with real padding

    def loss(self, output, inputs):
        labels = torch.tensor([2, 0, 1, 2, 0, 1, 2, 0], device=output.logits.device)
        return F.cross_entropy(output.logits, labels)


# --------------------------------------------------------------------------- #
# Padding and masking: the classic silent bug of text models
# --------------------------------------------------------------------------- #
@pytest.fixture
def model(torch_seed, torch_device):
    return TextClassifier(len(TOKENIZER)).to(torch_device).eval()


def test_padding_does_not_change_predictions(model):
    # Try pooling with `h.mean(1)` in model.py: padding leaks into `pooled` and this fails.
    assert_padding_invariant(model, TOKENIZER(SENTENCES), extra=8, atol=1e-5, rtol=1e-4)


def test_masked_tokens_are_ignored(model):
    # Try dropping `src_key_padding_mask=` in model.py: attention reads the padding.
    assert_mask_invariant(model, TOKENIZER(SENTENCES), vocab_size=len(TOKENIZER), atol=1e-5, rtol=1e-4)


hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings

from pytest_torch.strategies import padded_batches

torch.manual_seed(0)
HYPOTHESIS_MODEL = TextClassifier(len(TOKENIZER)).eval()


@settings(max_examples=25, deadline=None)
@given(batch=padded_batches(vocab_size=len(TOKENIZER), max_len=16, pad_id=0))
def test_any_padding_pattern_gives_valid_probabilities(batch):
    # Length-1 sentences, a full sentence among short ones, no padding at all...
    with torch.inference_mode():
        out = HYPOTHESIS_MODEL(**batch)
    assert_output_spec(out.logits.softmax(-1), Probabilities(("B", len(LABELS))))


# --------------------------------------------------------------------------- #
# Quality: the trained model must stay accurate
# --------------------------------------------------------------------------- #
def accuracy(output, batch):
    return (output.logits.argmax(-1) == batch[1]).float()  # per sample: paired comparisons


def test_validation_accuracy(trained_model, val_loader, metric_baseline):
    # The minimum always applies; record a baseline with --torch-update-metrics to also
    # catch smaller regressions.
    metric_baseline(trained_model, val_loader, {"accuracy": accuracy}, minimum={"accuracy": 0.9},
                    context={"dataset": "synthetic-sentiment", "split": "validation",
                             "preprocessing": f"whitespace-vocab-{len(TOKENIZER)}"})


def test_predictions_on_obvious_examples(trained_model):
    inputs = TOKENIZER(["love it really great", "awful and boring", "the plot was a story"])
    with torch.inference_mode():
        predicted = [LABELS[i] for i in trained_model(**inputs).logits.argmax(-1).tolist()]
    assert predicted == ["positive", "negative", "neutral"]


# --------------------------------------------------------------------------- #
# Releases: a checkpoint from the previous release must load into today's code
# --------------------------------------------------------------------------- #
def test_previous_release_checkpoint_loads(trained_model, tmp_path):
    # In a real project this file is committed (tests/data/classifier_v1.pt).
    path = tmp_path / "classifier_v1.pt"
    torch.save({"epoch": 3, "model_state_dict": trained_model.state_dict()}, path)
    restored = assert_checkpoint_loads(lambda: TextClassifier(len(TOKENIZER)), path).eval()
    inputs = TOKENIZER(SENTENCES)
    with torch.inference_mode():
        assert torch.equal(restored(**inputs).logits, trained_model(**inputs).logits)
