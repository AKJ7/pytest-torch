"""Testing a Hugging Face fine-tuning setup: the Trainer, its subclass and its metrics.

Run: pytest examples/test_hf_trainer.py -v          (needs: pip install "pytest-torch[hf]")
Try: drop `weight=self.class_weights` in WeightedTrainer.compute_loss, or return
     `accuracy * 100` from compute_metrics.

The model comes from a config built here, so the example runs offline; in a real project,
pass the hub id you fine-tune, e.g. model_loader.hf_tiny("google-bert/bert-base-uncased", ...).
"""
import copy

import pytest

transformers = pytest.importorskip("transformers")
pytest.importorskip("accelerate")

import torch
from torch.nn import functional as F

from pytest_torch import assert_optimizer_coverage

VOCAB = 200


# --------------------------------------------------------------------------- #
# The application code under test
# --------------------------------------------------------------------------- #
class WeightedTrainer(transformers.Trainer):
    """Cross-entropy with class weights, for an imbalanced dataset."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = None if self.class_weights is None else self.class_weights.to(outputs.logits.device)
        loss = F.cross_entropy(outputs.logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(prediction):
    logits, labels = prediction
    accuracy = float((logits.argmax(-1) == labels).mean())
    return {"accuracy": accuracy}


# --------------------------------------------------------------------------- #
# Test helpers
# --------------------------------------------------------------------------- #
@pytest.fixture
def model(model_loader):
    config = transformers.BertConfig(vocab_size=VOCAB, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    tiny = model_loader.hf_tiny(config, auto_class="AutoModelForSequenceClassification", num_labels=2)
    return copy.deepcopy(tiny).train()  # the loader's instances are shared: train a copy


@pytest.fixture
def dataset(make_batch):
    # A learnable task: the label says whether the first token is in the lower half of the vocabulary.
    data = make_batch.tokens(64, 16, vocab_size=VOCAB, padding="right")
    data["labels"] = (data["input_ids"][:, 0] < VOCAB // 2).long()
    return data


def logged_losses(trainer):
    return [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_fine_tuning_lowers_the_loss(hf_trainer, model, dataset):
    trainer = hf_trainer(model, dataset, max_steps=30, learning_rate=3e-3, per_device_train_batch_size=16)
    trainer.train()
    losses = logged_losses(trainer)  # one per step: hf_trainer logs every step
    assert len(losses) == 30
    assert sum(losses[-5:]) / 5 < 0.8 * sum(losses[:5]) / 5


def test_evaluation_reports_accuracy(hf_trainer, model, dataset):
    trainer = hf_trainer(model, dataset, eval_dataset=dataset, trainer_kwargs={"compute_metrics": compute_metrics})
    metrics = trainer.evaluate()
    assert 0.0 <= metrics["eval_accuracy"] <= 1.0
    assert torch.isfinite(torch.tensor(metrics["eval_loss"]))


def test_weighted_loss_uses_the_class_weights(hf_trainer, model, dataset):
    weights = torch.tensor([1.0, 4.0])
    trainer = hf_trainer(model, dataset, trainer_class=WeightedTrainer, trainer_kwargs={"class_weights": weights})
    batch = {key: value[:8] for key, value in dataset.items()}
    with torch.no_grad():
        loss = trainer.compute_loss(model, dict(batch))
        logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
    torch.testing.assert_close(loss, F.cross_entropy(logits, batch["labels"], weight=weights))


def test_every_parameter_is_optimized_once(hf_trainer, model, dataset):
    # The Trainer splits parameters into weight-decay and no-decay groups (biases, LayerNorm).
    trainer = hf_trainer(model, dataset, weight_decay=0.01)
    trainer.create_optimizer()
    assert_optimizer_coverage(model, trainer.optimizer)
    no_decay = [g for g in trainer.optimizer.param_groups if g["weight_decay"] == 0.0]
    assert no_decay and all(p.dim() == 1 for g in no_decay for p in g["params"])


def test_gradient_accumulation_matches_a_larger_batch(hf_trainer, model, dataset):
    big = copy.deepcopy(model)
    hf_trainer(big, dataset, max_steps=3, per_device_train_batch_size=8).train()
    hf_trainer(model, dataset, max_steps=3, per_device_train_batch_size=4, gradient_accumulation_steps=2).train()
    for (name, a), b in zip(big.named_parameters(), model.parameters()):
        torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-6, msg=lambda m, name=name: f"{name}: {m}")


def test_resuming_from_a_checkpoint_is_exact(hf_trainer, model, dataset, tmp_path):
    class StopAt(transformers.TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 3:  # "interrupt" right after saving checkpoint-3
                control.should_training_stop = True

    uninterrupted, interrupted, resumed = model, copy.deepcopy(model), copy.deepcopy(model)
    hf_trainer(uninterrupted, dataset, max_steps=6).train()
    hf_trainer(interrupted, dataset, max_steps=6, save_strategy="steps", save_steps=3,
               trainer_kwargs={"callbacks": [StopAt()]}).train()
    checkpoint = next((tmp_path / "hf-trainer").glob("checkpoint-3"))
    with torch.no_grad():  # only the checkpoint can bring these weights back
        for param in resumed.parameters():
            param.add_(1.0)
    trainer = hf_trainer(resumed, dataset, max_steps=6)
    trainer.train(resume_from_checkpoint=str(checkpoint))
    # Steps 1-3 come from the checkpoint's trainer state; 4-6 were trained after resuming.
    assert [e["step"] for e in trainer.state.log_history if "loss" in e] == [1, 2, 3, 4, 5, 6]
    for (name, a), b in zip(uninterrupted.named_parameters(), resumed.parameters()):
        torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-6, msg=lambda m, name=name: f"{name}: {m}")
