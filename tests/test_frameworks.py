import copy
import importlib.util

import pytest

torch = pytest.importorskip("torch")

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from pytest_torch import ModelLoader, from_lightning

needs_lightning = pytest.mark.skipif(
    importlib.util.find_spec("lightning") is None and importlib.util.find_spec("pytorch_lightning") is None,
    reason="needs lightning",
)
needs_transformers = pytest.mark.skipif(importlib.util.find_spec("transformers") is None,
                                        reason="needs transformers")


def lit_classifier(optimizers="plain", automatic=True):
    try:
        import lightning.pytorch as pl
    except ImportError:
        import pytorch_lightning as pl

    class LitClassifier(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Dropout(0.1), nn.Linear(16, 3))
            self.automatic_optimization = automatic

        def training_step(self, batch, batch_idx):
            x, y = batch
            loss = F.cross_entropy(self.net(x), y)
            self.log("train_loss", loss)
            return {"loss": loss} if optimizers == "dict" else loss

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(), lr=1e-2)
            if optimizers == "plain":
                return optimizer
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
            if optimizers == "step":
                return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
            return [optimizer], [scheduler]  # interval "epoch"

    return LitClassifier()


def batches(n=4, size=8):
    g = torch.Generator().manual_seed(0)
    return [(torch.randn(size, 8, generator=g), torch.randint(0, 3, (size,), generator=g)) for _ in range(n)]


# --------------------------------------------------------------------------- #
# Lightning
# --------------------------------------------------------------------------- #
@needs_lightning
def test_lightning_trainer_runs_quietly_on_the_test_device(lightning_trainer, torch_device, tmp_path):
    trainer = lightning_trainer()
    x, y = (torch.cat(t) for t in zip(*batches()))
    module = lit_classifier()
    trainer.fit(module, DataLoader(TensorDataset(x, y), batch_size=8))
    assert trainer.global_step == 5
    assert module.device.type == torch.device(torch_device).type
    assert not list(tmp_path.rglob("*.ckpt")) and not list(tmp_path.rglob("events.*"))
    assert lightning_trainer(max_steps=2).max_steps == 2


@needs_lightning
def test_lightning_module_under_the_training_runner(training_runner):
    make_setup, step = from_lightning(lambda: lit_classifier("step"))
    run = training_runner.run(make_setup, batches(n=1), step, steps=6).assert_healthy()
    run.assert_lr(lambda i: 1e-2 * 0.5**i)  # "step" interval: once per step
    training_runner.assert_resume_equivalent(make_setup, batches(), step, steps=6)


@needs_lightning
def test_lightning_epoch_schedulers_and_returned_dicts(training_runner):
    make_setup, step = from_lightning(lambda: lit_classifier("epoch"))
    run = training_runner.run(make_setup, batches(), step, steps=4)
    run.assert_lr([1e-2] * 4)  # no epochs in a step-based run
    make_setup, step = from_lightning(lambda: lit_classifier("epoch"), epoch_length=2)
    training_runner.run(make_setup, batches(), step, steps=4).assert_lr([1e-2, 1e-2, 5e-3, 5e-3])
    make_setup, step = from_lightning(lambda: lit_classifier("dict"))
    assert len(training_runner.run(make_setup, batches(), step, steps=2).losses) == 2


@needs_lightning
def test_lightning_manual_optimization_is_rejected(training_runner):
    make_setup, step = from_lightning(lambda: lit_classifier(automatic=False))
    with pytest.raises(TypeError, match="automatic optimization only"):
        training_runner.run(make_setup, batches(), step)


# --------------------------------------------------------------------------- #
# Hugging Face Trainer
# --------------------------------------------------------------------------- #
@needs_transformers
def test_hf_trainer_trains_a_tiny_model(hf_trainer, make_batch, tmp_path):
    import transformers

    model = copy.deepcopy(ModelLoader().hf_tiny(  # the loader's instances are shared: train a copy
        transformers.BertConfig(vocab_size=200), auto_class="AutoModelForSequenceClassification", num_labels=2))
    data = make_batch.tokens(16, 12, vocab_size=200, labels=2)
    trainer = hf_trainer(model.train(), data, learning_rate=1e-3)
    trainer.train()
    assert trainer.state.global_step == 5
    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    assert len(losses) == 5  # logged every step
    assert trainer.args.device.type == "cpu" and trainer.args.report_to == []
    assert not list(tmp_path.rglob("checkpoint-*"))
    assert hf_trainer(model, data, max_steps=2).args.max_steps == 2
