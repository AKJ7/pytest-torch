"""Testing a real training loop: your step function, your checkpoints, a few steps each.

Run: pytest examples/test_training_loops.py -v
Try: drop `/ self.accumulation_steps` in Trainer.step, remove the optimizer lines from
     save_checkpoint and load_checkpoint, or build `decay` from `model.head` in make_setup.
"""
import torch
from torch import nn
from torch.nn import functional as F

from pytest_torch import TrainingSetup, assert_optimizer_coverage


# --------------------------------------------------------------------------- #
# The application code under test
# --------------------------------------------------------------------------- #
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.LayerNorm(32), nn.Dropout(0.1))
        self.head = nn.Linear(32, 4)

    def forward(self, x):
        return self.head(self.encoder(x))


def make_setup(device="cpu"):
    model = Classifier().to(device)
    decay = [p for n, p in model.named_parameters() if p.dim() > 1]
    no_decay = [p for n, p in model.named_parameters() if p.dim() <= 1]
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": 0.01},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=3e-3)
    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / 4))
    return TrainingSetup(model, optimizer, schedulers=warmup)


class Trainer:
    """The training step of the application: accumulation and gradient clipping."""

    def __init__(self, accumulation_steps=1, max_grad_norm=1.0):
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm

    def step(self, setup, batch):
        x, y = batch
        loss = F.cross_entropy(setup.model(x), y)
        (loss / self.accumulation_steps).backward()
        if (setup.step_index + 1) % self.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(setup.model.parameters(), self.max_grad_norm)
            setup.optimizer.step()
            setup.optimizer.zero_grad(set_to_none=True)
            setup.scheduler.step()
        return loss.detach()


def save_checkpoint(setup, path):
    torch.save({
        "model": setup.model.state_dict(),
        "optimizer": setup.optimizer.state_dict(),
        "scheduler": setup.scheduler.state_dict(),
        "rng": torch.get_rng_state(),
    }, path)


def load_checkpoint(setup, path):
    state = torch.load(path)
    setup.model.load_state_dict(state["model"])
    setup.optimizer.load_state_dict(state["optimizer"])
    setup.scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["rng"])


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def batches(n=4, size=16):
    g = torch.Generator().manual_seed(0)
    return [(torch.randn(size, 16, generator=g), torch.randint(0, 4, (size,), generator=g)) for _ in range(n)]


def test_every_parameter_is_optimized_once():
    setup = make_setup()
    assert_optimizer_coverage(setup.model, setup.optimizers)


def test_short_run_is_healthy(training_runner):
    # Telemetry for every step: loss, gradient norm, update size, learning rate.
    run = training_runner.run(make_setup, batches(n=1), Trainer().step, steps=20)
    run.assert_healthy(max_grad_norm=50.0, min_relative_decrease=0.3)
    run.assert_lr(lambda step: 3e-3 * min(1.0, (step + 1) / 4))  # warmup over 4 steps
    print(run.table())  # pytest -s shows it


def test_resuming_from_a_checkpoint_changes_nothing(training_runner):
    # Dropout draws random numbers, so the RNG state must be in the checkpoint too.
    training_runner.assert_resume_equivalent(
        make_setup, batches(), Trainer().step, steps=8, save=save_checkpoint, load=load_checkpoint
    )


def test_gradient_accumulation_matches_a_full_batch(training_runner):
    # 4 micro-batches of 4 vs one batch of 16, with the clipping threshold never reached.
    training_runner.assert_accumulation_equivalent(
        make_setup, batches(), lambda k: Trainer(accumulation_steps=k, max_grad_norm=1e9).step,
        accumulation_steps=4,
    )


def test_bfloat16_autocast_training_stays_finite(training_runner):
    def bf16_step(setup, batch):
        x, y = batch
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = setup.model(x)
        loss = F.cross_entropy(logits.float(), y)  # the loss in float32
        loss.backward()
        setup.optimizer.step()
        setup.optimizer.zero_grad()
        return loss

    run = training_runner.assert_mixed_precision_training(make_setup, batches(), bf16_step, steps=6)
    assert all(r.optimizer_steps["optimizer"] == 1 for r in run.records)
