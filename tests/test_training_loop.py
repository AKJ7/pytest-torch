import copy

import pytest

torch = pytest.importorskip("torch")

from torch import nn
from torch.nn import functional as F

from pytest_torch import (
    TrainingSetup,
    assert_accumulation_equivalent,
    assert_mixed_precision_training,
    assert_optimizer_coverage,
    assert_resume_equivalent,
    run_training,
    split_batch,
    standard_step,
)


class Net(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(16, 3)

    def forward(self, x):
        return self.head(self.body(x))


def batches(n=4, size=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(size, 8, generator=g), torch.randint(0, 3, (size,), generator=g)) for _ in range(n)]


def loss_fn(model, batch):
    x, y = batch
    return F.cross_entropy(model(x), y)


def make_setup(dropout=0.0, lr=1e-2):
    def make():
        model = Net(dropout)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        return TrainingSetup(model, optimizer, schedulers=torch.optim.lr_scheduler.StepLR(optimizer, 3, 0.5))

    return make


def good_step(setup, batch):
    loss = loss_fn(setup.model, batch)
    loss.backward()
    setup.optimizer.step()
    setup.optimizer.zero_grad()
    setup.scheduler.step()
    return loss


# --------------------------------------------------------------------------- #
# runner and telemetry
# --------------------------------------------------------------------------- #
def test_run_records_telemetry_and_is_healthy():
    run = run_training(make_setup(), batches(n=1), good_step, steps=12)
    run.assert_healthy(max_grad_norm=100.0, min_relative_decrease=0.05)
    assert len(run.records) == 12 and len(run.losses) == 12
    first = run.records[0]
    assert first.optimizer_steps == {"optimizer": 1}
    assert first.grad_norm["optimizer"] > 0 and first.update_norm["optimizer"] > 0
    # StepLR(step_size=3, gamma=0.5): the lr used at each step
    run.assert_lr(lambda step: 1e-2 * 0.5 ** (step // 3))
    assert "grad norm" in run.table()


def test_runs_are_reproducible_and_take_the_device_and_a_standard_step():
    a = run_training(make_setup(dropout=0.3), batches(), loss_fn=loss_fn, steps=5, seed=3)
    b = run_training(lambda device: make_setup(dropout=0.3)(), batches(), loss_fn=loss_fn, steps=5, seed=3)
    assert a.losses == b.losses


def test_skipped_updates_are_reported():
    def forgets_some_steps(setup, batch):
        loss = loss_fn(setup.model, batch)
        loss.backward()
        if setup.step_index != 2:
            setup.optimizer.step()
        setup.optimizer.zero_grad()
        return loss

    run = run_training(make_setup(), batches(), forgets_some_steps, steps=4)
    with pytest.raises(AssertionError, match=r"step 2: no optimizer step"):
        run.assert_no_skipped_updates()
    run.assert_no_skipped_updates(max_skipped=1)


def test_zero_learning_rate_is_a_skipped_update():
    run = run_training(make_setup(lr=0.0), batches(), loss_fn=loss_fn, steps=3)
    with pytest.raises(AssertionError, match="stepped but no parameter changed"):
        run.assert_healthy()


def test_unexpected_learning_rate_and_exploding_gradients():
    def scheduler_first(setup, batch):
        setup.scheduler.step()  # before the optimizer: shifts the schedule by one step
        loss = loss_fn(setup.model, batch) * 1e4
        loss.backward()
        setup.optimizer.step()
        setup.optimizer.zero_grad()
        return loss

    with pytest.warns(UserWarning):  # torch warns about the order as well
        run = run_training(make_setup(), batches(), scheduler_first, steps=6)
    with pytest.raises(AssertionError, match=r"step 2: lr 0.005, expected 0.01") as info:
        run.assert_lr(lambda step: 1e-2 * 0.5 ** (step // 3))
    assert "Telemetry:" in str(info.value) and "scheduler.step() called before" in str(info.value)
    with pytest.raises(AssertionError, match="Exploding gradients"):
        run.assert_grad_norm_below(10.0)


def test_non_finite_loss_is_reported():
    def poisoned(setup, batch):
        loss = loss_fn(setup.model, batch)
        if setup.step_index == 1:
            loss = loss * float("nan")
        loss.backward()
        setup.optimizer.step()
        setup.optimizer.zero_grad()
        return loss

    run = run_training(make_setup(), batches(), poisoned, steps=3)
    with pytest.raises(AssertionError, match=r"(?s)step 1: non-finite loss.*applied non-finite gradients"):
        run.assert_finite()


def test_multiple_optimizers_and_named_losses():
    def make():
        model = Net()
        return TrainingSetup(model, {
            "body": torch.optim.SGD(model.body.parameters(), lr=0.1),
            "head": torch.optim.Adam(model.head.parameters(), lr=1e-3),
        })

    def step(setup, batch):
        loss = loss_fn(setup.model, batch)
        reg = sum(p.pow(2).sum() for p in setup.model.head.parameters()) * 1e-3
        (loss + reg).backward()
        for optimizer in setup.optimizers.values():
            optimizer.step()
            optimizer.zero_grad()
        return {"loss": loss, "reg": reg}

    run = run_training(make, batches(), step, steps=3).assert_healthy()
    assert set(run.records[0].losses) == {"loss", "reg"}
    assert run.records[0].optimizer_steps == {"body": 1, "head": 1}
    with pytest.raises(ValueError, match="several optimizers"):
        run.assert_lr([0.1])
    run.assert_lr([0.1, 0.1, 0.1], optimizer="body")


def test_telemetry_survives_an_earlier_torch_compile():
    # torch < 2.3: after torch.compile, new optimizers never call their step hooks
    from pytest_torch import is_compile_supported

    if not is_compile_supported():
        pytest.skip("torch.compile is not supported here")
    torch.compile(nn.Linear(2, 2), backend="eager")(torch.randn(1, 2))
    run = run_training(make_setup(), batches(), good_step, steps=2)
    assert [r.optimizer_steps for r in run.records] == [{"optimizer": 1}] * 2


def test_step_must_be_given_one_way():
    with pytest.raises(TypeError, match="either step"):
        run_training(make_setup(), batches(), good_step, loss_fn=loss_fn)


# --------------------------------------------------------------------------- #
# optimizer coverage
# --------------------------------------------------------------------------- #
def test_optimizer_coverage():
    model = Net()
    assert_optimizer_coverage(model, torch.optim.SGD(model.parameters(), lr=0.1))

    only_body = torch.optim.SGD(model.body.parameters(), lr=0.1)
    with pytest.raises(AssertionError, match=r"head.weight: trainable but in no optimizer"):
        assert_optimizer_coverage(model, only_body)
    assert_optimizer_coverage(model, only_body, ignore=["head.*"])

    twice = [torch.optim.SGD(model.parameters(), lr=0.1), torch.optim.SGD(model.head.parameters(), lr=0.1)]
    with pytest.raises(AssertionError, match=r"head.bias: in 'optimizer\[0\]', 'optimizer\[1\]': updated 2 times"):
        assert_optimizer_coverage(model, twice)

    stale = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(AssertionError, match="aren't parameters of the model") as info:
        assert_optimizer_coverage(copy.deepcopy(model), stale)
    assert "trainable but in no optimizer" in str(info.value)

    model.body.requires_grad_(False)
    assert_optimizer_coverage(model, torch.optim.SGD(model.parameters(), lr=0.1))
    with pytest.raises(AssertionError, match="frozen"):
        assert_optimizer_coverage(model, torch.optim.SGD(model.parameters(), lr=0.1), allow_frozen=False)


def test_coverage_is_part_of_a_healthy_run():
    def make():
        model = Net()
        return model, torch.optim.SGD(model.head.parameters(), lr=0.1)

    run = run_training(make, batches(), loss_fn=loss_fn, steps=2)
    with pytest.raises(AssertionError, match="Optimizer coverage"):
        run.assert_healthy()


# --------------------------------------------------------------------------- #
# resume
# --------------------------------------------------------------------------- #
def test_resume_with_the_reference_checkpoint_is_equivalent():
    report = assert_resume_equivalent(make_setup(dropout=0.2), batches(), good_step, steps=8)
    assert report.diverged_at is None and report.state_differences == []
    assert [r.step for r in report.resumed.records] == [4, 5, 6, 7]


def save_without(*skip):
    def save(setup, path):
        state = {"model": setup.model.state_dict()}
        if "optimizer" not in skip:
            state["optimizer"] = setup.optimizer.state_dict()
        if "scheduler" not in skip:
            state["scheduler"] = setup.scheduler.state_dict()
        torch.save(state, path)

    def load(setup, path):
        state = torch.load(path)
        setup.model.load_state_dict(state["model"])
        if "optimizer" in state:
            setup.optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            setup.scheduler.load_state_dict(state["scheduler"])

    return {"save": save, "load": load}


@pytest.mark.parametrize(
    ("skip", "message"),
    [
        (("optimizer",), r"optimizer 'optimizer': state differs \(e.g. state\[0\]\['step'\] missing\).*Adam moments"),
        (("scheduler",), r"scheduler 'scheduler': state differs"),
    ],
)
def test_resume_names_the_state_that_was_not_restored(skip, message):
    with pytest.raises(AssertionError, match=r"(?s)first divergence at step \d.*" + message):
        assert_resume_equivalent(make_setup(), batches(), good_step, steps=8, **save_without(*skip))


def test_resume_without_rng_state_matters_only_with_randomness():
    checkpoint = save_without()  # model, optimizer and scheduler, no RNG
    report = assert_resume_equivalent(make_setup(dropout=0.0), batches(), good_step, **checkpoint)
    assert any("random number generators" in d for d in report.state_differences)
    with pytest.raises(AssertionError, match=r"(?s)first divergence.*random number generators \(.*torch"):
        assert_resume_equivalent(make_setup(dropout=0.3), batches(), good_step, **checkpoint)


def test_resume_at_must_be_inside_the_run():
    with pytest.raises(ValueError, match="resume_at"):
        assert_resume_equivalent(make_setup(), batches(), good_step, steps=4, resume_at=4)


# --------------------------------------------------------------------------- #
# gradient accumulation
# --------------------------------------------------------------------------- #
def accumulating(divide=True, zero=True, premature=False):
    def make_step(k):
        def step(setup, batch):
            loss = loss_fn(setup.model, batch)
            (loss / k if divide else loss).backward()
            boundary = (setup.step_index + 1) % k == 0
            if boundary or (premature and setup.step_index % k == 0):
                setup.optimizer.step()
                if zero:
                    setup.optimizer.zero_grad()
            return loss

        return step

    return make_step


def sgd_setup():
    model = Net(dropout=0.5)  # disabled by the check
    return model, torch.optim.SGD(model.parameters(), lr=0.1)


def test_correct_accumulation_is_equivalent():
    assert_accumulation_equivalent(sgd_setup, batches(), accumulating(), accumulation_steps=4)
    assert_accumulation_equivalent(make_setup(), batches(), loss_fn=loss_fn, accumulation_steps=2, cycles=3)


@pytest.mark.parametrize(
    ("bug", "message"),
    [
        ({"divide": False}, "gradients are 4x the full-batch gradients: the loss isn't divided"),
        ({"zero": False}, "zero_grad\\(\\) isn't called after the optimizer step"),
        ({"premature": True}, "micro-batch 1 of 4: 'optimizer' stepped 1 time.*premature optimizer step"),
    ],
)
def test_accumulation_bugs_are_diagnosed(bug, message):
    with pytest.raises(AssertionError, match=message):
        assert_accumulation_equivalent(sgd_setup, batches(), accumulating(**bug), accumulation_steps=4)


def test_batch_norm_gets_a_hint():
    def make():
        model = nn.Sequential(nn.Linear(8, 16), nn.BatchNorm1d(16), nn.Linear(16, 3))
        return model, torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(AssertionError, match="BatchNorm computes statistics per micro-batch"):
        assert_accumulation_equivalent(make, batches(), loss_fn=loss_fn, accumulation_steps=4)


def test_split_batch():
    x, y = torch.arange(8.0).reshape(4, 2), torch.arange(4)
    parts = split_batch({"x": x, "y": y, "scale": 2.0}, 2)
    assert torch.equal(parts[1]["x"], x[2:]) and parts[1]["scale"] == 2.0
    with pytest.raises(ValueError, match="equal micro-batches"):
        split_batch(x, 3)


# --------------------------------------------------------------------------- #
# mixed precision
# --------------------------------------------------------------------------- #
def scaler_setup(init_scale=2.0**8):
    GradScaler = getattr(torch.amp, "GradScaler", None)
    if GradScaler is None:
        pytest.skip("CPU GradScaler needs torch >= 2.3")

    def make():
        model = Net()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        return TrainingSetup(model, optimizer, scaler=GradScaler("cpu", init_scale=init_scale))

    return make


def amp_step(direct=False, update=True, clip_first=False, scale=True, factor=1.0):
    def step(setup, batch):
        scaler, optimizer = setup.scaler, setup.optimizer
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = loss_fn(setup.model, batch) * factor
        (scaler.scale(loss) if scale else loss).backward()
        if clip_first:
            torch.nn.utils.clip_grad_norm_(setup.model.parameters(), 1.0)
        else:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(setup.model.parameters(), 1.0)
        if direct:
            optimizer.step()
        else:
            scaler.step(optimizer)
        if update:
            scaler.update()
        optimizer.zero_grad()
        return loss

    return step


def test_correct_mixed_precision_training_passes():
    run = assert_mixed_precision_training(scaler_setup(), batches(), amp_step(), steps=6)
    assert run.records[0].loss_scale == 2.0**8
    assert "scale" in run.table()


@pytest.mark.parametrize(
    ("bug", "message"),
    [
        ({"direct": True}, r"'optimizer'.step\(\) called directly instead of scaler.step"),
        ({"update": False}, r"scaler.step\(\) without scaler.update\(\)"),
        ({"clip_first": True}, r"clip_grad_norm_\(\) before scaler.unscale_\(\) for optimizer 'optimizer'"),
        ({"scale": False}, r"scaler.step\(\) without scaler.scale\(loss\)"),
    ],
)
def test_scaler_misuse_is_reported(bug, message):
    with pytest.raises(AssertionError, match=message):
        assert_mixed_precision_training(scaler_setup(), batches(), amp_step(**bug), steps=4)


def test_constant_overflow_is_reported():
    # 1e37 x the loss scale overflows float32: GradScaler skips steps while halving the scale
    with pytest.raises(AssertionError, match=r"(?s)steps skipped after inf/NaN gradients.*loss scale: 128, 64"):
        assert_mixed_precision_training(scaler_setup(), batches(), amp_step(factor=1e37), steps=4)


def test_clipping_is_restored_after_the_check():
    original = torch.nn.utils.clip_grad_norm_
    assert_mixed_precision_training(scaler_setup(), batches(), amp_step(), steps=2)
    assert torch.nn.utils.clip_grad_norm_ is original


def test_mixed_precision_without_a_scaler_checks_finiteness():
    def bf16_step(setup, batch):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = loss_fn(setup.model, batch)
        loss.backward()
        setup.optimizer.step()
        setup.optimizer.zero_grad()
        return loss

    assert_mixed_precision_training(make_setup(), batches(), bf16_step, steps=3)


# --------------------------------------------------------------------------- #
# fixture
# --------------------------------------------------------------------------- #
def test_training_runner_fixture(training_runner, torch_device, torch_seed):
    assert training_runner.device == torch_device and training_runner.seed == torch_seed
    run = training_runner.run(make_setup(), batches(), good_step, steps=4).assert_healthy()
    assert run.setup.model.head.weight.device.type == torch.device(torch_device).type
    training_runner.assert_resume_equivalent(make_setup(), batches(), good_step, steps=4)
    training_runner.assert_accumulation_equivalent(sgd_setup, batches(), accumulating(), accumulation_steps=2)


def test_standard_step_accumulates_and_clips():
    run = run_training(make_setup(), batches(), standard_step(loss_fn, accumulation_steps=2, max_grad_norm=0.5),
                       steps=6)
    run.assert_no_skipped_updates(every=2)
    assert [sum(r.optimizer_steps.values()) for r in run.records] == [0, 1, 0, 1, 0, 1]
