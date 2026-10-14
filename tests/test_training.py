import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.nn import functional as F

from pytest_torch import (
    assert_all_params_receive_grad,
    assert_frozen,
    assert_gradcheck,
    assert_gradients_finite,
    assert_loss_decreases,
    assert_overfits_one_batch,
)


def make():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 3))


@pytest.fixture
def data():
    g = torch.Generator().manual_seed(0)
    return torch.randn(16, 8, generator=g), torch.randint(0, 3, (16,), generator=g)


def ce(y):
    return lambda out: F.cross_entropy(out, y)


class UnusedHead(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.body = nn.Linear(8, 3)
        self.aux = nn.Linear(8, 3)  # defined, never used

    def forward(self, x):
        return self.body(x)


class Detached(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 3)

    def forward(self, x):
        return self.b(self.a(x).detach())


class LogOfZero(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 3)

    def forward(self, x):
        return torch.sqrt(self.lin(x) * 0)  # d/dx sqrt(0) = inf


# --------------------------------------------------------------------------- #
# gradients
# --------------------------------------------------------------------------- #
def test_finite_gradients_pass(data):
    x, y = data
    loss = assert_gradients_finite(make(), x, ce(y))
    assert loss.ndim == 0


def test_non_finite_gradients_are_reported(data):
    x, _ = data
    with pytest.raises(AssertionError, match=r"(?s)Non-finite gradients.*lin\.weight"):
        assert_gradients_finite(LogOfZero(), x, lambda out: out.sum())


def test_loss_fn_must_return_a_scalar(data):
    x, _ = data
    with pytest.raises(TypeError, match="scalar"):
        assert_gradients_finite(make(), x, lambda out: out)


def test_connected_model_receives_grads(data):
    x, y = data
    assert_all_params_receive_grad(make(), x, ce(y))


def test_unused_layer_is_reported(data):
    x, y = data
    with pytest.raises(AssertionError, match=r"aux\.weight: no gradient"):
        assert_all_params_receive_grad(UnusedHead(), x, ce(y))
    assert_all_params_receive_grad(UnusedHead(), x, ce(y), ignore=["aux.*"])


def test_detach_is_reported(data):
    x, y = data
    with pytest.raises(AssertionError, match=r"a\.weight: no gradient"):
        assert_all_params_receive_grad(Detached(), x, ce(y))


def test_zero_gradient_is_reported(data):
    x, _ = data
    model = make()

    def loss(out):
        return out.sum() * 0.0

    with pytest.raises(AssertionError, match="zero everywhere"):
        assert_all_params_receive_grad(model, x, loss)
    assert_all_params_receive_grad(model, x, loss, allow_zero=True)


# --------------------------------------------------------------------------- #
# gradcheck
# --------------------------------------------------------------------------- #
class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return 2 * x * grad


class WrongSquare(Square):
    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return x * grad  # missing factor 2


def test_gradcheck_passes_for_correct_backward():
    assert_gradcheck(Square.apply, torch.randn(3, 2))
    assert_gradcheck(nn.Linear(3, 2), torch.randn(4, 3))


def test_gradcheck_reports_wrong_backward():
    with pytest.raises(AssertionError, match=r"(?s)jacobian\[output_0\]\[input_0\].*max abs error"):
        assert_gradcheck(WrongSquare.apply, torch.randn(3, 2))


def test_gradcheck_needs_floating_inputs():
    with pytest.raises(ValueError, match="floating-point"):
        assert_gradcheck(lambda t: t * 2, torch.arange(3))


# --------------------------------------------------------------------------- #
# learning
# --------------------------------------------------------------------------- #
def test_loss_decreases_and_model_is_untouched(data):
    x, y = data
    model = make()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    run = assert_loss_decreases(model, x, ce(y), lr=1e-2)
    assert run.final < run.initial
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_frozen_model_does_not_learn(data):
    x, y = data
    model = make()
    with pytest.raises(AssertionError, match="did not decrease"):
        assert_loss_decreases(model, x, ce(y), optimizer=lambda params: torch.optim.SGD(params, lr=0.0))


def test_model_overfits_one_batch(data):
    x, y = data
    run = assert_overfits_one_batch(make(), x, ce(y))
    assert run.final <= 0.05 * run.initial
    assert len(run.losses) < 200  # stopped early


def test_too_weak_model_cannot_overfit(data):
    x, _ = data
    random_targets = torch.randn(16, 3)
    tiny = nn.Linear(8, 3, bias=False)
    with pytest.raises(AssertionError, match="Could not overfit one batch"):
        assert_overfits_one_batch(
            tiny, torch.zeros_like(x), lambda out: F.mse_loss(out, random_targets), steps=30
        )


# --------------------------------------------------------------------------- #
# frozen parameters
# --------------------------------------------------------------------------- #
class EncoderHead(nn.Module):
    def __init__(self, freeze="correct"):
        super().__init__()
        torch.manual_seed(0)
        self.encoder = nn.Sequential(nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU())
        self.head = nn.Linear(16, 3)
        self.freeze = freeze
        if freeze in ("correct", "bn_in_train_mode"):
            self.encoder.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze == "correct":
            self.encoder.eval()
        return self

    def forward(self, x):
        if self.freeze == "zero_gradient":
            # "Frozen" by cutting the encoder's influence: its gradient is all zeros, but it
            # exists, so AdamW's weight decay still moves the weights.
            return self.head(self.encoder(x) * 0.0 + 1.0)
        return self.head(self.encoder(x))


def test_correctly_frozen_encoder_passes(data):
    x, y = data
    model = EncoderHead()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    assert_frozen(model, x, ce(y), "encoder.*", steps=3)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())  # untouched


def test_frozen_batchnorm_in_train_mode_is_caught(data):
    x, y = data
    with pytest.raises(AssertionError, match=r"(?s)encoder\.1\.running_mean: buffer changed.*stay in eval mode"):
        assert_frozen(EncoderHead("bn_in_train_mode"), x, ce(y), "encoder.*")
    assert_frozen(EncoderHead("bn_in_train_mode"), x, ce(y), "encoder.*", check_buffers=False)


def test_unfrozen_parameters_are_caught(data):
    x, y = data
    with pytest.raises(AssertionError, match=r"(?s)encoder\.0\.weight: received a gradient.*changed by the optimizer"):
        assert_frozen(EncoderHead("not_frozen"), x, ce(y), ["encoder.0.*"])


def test_weight_decay_on_a_zero_gradient_is_caught(data):
    x, y = data
    model = EncoderHead("zero_gradient")
    with pytest.raises(AssertionError) as info:
        assert_frozen(model, x, ce(y), "encoder.0.weight", check_buffers=False)
    message = str(info.value)
    assert "changed by the optimizer" in message and "received a gradient" not in message
    assert "weight decay moves parameters even with a zero gradient" in message


def test_frozen_patterns_must_match_something(data):
    x, y = data
    with pytest.raises(ValueError, match=r"no parameter or buffer matches \['backbone\.\*'\]"):
        assert_frozen(EncoderHead(), x, ce(y), "backbone.*")
