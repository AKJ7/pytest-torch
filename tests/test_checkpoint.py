import pytest

torch = pytest.importorskip("torch")
from torch import nn

from pytest_torch import assert_checkpoint_loads, assert_checkpoint_roundtrip


class Tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 4)
        self.head = nn.Linear(4, 10, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, ids):
        return self.head(self.emb(ids))


class PlainAttribute(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.scale = torch.rand(4)  # not a buffer: lost on save

    def forward(self, x):
        return self.lin(x) * self.scale


class MutatedNonPersistent(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.register_buffer("running", torch.zeros(4), persistent=False)

    def forward(self, x):
        return self.lin(x) + self.running


def test_roundtrip_passes_and_keeps_tied_weights():
    assert_checkpoint_roundtrip(Tied, torch.randint(0, 10, (3, 5)))


def test_plain_attribute_is_named_as_culprit():
    with pytest.raises(AssertionError, match=r"(?s)outputs differ after reload.*plain tensor attribute \(not saved\): scale"):
        assert_checkpoint_roundtrip(PlainAttribute, torch.randn(2, 4))


def test_mutated_non_persistent_buffer_is_named_as_culprit():
    model = MutatedNonPersistent()
    model.running.fill_(1.0)  # e.g. statistics updated after construction
    with pytest.raises(AssertionError, match=r"non-persistent buffer \(not saved\): running"):
        assert_checkpoint_roundtrip(MutatedNonPersistent, torch.randn(2, 4), model=model)


def test_safetensors_format():
    pytest.importorskip("safetensors")
    assert_checkpoint_roundtrip(lambda: nn.Linear(4, 2), torch.randn(3, 4), format="safetensors")
    with pytest.raises(AssertionError, match="tied weights"):
        assert_checkpoint_roundtrip(Tied, torch.randint(0, 10, (3, 5)), format="safetensors")


def test_unknown_format():
    with pytest.raises(ValueError, match="format"):
        assert_checkpoint_roundtrip(lambda: nn.Linear(4, 2), torch.randn(3, 4), format="pickle")


@pytest.fixture
def ddp_checkpoint(tmp_path):
    torch.manual_seed(0)
    state = {f"module.{k}": v for k, v in nn.Linear(4, 4).state_dict().items()}
    path = tmp_path / "ckpt.pt"
    torch.save({"epoch": 3, "state_dict": state}, path)
    return path, state


def test_old_checkpoint_loads_after_stripping_the_prefix(ddp_checkpoint):
    path, state = ddp_checkpoint
    model = assert_checkpoint_loads(lambda: nn.Linear(4, 4), path, strip_prefix="module.")
    assert torch.equal(model.weight, state["module.weight"])


def test_prefix_mismatch_suggests_strip_prefix(ddp_checkpoint):
    path, _ = ddp_checkpoint
    with pytest.raises(AssertionError, match=r"(?s)missing in checkpoint: weight.*strip_prefix='module\.'"):
        assert_checkpoint_loads(lambda: nn.Linear(4, 4), path)


def test_shape_mismatch_fails_even_when_not_strict(ddp_checkpoint):
    path, _ = ddp_checkpoint
    with pytest.raises(AssertionError, match=r"shape mismatch: weight: checkpoint \(4, 4\), model \(5, 4\)"):
        assert_checkpoint_loads(lambda: nn.Linear(4, 5), path, strip_prefix="module.", strict=False)


def test_extra_keys_allowed_when_not_strict():
    state = {**nn.Linear(4, 4).state_dict(), "old_head.weight": torch.zeros(2)}
    assert_checkpoint_loads(lambda: nn.Linear(4, 4), state, strict=False)
    with pytest.raises(AssertionError, match=r"unexpected in checkpoint: old_head\.weight"):
        assert_checkpoint_loads(lambda: nn.Linear(4, 4), state)


def test_state_dict_location_must_be_found():
    with pytest.raises(ValueError, match="pass key="):
        assert_checkpoint_loads(lambda: nn.Linear(4, 4), {"epoch": 1, "weights_v2": {"w": 1}})
    nested = {"ema": nn.Linear(4, 4).state_dict()}
    assert_checkpoint_loads(lambda: nn.Linear(4, 4), nested, key="ema")
