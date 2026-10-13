import pytest

torch = pytest.importorskip("torch")

from pytest_torch import assert_batch_invariant, assert_finite, assert_tensors_close



def test_close_passes():
    a = torch.tensor([1.0, 2.0, 3.0])
    assert_tensors_close(a, a + 1e-8)


def test_mismatch_reports_worst_element():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 2.0, 3.5])
    with pytest.raises(AssertionError, match=r"max abs error.*index \(2,\)"):
        assert_tensors_close(a, b)


def test_shape_dtype_mismatch():
    with pytest.raises(AssertionError, match="Shape mismatch"):
        assert_tensors_close(torch.zeros(2), torch.zeros(3))
    with pytest.raises(AssertionError, match="Dtype mismatch"):
        assert_tensors_close(torch.zeros(2), torch.zeros(2, dtype=torch.float64))


def test_nan_positions_must_match():
    a = torch.tensor([1.0, float("nan")])
    b = torch.tensor([float("nan"), 1.0])
    with pytest.raises(AssertionError, match="NaN positions differ"):
        assert_tensors_close(a, b)


def test_assert_finite():
    assert_finite(torch.ones(3))
    with pytest.raises(AssertionError, match="1 NaN"):
        assert_finite(torch.tensor([1.0, float("nan")]))


def test_batch_invariance_in_eval_mode():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
    net.eval()
    assert_batch_invariant(net, torch.randn(8, 4))


def test_batch_invariance_fails_in_train_mode():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
    net.train()  # BatchNorm uses batch statistics -> output depends on batch
    with pytest.raises(Exception):
        assert_batch_invariant(net, torch.randn(8, 4))
