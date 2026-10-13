"""The same test across dtypes (and devices), with tolerances that follow the dtype.

Run: pytest examples/test_dtype_matrix.py -v
     pytest examples/test_dtype_matrix.py --torch-dtype=float16 --torch-dtype=bfloat16
"""
import pytest
import torch
from torch import nn

from pytest_torch import assert_batch_invariant, assert_outputs_close


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.LayerNorm(32), nn.Linear(32, 4))

    def forward(self, x):
        return self.net(x)


@pytest.fixture
def model(torch_seed, torch_device, torch_dtype):
    return Classifier().to(device=torch_device, dtype=torch_dtype).eval()


# Without a marker, `torch_dtype` is float32 (or whatever --torch-dtype lists), so the
# matrix stays opt-in. Combinations the device can't run at all (e.g. float64 on MPS) are
# skipped; a missing kernel for one op (float16 LayerNorm on CPU before torch 2.3) is a
# real failure, so list float16 where your target hardware supports it.
@pytest.mark.torch_dtypes(torch.float32, torch.bfloat16)
def test_batch_invariance_in_every_precision(model, make_tensor, torch_dtype):
    x = make_tensor(8, 16)  # follows torch_dtype: no dtype= needed
    assert x.dtype == torch_dtype
    # Tolerances follow the dtype too (bf16 is compared far more loosely than fp32),
    # and [tool.pytest_torch.tolerance] in pyproject.toml can change them per dtype.
    assert_batch_invariant(model, x)


@pytest.mark.torch_dtypes("float32", "bfloat16")
def test_low_precision_tracks_float32(torch_seed, torch_device, torch_dtype, make_tensor):
    reference = Classifier().to(torch_device).eval()
    candidate = Classifier().to(torch_device).eval()
    candidate.load_state_dict(reference.state_dict())
    candidate = candidate.to(torch_dtype)
    x = make_tensor(8, 16, dtype=torch.float32)  # explicit dtype wins over torch_dtype
    with torch.inference_mode():
        out = candidate(x.to(torch_dtype)).float()
        expected = reference(x)
    # Try rtol=1e-3, atol=1e-3: bf16 keeps ~3 significant digits, so the bfloat16 case fails.
    assert_outputs_close(out, expected, rtol=5e-2, atol=5e-2)
