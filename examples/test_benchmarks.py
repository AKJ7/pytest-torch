"""Benchmark suites: latency, throughput and memory over batch sizes, dtypes and modes.

Benchmarks are deselected in a normal run. Try:
    pytest examples/test_benchmarks.py --torch-benchmark                      # measure and report
    pytest examples/test_benchmarks.py --torch-benchmark --torch-update-perf  # record baselines
    pytest examples/test_benchmarks.py --torch-benchmark                      # compare against them
    pytest examples/test_benchmarks.py --torch-benchmark --torch-benchmark-json=bench.json
"""
import torch
from torch import nn

from pytest_torch import BenchmarkSuite


class TextEncoder(nn.Module):
    def __init__(self, vocab=1000, dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        layer = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(self.embed(input_ids), src_key_padding_mask=~attention_mask.bool())
        return hidden.mean(1)


class TestMLPBenchmark(BenchmarkSuite):
    # 3 batch sizes x 1 dtype x 1 mode = 3 tests per device
    batch_sizes = (1, 32, 256)

    def make_model(self):
        return nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))

    def make_inputs(self, batch_size, make_tensor):
        return make_tensor(batch_size, 128)  # cast to each benchmarked dtype for you


class TestEncoderBenchmark(BenchmarkSuite):
    # Every combination is one test; a dtype the device can't run is skipped, and so is
    # "compile" where torch.compile doesn't work (the time it takes is reported).
    batch_sizes = (1, 8)
    dtypes = (torch.float32, torch.bfloat16)
    modes = ("eager", "compile")
    latency_tolerance = 0.30  # fail when the median is more than 30% slower than the baseline
    iters = 10

    def make_model(self):
        return TextEncoder()

    def make_inputs(self, batch_size, make_tensor):
        # A dict becomes keyword arguments; integer tensors keep their dtype.
        return {
            "input_ids": make_tensor(batch_size, 32, dtype=torch.long, low=0, high=1000),
            "attention_mask": make_tensor(batch_size, 32, dtype=torch.long, low=1, high=2),  # all ones
        }
