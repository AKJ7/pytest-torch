"""Every config in configs/ gets the full suite: build, contract, outputs, checkpoint,
frozen encoder, training sanity, and rollout-vs-episode equivalence.

Run: pytest examples/integration/navigation_policy -v
     (ids read like test_frozen[transformer_rope_frozen-cpu])
"""
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from pytest_torch import ModelSuite, TensorSpec, assert_streaming_equivalent

from .model import ACTION_DIM, NavPolicy

pytest.importorskip("yaml")
CONFIGS = sorted((Path(__file__).parent / "configs").glob("*.yaml"))


class TestNavPolicy(ModelSuite):
    configs = CONFIGS
    # Real configs are sized for training runs; shrink every config so the suite stays fast.
    config_overrides = {"habitat_baselines.rl.ddppo.hidden_size": 32}
    checks = ("contract", "output_spec", "checkpoint", "frozen", "training")
    output_spec = {
        "mean": TensorSpec(("B", "T", ACTION_DIM), min=-1, max=1),
        "std": TensorSpec(("B", "T", ACTION_DIM), min=1e-6),
        "value": TensorSpec(("B", "T")),
    }
    check_options = {
        "inference_contract": {"atol": 1e-5, "rtol": 1e-4},
        "overfits_one_batch": {"steps": 300, "lr": 3e-3},
    }

    def make_model(self):
        return NavPolicy(self.config)  # self.config: the current config, overrides applied

    def make_inputs(self, make_tensor):
        categories = self.config["habitat"]["task"]["num_object_categories"]
        return {
            "rgb": make_tensor(4, 6, 3, 32, 32, kind="uniform", low=0, high=1),
            "objectgoal": make_tensor(4, 6, dtype=torch.long, low=0, high=categories),
        }

    def loss(self, output, inputs):
        # A PPO-flavoured stand-in: the log-likelihood of fixed actions under the policy's
        # Gaussian (so mean *and* std learn) plus a value regression.
        # Try `F.mse_loss(output.mean, actions)` alone: test_all_params_receive_grad fails,
        # naming `action_log_std`, because the std no longer gets a gradient.
        actions = torch.linspace(-0.5, 0.5, ACTION_DIM, device=output.mean.device).expand_as(output.mean)
        nll = F.gaussian_nll_loss(output.mean, actions, output.std.pow(2))
        return nll + F.mse_loss(output.value, torch.zeros_like(output.value))

    def frozen_parameters(self):
        # Mirrors `train_encoder: false` in the config; nothing is frozen otherwise.
        return [] if self.config["habitat_baselines"]["rl"]["ddppo"]["train_encoder"] else ["visual_encoder.*"]

    # A hand-written test runs once per config too: request `model_config`.
    def test_rollout_matches_the_full_episode(self, torch_seed, torch_device, make_tensor, model_config):
        policy = NavPolicy(model_config).to(torch_device).eval()
        episode = self.make_inputs(make_tensor)
        # Acting step by step with the recurrent state (KV cache or GRU hidden state) must
        # give what the PPO update computes on the whole episode.
        assert_streaming_equivalent(
            policy,       # called like a model: the dict of observations becomes keyword arguments
            policy.step,  # (chunk, state) -> (output, state)
            episode,
            chunk_sizes=[1, 4],
            init_state=lambda: None,
        )
