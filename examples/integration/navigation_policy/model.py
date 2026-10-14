"""Project code: an object-navigation policy built from a habitat-baselines-style config.

The recurrent core runs over a whole episode (training) or one step at a time with a cache
(acting in the environment); both paths must agree.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn

ACTION_DIM = 2  # linear and angular velocity


class PolicyOutput(NamedTuple):
    mean: torch.Tensor   # (B, T, ACTION_DIM)
    std: torch.Tensor    # (B, T, ACTION_DIM), positive
    value: torch.Tensor  # (B, T)


class VisualEncoder(nn.Module):
    """Stands in for VC-1: RGB frames -> features."""

    def __init__(self, out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, out),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:  # (B, T, 3, H, W) -> (B, T, out)
        b, t = rgb.shape[:2]
        return self.net(rgb.flatten(0, 1)).view(b, t, -1)


def rotary(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """RoPE on (B, heads, T, head_dim) at absolute `positions` (T,)."""
    half = x.shape[-1] // 2
    freqs = 10000 ** (-torch.arange(half, device=x.device, dtype=x.dtype) / half)
    angles = positions.to(x.dtype)[:, None] * freqs[None]
    cos, sin = angles.cos(), angles.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalBlock(nn.Module):
    def __init__(self, dim: int, heads: int, rope: bool):
        super().__init__()
        self.heads, self.rope = heads, rope
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.out = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, cache, start):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        if self.rope:
            positions = torch.arange(start, start + t, device=x.device)
            q, k = rotary(q, positions), rotary(k, positions)
        if cache is not None:
            k, v = torch.cat([cache[0], k], dim=2), torch.cat([cache[1], v], dim=2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        future = torch.ones(t, k.shape[2], dtype=torch.bool, device=x.device).triu(k.shape[2] - t + 1)
        attn = scores.masked_fill(future, float("-inf")).softmax(-1)
        x = x + self.out((attn @ v).transpose(1, 2).reshape(b, t, d))
        return x + self.mlp(self.ln2(x)), (k, v)


class NavPolicy(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        ddppo = config["habitat_baselines"]["rl"]["ddppo"]
        agent = config["habitat_baselines"]["rl"]["policy"]["main_agent"]
        hidden = ddppo["hidden_size"]
        self.rnn_type, self.train_encoder = ddppo["rnn_type"], ddppo["train_encoder"]
        self.visual_encoder = VisualEncoder(hidden)
        self.goal_embedding = nn.Embedding(config["habitat"]["task"]["num_object_categories"], hidden)
        if self.rnn_type == "transformer":
            tcfg = agent["transformer_config"]
            self.rope = tcfg["pos_emb"] == "rope"
            if not self.rope:
                self.positions = nn.Embedding(config["habitat"]["environment"]["max_episode_steps"], hidden)
            self.blocks = nn.ModuleList(CausalBlock(hidden, tcfg["n_heads"], self.rope) for _ in range(tcfg["n_layers"]))
        else:
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.action_mean = nn.Linear(hidden, ACTION_DIM)
        self.action_log_std = nn.Parameter(torch.full((ACTION_DIM,), -0.5))
        self.critic = nn.Linear(hidden, 1)
        if not self.train_encoder:
            self.visual_encoder.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.train_encoder:
            # A frozen encoder must also keep its BatchNorm statistics frozen.
            # Try deleting this line: test_frozen fails on encoder running_mean/running_var.
            self.visual_encoder.eval()
        return self

    def _core(self, x, state):
        if self.rnn_type == "gru":
            out, h = self.gru(x, state)
            return out, h
        # Try `start = 0` (positions restart at every step): the transformer configs fail
        # test_rollout_matches_the_full_episode; the GRU, which has no positions, passes.
        start = 0 if state is None else state[0][0].shape[2]
        if not self.rope:
            x = x + self.positions(torch.arange(start, start + x.shape[1], device=x.device))
        caches = []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, None if state is None else state[i], start)
            caches.append(cache)
        return x, tuple(caches)

    def step(self, obs: dict, state=None):
        """Any number of new steps with the recurrent state: returns (output, new_state)."""
        x = self.visual_encoder(obs["rgb"]) + self.goal_embedding(obs["objectgoal"])
        h, state = self._core(x, state)
        mean = torch.tanh(self.action_mean(h))
        std = self.action_log_std.exp().expand_as(mean)
        return PolicyOutput(mean, std, self.critic(h).squeeze(-1)), state

    def forward(self, rgb: torch.Tensor, objectgoal: torch.Tensor) -> PolicyOutput:
        """A whole episode segment at once (what PPO's update sees)."""
        return self.step({"rgb": rgb, "objectgoal": objectgoal})[0]
