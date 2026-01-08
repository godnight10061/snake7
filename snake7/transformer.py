from __future__ import annotations

from typing import Literal, Optional

import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    Transformer encoder over a short history of vector observations.

    Expects observations shaped like (seq_len, obs_dim) (or batched as (batch, seq_len, obs_dim)).
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        *,
        d_model: int = 64,
        n_head: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
        dim_feedforward: Optional[int] = None,
        pooling: Literal["last", "mean"] = "last",
    ) -> None:
        if observation_space.shape is None or len(observation_space.shape) != 2:
            raise ValueError("TransformerFeaturesExtractor expects observation shape (seq_len, obs_dim)")

        seq_len, obs_dim = observation_space.shape
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if n_head <= 0:
            raise ValueError("n_head must be > 0")
        if n_layers <= 0:
            raise ValueError("n_layers must be > 0")
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if pooling not in ("last", "mean"):
            raise ValueError("pooling must be 'last' or 'mean'")

        super().__init__(observation_space, features_dim=d_model)

        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.pooling = pooling

        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        self.embed = nn.Linear(self.obs_dim, d_model)
        self.pos_embed = nn.Parameter(th.zeros(self.seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.embed(observations)
        x = x + self.pos_embed.unsqueeze(0)
        x = self.encoder(x)

        if self.pooling == "mean":
            x = x.mean(dim=1)
        else:
            x = x[:, -1, :]

        return self.norm(x)

