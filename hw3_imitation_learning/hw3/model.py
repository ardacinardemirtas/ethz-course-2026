"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """Compute training loss for a batch."""
        raise NotImplementedError

    @abc.abstractmethod
    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""
        raise NotImplementedError


# TODO: Students implement ObstaclePolicy here.
class ObstaclePolicy(BasePolicy):
    """Predicts action chunks with an MSE loss.

    A simple MLP that maps a state vector to a flat action chunk
    (chunk_size * action_dim) and reshapes to (B, chunk_size, action_dim).
    """

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int, d_model: int, depth: int, dropout: float = 0.1, use_layer_norm: bool = True) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers = []
        in_dim = self.state_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, d_model))
            if use_layer_norm:
                layers.append(nn.LayerNorm(d_model))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = d_model
        layers.append(nn.Linear(d_model, self.chunk_size * self.action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        return self.net(state).view(-1, self.chunk_size, self.action_dim)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(self.forward(state), action_chunk)

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state)


# TODO: Students implement MultiTaskPolicy here.
class MultiTaskPolicy(BasePolicy):
    """Goal-conditioned policy for the multicube scene."""

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        raise NotImplementedError


PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    # TODO,
    chunk_size: int = 10,
    d_model: int = 512,
    depth: int = 4,
    dropout: float = 0.1,
    use_layer_norm: bool = True,
) -> BasePolicy:
    if policy_type == "obstacle":
        return ObstaclePolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            # TODO: Build with your chosen specifications
            chunk_size=chunk_size,
            depth=depth,
            d_model=d_model,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=chunk_size,
            # TODO: Build with your chosen specifications
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
