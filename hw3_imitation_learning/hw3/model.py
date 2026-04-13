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


class MultiTaskPolicy(BasePolicy):
    """Goal-conditioned policy for the multicube scene.
    """

    # Slices into the 15-dim raw/normalized state vector
    _EE_XYZ = slice(0, 3)   # ee position (x, y, z)
    _GRIPPER = 3             # gripper scalar
    _ROBOT = slice(0, 4)    # ee_xyz + gripper
    # Each cube contributes 2 dims (xy); we take index 1 (y) from each pair
    _RED_XY = slice(4, 6)   # red cube xy
    _GREEN_XY = slice(6, 8) # green cube xy
    _BLUE_XY = slice(8, 10) # blue cube xy
    _RED_Y = 5              # index of red cube's y in state vec (4 + 1)
    _GREEN_Y = 7            # index of green cube's y in state vec (6 + 1)
    _BLUE_Y = 9             # index of blue cube's y in state vec (8 + 1)
    _GOAL = slice(10, 13)   # one-hot [red, green, blue]
    _GOAL_POS_XY = slice(13, 15)  # bin position xy
    _GOAL_POS_Y = 14        # index of bin y in state vec (13 + 1)
    _NET_INPUT_DIM = 7  # (ee-bin)(3) + (ee-cube)(3) + gripper(1)

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        d_model: int = 512,
        depth: int = 4,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        use_full_state: bool = False,
        use_relative_state: bool = False,  # kept for backwards compat, always True when not use_full_state
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.use_full_state = use_full_state
        layers = []
        if use_full_state:
            in_dim = state_dim
        else:
            in_dim = self._NET_INPUT_DIM
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, d_model))
            if use_layer_norm:
                layers.append(nn.LayerNorm(d_model))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = d_model
        layers.append(nn.Linear(d_model, chunk_size * action_dim))
        self.net = nn.Sequential(*layers)

    def _to_net_input(self, state: torch.Tensor) -> torch.Tensor:
        """Reduce 15-dim state to 7-dim relative input.

        Computes:
            ee_xyz - bin_xyz  (bin z assumed 0)  →  3 dims
            ee_xyz - cube_xyz (cube z assumed 0) →  3 dims
            gripper                               →  1 dim
        """
        ee_xyz = state[:, self._EE_XYZ]                        # (B, 3)
        gripper = state[:, self._GRIPPER].unsqueeze(1)         # (B, 1)
        bin_xy = state[:, self._GOAL_POS_XY]                   # (B, 2)

        cube_xys = torch.stack([                               # (B, 3, 2)
            state[:, self._RED_XY],
            state[:, self._GREEN_XY],
            state[:, self._BLUE_XY],
        ], dim=1)
        goal_idx = state[:, self._GOAL].argmax(dim=-1)         # (B,)
        target_xy = cube_xys[torch.arange(state.shape[0], device=state.device), goal_idx]  # (B, 2)

        ee_minus_bin = ee_xyz - torch.cat([bin_xy, torch.zeros_like(gripper)], dim=-1)    # (B, 3)
        ee_minus_cube = ee_xyz - torch.cat([target_xy, torch.zeros_like(gripper)], dim=-1)  # (B, 3)
        return torch.cat([ee_minus_bin, ee_minus_cube, gripper], dim=-1)  # (B, 7)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        net_input = state if self.use_full_state else self._to_net_input(state)
        return self.net(net_input).view(-1, self.chunk_size, self.action_dim)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(self.forward(state), action_chunk)

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state)


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
    use_full_state: bool = False,
    use_relative_state: bool = False,
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
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            use_full_state=use_full_state,
            use_relative_state=use_relative_state,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
