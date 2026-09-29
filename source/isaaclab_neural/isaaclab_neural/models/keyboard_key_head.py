# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factorized robot/key state model components for articulated keyboards.

Each key is a one-degree-of-freedom joint that interacts with the robot only
through contacts. The temporal trunk sees robot coordinates, non-key contact
latents and an order-independent summary of all keys; one key decoder with
shared parameters then predicts every key's transition from the trunk context,
that key's own short history and its own contact latent. Key inputs and outputs
use statistics pooled over all keys so that shared parameters see common units.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from isaaclab_neural.utils.running_mean_std import RunningMeanStd

REQUIRED_LOW_DIM_INPUTS = ("states_embedding", "joint_f", "gravity_dir")
_EPSILON = 1.0e-5


def _mlp(in_features: int, hidden_sizes: list[int], out_features: int, device: Any) -> nn.Sequential:
    layers: list[nn.Module] = []
    for size in hidden_sizes:
        layers += [nn.Linear(in_features, size, device=device), nn.SiLU()]
        in_features = size
    layers.append(nn.Linear(in_features, out_features, device=device))
    return nn.Sequential(*layers)


def pooled_moments(mean: torch.Tensor, var: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the mean and variance of the mixture of equally weighted per-coordinate distributions."""
    pooled_mean = mean.mean()
    return pooled_mean, (var + mean.square()).mean() - pooled_mean.square()


class KeyboardKeyHead(nn.Module):
    """Trunk-input builder and shared per-key decoder for fixed-base keyboard articulations.

    Args:
        cfg: ``network.key_head`` settings. ``key_q_start`` and ``num_keys`` locate the
            contiguous key joints in the generalized position block; ``key_body_slot_start``
            locates their contact-latent body slots.
        output_dim: Full prediction dimension, twice the number of scalar joints.
        num_bodies: Number of contact-latent body slots.
        body_latent_dim: Contact latent size per body.
        trunk_dim: Temporal trunk output size.
        device: Device for parameters.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        output_dim: int,
        num_bodies: int,
        body_latent_dim: int,
        trunk_dim: int,
        device: Any = None,
    ):
        super().__init__()
        if output_dim % 2:
            raise ValueError("Keyboard key head requires scalar joints with equal position and velocity sizes.")
        self.num_q = output_dim // 2
        self.num_keys = int(cfg["num_keys"])
        key_start = int(cfg["key_q_start"])
        slot_start = int(cfg["key_body_slot_start"])
        if self.num_keys <= 0 or key_start < 0 or key_start + self.num_keys > self.num_q:
            raise ValueError("Key joints must lie inside the generalized position block.")
        if self.num_keys >= self.num_q:
            raise ValueError("Keyboard key head requires at least one robot joint.")
        if slot_start < 0 or slot_start + self.num_keys > num_bodies:
            raise ValueError("Key contact body slots must lie inside the contact-latent bodies.")
        self.history_steps = int(cfg.get("history_steps", 3))
        if self.history_steps <= 0:
            raise ValueError("Key head history_steps must be positive.")
        key_q = torch.arange(key_start, key_start + self.num_keys)
        robot_q = torch.tensor(
            [index for index in range(self.num_q) if not key_start <= index < key_start + self.num_keys]
        )
        slots = torch.arange(slot_start, slot_start + self.num_keys)
        other_slots = torch.tensor(
            [slot for slot in range(num_bodies) if not slot_start <= slot < slot_start + self.num_keys]
        )
        for name, value in (
            ("key_q", key_q),
            ("key_qd", key_q + self.num_q),
            ("robot_q", robot_q),
            ("robot_state", torch.cat([robot_q, robot_q + self.num_q])),
            ("key_slots", slots),
            ("other_slots", other_slots),
        ):
            self.register_buffer(name, value.to(device), persistent=False)
        self.robot_state_dim = 2 * robot_q.numel()
        embed_dim = int(cfg.get("embed_dim", 8))
        summary_dim = int(cfg.get("summary_dim", 64))
        context_dim = int(cfg.get("context_dim", 64))
        hidden = [int(size) for size in cfg.get("hidden_sizes", [128, 128])]
        key_feature_dim = 2 + body_latent_dim
        self.key_embedding = nn.Embedding(self.num_keys, embed_dim, device=device)
        self.summary_net = _mlp(key_feature_dim + embed_dim, [summary_dim], summary_dim, device)
        self.context_proj = nn.Linear(trunk_dim, context_dim, device=device)
        self.decoder = _mlp(context_dim + self.history_steps * key_feature_dim + embed_dim, hidden, 2, device)
        self.trunk_input_dim = (
            self.robot_state_dim + robot_q.numel() + 3 + other_slots.numel() * body_latent_dim + summary_dim
        )

    def _raw(self, value: torch.Tensor, rms: RunningMeanStd | None, index: torch.Tensor) -> torch.Tensor:
        """Undo per-coordinate input normalization for the selected coordinates."""
        selected = value.index_select(-1, index)
        if rms is None:
            return selected
        return selected * torch.sqrt(rms.var.index_select(0, index) + _EPSILON) + rms.mean.index_select(0, index)

    def key_features(
        self, states_embedding: torch.Tensor, contact_latents: torch.Tensor, states_rms: RunningMeanStd | None
    ) -> torch.Tensor:
        """Return pooled-normalized key position, velocity and contact latent, shape ``[..., keys, 2 + latent]``."""
        channels = []
        for index in (self.key_q, self.key_qd):
            raw = self._raw(states_embedding, states_rms, index)
            if states_rms is not None:
                mean, var = pooled_moments(
                    states_rms.mean.index_select(0, index), states_rms.var.index_select(0, index)
                )
                raw = (raw - mean) / torch.sqrt(var + _EPSILON**2)
            channels.append(raw)
        latents = contact_latents.index_select(-2, self.key_slots)
        return torch.cat([torch.stack(channels, dim=-1), latents], dim=-1)

    def trunk_input(
        self, input_dict: dict[str, torch.Tensor], contact_latents: torch.Tensor, key_features: torch.Tensor
    ):
        """Concatenate robot coordinates, non-key contact latents and the key summary for each timestep."""
        embedding = self.key_embedding.weight.expand(*key_features.shape[:-1], -1)
        summary = self.summary_net(torch.cat([key_features, embedding], dim=-1)).sum(dim=-2)
        others = contact_latents.index_select(-2, self.other_slots).flatten(start_dim=-2)
        return torch.cat(
            [
                input_dict["states_embedding"].index_select(-1, self.robot_state),
                input_dict["joint_f"].index_select(-1, self.robot_q),
                input_dict["gravity_dir"],
                others,
                summary,
            ],
            dim=-1,
        )

    def decode(self, trunk: torch.Tensor, key_features: torch.Tensor) -> torch.Tensor:
        """Return pooled-normalized key position and velocity changes, shape ``[B, T, keys, 2]``."""
        history = []
        for lag in range(self.history_steps):
            if lag == 0:
                shifted = key_features
            else:
                # Early timesteps repeat the first available frame, matching the causal trunk window.
                pad = key_features[:, :1].expand(-1, min(lag, key_features.shape[1]), -1, -1)
                shifted = torch.cat([pad, key_features[:, :-lag]], dim=1)[:, : key_features.shape[1]]
            history.append(shifted)
        context = self.context_proj(trunk).unsqueeze(-2).expand(-1, -1, self.num_keys, -1)
        embedding = self.key_embedding.weight.expand(*key_features.shape[:-1], -1)
        return self.decoder(torch.cat([context, *history, embedding], dim=-1))

    def assemble(
        self, robot_normalized: torch.Tensor, key_normalized: torch.Tensor, output_rms: RunningMeanStd
    ) -> torch.Tensor:
        """Un-normalize robot outputs per coordinate and key outputs with pooled statistics, in original order."""
        output = robot_normalized.new_zeros((*robot_normalized.shape[:-1], 2 * self.num_q))
        mean, var = output_rms.mean, output_rms.var
        robot = robot_normalized * torch.sqrt(var.index_select(0, self.robot_state) + _EPSILON)
        output[..., self.robot_state] = robot + mean.index_select(0, self.robot_state)
        for channel, index in enumerate((self.key_q, self.key_qd)):
            pooled_mean, pooled_var = pooled_moments(mean.index_select(0, index), var.index_select(0, index))
            output[..., index] = key_normalized[..., channel] * torch.sqrt(pooled_var + _EPSILON**2) + pooled_mean
        return output
