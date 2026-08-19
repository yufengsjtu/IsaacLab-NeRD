# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Body-routed Deep Sets model for owner-frame Active15 contact tokens."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from isaaclab_neural.contacts.contact_set_schema import (
    ACTIVE15_BODY_SLOT_INDEX,
    ACTIVE15_FEATURE_SLICE,
    ACTIVE15_TOKEN_DIM,
    ACTIVE15_VALID_INDEX,
)


class BodyRoutedActive15Encoder(nn.Module):
    r"""Apply shared phi, owner-body sum pooling, then shared rho."""

    def __init__(
        self,
        *,
        num_bodies: int,
        body_latent_dim: int = 64,
        hidden_dim: int = 32,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        if num_bodies <= 0 or body_latent_dim <= 0 or hidden_dim <= 0:
            raise ValueError("num_bodies, body_latent_dim, and hidden_dim must be positive.")
        self.num_bodies = int(num_bodies)
        self.body_latent_dim = int(body_latent_dim)
        self.out_features = self.num_bodies * self.body_latent_dim
        self.phi = nn.Sequential(
            nn.Linear(ACTIVE15_FEATURE_SLICE.stop - ACTIVE15_FEATURE_SLICE.start, hidden_dim, device=device),
            nn.SiLU(),
            nn.Linear(hidden_dim, body_latent_dim, device=device),
        )
        self.rho = nn.Linear(body_latent_dim, body_latent_dim, device=device)

    def forward(self, contact_tokens: torch.Tensor) -> torch.Tensor:
        """Return owner-body latents with the leading input dimensions preserved."""
        if contact_tokens.ndim < 3 or contact_tokens.shape[-1] != ACTIVE15_TOKEN_DIM:
            raise ValueError(f"Expected contact tokens with shape (..., K, {ACTIVE15_TOKEN_DIM}).")
        if not contact_tokens.is_floating_point():
            raise TypeError("contact_tokens must be a floating-point tensor.")

        *leading, num_tokens, _ = contact_tokens.shape
        num_sets = math.prod(leading)
        flat_tokens = contact_tokens.reshape(num_sets, num_tokens, ACTIVE15_TOKEN_DIM)
        body_slots = flat_tokens[..., ACTIVE15_BODY_SLOT_INDEX].round().long()
        valid = flat_tokens[..., ACTIVE15_VALID_INDEX] > 0.5
        valid = valid & (body_slots >= 0) & (body_slots < self.num_bodies)
        encoded = self.phi(flat_tokens[..., ACTIVE15_FEATURE_SLICE])
        weights = valid.unsqueeze(-1).to(encoded.dtype)

        safe_slots = body_slots.clamp(min=0, max=self.num_bodies - 1)
        set_offsets = torch.arange(num_sets, device=contact_tokens.device).unsqueeze(-1) * self.num_bodies
        segment_ids = set_offsets + safe_slots
        pooled = encoded.new_zeros((num_sets * self.num_bodies, self.body_latent_dim))
        pooled.index_add_(
            0,
            segment_ids.reshape(-1),
            (encoded * weights).reshape(-1, self.body_latent_dim),
        )
        counts = encoded.new_zeros((num_sets * self.num_bodies, 1))
        counts.index_add_(0, segment_ids.reshape(-1), weights.reshape(-1, 1))

        latents = self.rho(pooled)
        latents = torch.where(counts > 0, latents, torch.zeros_like(latents))
        return latents.reshape(*leading, self.num_bodies, self.body_latent_dim)
