# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared per-contact encoding with masked per-body pooling."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from isaaclab_neural.contacts.contact_set_schema import (
    CONTACT_TOKEN_BODY_SLOT_INDEX,
    CONTACT_TOKEN_DIM,
    CONTACT_TOKEN_GEOMETRY_SLICE,
    CONTACT_TOKEN_OTHER_BODY_SLOT_INDEX,
    CONTACT_TOKEN_OTHER_DYNAMIC_INDEX,
    CONTACT_TOKEN_VALID_INDEX,
)


class SharedPerBodyContactEncoder(nn.Module):
    """Encode canonical contact tokens into low-dimensional per-body latents.

    The per-contact MLP is shared by all primary bodies. Valid contacts are
    grouped by their local primary body slot and pooled with a masked mean plus
    a contact-count projection. Bodies without valid contacts produce zeros.
    """

    def __init__(
        self,
        *,
        num_bodies: int,
        body_latent_dim: int = 16,
        hidden_dim: int = 64,
        max_other_bodies: int = 32,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize the shared per-body contact encoder.

        Args:
            num_bodies: Number of primary bodies in stable local-slot order.
            body_latent_dim: Output feature count for each primary body.
            hidden_dim: Hidden feature count in the shared per-contact MLP.
            max_other_bodies: Number of known other-body identity embeddings.
            device: Device on which to create module parameters.
        """
        super().__init__()
        if num_bodies <= 0:
            raise ValueError("num_bodies must be a positive integer.")
        if body_latent_dim <= 0:
            raise ValueError("body_latent_dim must be a positive integer.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer.")
        if max_other_bodies <= 0:
            raise ValueError("max_other_bodies must be a positive integer.")

        self.num_bodies = num_bodies
        self.body_latent_dim = body_latent_dim
        self.out_features = num_bodies * body_latent_dim

        geometry_dim = CONTACT_TOKEN_DIM - CONTACT_TOKEN_GEOMETRY_SLICE.start
        self.contact_encoder = nn.Sequential(
            nn.Linear(geometry_dim, hidden_dim, device=device),
            nn.SiLU(),
            nn.Linear(hidden_dim, body_latent_dim, device=device),
        )
        self.other_body_embed = nn.Embedding(max_other_bodies, body_latent_dim, device=device)
        self.other_static_embed = nn.Parameter(torch.zeros(body_latent_dim, device=device))
        self.other_foreign_embed = nn.Parameter(torch.zeros(body_latent_dim, device=device))
        self.count_proj = nn.Linear(1, body_latent_dim, device=device)

    def forward(self, contact_tokens: torch.Tensor) -> torch.Tensor:
        """Return per-body latents with shape ``(..., num_bodies, body_latent_dim)``.

        Args:
            contact_tokens: Canonical 17-D tokens with shape ``(..., K, 17)``.

        Returns:
            Per-body contact latents in stable local body-slot order.
        """
        if contact_tokens.ndim < 3:
            raise ValueError("contact_tokens must have shape (..., K, 17).")
        if contact_tokens.shape[-1] != CONTACT_TOKEN_DIM:
            raise ValueError(f"Expected contact_dim={CONTACT_TOKEN_DIM}, got {contact_tokens.shape[-1]}.")
        if not contact_tokens.is_floating_point():
            raise TypeError("contact_tokens must be a floating-point tensor.")

        *leading, num_tokens, _ = contact_tokens.shape
        num_sets = math.prod(leading)
        flat_tokens = contact_tokens.reshape(num_sets, num_tokens, CONTACT_TOKEN_DIM)

        body_slots = flat_tokens[..., CONTACT_TOKEN_BODY_SLOT_INDEX].long()
        valid = flat_tokens[..., CONTACT_TOKEN_VALID_INDEX] > 0.5
        valid = valid & (body_slots >= 0) & (body_slots < self.num_bodies)
        contact_features = self._encode_contacts(flat_tokens)

        weights = valid.unsqueeze(-1).to(contact_features.dtype)
        safe_body_slots = body_slots.clamp(min=0, max=self.num_bodies - 1)
        set_offsets = torch.arange(num_sets, device=contact_tokens.device).unsqueeze(-1) * self.num_bodies
        segment_ids = set_offsets + safe_body_slots

        pooled_sums = contact_features.new_zeros((num_sets * self.num_bodies, self.body_latent_dim))
        pooled_sums.index_add_(
            0, segment_ids.reshape(-1), (contact_features * weights).reshape(-1, self.body_latent_dim)
        )
        counts = contact_features.new_zeros((num_sets * self.num_bodies, 1))
        counts.index_add_(0, segment_ids.reshape(-1), weights.reshape(-1, 1))

        pooled_means = pooled_sums / counts.clamp_min(1.0)
        latents = pooled_means + self.count_proj(torch.log1p(counts))
        latents = torch.where(counts > 0, latents, torch.zeros_like(latents))
        return latents.reshape(*leading, self.num_bodies, self.body_latent_dim)

    def _encode_contacts(self, tokens: torch.Tensor) -> torch.Tensor:
        geometry = tokens[..., CONTACT_TOKEN_GEOMETRY_SLICE]
        encoded = self.contact_encoder(geometry)

        other_ids = tokens[..., CONTACT_TOKEN_OTHER_BODY_SLOT_INDEX].long()
        other_dynamic = tokens[..., CONTACT_TOKEN_OTHER_DYNAMIC_INDEX] > 0.5
        other_is_known = other_ids >= 0
        safe_other_ids = other_ids.clamp(min=0, max=self.other_body_embed.num_embeddings - 1)
        known_other_embed = self.other_body_embed(safe_other_ids)
        unknown_other_embed = torch.where(
            other_dynamic.unsqueeze(-1),
            self.other_foreign_embed.expand_as(known_other_embed),
            self.other_static_embed.expand_as(known_other_embed),
        )
        return encoded + torch.where(other_is_known.unsqueeze(-1), known_other_embed, unknown_other_embed)
