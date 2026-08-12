# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Masked set encoder for contact token inputs."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from isaaclab_neural.contacts.contact_set_encoder import contact_token_validity_mask
from isaaclab_neural.contacts.contact_set_schema import (
    CONTACT_TOKEN_CATEGORICAL_CHANNELS,
    CONTACT_TOKEN_DIM,
    CONTACT_TOKEN_GEOMETRY_SLICE,
)


class ContactSetEncoderBlock(nn.Module):
    """Encode a variable-size contact token set into a fixed-width embedding."""

    def __init__(
        self,
        *,
        contact_dim: int = CONTACT_TOKEN_DIM,
        hidden_size: int = 192,
        num_layers: int = 2,
        num_heads: int = 4,
        num_latent_queries: int = 8,
        max_bodies: int = 32,
        max_other_bodies: int = 32,
        dropout: float = 0.0,
        device: str | torch.device = "cuda:0",
    ) -> None:
        super().__init__()
        if contact_dim < 5:
            raise ValueError("contact_dim must include validity, identity, and geometry channels.")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        if num_layers <= 0:
            raise ValueError("num_layers must be a positive integer.")
        if num_latent_queries <= 0:
            raise ValueError("num_latent_queries must be a positive integer.")

        self.contact_dim = contact_dim
        self.hidden_size = hidden_size
        self.geometry_dim = contact_dim - len(CONTACT_TOKEN_CATEGORICAL_CHANNELS)
        self.num_latent_queries = num_latent_queries
        self.out_features = hidden_size

        self.body_embed = nn.Embedding(max_bodies, hidden_size, device=device)
        self.other_body_embed = nn.Embedding(max_other_bodies, hidden_size, device=device)
        self.other_static_embed = nn.Parameter(torch.zeros(hidden_size, device=device))
        self.other_foreign_embed = nn.Parameter(torch.zeros(hidden_size, device=device))
        self.null_contact = nn.Parameter(torch.zeros(hidden_size, device=device))
        self.latent_queries = nn.Parameter(torch.empty(num_latent_queries, hidden_size, device=device))
        self.pool_query = nn.Parameter(torch.empty(1, hidden_size, device=device))
        self.geometry_proj = nn.Linear(self.geometry_dim, hidden_size, device=device)
        self.count_proj = nn.Linear(1, hidden_size, device=device)
        self.sum_proj = nn.Linear(self.geometry_dim, hidden_size, device=device)
        self.contact_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            device=device,
        )
        self.contact_attention_norm = nn.LayerNorm(hidden_size, device=device)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(device)
        self.pool_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            device=device,
        )
        self.pool_attention_norm = nn.LayerNorm(hidden_size, device=device)
        # Fuse attention pool (H) with cardinality branch (H).
        self.output_proj = nn.Linear(hidden_size * 2, hidden_size, device=device)
        nn.init.normal_(self.latent_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.pool_query, mean=0.0, std=0.02)

    def forward(self, contact_tokens: torch.Tensor) -> torch.Tensor:
        """Return contact-set embeddings with shape ``(..., out_features)``.

        Args:
            contact_tokens: Contact tokens with shape ``(..., K, contact_dim)``.
        """
        *leading, num_tokens, contact_dim = contact_tokens.shape
        if contact_dim != self.contact_dim:
            raise ValueError(f"Expected contact_dim={self.contact_dim}, got {contact_dim}.")
        # Keep the set axis so attention pools over contacts, not flattened rows.
        flat_tokens = contact_tokens.reshape(math.prod(leading), num_tokens, self.contact_dim)
        valid = contact_token_validity_mask(flat_tokens)
        token_embed = self._embed_tokens(flat_tokens, valid)
        pooled = self._encode_set(token_embed, valid)  # [B*T, H]
        cardinality = self._cardinality_branch(flat_tokens, valid)
        fused = torch.cat((pooled, cardinality), dim=-1)  # [B*T, 2*H]
        return self.output_proj(fused).reshape(*leading, self.out_features)  # [B, T, H]

    def _embed_tokens(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        geometry = tokens[..., CONTACT_TOKEN_GEOMETRY_SLICE]
        projected = self.geometry_proj(geometry)

        body_ids = tokens[..., 1].long().clamp(min=0, max=self.body_embed.num_embeddings - 1)
        other_ids = tokens[..., 2].long()
        other_dynamic = tokens[..., 3].long().clamp(min=0, max=1)

        body_embed = self.body_embed(body_ids)
        other_is_known = other_ids >= 0
        safe_other = other_ids.clamp(min=0, max=self.other_body_embed.num_embeddings - 1)
        other_embed = self.other_body_embed(safe_other)
        # Match PhysicsNeMo: known others use entity embeddings only; unknown
        # others use dedicated static / foreign parameters.
        other_embed = torch.where(
            other_is_known.unsqueeze(-1),
            other_embed,
            torch.where(
                other_dynamic.bool().unsqueeze(-1),
                self.other_foreign_embed.expand_as(other_embed),
                self.other_static_embed.expand_as(other_embed),
            ),
        )
        token_embed = projected + body_embed + other_embed
        return torch.where(valid.unsqueeze(-1), token_embed, torch.zeros_like(token_embed))

    def _encode_set(self, token_embed: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch = token_embed.shape[0]
        null_contact = self.null_contact.view(1, 1, -1).expand(batch, 1, -1)
        contacts = torch.cat((token_embed, null_contact), dim=1)
        # The learned null contact is visible only for empty sets. This avoids
        # all-masked attention rows without changing non-empty contact pooling.
        null_padding = valid.any(dim=1, keepdim=True)
        key_padding = torch.cat((~valid, null_padding), dim=1)

        latent_queries = self.latent_queries.unsqueeze(0).expand(batch, -1, -1)
        attended, _ = self.contact_attention(
            query=latent_queries,
            key=contacts,
            value=contacts,
            key_padding_mask=key_padding,
            need_weights=False,
        )
        latents = self.contact_attention_norm(latent_queries + attended)
        latents = self.encoder(latents)

        pool_query = self.pool_query.unsqueeze(0).expand(batch, -1, -1)
        pooled, _ = self.pool_attention(
            query=pool_query,
            key=latents,
            value=latents,
            need_weights=False,
        )
        return self.pool_attention_norm(pool_query + pooled)[:, 0, :]

    def _cardinality_branch(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        geometry = tokens[..., CONTACT_TOKEN_GEOMETRY_SLICE]
        weights = valid.unsqueeze(-1).to(geometry.dtype)
        masked_sum = (geometry * weights).sum(dim=-2)
        sum_embed = self.sum_proj(masked_sum)
        counts = valid.sum(dim=-1, keepdim=True).to(geometry.dtype)
        count_embed = self.count_proj(torch.log1p(counts))
        # Match PhysicsNeMo: add sum/count into one H-dim branch before fusion.
        return sum_embed + count_embed  # [B*T, H]
