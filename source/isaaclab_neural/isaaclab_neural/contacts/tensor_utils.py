# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tensor helpers for fixed-slot contact inputs."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import torch

from isaaclab_neural.utils.running_mean_std import RunningMeanStd

MIN_CONTACT_RMS_SAMPLES = 1024


class MaskedContactMoments:
    """Accumulate independent per-slot moments from valid contact values."""

    def __init__(self, num_contacts: int, channels: int, device: torch.device | str):
        self.count = torch.zeros((num_contacts, 1), dtype=torch.float64, device=device)
        self.sum = torch.zeros((num_contacts, channels), dtype=torch.float64, device=device)
        self.square_sum = torch.zeros((num_contacts, channels), dtype=torch.float64, device=device)

    @classmethod
    def from_batch(cls, value: torch.Tensor, contact_masks: torch.Tensor) -> MaskedContactMoments:
        """Create an accumulator matching one flattened contact field."""
        reshaped = reshape_contact_field(value, contact_masks)
        return cls(contact_masks.shape[-1], reshaped.shape[-1], value.device)

    def update(self, value: torch.Tensor, contact_masks: torch.Tensor) -> None:
        """Accumulate valid values without mixing contact slots."""
        reshaped = reshape_contact_field(value, contact_masks).to(torch.float64)
        weights = contact_masks.bool().unsqueeze(-1).to(torch.float64)
        reduce_dims = tuple(range(reshaped.ndim - 2))
        self.count += weights.sum(dim=reduce_dims)
        self.sum += (reshaped * weights).sum(dim=reduce_dims)
        self.square_sum += (reshaped.square() * weights).sum(dim=reduce_dims)

    def synchronize(self) -> None:
        """Sum raw moments across an initialized distributed process group."""
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return
        for tensor in (self.count, self.sum, self.square_sum):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    def finalize(
        self,
        device: torch.device | str,
        *,
        min_samples: int = MIN_CONTACT_RMS_SAMPLES,
    ) -> tuple[RunningMeanStd, torch.Tensor]:
        """Create per-slot RMS, using pooled statistics for sparse valid slots."""
        valid = self.count > 0
        safe_count = self.count.clamp_min(1.0)
        mean = self.sum / safe_count
        variance = (self.square_sum / safe_count - mean.square()).clamp_min(0.0)

        pooled_count = self.count.sum()
        if pooled_count > 0:
            pooled_mean = self.sum.sum(dim=0, keepdim=True) / pooled_count
            pooled_variance = (
                self.square_sum.sum(dim=0, keepdim=True) / pooled_count - pooled_mean.square()
            ).clamp_min(0.0)
            sparse = valid & (self.count < min_samples)
            mean = torch.where(sparse, pooled_mean.expand_as(mean), mean)
            variance = torch.where(sparse, pooled_variance.expand_as(variance), variance)

        mean = torch.where(valid, mean, torch.zeros_like(mean))
        variance = torch.where(valid, variance, torch.ones_like(variance))

        rms = RunningMeanStd(shape=tuple(mean.shape), device=device)
        rms.mean.copy_(mean.to(dtype=torch.float32, device=device))
        rms.var.copy_(variance.to(dtype=torch.float32, device=device))
        rms.count.fill_(float(self.count.sum()))
        return rms, self.count.squeeze(-1).to(dtype=torch.float32, device=device)


def reshape_contact_field(value: torch.Tensor, contact_masks: torch.Tensor) -> torch.Tensor:
    """Reshape a flattened contact field to ``[..., num_contacts, channels]``."""
    num_contacts = contact_masks.shape[-1]
    if num_contacts <= 0:
        raise ValueError("Contact tensor operations require at least one contact slot.")
    if value.shape[:-1] != contact_masks.shape[:-1] or value.shape[-1] % num_contacts != 0:
        raise ValueError(
            f"Contact input shape {tuple(value.shape)} is incompatible with "
            f"contact mask shape {tuple(contact_masks.shape)}."
        )
    return value.reshape(*contact_masks.shape, -1)


def mask_inactive_contact_fields(input_dict: MutableMapping[str, torch.Tensor]) -> None:
    """Zero every inactive slot in all contact fields in-place."""
    contact_masks = input_dict.get("contact_masks")
    if contact_masks is None or contact_masks.shape[-1] == 0:
        return

    expanded_mask = contact_masks.bool().unsqueeze(-1)
    for input_name, value in list(input_dict.items()):
        if not input_name.startswith("contact_") or input_name == "contact_masks":
            continue
        reshaped = reshape_contact_field(value, contact_masks)
        input_dict[input_name] = torch.where(expanded_mask, reshaped, 0.0).reshape_as(value)


def normalize_contact_field(
    value: torch.Tensor,
    contact_masks: torch.Tensor,
    normalizer: Any,
) -> torch.Tensor:
    """Normalize each contact with independent per-slot channel statistics."""
    reshaped = reshape_contact_field(value, contact_masks)
    return normalizer.normalize(reshaped).reshape_as(value)


class ContactTokenMoments:
    """Accumulate per-channel statistics over valid contact tokens only."""

    def __init__(
        self,
        channels: int,
        device: torch.device | str,
        categorical_channels: tuple[int, ...] | None = None,
    ):
        from isaaclab_neural.contacts.contact_set_schema import CONTACT_TOKEN_CATEGORICAL_CHANNELS

        self.channels = channels
        self.categorical = set(
            CONTACT_TOKEN_CATEGORICAL_CHANNELS if categorical_channels is None else categorical_channels
        )
        self.count = torch.zeros((), dtype=torch.float64, device=device)
        self.sum = torch.zeros(channels, dtype=torch.float64, device=device)
        self.square_sum = torch.zeros(channels, dtype=torch.float64, device=device)

    def update(self, contact_tokens: torch.Tensor) -> None:
        """Update moments from ``[..., max_tokens, contact_dim]`` tensors."""
        flat = contact_tokens.reshape(-1, self.channels)
        valid = flat[:, 0] > 0.5
        if not valid.any():
            return
        values = flat[valid].to(torch.float64)
        self.count += valid.sum().to(torch.float64)
        self.sum += values.sum(dim=0)
        self.square_sum += values.square().sum(dim=0)

    def synchronize(self) -> None:
        """Sum raw moments across an initialized distributed process group."""
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return
        for tensor in (self.count, self.sum, self.square_sum):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    def finalize(self, device: torch.device | str) -> RunningMeanStd:
        """Return channel-wise RMS with identity channels pinned to mean=0, std=1."""
        safe_count = self.count.clamp_min(1.0)
        mean = self.sum / safe_count
        variance = (self.square_sum / safe_count - mean.square()).clamp_min(0.0)
        for channel in self.categorical:
            mean[channel] = 0.0
            variance[channel] = 1.0
        rms = RunningMeanStd(shape=(self.channels,), device=device)
        rms.mean.copy_(mean.to(dtype=torch.float32, device=device))
        rms.var.copy_(variance.to(dtype=torch.float32, device=device))
        rms.count.copy_(self.count.to(dtype=rms.count.dtype, device=device))
        return rms


def normalize_contact_tokens(
    contact_tokens: torch.Tensor,
    normalizer: RunningMeanStd,
) -> torch.Tensor:
    """Normalize valid contact tokens and re-zero padding."""
    normalized = normalizer.normalize(contact_tokens)
    valid = contact_tokens[..., 0:1] > 0.5
    return normalized * valid.to(normalized.dtype)


def normalize_active15_contact_tokens(
    contact_tokens: torch.Tensor,
    normalizer: RunningMeanStd,
) -> torch.Tensor:
    """Normalize Active15 features while preserving valid and owner-slot values exactly."""
    normalized = normalizer.normalize(contact_tokens)
    normalized[..., :2] = contact_tokens[..., :2]
    valid = contact_tokens[..., 0:1] > 0.5
    return normalized * valid.to(normalized.dtype)
