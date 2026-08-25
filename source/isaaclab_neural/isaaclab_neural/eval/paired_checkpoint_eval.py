# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone fixed-checkpoint evaluation on transition-paired datasets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

_DESIGN_CONTRACTS = {
    "a_active": {
        "solver_contact_representation": "active15_tokens",
        "dataset_contact_representation": "raw15_tokens",
        "contact_filter": "solver_active",
        "encoder_type": "body_routed_active15",
        "contact_dim": 17,
        "body_latent_dim": 64,
        "hidden_dim": 32,
        "pooling": "sum",
        "use_count_projection": False,
    },
    "a_mean_count": {
        "solver_contact_representation": "active15_tokens",
        "dataset_contact_representation": "raw15_tokens",
        "contact_filter": "solver_active",
        "encoder_type": "body_routed_active15",
        "contact_dim": 17,
        "body_latent_dim": 64,
        "hidden_dim": 32,
        "pooling": "mean",
        "use_count_projection": True,
    },
    "c_active": {
        "solver_contact_representation": "contact_tokens",
        "dataset_contact_representation": "contact_tokens",
        "contact_filter": "solver_active",
        "encoder_type": "shared_per_body",
        "contact_dim": 17,
        "body_latent_dim": 16,
        "hidden_dim": 64,
        "max_other_bodies": 32,
        "use_other_body_embeddings": True,
    },
    "c_no_other": {
        "solver_contact_representation": "contact_tokens",
        "dataset_contact_representation": "contact_tokens",
        "contact_filter": "solver_active",
        "encoder_type": "shared_per_body",
        "contact_dim": 17,
        "body_latent_dim": 16,
        "hidden_dim": 64,
        "max_other_bodies": 32,
        "use_other_body_embeddings": False,
    },
}


def validate_paired_eval_design_contract(design: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Fail loudly if a manifest label does not match its checkpoint configuration."""
    expected = _DESIGN_CONTRACTS.get(design)
    if expected is None:
        raise ValueError(f"Unsupported paired-eval design: {design!r}.")
    try:
        solver_cfg = cfg["env"]["neural_solver_cfg"]
        dataset_cfg = cfg["algorithm"]["dataset"]
        contact_cfg = cfg["inputs"]["contact_set"]
        actual = {
            "solver_contact_representation": solver_cfg["contact_representation"],
            "dataset_contact_representation": dataset_cfg["contact_representation"],
            "contact_filter": solver_cfg.get("contact_filter", "none"),
            "encoder_type": contact_cfg["encoder_type"],
            "contact_dim": int(contact_cfg["dim"]),
            "body_latent_dim": int(contact_cfg["body_latent_dim"]),
            "hidden_dim": int(contact_cfg["hidden_dim"]),
            "pooling": contact_cfg.get("pooling", "sum"),
            "use_count_projection": bool(contact_cfg.get("use_count_projection", False)),
            "max_other_bodies": int(contact_cfg.get("max_other_bodies", 32)),
            "use_other_body_embeddings": bool(contact_cfg.get("use_other_body_embeddings", True)),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Checkpoint configuration for design {design!r} is incomplete.") from exc

    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Checkpoint configuration does not match design {design!r}: {mismatches}.")
    return actual


@torch.no_grad()
def window_contact_count_metrics(contact_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return post-filter valid-token density for each dataset window."""
    if contact_tokens.ndim != 4:
        raise ValueError("contact_tokens must have shape [batch, time, token, feature].")
    valid_counts = (contact_tokens[..., 0] > 0.5).sum(dim=-1).float()
    return {
        "contact_count_mean": valid_counts.mean(dim=-1),
        "contact_count_max": valid_counts.max(dim=-1).values,
        "contact_nonempty_fraction": (valid_counts > 0).float().mean(dim=-1),
    }


def _mean_non_batch(values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 1:
        return values
    return values.mean(dim=tuple(range(1, values.ndim)))


@torch.no_grad()
def window_state_error_metrics(
    predicted_next_states: torch.Tensor,
    target_next_states: torch.Tensor,
    *,
    dof_q: int,
) -> dict[str, torch.Tensor]:
    """Return one error value per dataset window for every physical-state metric."""
    if predicted_next_states.shape != target_next_states.shape:
        raise ValueError("Predicted and target next-state tensors must have the same shape.")
    if predicted_next_states.ndim < 2:
        raise ValueError("State tensors must include batch and feature dimensions.")
    if dof_q <= 0 or dof_q >= predicted_next_states.shape[-1]:
        raise ValueError("dof_q must split the final state dimension into non-empty q and qd parts.")

    state_error = predicted_next_states - target_next_states
    q_error = state_error[..., :dof_q]
    qd_error = state_error[..., dof_q:]
    return {
        "state_MSE": _mean_non_batch(state_error.square()),
        "q_MSE": _mean_non_batch(q_error.square()),
        "qd_MSE": _mean_non_batch(qd_error.square()),
        "state_L2": _mean_non_batch(state_error.norm(dim=-1)),
        "q_error_norm": _mean_non_batch(q_error.norm(dim=-1)),
        "qd_error_norm": _mean_non_batch(qd_error.norm(dim=-1)),
    }


@torch.no_grad()
def window_state_dimension_mse(
    predicted_next_states: torch.Tensor,
    target_next_states: torch.Tensor,
) -> torch.Tensor:
    """Return per-window MSE while preserving the physical-state dimension."""
    if predicted_next_states.shape != target_next_states.shape:
        raise ValueError("Predicted and target next-state tensors must have the same shape.")
    if predicted_next_states.ndim < 2:
        raise ValueError("State tensors must include batch and feature dimensions.")
    squared_error = (predicted_next_states - target_next_states).square()
    reduction_dims = tuple(range(1, squared_error.ndim - 1))
    return squared_error.mean(dim=reduction_dims) if reduction_dims else squared_error


def aggregate_by_trajectory(
    values: np.ndarray,
    trajectory_index: np.ndarray,
) -> np.ndarray:
    """Average window metrics by trajectory, preserving an optional seed axis."""
    values_array = np.asarray(values, dtype=np.float64)
    indices = np.asarray(trajectory_index)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("trajectory_index must be a one-dimensional integer array.")
    if indices.size == 0:
        raise ValueError("trajectory_index must not be empty.")
    if np.any(indices < 0):
        raise ValueError("trajectory_index values must be non-negative.")
    if values_array.ndim not in {1, 2}:
        raise ValueError("values must have shape [window] or [seed, window].")
    if values_array.shape[-1] != indices.size:
        raise ValueError("values and trajectory_index must contain the same number of windows.")
    if not np.isfinite(values_array).all():
        raise ValueError("values must be finite.")

    num_trajectories = int(indices.max()) + 1
    counts = np.bincount(indices, minlength=num_trajectories)
    if np.any(counts == 0):
        raise ValueError("trajectory_index must contain every trajectory from zero through its maximum.")
    flattened = values_array.reshape(-1, values_array.shape[-1])
    means = np.stack([np.bincount(indices, weights=row, minlength=num_trajectories) / counts for row in flattened])
    if values_array.ndim == 1:
        return means[0]
    return means


def _ratio(candidate: float, reference: float) -> float | None:
    return candidate / reference if reference != 0.0 else None


def _relative_change_percent(candidate: float, reference: float) -> float | None:
    ratio = _ratio(candidate, reference)
    return None if ratio is None else (ratio - 1.0) * 100.0


def summarize_paired_trajectory_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    trajectory_index: np.ndarray,
    *,
    seed_ids: Sequence[int],
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Summarize a candidate-minus-reference metric on paired trajectories.

    The confidence interval resamples trajectories after averaging the fixed
    checkpoint ensemble. It therefore measures dataset uncertainty conditional
    on these checkpoints, not uncertainty across hypothetical training runs.
    """
    candidate_array = np.asarray(candidate, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if candidate_array.shape != reference_array.shape:
        raise ValueError("candidate and reference must have the same shape.")
    if candidate_array.ndim != 2:
        raise ValueError("candidate and reference must have shape [seed, window].")
    if len(seed_ids) != candidate_array.shape[0]:
        raise ValueError("seed_ids must match the seed axis.")
    if len(seed_ids) != len(set(seed_ids)):
        raise ValueError("seed_ids must be unique.")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive.")

    candidate_trajectory = aggregate_by_trajectory(candidate_array, trajectory_index)
    reference_trajectory = aggregate_by_trajectory(reference_array, trajectory_index)
    per_seed = []
    for seed, candidate_seed, reference_seed in zip(
        seed_ids,
        candidate_trajectory,
        reference_trajectory,
        strict=True,
    ):
        candidate_mean = float(candidate_seed.mean())
        reference_mean = float(reference_seed.mean())
        per_seed.append(
            {
                "seed": int(seed),
                "candidate_mean": candidate_mean,
                "reference_mean": reference_mean,
                "mean_delta": candidate_mean - reference_mean,
                "ratio": _ratio(candidate_mean, reference_mean),
                "relative_change_percent": _relative_change_percent(candidate_mean, reference_mean),
            }
        )

    candidate_ensemble = candidate_trajectory.mean(axis=0)
    reference_ensemble = reference_trajectory.mean(axis=0)
    trajectory_delta = candidate_ensemble - reference_ensemble
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_means = np.empty(bootstrap_samples, dtype=np.float64)
    chunk_size = 1024
    for start in range(0, bootstrap_samples, chunk_size):
        count = min(chunk_size, bootstrap_samples - start)
        sampled = rng.integers(
            0,
            trajectory_delta.size,
            size=(count, trajectory_delta.size),
        )
        bootstrap_means[start : start + count] = trajectory_delta[sampled].mean(axis=1)

    candidate_mean = float(candidate_ensemble.mean())
    reference_mean = float(reference_ensemble.mean())
    return {
        "num_seeds": int(candidate_array.shape[0]),
        "num_trajectories": int(trajectory_delta.size),
        "candidate_mean": candidate_mean,
        "reference_mean": reference_mean,
        "mean_delta": candidate_mean - reference_mean,
        "ratio": _ratio(candidate_mean, reference_mean),
        "relative_change_percent": _relative_change_percent(candidate_mean, reference_mean),
        "trajectory_bootstrap_ci95": [
            float(np.percentile(bootstrap_means, 2.5)),
            float(np.percentile(bootstrap_means, 97.5)),
        ],
        "trajectory_bootstrap_scope": "fixed_checkpoint_ensemble",
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "per_seed": per_seed,
    }
