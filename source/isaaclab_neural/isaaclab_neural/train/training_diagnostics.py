# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small metric helpers for NeRD training diagnostics."""

from __future__ import annotations

import torch


@torch.no_grad()
def state_error_metrics(
    predicted_next_states: torch.Tensor,
    target_next_states: torch.Tensor,
    *,
    dof_q: int,
) -> dict[str, float]:
    """Return unweighted one-step state errors split into q and qd."""
    if predicted_next_states.shape != target_next_states.shape:
        raise ValueError("Predicted and target next-state tensors must have the same shape.")
    if dof_q <= 0 or dof_q >= predicted_next_states.shape[-1]:
        raise ValueError("dof_q must split the final state dimension into non-empty q and qd parts.")

    state_error = predicted_next_states - target_next_states
    q_error = state_error[..., :dof_q]
    qd_error = state_error[..., dof_q:]
    metric_names = (
        "state_MSE",
        "q_MSE",
        "qd_MSE",
        "state_L2",
        "q_error_norm",
        "qd_error_norm",
    )
    metric_values = (
        torch.stack(
            (
                state_error.square().mean(),
                q_error.square().mean(),
                qd_error.square().mean(),
                state_error.norm(dim=-1).mean(),
                q_error.norm(dim=-1).mean(),
                qd_error.norm(dim=-1).mean(),
            )
        )
        .detach()
        .cpu()
    )
    return dict(zip(metric_names, (float(value) for value in metric_values), strict=True))


@torch.no_grad()
def tensor_distribution_metrics(
    values: torch.Tensor,
    prefix: str,
    *,
    include_positive_fraction: bool = False,
) -> dict[str, float]:
    """Summarize scalar spread and across-sample variance for one representation."""
    if values.numel() == 0 or values.ndim == 0:
        raise ValueError("Diagnostic tensors must be non-empty and have a feature dimension.")

    flattened = values.detach().float().reshape(-1, values.shape[-1])
    metrics = {
        f"{prefix}_mean": float(flattened.mean().cpu()),
        f"{prefix}_std": float(flattened.std(unbiased=False).cpu()),
        f"{prefix}_sample_variance": float(flattened.var(dim=0, unbiased=False).mean().cpu()),
    }
    if include_positive_fraction:
        metrics[f"{prefix}_positive_fraction"] = float((flattened > 0).float().mean().cpu())
    return metrics


@torch.no_grad()
def snapshot_trainable_parameters(module: torch.nn.Module) -> list[torch.Tensor]:
    """Clone trainable parameters for a later epoch-displacement measurement."""
    return [parameter.detach().clone() for parameter in module.parameters() if parameter.requires_grad]


@torch.no_grad()
def parameter_change_metrics(
    snapshot: list[torch.Tensor],
    module: torch.nn.Module,
) -> dict[str, float]:
    """Return current parameter norm and displacement from a prior snapshot."""
    parameters = [parameter.detach() for parameter in module.parameters() if parameter.requires_grad]
    if len(snapshot) != len(parameters):
        raise ValueError("Parameter snapshot does not match the current trainable parameter set.")

    update_squared = torch.zeros((), dtype=torch.float64, device=parameters[0].device)
    parameter_squared = torch.zeros_like(update_squared)
    for reference, parameter in zip(snapshot, parameters, strict=True):
        if reference.shape != parameter.shape:
            raise ValueError("Parameter snapshot shape does not match the current model.")
        update_squared += (parameter - reference).float().square().sum().double()
        parameter_squared += parameter.float().square().sum().double()

    update_norm = float(update_squared.sqrt().cpu())
    parameter_norm = float(parameter_squared.sqrt().cpu())
    relative_update_norm = update_norm / max(parameter_norm, torch.finfo(torch.float64).eps)
    return {
        "parameter_update_norm": update_norm,
        "parameter_norm": parameter_norm,
        "relative_parameter_update_norm": relative_update_norm,
    }


@torch.no_grad()
def mean_predictor_loss(
    target_variance: torch.Tensor,
    loss_weights: torch.Tensor | float,
) -> float:
    """Return the weighted MSE of the current dataset's target-mean predictor."""
    weights = torch.as_tensor(loss_weights, device=target_variance.device, dtype=target_variance.dtype)
    return float((target_variance * weights.square()).mean().detach().cpu())
