# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for standalone paired-checkpoint evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from isaaclab_neural.eval.paired_checkpoint_eval import (
    aggregate_by_trajectory,
    summarize_paired_trajectory_metrics,
    validate_paired_eval_design_contract,
    window_contact_count_metrics,
    window_state_dimension_mse,
    window_state_error_metrics,
)
from isaaclab_neural.train.training_diagnostics import state_error_metrics


def _design_cfg(
    *,
    solver_representation: str,
    dataset_representation: str,
    encoder_type: str,
    body_latent_dim: int,
    hidden_dim: int,
    pooling: str | None = None,
    use_count_projection: bool | None = None,
    use_other_body_embeddings: bool | None = None,
) -> dict:
    contact_set = {
        "dim": 17,
        "encoder_type": encoder_type,
        "body_latent_dim": body_latent_dim,
        "hidden_dim": hidden_dim,
    }
    if encoder_type == "shared_per_body":
        contact_set["max_other_bodies"] = 32
    if pooling is not None:
        contact_set["pooling"] = pooling
    if use_count_projection is not None:
        contact_set["use_count_projection"] = use_count_projection
    if use_other_body_embeddings is not None:
        contact_set["use_other_body_embeddings"] = use_other_body_embeddings
    return {
        "env": {
            "neural_solver_cfg": {
                "contact_representation": solver_representation,
                "contact_filter": "solver_active",
            }
        },
        "algorithm": {"dataset": {"contact_representation": dataset_representation}},
        "inputs": {"contact_set": contact_set},
    }


@pytest.mark.parametrize(
    ("design", "cfg"),
    [
        (
            "a_active",
            _design_cfg(
                solver_representation="active15_tokens",
                dataset_representation="raw15_tokens",
                encoder_type="body_routed_active15",
                body_latent_dim=64,
                hidden_dim=32,
            ),
        ),
        (
            "a_mean_count",
            _design_cfg(
                solver_representation="active15_tokens",
                dataset_representation="raw15_tokens",
                encoder_type="body_routed_active15",
                body_latent_dim=64,
                hidden_dim=32,
                pooling="mean",
                use_count_projection=True,
            ),
        ),
        (
            "c_active",
            _design_cfg(
                solver_representation="contact_tokens",
                dataset_representation="contact_tokens",
                encoder_type="shared_per_body",
                body_latent_dim=16,
                hidden_dim=64,
            ),
        ),
        (
            "c_no_other",
            _design_cfg(
                solver_representation="contact_tokens",
                dataset_representation="contact_tokens",
                encoder_type="shared_per_body",
                body_latent_dim=16,
                hidden_dim=64,
                use_other_body_embeddings=False,
            ),
        ),
    ],
)
def test_validate_paired_eval_design_contract_accepts_exact_designs(design: str, cfg: dict) -> None:
    assert (
        validate_paired_eval_design_contract(design, cfg)["encoder_type"]
        == cfg["inputs"]["contact_set"]["encoder_type"]
    )


def test_validate_paired_eval_design_contract_rejects_swapped_or_unknown_labels() -> None:
    mean_count_cfg = _design_cfg(
        solver_representation="active15_tokens",
        dataset_representation="raw15_tokens",
        encoder_type="body_routed_active15",
        body_latent_dim=64,
        hidden_dim=32,
        pooling="mean",
        use_count_projection=True,
    )

    with pytest.raises(ValueError, match="does not match design 'a_active'"):
        validate_paired_eval_design_contract("a_active", mean_count_cfg)
    with pytest.raises(ValueError, match="Unsupported paired-eval design"):
        validate_paired_eval_design_contract("unknown", mean_count_cfg)


def test_window_metrics_average_to_existing_state_metrics() -> None:
    target = torch.arange(2 * 3 * 7, dtype=torch.float32).reshape(2, 3, 7)
    predicted = target + torch.tensor([0.5, -1.0, 2.0, 0.0, 1.5, -0.5, 3.0])

    per_window = window_state_error_metrics(predicted, target, dof_q=4)
    aggregate = state_error_metrics(predicted, target, dof_q=4)

    for name, values in per_window.items():
        assert values.shape == (2,)
        assert float(values.mean()) == pytest.approx(aggregate[name])


def test_window_metrics_validate_shapes_and_dof_split() -> None:
    states = torch.zeros(2, 3, 7)

    with pytest.raises(ValueError, match="same shape"):
        window_state_error_metrics(states, states[..., :-1], dof_q=4)
    with pytest.raises(ValueError, match="dof_q"):
        window_state_error_metrics(states, states, dof_q=0)
    with pytest.raises(ValueError, match="dof_q"):
        window_state_error_metrics(states, states, dof_q=7)


def test_window_contact_count_metrics_use_post_filter_valid_channel() -> None:
    tokens = torch.zeros(2, 3, 4, 17)
    tokens[0, 0, :2, 0] = 1.0
    tokens[0, 1, :1, 0] = 1.0
    tokens[1, :, :3, 0] = 1.0

    metrics = window_contact_count_metrics(tokens)

    torch.testing.assert_close(metrics["contact_count_mean"], torch.tensor([1.0, 3.0]))
    torch.testing.assert_close(metrics["contact_count_max"], torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(metrics["contact_nonempty_fraction"], torch.tensor([2.0 / 3.0, 1.0]))

    with pytest.raises(ValueError, match="batch, time, token, feature"):
        window_contact_count_metrics(tokens[0])


def test_window_state_dimension_mse_preserves_each_physical_dimension() -> None:
    target = torch.zeros(2, 3, 4)
    predicted = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0]],
            [[4.0, 3.0, 2.0, 1.0], [5.0, 4.0, 3.0, 2.0], [6.0, 5.0, 4.0, 3.0]],
        ]
    )

    per_window = window_state_dimension_mse(predicted, target)

    assert per_window.shape == (2, 4)
    torch.testing.assert_close(per_window, predicted.square().mean(dim=1))
    assert float(per_window.mean()) == pytest.approx(
        window_state_error_metrics(predicted, target, dof_q=2)["state_MSE"].mean().item()
    )


def test_aggregate_by_trajectory_handles_seed_axis_and_unsorted_windows() -> None:
    trajectory_index = np.array([1, 0, 2, 1, 0, 2], dtype=np.int64)
    values = np.array(
        [
            [3.0, 1.0, 5.0, 7.0, 9.0, 11.0],
            [4.0, 2.0, 6.0, 8.0, 10.0, 12.0],
        ],
        dtype=np.float64,
    )

    means = aggregate_by_trajectory(values, trajectory_index)

    np.testing.assert_allclose(means, np.array([[5.0, 5.0, 8.0], [6.0, 6.0, 9.0]]))


def test_paired_trajectory_summary_uses_paired_seed_and_trajectory_differences() -> None:
    trajectory_index = np.repeat(np.arange(4, dtype=np.int64), 2)
    reference = np.array(
        [
            [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
            [2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
        ]
    )
    candidate = reference - 0.25

    summary = summarize_paired_trajectory_metrics(
        candidate,
        reference,
        trajectory_index,
        seed_ids=[3, 7],
        bootstrap_samples=500,
        bootstrap_seed=11,
    )

    assert summary["num_seeds"] == 2
    assert summary["num_trajectories"] == 4
    assert summary["candidate_mean"] == pytest.approx(2.75)
    assert summary["reference_mean"] == pytest.approx(3.0)
    assert summary["mean_delta"] == pytest.approx(-0.25)
    assert summary["ratio"] == pytest.approx(2.75 / 3.0)
    assert summary["relative_change_percent"] == pytest.approx(-100.0 / 12.0)
    assert summary["trajectory_bootstrap_ci95"] == pytest.approx([-0.25, -0.25])
    assert [row["seed"] for row in summary["per_seed"]] == [3, 7]
    assert all(row["mean_delta"] == pytest.approx(-0.25) for row in summary["per_seed"])


def test_paired_trajectory_summary_rejects_unpaired_inputs() -> None:
    trajectory_index = np.array([0, 0, 1, 1], dtype=np.int64)
    values = np.ones((2, 4), dtype=np.float64)

    with pytest.raises(ValueError, match="same shape"):
        summarize_paired_trajectory_metrics(values[:, :-1], values, trajectory_index, seed_ids=[0, 1])
    with pytest.raises(ValueError, match="seed_ids"):
        summarize_paired_trajectory_metrics(values, values, trajectory_index, seed_ids=[0])
    with pytest.raises(ValueError, match="non-negative"):
        aggregate_by_trajectory(values, np.array([0, -1, 1, 1]))
