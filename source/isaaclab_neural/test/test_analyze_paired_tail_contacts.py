# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import h5py
import numpy as np
from isaaclab_neural.eval.analyze_paired_tail_contacts import analyze_tail_contacts


def _write_metrics(
    path: Path,
    *,
    counts: np.ndarray,
    seed_mse: np.ndarray,
) -> None:
    trajectory_index = np.repeat(np.arange(2), 3)
    start_step = np.tile(np.arange(3), 2)
    payload = {
        "trajectory_index": trajectory_index,
        "start_step": start_step,
        "contact_count_mean": counts,
    }
    for seed in range(3):
        payload[f"seed{seed}_state_MSE"] = seed_mse
        payload[f"seed{seed}_state_L2"] = np.sqrt(seed_mse)
    np.savez_compressed(path, **payload)


def test_tail_self_collision_analysis_reconstructs_extra_contacts(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.hdf5"
    tokens = np.zeros((2, 4, 4, 17), dtype=np.float32)
    solver_active = np.zeros((2, 4, 4), dtype=bool)
    tokens[:, :, 0, 0] = 1
    tokens[:, :, 0, 1] = 0
    tokens[:, :, 0, 2] = -1
    solver_active[:, :, 0] = True
    tokens[0, 1, 1, :4] = (1, 0, 1, 1)
    tokens[0, 1, 2, :4] = (1, 1, 0, 1)
    tokens[0, 1, 1:3, 13] = -0.02
    tokens[0, 1, 1:3, 14] = 2.0
    solver_active[0, 1, 1:3] = True
    with h5py.File(dataset, "w") as handle:
        data = handle.create_group("data")
        data.create_dataset("contact_tokens", data=tokens)
        data.create_dataset("contact_token_solver_active", data=solver_active)
        data.create_dataset(
            "contact_token_overflow",
            data=np.zeros((2, 4), dtype=np.int64),
        )

    a_metrics = tmp_path / "a.npz"
    c_metrics = tmp_path / "c.npz"
    a_counts = np.ones(6, dtype=np.float32)
    c_counts = np.array([2, 2, 1, 1, 1, 1], dtype=np.float32)
    a_mse = np.array([10, 1, 1, 1, 1, 1], dtype=np.float32)
    c_mse = np.ones(6, dtype=np.float32)
    _write_metrics(a_metrics, counts=a_counts, seed_mse=a_mse)
    _write_metrics(c_metrics, counts=c_counts, seed_mse=c_mse)

    result = analyze_tail_contacts(
        dataset,
        a_metrics,
        c_metrics,
        tail_fraction=0.001,
        chunk_trajectories=1,
    )

    assert result["scope"]["sequence_length"] == 2
    assert result["scope"]["tail_windows"] == 1
    assert result["count_reconstruction"]["max_abs_derived_c_count_error"] == 0
    assert result["count_reconstruction"]["all_exact_within_1e_5_fraction"] == 1
    assert result["self_collision_exposure"]["self_active_directed_tokens"] == 2
    assert result["self_collision_exposure"]["tail_windows_with_self_collision"] == 1
    assert result["self_collision_exposure"]["odd_directed_self_count_frames"] == 0
    assert result["self_body_pairs"]["all"] == {"0:1": 2}
    assert result["error_by_self_collision"]["tail_self_present"]["c_over_a_state_MSE"] == 0.1
