# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fail-closed cache tests for the native15 self-collision dataset."""

from pathlib import Path

import h5py
import numpy as np
from run_experiment import dataset_cache_complete


def _experiment() -> dict:
    return {
        "env_name": "Anymal-C-Rough-Native-Raw15Self",
        "contact_representation": "raw15_self_tokens",
        "max_contact_tokens": 4,
        "train_transitions": 6,
        "valid_transitions": 6,
        "trajectory_length": 3,
        "require_zero_contact_token_overflow": True,
        "require_self_collision_tokens": True,
        "dataset_specs": [{"filename": "dataset_train.hdf5", "seed": 0, "split": "train"}],
    }


def _write_cache(
    path: Path,
    *,
    include_self_sidecar: bool = True,
    has_self_collision: bool = True,
    overflow: int = 0,
) -> None:
    step_shape = (2, 3)
    capacity = 4
    tokens = np.zeros((*step_shape, capacity, 17), dtype=np.float32)
    tokens[0, 0, 0, 0] = 1.0
    body_ids = np.full((*step_shape, capacity), -1, dtype=np.int64)
    body_ids[0, 0, 0] = 0
    world_ids = np.full((*step_shape, capacity), -1, dtype=np.int64)
    world_ids[0, 0, 0] = 0

    with h5py.File(path, "w") as handle:
        data_group = handle.create_group("data")
        data_group.attrs.update(
            {
                "mode": "trajectory",
                "env": "Anymal-C-Rough-Native-Raw15Self",
                "total_transitions": 6,
                "contact_representation": "raw15_self_tokens",
                "contact_token_dim": 17,
                "max_contact_tokens": capacity,
                "contact_identity_schema": "world_owner_v1",
                "contact_token_frame": "owner_body_v1",
                "contact_schema": "raw15_self_owner_body_v1",
                "contact_frame": "owner_body_v1",
                "contact_velocity_point": "raw_point_midpoint_v1",
                "contact_selection": "newton_raw_candidates_with_directed_robot_self_v1",
                "contact_token_self_collision_schema": "directed_robot_self_v1",
            }
        )
        data_group.create_dataset("states", data=np.zeros((*step_shape, 37), dtype=np.float32))
        data_group.create_dataset("next_states", data=np.zeros((*step_shape, 37), dtype=np.float32))
        data_group.create_dataset("joint_f", data=np.zeros((*step_shape, 18), dtype=np.float32))
        data_group.create_dataset("root_body_q", data=np.zeros((*step_shape, 7), dtype=np.float32))
        data_group.create_dataset("root_body_qd", data=np.zeros((*step_shape, 6), dtype=np.float32))
        data_group.create_dataset("gravity_dir", data=np.zeros((*step_shape, 3), dtype=np.float32))
        data_group.create_dataset("contact_tokens", data=tokens)
        data_group.create_dataset("contact_token_body_ids", data=body_ids)
        data_group.create_dataset("contact_token_world_ids", data=world_ids)
        overflow_data = np.zeros(step_shape, dtype=np.int64)
        overflow_data[0, 0] = overflow
        data_group.create_dataset("contact_token_overflow", data=overflow_data)
        if include_self_sidecar:
            self_collision = np.zeros((*step_shape, capacity), dtype=np.bool_)
            self_collision[0, 0, 0] = has_self_collision
            data_group.create_dataset("contact_token_self_collision", data=self_collision)


def test_raw15_self_cache_requires_sidecar_self_tokens_and_zero_overflow(tmp_path: Path) -> None:
    filename = "dataset_train.hdf5"
    path = tmp_path / filename
    experiment = _experiment()

    _write_cache(path)
    assert dataset_cache_complete(tmp_path, [filename], experiment)

    _write_cache(path, include_self_sidecar=False)
    assert not dataset_cache_complete(tmp_path, [filename], experiment)

    _write_cache(path, has_self_collision=False)
    assert not dataset_cache_complete(tmp_path, [filename], experiment)

    _write_cache(path, overflow=1)
    assert not dataset_cache_complete(tmp_path, [filename], experiment)
