# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for per-primary-body resultant linear contact-force dataset labels."""

from types import SimpleNamespace

import h5py
import pytest
import torch
from isaaclab_neural.contacts.contact_set_schema import CONTACT_TOKEN_DIM
from isaaclab_neural.data import (
    BatchTransitionDataset,
    TrajectoryDataset,
    append_rollouts_to_hdf5,
    write_rollouts_to_hdf5,
)
from isaaclab_neural.generate.adapter import BodyNetContactForceRecorder, build_primary_body_sensor_indices
from isaaclab_neural.generate.sampler import ActionTrajectorySampler


def test_sensor_names_map_to_primary_body_slot_order():
    indices = build_primary_body_sensor_indices(
        ["LF_SHANK", "/World/envs/env_0/Robot/base", "extra", "RF_SHANK"],
        ["/World/envs/env_0/Robot/base", "LF_SHANK", "RF_SHANK"],
    )

    torch.testing.assert_close(indices, torch.tensor([1, 0, 3]))

    with pytest.raises(ValueError, match="missing=.*RH_SHANK"):
        build_primary_body_sensor_indices(["base", "LF_SHANK"], ["base", "RH_SHANK"])
    with pytest.raises(ValueError, match="ambiguous"):
        build_primary_body_sensor_indices(["base", "/other/base"], ["base"])


def test_recorder_reorders_sensor_forces_with_generic_body_count():
    force_values = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3)
    sensor = SimpleNamespace(
        sensor_names=["body_3", "body_1", "extra", "body_4", "body_0", "body_2"],
        data=SimpleNamespace(net_forces_w=force_values),
    )
    contact_adapter = SimpleNamespace(
        primary_body_ids_by_world=torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
        model=SimpleNamespace(body_label=[f"/Robot/body_{index % 5}" for index in range(10)]),
    )
    recorder = BodyNetContactForceRecorder(sensor, contact_adapter, num_envs=2, device="cpu")

    actual = recorder.read_post_solve()

    assert actual.shape == (2, 5, 3)
    torch.testing.assert_close(actual, force_values[:, [4, 1, 5, 0, 3]])


def test_sampler_allocates_and_copies_labels_only_after_step():
    post_solve_forces = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    adapter = SimpleNamespace(
        num_envs=2,
        num_contacts_per_env=8,
        state_dim=4,
        joint_f_dim=3,
        action_dim=2,
        num_primary_bodies=5,
        body_net_contact_force_recorder=object(),
        solver=SimpleNamespace(contact_representation="contact_tokens", max_contact_tokens=7),
        model=SimpleNamespace(up_axis=2),
        env=SimpleNamespace(scene=None),
        state_world_ids=torch.tensor([0, 1]),
        root_world_ids=torch.tensor([0, 1]),
        contact_world_ids=torch.tensor([0, 1]),
        raw_neural_inputs=lambda: {
            "states": torch.ones(2, 4),
            "joint_f": torch.ones(2, 3),
        },
        body_net_contact_forces_w=lambda: post_solve_forces,
    )
    sampler = ActionTrajectorySampler.__new__(ActionTrajectorySampler)
    sampler.adapter = adapter
    sampler.trajectory_length = 3
    sampler.data_device = torch.device("cpu")

    buffers = sampler.allocate_batch_buffers()
    assert buffers["body_net_contact_forces_w"].shape == (2, 3, 5, 3)

    sampler._copy_after_step(buffers, 1, torch.zeros(2, 2))
    post_solve_forces.add_(1000.0)

    torch.testing.assert_close(
        buffers["body_net_contact_forces_w"][:, 1],
        post_solve_forces - 1000.0,
    )


def _token_rollouts(num_primary_bodies: int = 5):
    num_trajectories, num_steps, max_tokens = 2, 3, 4
    return {
        "states": torch.zeros(num_trajectories, num_steps, 4),
        "next_states": torch.zeros(num_trajectories, num_steps, 4),
        "joint_f": torch.zeros(num_trajectories, num_steps, 2),
        "root_body_q": torch.zeros(num_trajectories, num_steps, 7),
        "root_body_qd": torch.zeros(num_trajectories, num_steps, 6),
        "gravity_dir": torch.zeros(num_trajectories, num_steps, 3),
        "contacts": {
            "contact_tokens": torch.zeros(num_trajectories, num_steps, max_tokens, CONTACT_TOKEN_DIM),
            "contact_token_overflow": torch.zeros(num_trajectories, num_steps, dtype=torch.long),
        },
        "contact_token_body_ids": torch.full((num_trajectories, num_steps, max_tokens), -1, dtype=torch.long),
        "contact_token_world_ids": torch.full((num_trajectories, num_steps, max_tokens), -1, dtype=torch.long),
        "body_net_contact_forces_w": torch.arange(
            num_trajectories * num_steps * num_primary_bodies * 3, dtype=torch.float32
        ).reshape(num_trajectories, num_steps, num_primary_bodies, 3),
        "trajectory_context": {
            "state_world_id": torch.arange(num_trajectories, dtype=torch.long),
            "root_world_id": torch.arange(num_trajectories, dtype=torch.long),
            "contact_world_id": torch.arange(num_trajectories, dtype=torch.long),
        },
    }


def test_hdf5_round_trip_and_missing_label_rejection(tmp_path):
    dataset_path = tmp_path / "force_labels.hdf5"
    rollouts = _token_rollouts(num_primary_bodies=5)
    body_names = [f"body_{index}" for index in range(5)]
    write_rollouts_to_hdf5(
        dataset_path,
        rollouts,
        "rough-anymal",
        body_net_contact_force_body_names=body_names,
    )

    dataset = TrajectoryDataset(dataset_path, sample_sequence_length=2)
    sample = dataset[0]
    assert "body_net_contact_forces_w" in dataset.data_keys
    assert sample["body_net_contact_forces_w"].shape == (2, 5, 3)
    torch.testing.assert_close(
        sample["body_net_contact_forces_w"],
        rollouts["body_net_contact_forces_w"][0, :2],
    )
    with h5py.File(dataset_path, "r") as dataset_file:
        data = dataset_file["data"]
        assert data.attrs["body_net_contact_force_frame"] == "world"
        assert data.attrs["body_net_contact_force_units"] == "N"
        assert data.attrs["body_net_contact_force_timing"] == "post_solve_t_to_t_plus_1"
        assert data.attrs["num_primary_bodies"] == 5
        stored_body_names = [
            name.decode() if isinstance(name, bytes) else str(name)
            for name in data.attrs["body_net_contact_force_body_names"]
        ]
        assert stored_body_names == body_names

    transition_dataset = BatchTransitionDataset(batch_size=2, hdf5_dataset_path=dataset_path)
    assert "body_net_contact_forces_w" in transition_dataset.data_keys

    with h5py.File(dataset_path, "a") as dataset_file:
        del dataset_file["data"]["body_net_contact_forces_w"]
    with pytest.raises(ValueError, match="missing field 'body_net_contact_forces_w'"):
        TrajectoryDataset(dataset_path, sample_sequence_length=2)


def test_hdf5_rejects_missing_or_invalid_body_name_order(tmp_path):
    dataset_path = tmp_path / "force_label_body_names.hdf5"
    body_names = [f"body_{index}" for index in range(5)]
    write_rollouts_to_hdf5(
        dataset_path,
        _token_rollouts(num_primary_bodies=5),
        "rough-anymal",
        body_net_contact_force_body_names=body_names,
    )

    with h5py.File(dataset_path, "a") as dataset_file:
        del dataset_file["data"].attrs["body_net_contact_force_body_names"]
    with pytest.raises(ValueError, match="body-name"):
        TrajectoryDataset(dataset_path, sample_sequence_length=2)

    with h5py.File(dataset_path, "a") as dataset_file:
        dataset_file["data"].attrs["body_net_contact_force_body_names"] = ["body_0"] * 5
    with pytest.raises(ValueError, match="unique"):
        TrajectoryDataset(dataset_path, sample_sequence_length=2)


def test_hdf5_append_preserves_body_name_order(tmp_path):
    dataset_path = tmp_path / "force_labels_append.hdf5"
    rollouts = _token_rollouts(num_primary_bodies=5)
    body_names = [f"body_{index}" for index in range(5)]

    append_rollouts_to_hdf5(
        dataset_path,
        rollouts,
        "rough-anymal",
        body_net_contact_force_body_names=body_names,
    )
    append_rollouts_to_hdf5(
        dataset_path,
        rollouts,
        "rough-anymal",
        body_net_contact_force_body_names=body_names,
    )

    with h5py.File(dataset_path, "r") as dataset_file:
        data = dataset_file["data"]
        stored_body_names = [
            name.decode() if isinstance(name, bytes) else str(name)
            for name in data.attrs["body_net_contact_force_body_names"]
        ]
        assert stored_body_names == body_names
        assert data["states"].shape[0] == 4

    with pytest.raises(ValueError, match="different body-name order"):
        append_rollouts_to_hdf5(
            dataset_path,
            rollouts,
            "rough-anymal",
            body_net_contact_force_body_names=list(reversed(body_names)),
        )
