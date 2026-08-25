# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for transition-paired Raw15 and ContactTokens datasets."""

from __future__ import annotations

from types import SimpleNamespace

import h5py
import pytest
import torch
from isaaclab_neural.contacts.contact_set_schema import CONTACT_TOKEN_DIM
from isaaclab_neural.data import write_rollouts_to_hdf5
from isaaclab_neural.eval.paired_contact_dataset import (
    PairedActionTrajectorySampler,
    PairedDataGenerationAdapter,
    split_paired_rollouts,
    truncation_summary_delta,
    validate_paired_hdf5_files,
    validate_paired_policy_suite_manifest,
)


def _paired_rollouts(num_trajectories: int = 2, num_steps: int = 3) -> dict:
    shared = {
        "states": torch.arange(num_trajectories * num_steps * 4, dtype=torch.float32).reshape(
            num_trajectories, num_steps, 4
        ),
        "next_states": torch.ones(num_trajectories, num_steps, 4),
        "joint_f": torch.ones(num_trajectories, num_steps, 2),
        "actions": torch.ones(num_trajectories, num_steps, 2) * 2.0,
        "root_body_q": torch.zeros(num_trajectories, num_steps, 7),
        "root_body_qd": torch.zeros(num_trajectories, num_steps, 6),
        "gravity_dir": torch.zeros(num_trajectories, num_steps, 3),
    }
    token_shape = (num_trajectories, num_steps, 2)
    world_ids = torch.arange(num_trajectories, dtype=torch.long)
    return {
        **shared,
        "contacts": {
            "contact_tokens": torch.zeros(*token_shape, CONTACT_TOKEN_DIM),
            "contact_token_overflow": torch.zeros(num_trajectories, num_steps, dtype=torch.long),
        },
        "contact_token_body_ids": torch.full(token_shape, -1, dtype=torch.long),
        "contact_token_world_ids": torch.full(token_shape, -1, dtype=torch.long),
        "paired_contacts": {
            "contact_tokens": torch.zeros(*token_shape, CONTACT_TOKEN_DIM),
            "contact_token_overflow": torch.zeros(num_trajectories, num_steps, dtype=torch.long),
            "contact_token_solver_active": torch.zeros(token_shape, dtype=torch.bool),
        },
        "paired_contact_token_body_ids": torch.full(token_shape, -1, dtype=torch.long),
        "paired_contact_token_world_ids": torch.full(token_shape, -1, dtype=torch.long),
        "trajectory_context": {
            "source_env_id": world_ids,
            "state_world_id": world_ids,
            "root_world_id": world_ids,
            "contact_world_id": world_ids,
            "terrain_level": torch.zeros(num_trajectories, dtype=torch.long),
            "terrain_type": torch.ones(num_trajectories, dtype=torch.long),
            "env_origin": torch.zeros(num_trajectories, 3),
        },
    }


def _terrain_context() -> dict:
    return {
        "schema_version": 1,
        "seed": 40,
        "config_sha256": "config",
        "mesh_sha256": "mesh",
        "origins_sha256": "origins",
        "num_rows": 10,
        "num_cols": 20,
    }


def _policy_suite_manifest() -> dict:
    return {
        "schema_version": 1,
        "requested_transitions": 1_000_000,
        "actual_transitions": 1_228_800,
        "num_envs": 1024,
        "trajectory_length": 400,
        "seed": 40,
        "randomize_pd_gains": False,
        "task": "Isaac-Velocity-Rough-Anymal-C-v0",
        "contact_packing_policy": "body_round_robin_pair_atomic",
        "max_contact_tokens": 64,
        "policy_checkpoint": {
            "size_bytes": 6_882_293,
            "sha256": "833870337db02e88a660dd0d0b7921ea2a19dcb9c62bffe9660b3775976f3d3a",
        },
        "policy_agent_config": {
            "size_bytes": 1_326,
            "sha256": "9d5feef5b5ad6e9097ea3d740a14d74460cf54108510954d78b173950f7e852e",
        },
        "pairing": {
            "num_trajectories": 3072,
            "steps_per_trajectory": 400,
            "total_transitions": 1_228_800,
        },
    }


def test_validate_paired_policy_suite_manifest_freezes_rollout_provenance() -> None:
    manifest = _policy_suite_manifest()

    validated = validate_paired_policy_suite_manifest(manifest)

    assert validated["actual_transitions"] == 1_228_800
    assert validated["policy_checkpoint"]["sha256"] == manifest["policy_checkpoint"]["sha256"]

    with pytest.raises(ValueError, match="frozen contract"):
        validate_paired_policy_suite_manifest({**manifest, "seed": 41})
    with pytest.raises(ValueError, match="fewer transitions"):
        validate_paired_policy_suite_manifest({**manifest, "actual_transitions": 999_600})


def test_split_paired_rollouts_reuses_every_non_contact_tensor() -> None:
    combined = _paired_rollouts()

    primary, paired = split_paired_rollouts(combined)

    for key in (
        "states",
        "next_states",
        "joint_f",
        "actions",
        "root_body_q",
        "root_body_qd",
        "gravity_dir",
        "trajectory_context",
    ):
        assert primary[key] is combined[key]
        assert paired[key] is combined[key]
    assert primary["contacts"] is combined["contacts"]
    assert paired["contacts"] is combined["paired_contacts"]
    assert primary["contact_token_body_ids"] is combined["contact_token_body_ids"]
    assert paired["contact_token_body_ids"] is combined["paired_contact_token_body_ids"]
    assert not any(key.startswith("paired_") for key in primary)
    assert not any(key.startswith("paired_") for key in paired)


def test_paired_adapter_sync_uses_one_contact_object_for_both_encoders() -> None:
    contacts = object()
    state = object()
    primary_solver = object()
    primary_calls = []
    secondary_calls = []

    adapter = PairedDataGenerationAdapter.__new__(PairedDataGenerationAdapter)
    adapter.solver = primary_solver
    adapter.state = state
    adapter._prepare_contacts = lambda: contacts
    adapter.backend = SimpleNamespace(sync_solver=lambda solver, received: primary_calls.append((solver, received)))
    adapter.paired_contact_adapter = SimpleNamespace(
        update=lambda received, received_state: secondary_calls.append((received, received_state))
    )

    adapter.sync()

    assert primary_calls == [(primary_solver, contacts)]
    assert secondary_calls == [(contacts, state)]


def test_paired_sampler_allocates_aligned_contact_token_buffers() -> None:
    sampler = PairedActionTrajectorySampler.__new__(PairedActionTrajectorySampler)
    sampler.trajectory_length = 3
    sampler.data_device = torch.device("cpu")
    sampler.adapter = SimpleNamespace(
        num_envs=2,
        state_dim=4,
        joint_f_dim=2,
        action_dim=2,
        num_contacts_per_env=2,
        solver=SimpleNamespace(contact_representation="raw15_tokens", max_contact_tokens=2),
        paired_contact_adapter=SimpleNamespace(max_contact_tokens=2),
        model=SimpleNamespace(up_axis=2),
        state_world_ids=torch.arange(2),
        root_world_ids=torch.arange(2),
        contact_world_ids=torch.arange(2),
        env=SimpleNamespace(scene=None),
    )

    buffers = sampler.allocate_batch_buffers()

    assert buffers["paired_contacts"]["contact_tokens"].shape == (2, 3, 2, CONTACT_TOKEN_DIM)
    assert buffers["paired_contacts"]["contact_token_solver_active"].dtype == torch.bool
    assert buffers["paired_contact_token_body_ids"].shape == (2, 3, 2)
    assert buffers["paired_contact_token_world_ids"].shape == (2, 3, 2)


def test_validate_paired_hdf5_files_hashes_shared_arrays_and_context(tmp_path) -> None:
    primary, paired = split_paired_rollouts(_paired_rollouts())
    raw_path = tmp_path / "raw15.hdf5"
    token_path = tmp_path / "contact_tokens.hdf5"
    write_rollouts_to_hdf5(
        raw_path,
        primary,
        "Raw15-Paired",
        terrain_context=_terrain_context(),
        contact_representation="raw15_tokens",
    )
    write_rollouts_to_hdf5(
        token_path,
        paired,
        "ContactTokens-Paired",
        terrain_context=_terrain_context(),
        contact_representation="contact_tokens",
    )

    report = validate_paired_hdf5_files(raw_path, token_path)

    assert report["num_trajectories"] == 2
    assert report["steps_per_trajectory"] == 3
    assert report["total_transitions"] == 6
    assert len(report["shared_data_sha256"]) == 64
    assert len(report["shared_context_sha256"]) == 64
    assert report["shared_data_keys"] == [
        "actions",
        "gravity_dir",
        "joint_f",
        "next_states",
        "root_body_q",
        "root_body_qd",
        "states",
    ]

    with h5py.File(token_path, "r+") as handle:
        handle["data"]["contact_tokens"][0, 0, 0, 4] += 1.0
    validate_paired_hdf5_files(raw_path, token_path)

    with h5py.File(token_path, "r+") as handle:
        handle["data"]["actions"][0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="shared rollout data differ"):
        validate_paired_hdf5_files(raw_path, token_path)


def test_truncation_summary_delta_excludes_adapter_initialization_frames() -> None:
    before = {
        "frames": 3,
        "raw_contacts": 7,
        "packed_contacts": 6,
        "dropped_contacts": 1,
        "truncated_frames": 1,
        "truncated_frame_ratio": 1.0 / 3.0,
        "overflow_frames": 1,
        "overflow_tokens_total": 2,
    }
    after = {
        "frames": 13,
        "raw_contacts": 32,
        "packed_contacts": 29,
        "dropped_contacts": 3,
        "truncated_frames": 2,
        "truncated_frame_ratio": 2.0 / 13.0,
        "overflow_frames": 2,
        "overflow_tokens_total": 5,
    }

    delta = truncation_summary_delta(after, before)

    assert delta == {
        "frames": 10,
        "raw_contacts": 25,
        "packed_contacts": 23,
        "dropped_contacts": 2,
        "truncated_frames": 1,
        "truncated_frame_ratio": 0.1,
        "overflow_frames": 1,
        "overflow_tokens_total": 3,
    }
