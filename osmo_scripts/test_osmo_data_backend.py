# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the credential-backed OSMO data storage path."""

import argparse
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import yaml
from run_experiment import dataset_cache_complete, run_training, stage_generated_datasets


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_stage_generated_datasets_with_osmo_data(tmp_path: Path):
    dataset_dir = tmp_path / "Anymal-C-Rough-Native-ContactTokens"
    dataset_dir.mkdir()
    (dataset_dir / "dataset_train.hdf5").touch()

    with patch("run_experiment.subprocess.run") as run_mock:
        stage_generated_datasets(
            local_env_dir=dataset_dir,
            dataset_subdir="rough-token-20m",
            env_name=dataset_dir.name,
            storage_backend="osmo_data",
            swift_data_container="",
            nvdataset_data_dataset="",
            nvdataset_data_description="",
            osmo_data_dataset_url=(
                "swift://pdx.s8k.io/AUTH_team-nvr-srl/users/jiex/isaaclab-nerd-rowan/data/datasets/rough-token-20m/"
            ),
        )

    run_mock.assert_called_once_with(
        [
            "osmo",
            "data",
            "upload",
            ("swift://pdx.s8k.io/AUTH_team-nvr-srl/users/jiex/isaaclab-nerd-rowan/data/datasets/rough-token-20m/"),
            str(dataset_dir),
        ],
        check=True,
    )


def test_shared_per_body_lr_variant_only_changes_learning_rate():
    root = Path(__file__).resolve().parents[1]
    config_dir = root / "source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal"
    author = load_yaml(config_dir / "transformer_rough_native_shared_per_body.yaml")
    variant = load_yaml(config_dir / "transformer_rough_native_shared_per_body_lr_1e-3.yaml")

    assert variant["algorithm"]["optimizer"]["lr_start"] == "1e-3"
    assert variant["algorithm"]["optimizer"]["lr_end"] == "1e-4"

    normalized = deepcopy(variant)
    normalized["algorithm"]["optimizer"]["lr_start"] = author["algorithm"]["optimizer"]["lr_start"]
    normalized["algorithm"]["optimizer"]["lr_end"] = author["algorithm"]["optimizer"]["lr_end"]
    assert normalized == author


def test_shared_per_body_lr_preset_only_changes_names_and_config():
    preset_dir = Path(__file__).resolve().parent / "presets"
    author = load_yaml(preset_dir / "anymal_rough_newton_native_shared_per_body.yaml")
    variant = load_yaml(preset_dir / "anymal_rough_newton_native_shared_per_body_lr_1e-3.yaml")

    normalized = deepcopy(variant)
    normalized["workflow"]["base_name"] = author["workflow"]["base_name"]
    normalized["experiment"]["train_cfg"] = author["experiment"]["train_cfg"]
    assert normalized == author


def test_run_training_forwards_training_seed(tmp_path: Path):
    experiment = {
        "train_task": "test-task",
        "train_cfg": "test.yaml",
        "env_name": "test-env",
        "train_num_envs": 1,
        "num_gpus": 1,
    }
    args = argparse.Namespace(
        train_seed=2,
        enable_wandb=False,
    )

    with patch("run_experiment.subprocess.run") as run_mock:
        run_training(experiment, tmp_path, args)

    command = run_mock.call_args.args[0]
    seed_index = command.index("--seed")
    assert command[seed_index + 1] == "2"


def _write_active15_cache(path: Path) -> None:
    num_trajectories, trajectory_length, capacity = 2, 3, 4
    step_shape = (num_trajectories, trajectory_length)
    with h5py.File(path, "w") as handle:
        data_group = handle.create_group("data")
        data_group.attrs.update(
            {
                "mode": "trajectory",
                "env": "Anymal-C-Rough-Native-Active15",
                "total_transitions": 6,
                "contact_representation": "active15_tokens",
                "contact_token_dim": 17,
                "max_contact_tokens": capacity,
                "contact_identity_schema": "world_owner_v1",
                "contact_token_frame": "owner_body_v1",
                "contact_schema": "active15_owner_body_v1",
                "contact_frame": "owner_body_v1",
                "contact_velocity_point": "raw_point_midpoint_v1",
                "contact_selection": "mujoco_solver_included_v1",
            }
        )
        data_group.create_dataset("states", data=np.zeros((*step_shape, 37), dtype=np.float32))
        data_group.create_dataset("next_states", data=np.zeros((*step_shape, 37), dtype=np.float32))
        data_group.create_dataset("joint_f", data=np.zeros((*step_shape, 18), dtype=np.float32))
        data_group.create_dataset("root_body_q", data=np.zeros((*step_shape, 7), dtype=np.float32))
        data_group.create_dataset("root_body_qd", data=np.zeros((*step_shape, 6), dtype=np.float32))
        data_group.create_dataset("gravity_dir", data=np.zeros((*step_shape, 3), dtype=np.float32))
        data_group.create_dataset(
            "contact_tokens",
            data=np.zeros((*step_shape, capacity, 17), dtype=np.float32),
        )
        data_group.create_dataset(
            "contact_token_body_ids",
            data=np.full((*step_shape, capacity), -1, dtype=np.int64),
        )
        data_group.create_dataset(
            "contact_token_world_ids",
            data=np.full((*step_shape, capacity), -1, dtype=np.int64),
        )
        data_group.create_dataset("contact_token_overflow", data=np.zeros(step_shape, dtype=np.int64))


def test_active15_cache_contract_rejects_incomplete_or_wrong_shape(tmp_path: Path):
    filename = "dataset_train.hdf5"
    path = tmp_path / filename
    experiment = {
        "env_name": "Anymal-C-Rough-Native-Active15",
        "contact_representation": "active15_tokens",
        "max_contact_tokens": 4,
        "train_transitions": 6,
        "valid_transitions": 6,
        "trajectory_length": 3,
        "dataset_specs": [{"filename": filename, "seed": 0, "split": "train"}],
    }

    _write_active15_cache(path)
    assert dataset_cache_complete(tmp_path, [filename], experiment)

    with h5py.File(path, "r+") as handle:
        handle["data"].attrs["total_transitions"] = 5
    assert not dataset_cache_complete(tmp_path, [filename], experiment)

    _write_active15_cache(path)
    with h5py.File(path, "r+") as handle:
        del handle["data"]["contact_token_overflow"]
    assert not dataset_cache_complete(tmp_path, [filename], experiment)

    _write_active15_cache(path)
    with h5py.File(path, "r+") as handle:
        del handle["data"]["contact_tokens"]
        handle["data"].create_dataset(
            "contact_tokens",
            data=np.zeros((2, 3, 4, 16), dtype=np.float32),
        )
    assert not dataset_cache_complete(tmp_path, [filename], experiment)
