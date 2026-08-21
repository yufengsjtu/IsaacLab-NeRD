# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the temporary Phase-1 Active15 speed-pilot configs."""

from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal"
PRESET_DIR = ROOT / "osmo_scripts/presets"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _normalize_speed_config(config: dict) -> dict:
    normalized = deepcopy(config)
    algorithm = normalized["algorithm"]
    algorithm["num_epochs"] = 1000
    algorithm["num_iters_per_epoch"] = 5000
    algorithm["batch_size"] = 512
    algorithm["num_valid_batches"] = 100
    algorithm.pop("profiling")
    algorithm.pop("diagnostics")
    algorithm["dataset"]["max_capacity"] = 20_000_000
    algorithm["dataset"]["valid_datasets"] = {
        "exp_trajectory": "./data/datasets/Anymal-C-Rough-Native-Raw15/dataset_valid.hdf5",
        "zero_action_trajectory": (
            "./data/datasets/Anymal-C-Rough-Native-Raw15/dataset_zero_action_valid.hdf5"
        ),
        "lstm_actuator_zero_action_trajectory": (
            "./data/datasets/Anymal-C-Rough-Native-Raw15/dataset_lstm_actuator_zero_action_valid.hdf5"
        ),
        "lstm_actuator_policy_trajectory": (
            "./data/datasets/Anymal-C-Rough-Native-Raw15/dataset_lstm_actuator_policy_valid.hdf5"
        ),
    }
    algorithm["eval"] = _load_yaml(
        CONFIG_DIR / "transformer_rough_native_body_routed_active15.yaml"
    )["algorithm"]["eval"]
    return normalized


def _normalize_speed_preset(preset: dict) -> dict:
    normalized = deepcopy(preset)
    standard = _load_yaml(PRESET_DIR / "anymal_rough_newton_native_active15.yaml")
    normalized["workflow"]["base_name"] = standard["workflow"]["base_name"]
    normalized["resources"] = deepcopy(standard["resources"])
    normalized["experiment"]["train_cfg"] = standard["experiment"]["train_cfg"]
    normalized["experiment"]["num_gpus"] = standard["experiment"]["num_gpus"]
    normalized["experiment"].pop("eval_interval")
    return normalized


def test_speed_configs_only_change_the_approved_pilot_controls() -> None:
    standard = _load_yaml(CONFIG_DIR / "transformer_rough_native_body_routed_active15.yaml")
    batch_512 = _load_yaml(CONFIG_DIR / "transformer_rough_native_body_routed_active15_speed_b512.yaml")
    batch_64 = _load_yaml(CONFIG_DIR / "transformer_rough_native_body_routed_active15_speed_b64.yaml")

    for config, batch_size in ((batch_512, 512), (batch_64, 64)):
        algorithm = config["algorithm"]
        assert algorithm["num_epochs"] == 1
        assert algorithm["num_iters_per_epoch"] == 1000
        assert algorithm["batch_size"] == batch_size
        assert algorithm["num_valid_batches"] == 0
        assert algorithm["profiling"] == {
            "cuda_event_timing": True,
            "record_samples": True,
            "warmup_steps": 50,
            "phase_steps": 100,
            "wandb_system_stats_interval_seconds": 1.0,
        }
        assert algorithm["diagnostics"] == {"enabled": False}
        assert algorithm["dataset"]["max_capacity"] == 1_024_000
        assert algorithm["dataset"]["valid_datasets"] == {}
        assert algorithm["eval"] == {"interval": 0, "require_terrain_context": False}
        assert _normalize_speed_config(config) == standard


def test_speed_configs_have_the_expected_subset_and_iterator_lengths() -> None:
    trajectory_length = 400
    sample_sequence_length = 10
    num_trajectories = 1_024_000 // trajectory_length
    windows_per_trajectory = trajectory_length - sample_sequence_length + 1
    global_windows = num_trajectories * windows_per_trajectory

    assert num_trajectories == 2_560
    assert global_windows == 1_000_960
    assert global_windows // 512 == 1_955
    assert (global_windows // 8) // 64 == 1_955
    assert (global_windows // 8) // 512 == 244
    assert [step for step in range(1, 1_000) if step % 244 == 0] == [244, 488, 732, 976]


def test_speed_presets_share_data_model_and_seed_contract() -> None:
    names = (
        "anymal_rough_newton_native_active15_speed_1g_b512",
        "anymal_rough_newton_native_active15_speed_8g_b64",
        "anymal_rough_newton_native_active15_speed_8g_b512",
    )
    expected = (
        (1, 512, {"num_gpu": 1, "num_cpu": 12, "memory": "64Gi", "storage": "256Gi", "platform": "ovx-l40"}),
        (8, 64, {"num_gpu": 8, "num_cpu": 96, "memory": "512Gi", "storage": "512Gi", "platform": "ovx-l40"}),
        (8, 512, {"num_gpu": 8, "num_cpu": 96, "memory": "512Gi", "storage": "512Gi", "platform": "ovx-l40"}),
    )

    standard = _load_yaml(PRESET_DIR / "anymal_rough_newton_native_active15.yaml")
    for name, (num_gpus, batch_size, resources) in zip(names, expected, strict=True):
        preset = _load_yaml(PRESET_DIR / f"{name}.yaml")
        experiment = preset["experiment"]
        assert preset["resources"] == resources
        assert preset["resources"]["num_gpu"] == num_gpus
        assert experiment["num_gpus"] == num_gpus
        assert experiment["dataset_env_name"] == "Anymal-C-Rough-Native-Raw15"
        assert experiment["dataset_contact_representation"] == "raw15_tokens"
        assert experiment["contact_representation"] == "active15_tokens"
        assert experiment["contact_filter"] == "solver_active"
        assert experiment["eval_interval"] == 0
        assert experiment["train_cfg"].endswith(f"_speed_b{batch_size}.yaml")
        assert preset["workflow"]["dataset_subdir"] == "anymal-c-rough-newton-native-raw15"
        assert _normalize_speed_preset(preset) == standard
