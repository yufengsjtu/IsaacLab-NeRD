# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contract tests for the A-active plus self-collision dataset and training campaign."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal"
PRESET_DIR = ROOT / "osmo_scripts/presets"
CAMPAIGN_PATH = ROOT / "osmo_scripts/a_active_self_collision_campaign.yaml"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_a_active_self_collision_config_is_architecture_matched() -> None:
    baseline = _load_yaml(CONFIG_DIR / "transformer_rough_native_body_routed_active15.yaml")
    experiment = _load_yaml(CONFIG_DIR / "transformer_rough_native_body_routed_active15_self.yaml")

    assert experiment["env"]["neural_solver_cfg"]["contact_representation"] == "active15_self_tokens"
    assert experiment["env"]["neural_solver_cfg"]["contact_filter"] == "solver_active"
    assert experiment["algorithm"]["dataset"]["contact_representation"] == "raw15_self_tokens"
    assert experiment["inputs"]["contact_set"] == baseline["inputs"]["contact_set"]
    assert experiment["network"] == baseline["network"]

    normalized = copy.deepcopy(experiment)
    normalized["env"]["env_name"] = baseline["env"]["env_name"]
    normalized["env"]["neural_solver_cfg"]["contact_representation"] = "active15_tokens"
    normalized["algorithm"]["dataset"]["contact_representation"] = "raw15_tokens"
    baseline_root = "./data/datasets/Anymal-C-Rough-Native-Raw15"
    experiment_root = "./data/datasets/Anymal-C-Rough-Native-Raw15Self"
    dataset = normalized["algorithm"]["dataset"]
    dataset["train_dataset_path"] = dataset["train_dataset_path"].replace(experiment_root, baseline_root)
    dataset["valid_datasets"] = {
        key: value.replace(experiment_root, baseline_root) for key, value in dataset["valid_datasets"].items()
    }
    normalized["algorithm"]["eval"]["dataset_path"] = normalized["algorithm"]["eval"]["dataset_path"].replace(
        experiment_root,
        baseline_root,
    )
    assert normalized == baseline


def test_dataset_and_training_presets_freeze_the_new_representation() -> None:
    dataset_preset = _load_yaml(PRESET_DIR / "anymal_rough_newton_native_raw15_self_dataset.yaml")
    training_preset = _load_yaml(PRESET_DIR / "anymal_rough_newton_native_active15_self.yaml")
    dataset_experiment = dataset_preset["experiment"]
    training_experiment = training_preset["experiment"]

    assert dataset_experiment["dataset_only"] is True
    assert dataset_experiment["contact_representation"] == "raw15_self_tokens"
    assert dataset_experiment["require_zero_contact_token_overflow"] is True
    assert dataset_experiment["require_self_collision_tokens"] is True
    assert dataset_experiment["train_transitions"] == 20_000_000
    assert dataset_experiment["valid_transitions"] == 1_000_000
    assert len(dataset_experiment["dataset_specs"]) == 5

    assert training_preset["resources"] == {
        "num_gpu": 8,
        "num_cpu": 96,
        "memory": "512Gi",
        "storage": "512Gi",
        "platform": "ovx-l40",
    }
    assert training_experiment["dataset_env_name"] == "Anymal-C-Rough-Native-Raw15Self"
    assert training_experiment["dataset_contact_representation"] == "raw15_self_tokens"
    assert training_experiment["contact_representation"] == "active15_self_tokens"
    assert training_experiment["contact_filter"] == "solver_active"
    assert training_experiment["num_gpus"] == 8
    assert training_experiment["save_interval"] == 100
    assert training_experiment["persist_periodic_checkpoints"] is True


def test_campaign_expands_to_three_seed_jobs_on_one_immutable_dataset() -> None:
    manifest = _load_yaml(CAMPAIGN_PATH)
    campaign = manifest["campaign"]
    row = manifest["rows"][0]

    assert campaign["dataset_subdir"].startswith("anymal-c-rough-newton-native-raw15-self-")
    assert campaign["dataset_cache_mode"] == "require"
    assert campaign["storage_backend"] == "osmo_data"
    assert campaign["priority"] == "NORMAL"
    assert campaign["resources"] == {
        "num_gpu": 8,
        "num_cpu": 96,
        "memory": "512Gi",
        "storage": "512Gi",
        "platform": "ovx-l40",
    }
    assert row == {
        "id": "A-ACTIVE-SELF-COLLISION",
        "preset": "anymal_rough_newton_native_active15_self",
        "dataset_family": "raw15_self",
        "seeds": [0, 1, 2],
    }
