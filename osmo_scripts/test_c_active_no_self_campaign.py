# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contract tests for the C-active no-self-collision campaign."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import yaml
from isaaclab_neural.train.trainers import VanillaTrainer

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal"
PRESET_DIR = ROOT / "osmo_scripts/presets"
CAMPAIGN_PATH = ROOT / "osmo_scripts/c_active_no_self_campaign.yaml"

BASE_CONFIG = "transformer_rough_native_shared_per_body_active_filter.yaml"
EXPERIMENT_CONFIG = "transformer_rough_native_shared_per_body_active_filter_no_self_collision.yaml"
BASE_PRESET = "anymal_rough_newton_native_shared_per_body_active_filter.yaml"
EXPERIMENT_PRESET = "anymal_rough_newton_native_shared_per_body_active_filter_no_self_collision.yaml"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_no_self_collision_config_and_preset_are_surgical() -> None:
    baseline_config = _load_yaml(CONFIG_DIR / BASE_CONFIG)
    experiment_config = _load_yaml(CONFIG_DIR / EXPERIMENT_CONFIG)
    baseline_preset = _load_yaml(PRESET_DIR / BASE_PRESET)
    experiment_preset = _load_yaml(PRESET_DIR / EXPERIMENT_PRESET)

    solver = experiment_config["env"]["neural_solver_cfg"]
    assert solver["contact_representation"] == "contact_tokens"
    assert solver["contact_filter"] == "solver_active"
    assert solver["exclude_robot_self_collisions"] is True
    assert experiment_config["inputs"]["contact_set"] == baseline_config["inputs"]["contact_set"]
    assert "categorical_decoding" not in experiment_config["inputs"]["contact_set"]
    assert "categorical_rounding" not in experiment_config["inputs"]["contact_set"]

    normalized_config = copy.deepcopy(experiment_config)
    normalized_config["env"]["neural_solver_cfg"].pop("exclude_robot_self_collisions")
    assert normalized_config == baseline_config

    assert experiment_preset["resources"] == baseline_preset["resources"]
    assert experiment_preset["experiment"]["num_gpus"] == 8
    assert experiment_preset["experiment"]["save_interval"] == 100
    assert experiment_preset["experiment"]["persist_periodic_checkpoints"] is True
    normalized_preset = copy.deepcopy(experiment_preset)
    normalized_preset["workflow"]["base_name"] = baseline_preset["workflow"]["base_name"]
    normalized_preset["experiment"]["train_cfg"] = baseline_preset["experiment"]["train_cfg"]
    assert normalized_preset == baseline_preset


def test_no_self_collision_manifest_expands_to_three_seed_matched_jobs() -> None:
    manifest = _load_yaml(CAMPAIGN_PATH)
    campaign = manifest["campaign"]
    row = manifest["rows"][0]

    assert len(manifest["rows"]) == 1
    assert row == {
        "id": "C-ACTIVE-NO-SELF-COLLISION",
        "preset": "anymal_rough_newton_native_shared_per_body_active_filter_no_self_collision",
        "dataset_family": "contact_tokens",
        "seeds": [0, 1, 2],
    }
    assert campaign["dataset_subdir"] == ("anymal-c-rough-newton-native-contact-tokens-sampling-v2-20260821-0630")
    assert campaign["dataset_cache_mode"] == "require"
    assert campaign["storage_backend"] == "osmo_data"
    assert campaign["priority"] == "NORMAL"
    assert campaign["wandb_entity"] == "nvr-srl"
    assert campaign["wandb_project"] == "isaaclab-nerd-rowan-contact-ablation"
    assert campaign["resources"] == {
        "num_gpu": 8,
        "num_cpu": 96,
        "memory": "512Gi",
        "storage": "512Gi",
        "platform": "ovx-l40",
    }
    assert campaign["training"] == {
        "num_epochs": 1000,
        "num_iters_per_epoch": 5000,
        "save_interval": 100,
        "persist_periodic_checkpoints": True,
        "early_stop_checkpoint_gate": "verify_wandb_file_download",
    }


def test_no_self_collision_axis_is_visible_in_compact_wandb_metadata() -> None:
    config = _load_yaml(CONFIG_DIR / EXPERIMENT_CONFIG)
    config["algorithm"]["seed"] = 0
    trainer = VanillaTrainer.__new__(VanillaTrainer)
    trainer.cfg = config
    trainer.seed = 0
    trainer.num_iters_per_epoch = config["algorithm"]["num_iters_per_epoch"]
    trainer.dataset_max_capacity = config["algorithm"]["dataset"]["max_capacity"]
    trainer.neural_model = torch.nn.Linear(3, 2)

    metadata = trainer._wandb_config()

    assert metadata["exclude_robot_self_collisions"] is True
