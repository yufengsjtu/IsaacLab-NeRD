# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Phase-2 Active15 quality-control configs."""

from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal"
PRESET_DIR = ROOT / "osmo_scripts/presets"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _normalize_quality_config(config: dict) -> dict:
    normalized = deepcopy(config)
    algorithm = normalized["algorithm"]
    algorithm["batch_size"] = 512
    algorithm["num_valid_batches"] = 100
    algorithm["dataset"]["max_capacity"] = 20_000_000
    algorithm["diagnostics"] = {"enabled": True, "batches_per_epoch": 1}
    return normalized


def _normalize_quality_preset(preset: dict) -> dict:
    normalized = deepcopy(preset)
    standard = _load_yaml(PRESET_DIR / "anymal_rough_newton_native_active15.yaml")
    normalized["workflow"]["base_name"] = standard["workflow"]["base_name"]
    normalized["resources"] = deepcopy(standard["resources"])
    normalized["experiment"]["train_cfg"] = standard["experiment"]["train_cfg"]
    normalized["experiment"]["num_gpus"] = standard["experiment"]["num_gpus"]
    normalized["experiment"]["save_interval"] = standard["experiment"]["save_interval"]
    normalized["experiment"]["persist_periodic_checkpoints"] = True
    return normalized


def test_phase2_quality_configs_only_change_capacity_or_batch_size() -> None:
    standard = _load_yaml(CONFIG_DIR / "transformer_rough_native_body_routed_active15.yaml")
    subset_1m = _load_yaml(CONFIG_DIR / "transformer_rough_native_body_routed_active15_phase2_1m_b512.yaml")
    full20m_batch_1024 = _load_yaml(
        CONFIG_DIR / "transformer_rough_native_body_routed_active15_phase2_full20m_b1024.yaml"
    )

    for config, batch_size, max_capacity, num_valid_batches in (
        (subset_1m, 512, 1_024_000, 100),
        (full20m_batch_1024, 1024, 20_000_000, 50),
    ):
        algorithm = config["algorithm"]
        assert algorithm["num_epochs"] == 1000
        assert algorithm["num_iters_per_epoch"] == 5000
        assert algorithm["batch_size"] == batch_size
        assert algorithm["num_valid_batches"] == num_valid_batches
        assert algorithm["dataset"]["max_capacity"] == max_capacity
        assert _normalize_quality_config(config) == standard


def test_phase2_quality_presets_share_the_standard_contract() -> None:
    names = (
        "anymal_rough_newton_native_active15_phase2_1m_8g_b512",
        "anymal_rough_newton_native_active15_phase2_full20m_8g_b512",
        "anymal_rough_newton_native_active15_phase2_full20m_4g_b1024",
        "anymal_rough_newton_native_active15_phase2_full20m_4g_b512",
    )
    expected = (
        (
            8,
            512,
            1_024_000,
            {"num_gpu": 8, "num_cpu": 96, "memory": "512Gi", "storage": "512Gi", "platform": "ovx-l40"},
        ),
        (
            8,
            512,
            20_000_000,
            {"num_gpu": 8, "num_cpu": 96, "memory": "512Gi", "storage": "512Gi", "platform": "ovx-l40"},
        ),
        (
            4,
            1024,
            20_000_000,
            {"num_gpu": 4, "num_cpu": 48, "memory": "440Gi", "storage": "512Gi", "platform": "ovx-l40"},
        ),
        (
            4,
            512,
            20_000_000,
            {"num_gpu": 4, "num_cpu": 48, "memory": "440Gi", "storage": "512Gi", "platform": "ovx-l40"},
        ),
    )

    standard = _load_yaml(PRESET_DIR / "anymal_rough_newton_native_active15.yaml")
    for name, (num_gpus, batch_size, max_capacity, resources) in zip(names, expected, strict=True):
        preset = _load_yaml(PRESET_DIR / f"{name}.yaml")
        experiment = preset["experiment"]
        config = _load_yaml(ROOT / experiment["train_cfg"])
        assert preset["resources"] == resources
        assert experiment["num_gpus"] == num_gpus
        assert config["algorithm"]["batch_size"] == batch_size
        assert config["algorithm"]["dataset"]["max_capacity"] == max_capacity
        assert experiment["dataset_contact_representation"] == "raw15_tokens"
        assert experiment["contact_representation"] == "active15_tokens"
        assert experiment["contact_filter"] == "solver_active"
        assert preset["workflow"]["dataset_subdir"] == "anymal-c-rough-newton-native-raw15"
        assert _normalize_quality_preset(preset) == standard


def test_phase2_quality_comparison_axes_have_expected_sample_budgets() -> None:
    updates_per_epoch = 5_000
    sequence_length = 10

    assert 8 * 512 == 4 * 1024 == 4_096
    assert 4 * 512 == 2_048
    assert updates_per_epoch * 4_096 == 20_480_000
    assert updates_per_epoch * 2_048 == 10_240_000
    assert updates_per_epoch * 4_096 * sequence_length == 204_800_000
    assert 100 * 4_096 == 200 * 2_048
