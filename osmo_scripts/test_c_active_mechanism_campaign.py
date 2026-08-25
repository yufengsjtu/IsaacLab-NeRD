# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contract tests for the C-active mechanism ablation campaign."""

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal"
PRESET_DIR = ROOT / "osmo_scripts/presets"
CAMPAIGN_PATH = ROOT / "osmo_scripts/c_active_mechanism_campaign.yaml"

A_BASE_CONFIG = "transformer_rough_native_body_routed_active15.yaml"
C_BASE_CONFIG = "transformer_rough_native_shared_per_body_active_filter.yaml"
A_BASE_PRESET = "anymal_rough_newton_native_active15"
C_BASE_PRESET = "anymal_rough_newton_native_shared_per_body_active_filter"

EXPECTED_ROWS = {
    "A-ACTIVE-MEAN": (
        "anymal_rough_newton_native_active15_mean",
        "raw15",
        "mean",
        False,
        None,
    ),
    "A-ACTIVE-SUM-COUNT": (
        "anymal_rough_newton_native_active15_sum_count",
        "raw15",
        "sum",
        True,
        None,
    ),
    "A-ACTIVE-MEAN-COUNT": (
        "anymal_rough_newton_native_active15_mean_count",
        "raw15",
        "mean",
        True,
        None,
    ),
    "C-ACTIVE-NO-OTHER-EMBEDDING": (
        "anymal_rough_newton_native_shared_per_body_active_filter_no_other_embedding",
        "contact_tokens",
        None,
        None,
        False,
    ),
}


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_c_active_mechanism_manifest_expands_to_12_unique_jobs() -> None:
    manifest = _load_yaml(CAMPAIGN_PATH)
    campaign = manifest["campaign"]
    rows = manifest["rows"]

    assert campaign["dataset_subdirs"] == {
        "raw15": "anymal-c-rough-newton-native-raw15-sampling-v2-20260821-0630",
        "contact_tokens": "anymal-c-rough-newton-native-contact-tokens-sampling-v2-20260821-0630",
    }
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

    assert {row["id"] for row in rows} == set(EXPECTED_ROWS)
    jobs = [(row["id"], seed) for row in rows for seed in row["seeds"]]
    assert all(row["seeds"] == [0, 1, 2] for row in rows)
    assert len(jobs) == len(set(jobs)) == 12
    assert len(jobs) * campaign["resources"]["num_gpu"] == 96


def test_c_active_mechanism_configs_and_presets_are_surgical() -> None:
    rows = {row["id"]: row for row in _load_yaml(CAMPAIGN_PATH)["rows"]}
    a_base_config = _load_yaml(CONFIG_DIR / A_BASE_CONFIG)
    c_base_config = _load_yaml(CONFIG_DIR / C_BASE_CONFIG)

    for row_id, expected in EXPECTED_ROWS.items():
        preset_name, dataset_family, pooling, use_count_projection, use_other_body_embeddings = expected
        row = rows[row_id]
        assert row["preset"] == preset_name
        assert row["dataset_family"] == dataset_family

        preset = _load_yaml(PRESET_DIR / f"{preset_name}.yaml")
        baseline_preset_name = A_BASE_PRESET if dataset_family == "raw15" else C_BASE_PRESET
        baseline_preset = _load_yaml(PRESET_DIR / f"{baseline_preset_name}.yaml")
        config = _load_yaml(ROOT / preset["experiment"]["train_cfg"])
        contact_set = config["inputs"]["contact_set"]
        algorithm = config["algorithm"]

        normalized_preset = copy.deepcopy(preset)
        normalized_preset["workflow"]["base_name"] = baseline_preset["workflow"]["base_name"]
        normalized_preset["experiment"]["train_cfg"] = baseline_preset["experiment"]["train_cfg"]
        assert normalized_preset == baseline_preset
        assert preset["resources"] == {
            "num_gpu": 8,
            "num_cpu": 96,
            "memory": "512Gi",
            "storage": "512Gi",
            "platform": "ovx-l40",
        }
        assert preset["experiment"]["num_gpus"] == 8
        assert preset["experiment"]["train_num_envs"] == 64
        assert preset["experiment"]["save_interval"] == 100
        assert preset["experiment"]["persist_periodic_checkpoints"] is True
        assert preset["experiment"]["contact_filter"] == "solver_active"

        assert algorithm["num_epochs"] == 1000
        assert algorithm["num_iters_per_epoch"] == 5000
        assert algorithm["batch_size"] == 512
        assert algorithm["num_valid_batches"] == 100
        assert algorithm["dataset"]["max_capacity"] == 20_000_000
        assert algorithm["dataset"]["load_mode"] == "eager"
        assert algorithm["eval"]["num_rollouts"] == 1024
        assert float(algorithm["optimizer"]["lr_start"]) == 1e-4
        assert float(algorithm["optimizer"]["lr_end"]) == 1e-5
        assert config["env"]["neural_solver_cfg"]["contact_filter"] == "solver_active"

        if dataset_family == "raw15":
            assert contact_set["encoder_type"] == "body_routed_active15"
            assert contact_set["body_latent_dim"] == 64
            assert contact_set["hidden_dim"] == 32
            assert contact_set["pooling"] == pooling
            assert contact_set["use_count_projection"] is use_count_projection
            assert algorithm["diagnostics"] == a_base_config["algorithm"]["diagnostics"]
            normalized_config = copy.deepcopy(config)
            normalized_config["inputs"]["contact_set"].pop("pooling")
            normalized_config["inputs"]["contact_set"].pop("use_count_projection")
            assert normalized_config == a_base_config
        else:
            assert contact_set["encoder_type"] == "shared_per_body"
            assert contact_set["body_latent_dim"] == 16
            assert contact_set["hidden_dim"] == 64
            assert contact_set["use_other_body_embeddings"] is use_other_body_embeddings
            assert "categorical_decoding" not in contact_set
            assert "categorical_rounding" not in contact_set
            normalized_config = copy.deepcopy(config)
            normalized_config["inputs"]["contact_set"].pop("use_other_body_embeddings")
            assert normalized_config == c_base_config
