# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contract tests for the combined Phase-3/4/5 native-contact campaign."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal"
PRESET_DIR = ROOT / "osmo_scripts/presets"
CAMPAIGN_PATH = ROOT / "osmo_scripts/phase345_campaign.yaml"

EXPECTED_ROWS = {
    "A-RAW-D64": ("anymal_rough_newton_native_raw15", (0, 1, 2), "raw15", 512, 1e-4, None, 64, True),
    "A-ACTIVE-D64": (
        "anymal_rough_newton_native_active15",
        (0, 1, 2),
        "raw15",
        512,
        1e-4,
        "solver_active",
        64,
        True,
    ),
    "C-RAW": (
        "anymal_rough_newton_native_shared_per_body",
        (0, 1, 2),
        "contact_tokens",
        512,
        1e-4,
        None,
        16,
        False,
    ),
    "C-ACTIVE": (
        "anymal_rough_newton_native_shared_per_body_active_filter",
        (0, 1, 2),
        "contact_tokens",
        512,
        1e-4,
        "solver_active",
        16,
        False,
    ),
    "D-RAW": (
        "anymal_rough_newton_native_contact_tokens",
        (0, 1, 2),
        "contact_tokens",
        512,
        1e-4,
        None,
        None,
        False,
    ),
    "D-ACTIVE": (
        "anymal_rough_newton_native_contact_tokens_active_filter",
        (0, 1, 2),
        "contact_tokens",
        512,
        1e-4,
        "solver_active",
        None,
        False,
    ),
    "LR-EARLIER-8G-B512": (
        "anymal_rough_newton_native_active15_lr_1e-3",
        (0, 1, 2),
        "raw15",
        512,
        1e-3,
        "solver_active",
        64,
        True,
    ),
    "LR-EARLIER-8G-B64": (
        "anymal_rough_newton_native_active15_lr_1e-3_b64",
        (0,),
        "raw15",
        64,
        1e-3,
        "solver_active",
        64,
        True,
    ),
    "A-ACTIVE-D16": (
        "anymal_rough_newton_native_active15_d16",
        (0, 1, 2),
        "raw15",
        512,
        1e-4,
        "solver_active",
        16,
        True,
    ),
    "A-ACTIVE-D32": (
        "anymal_rough_newton_native_active15_d32",
        (0, 1, 2),
        "raw15",
        512,
        1e-4,
        "solver_active",
        32,
        True,
    ),
}

EXPECTED_PHASES = {
    "A-RAW-D64": (3,),
    "A-ACTIVE-D64": (3, 4, 5),
    "C-RAW": (3,),
    "C-ACTIVE": (3,),
    "D-RAW": (3,),
    "D-ACTIVE": (3,),
    "LR-EARLIER-8G-B512": (4,),
    "LR-EARLIER-8G-B64": (4,),
    "A-ACTIVE-D16": (5,),
    "A-ACTIVE-D32": (5,),
}


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _stored_representation(experiment: dict) -> str:
    return experiment.get("dataset_contact_representation", experiment["contact_representation"])


def test_phase345_manifest_expands_to_28_unique_jobs() -> None:
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
    assert len(jobs) == 28
    assert len(set(jobs)) == 28
    assert len(jobs) * campaign["resources"]["num_gpu"] == 224


def test_phase345_presets_and_configs_match_the_frozen_matrix() -> None:
    rows = {row["id"]: row for row in _load_yaml(CAMPAIGN_PATH)["rows"]}
    resources = _load_yaml(CAMPAIGN_PATH)["campaign"]["resources"]

    for row_id, expected in EXPECTED_ROWS.items():
        preset_name, seeds, dataset_family, batch_size, lr_start, contact_filter, latent_dim, diagnostics = expected
        row = rows[row_id]
        assert row["preset"] == preset_name
        assert tuple(row["seeds"]) == seeds
        assert tuple(row["phases"]) == EXPECTED_PHASES[row_id]
        assert row["dataset_family"] == dataset_family

        preset = _load_yaml(PRESET_DIR / f"{preset_name}.yaml")
        experiment = preset["experiment"]
        config_path = ROOT / experiment["train_cfg"]
        assert config_path.is_file()
        config = _load_yaml(config_path)
        algorithm = config["algorithm"]
        solver = config["env"]["neural_solver_cfg"]
        contact_set = config["inputs"]["contact_set"]

        assert preset["resources"] == resources
        assert experiment["num_gpus"] == 8
        assert experiment["train_num_envs"] == 64
        assert experiment["max_contact_tokens"] == 64
        assert experiment["save_interval"] == 100
        assert experiment["persist_periodic_checkpoints"] is True
        assert algorithm["num_epochs"] == 1000
        assert algorithm["num_iters_per_epoch"] == 5000
        assert algorithm["batch_size"] == batch_size
        assert algorithm["batch_size"] * algorithm["num_valid_batches"] == 51_200
        assert algorithm["dataset"]["max_capacity"] == 20_000_000
        assert algorithm["dataset"]["load_mode"] == "eager"
        assert algorithm["eval"]["num_rollouts"] == 1024
        assert float(algorithm["optimizer"]["lr_start"]) == lr_start
        assert float(algorithm["optimizer"]["lr_end"]) == lr_start / 10
        assert solver.get("contact_filter") == contact_filter
        assert experiment.get("contact_filter") == contact_filter
        assert algorithm.get("diagnostics", {}).get("enabled", False) is diagnostics
        if diagnostics:
            assert algorithm["diagnostics"]["batches_per_epoch"] == 1

        if dataset_family == "raw15":
            assert experiment.get("dataset_env_name", experiment["env_name"]) == "Anymal-C-Rough-Native-Raw15"
            assert _stored_representation(experiment) == "raw15_tokens"
            assert preset["workflow"]["dataset_subdir"] == "anymal-c-rough-newton-native-raw15"
        else:
            assert experiment.get("dataset_env_name", experiment["env_name"]) == "Anymal-C-Rough-Native-ContactTokens"
            assert _stored_representation(experiment) == "contact_tokens"
            assert preset["workflow"]["dataset_subdir"] == "anymal-c-rough-newton-native-contact-tokens"

        if row_id.startswith("D-"):
            assert "encoder_type" not in contact_set
            assert contact_set["num_latent_queries"] == 8
            assert contact_set["hidden_size"] == 384
        else:
            assert contact_set.get("body_latent_dim") == latent_dim


def test_phase345_filter_pairs_share_dataset_and_seed_contracts() -> None:
    rows = {row["id"]: row for row in _load_yaml(CAMPAIGN_PATH)["rows"]}
    for raw_id, active_id in (
        ("A-RAW-D64", "A-ACTIVE-D64"),
        ("C-RAW", "C-ACTIVE"),
        ("D-RAW", "D-ACTIVE"),
    ):
        raw = rows[raw_id]
        active = rows[active_id]
        assert raw["dataset_family"] == active["dataset_family"]
        assert raw["seeds"] == active["seeds"]

        raw_preset = _load_yaml(PRESET_DIR / f"{raw['preset']}.yaml")
        active_preset = _load_yaml(PRESET_DIR / f"{active['preset']}.yaml")
        assert raw_preset["workflow"]["dataset_subdir"] == active_preset["workflow"]["dataset_subdir"]

        raw_config = _load_yaml(ROOT / raw_preset["experiment"]["train_cfg"])
        active_config = _load_yaml(ROOT / active_preset["experiment"]["train_cfg"])
        assert raw_config["algorithm"].get("diagnostics") == active_config["algorithm"].get("diagnostics")
