# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contract tests for the public OSMO preset surface."""

from __future__ import annotations

from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
PRESET_DIR = REPOSITORY_ROOT / "osmo_scripts" / "presets"

PUBLIC_PRESETS = {
    "cartpole_newton_native.yaml": "Cartpole",
    "anymal_newton_native.yaml": "Anymal-C-Flat-Native-v2",
    "anymal_rough_newton_native.yaml": "Anymal-C-Rough-Native-v2",
    "anymal_newton_native_contact_tokens.yaml": "Anymal-C-Flat-Native-v2",
    "anymal_rough_newton_native_contact_tokens.yaml": "Anymal-C-Rough-Native-v2",
    "anymal_newton_native_shared_per_body.yaml": "Anymal-C-Flat-Native-v2",
    "anymal_rough_formal.yaml": "Anymal-C-Rough-Native-v2",
    "anymal_rough_newton_native_shared_per_body_contact_force_formal.yaml": ("Anymal-C-Rough-Native-Force-v2"),
    "anymal_rough_newton_native_shared_per_body_rope_formal.yaml": "Anymal-C-Rough-Native-v2",
    "anymal_rough_newton_native_shared_per_body_contact_force_rope.yaml": ("Anymal-C-Rough-Native-Force-v2"),
}
G1_PRESET_FILES = {
    "g1_newton_native_shared_per_body.yaml",
    "g1_newton_native_shared_per_body_perturb.yaml",
    "g1_newton_native_smoke.yaml",
}
ADDITIONAL_TRACKED_PRESETS = {
    "anymal_rough_newton_native_shared_per_body_contact_force_rope_active_filter.yaml",
}


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_public_preset_surface_is_intentional() -> None:
    assert {path.name for path in PRESET_DIR.glob("*.yaml")} == (
        set(PUBLIC_PRESETS) | G1_PRESET_FILES | ADDITIONAL_TRACKED_PRESETS
    )


def test_public_presets_have_valid_unique_training_mappings() -> None:
    train_cfgs = set()
    for filename, env_name in PUBLIC_PRESETS.items():
        preset = _load_yaml(PRESET_DIR / filename)
        workflow = preset["workflow"]
        experiment = preset["experiment"]
        train_cfg = REPOSITORY_ROOT / experiment["train_cfg"]
        dataset_specs = experiment["dataset_specs"]

        assert experiment["env_name"] == env_name
        assert workflow["dataset_subdir"]
        assert train_cfg.is_file()
        assert train_cfg not in train_cfgs
        train_cfgs.add(train_cfg)
        train_config = _load_yaml(train_cfg)
        dataset_config = train_config["algorithm"]["dataset"]
        configured_paths = [
            dataset_config["train_dataset_path"],
            *dataset_config["valid_datasets"].values(),
            train_config["algorithm"]["eval"]["dataset_path"],
        ]
        assert all(Path(path).parent.name == env_name for path in configured_paths)
        generated_valid_files = {item["filename"] for item in dataset_specs if item["split"] == "valid"}
        configured_valid_files = {Path(path).name for path in dataset_config["valid_datasets"].values()}
        assert configured_valid_files == generated_valid_files
        if env_name != "Cartpole":
            algorithm = train_config["algorithm"]
            optimizer = algorithm["optimizer"]
            assert algorithm["num_epochs"] == 1000
            assert algorithm["num_iters_per_epoch"] == 1250
            assert algorithm["batch_size"] == 1024
            assert dataset_config["max_capacity"] == 10_000_000
            if env_name == "Anymal-C-Flat-Native-v2":
                assert Path(algorithm["eval"]["dataset_path"]).name == "dataset_lstm_actuator_policy_valid.hdf5"
                assert float(optimizer["lr_start"]) == 2e-3
                assert float(optimizer["lr_end"]) == 2e-4
            else:
                assert float(optimizer["lr_start"]) == 2e-4
                assert float(optimizer["lr_end"]) == 2e-5
        assert dataset_specs
        assert len({item["filename"] for item in dataset_specs}) == len(dataset_specs)
        if env_name == "Cartpole":
            assert experiment["num_contacts_per_env"] == 0
        else:
            assert experiment["num_contacts_per_env"] > 0


def test_compatible_anymal_presets_share_native_dataset_buckets() -> None:
    buckets = {
        filename: (
            preset["workflow"]["dataset_subdir"],
            preset["experiment"]["env_name"],
        )
        for filename in PUBLIC_PRESETS
        if filename != "cartpole_newton_native.yaml"
        for preset in [_load_yaml(PRESET_DIR / filename)]
    }
    flat = {key: value for key, value in buckets.items() if "rough" not in key}
    rough_force = {key: value for key, value in buckets.items() if "contact_force" in key}
    rough = {key: value for key, value in buckets.items() if "rough" in key and "contact_force" not in key}

    assert set(flat.values()) == {("anymal-c-flat-native-v2", "Anymal-C-Flat-Native-v2")}
    assert set(rough.values()) == {("anymal-c-rough-native-v2", "Anymal-C-Rough-Native-v2")}
    assert set(rough_force.values()) == {("anymal-c-rough-native-force-v2", "Anymal-C-Rough-Native-Force-v2")}
    assert {_load_yaml(PRESET_DIR / filename)["experiment"]["train_transitions"] for filename in buckets} == {
        10_000_000
    }
