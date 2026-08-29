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
    "anymal_newton_native.yaml": "Anymal-C-Native",
    "anymal_rough_newton_native.yaml": "Anymal-C-Rough-Native",
    "anymal_newton_native_contact_tokens.yaml": "Anymal-C-Native-ContactTokens",
    "anymal_rough_newton_native_contact_tokens.yaml": "Anymal-C-Rough-Native-ContactTokens",
    "anymal_newton_native_shared_per_body.yaml": "Anymal-C-Native-ContactTokens",
    "anymal_rough_formal.yaml": "Anymal-C-Rough-Native-ContactTokens",
    "anymal_rough_newton_native_shared_per_body_contact_force_formal.yaml": (
        "Anymal-C-Rough-Native-ContactTokens-ContactForce"
    ),
    "anymal_rough_newton_native_shared_per_body_rope_formal.yaml": "Anymal-C-Rough-Native-ContactTokens",
    "anymal_rough_newton_native_shared_per_body_contact_force_rope.yaml": (
        "Anymal-C-Rough-Native-ContactTokens-ContactForce"
    ),
}


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_public_preset_surface_is_intentional() -> None:
    assert {path.name for path in PRESET_DIR.glob("*.yaml")} == set(PUBLIC_PRESETS)


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
        assert dataset_specs
        assert len({item["filename"] for item in dataset_specs}) == len(dataset_specs)
        if env_name == "Cartpole":
            assert experiment["num_contacts_per_env"] == 0
        else:
            assert experiment["num_contacts_per_env"] > 0
