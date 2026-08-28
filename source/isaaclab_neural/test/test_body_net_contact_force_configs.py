# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Stage 1 force-head configs and OSMO integration."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import yaml

PACKAGE_ROOT = Path(__file__).parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
CFG_DIR = PACKAGE_ROOT / "isaaclab_neural" / "train" / "cfg" / "Anymal"
PRESET_DIR = REPOSITORY_ROOT / "osmo_scripts" / "presets"
ENTRYPOINT_PATH = REPOSITORY_ROOT / "osmo_scripts" / "entrypoint.sh"
sys.path.insert(0, str(REPOSITORY_ROOT / "osmo_scripts"))

from run_experiment import DatasetSpec, build_dataset_args, dataset_cache_complete  # noqa: E402

FORCE_VARIANTS = {
    "smoke": (
        "transformer_rough_native_shared_per_body_contact_force_smoke.yaml",
        "anymal_rough_newton_native_shared_per_body_contact_force_smoke.yaml",
        100_000,
        1,
        64,
    ),
    "formal": (
        "transformer_rough_native_shared_per_body_contact_force_formal.yaml",
        "anymal_rough_newton_native_shared_per_body_contact_force_formal.yaml",
        10_000_000,
        1000,
        1024,
    ),
}


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_force_variants_preserve_shared_per_body_contract() -> None:
    for cfg_filename, preset_filename, capacity, epochs, batch_size in FORCE_VARIANTS.values():
        cfg = _load_yaml(CFG_DIR / cfg_filename)
        preset = _load_yaml(PRESET_DIR / preset_filename)
        solver_cfg = cfg["env"]["neural_solver_cfg"]
        contact_cfg = cfg["inputs"]["contact_set"]
        force_cfg = cfg["network"]["body_net_contact_force_head"]
        force_loss_cfg = cfg["algorithm"]["loss"]["body_net_contact_force"]

        assert solver_cfg["contact_representation"] == "contact_tokens"
        assert solver_cfg["max_contact_tokens"] == 64
        assert solver_cfg["states_frame"] == "body"
        assert solver_cfg["anchor_frame_step"] == "every"
        assert solver_cfg["num_states_history"] == 10
        assert contact_cfg["encoder_type"] == "shared_per_body"
        assert force_cfg["enabled"] is True
        assert "decoder_type" not in force_cfg
        assert "body_embedding_dim" not in force_cfg
        assert force_cfg["model"]["mlp"]["layer_sizes"] == [64]
        assert force_loss_cfg == {"weight": 0.1, "active_contact_threshold_n": 1.0}
        assert cfg["algorithm"]["dataset"]["max_capacity"] == capacity
        assert cfg["algorithm"]["num_epochs"] == epochs
        assert cfg["algorithm"]["batch_size"] == batch_size

        experiment = preset["experiment"]
        assert preset["workflow"]["dataset_subdir"].endswith("-v1")
        assert experiment["record_body_net_contact_forces"] is True
        assert experiment["contact_representation"] == "contact_tokens"
        assert experiment["max_contact_tokens"] == 64
        assert experiment["num_primary_bodies"] == 17
        assert Path(experiment["train_cfg"]).name == cfg_filename


def test_force_preset_adds_generation_flag_and_old_preset_does_not(tmp_path: Path) -> None:
    spec = DatasetSpec(
        filename="dataset_train.hdf5",
        seed=0,
        split="train",
        task_kind="default",
        zero_actions=False,
        use_policy=False,
    )
    force_experiment = _load_yaml(PRESET_DIR / "anymal_rough_newton_native_shared_per_body_contact_force_smoke.yaml")[
        "experiment"
    ]
    old_experiment = _load_yaml(PRESET_DIR / "anymal_rough_newton_native_shared_per_body.yaml")["experiment"]

    force_args = build_dataset_args(force_experiment, spec, tmp_path)
    old_args = build_dataset_args(old_experiment, spec, tmp_path)

    assert force_args is not None
    assert "--record-body-net-contact-forces" in force_args
    assert old_args is not None
    assert "--record-body-net-contact-forces" not in old_args
    assert "record_body_net_contact_forces" not in old_experiment


def test_osmo_entrypoint_traces_only_after_credentials_are_exported() -> None:
    entrypoint = ENTRYPOINT_PATH.read_text(encoding="utf-8")
    trace_offset = entrypoint.index("\nset -x\n")

    assert trace_offset > entrypoint.index('export NGC_API_KEY="${NGC_API_KEY:-}"')
    assert trace_offset > entrypoint.index('export WANDB_API_KEY="${WANDB_API_KEY:-}"')


def test_force_preset_rejects_old_cache_without_force_labels(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset_train.hdf5"
    experiment = {
        "env_name": "force-test",
        "train_transitions": 4,
        "valid_transitions": 2,
        "trajectory_length": 2,
        "record_body_net_contact_forces": True,
        "dataset_specs": [
            {
                "filename": dataset_path.name,
                "seed": 0,
                "split": "train",
                "task_kind": "default",
                "zero_actions": False,
                "use_policy": False,
            }
        ],
    }
    with h5py.File(dataset_path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["mode"] = "trajectory"
        data.attrs["env"] = experiment["env_name"]
        data.attrs["total_transitions"] = 4
        data.create_dataset("states", shape=(2, 2, 3), dtype="f4")

    assert not dataset_cache_complete(tmp_path, [dataset_path.name], experiment)

    with h5py.File(dataset_path, "a") as handle:
        data = handle["data"]
        data.create_dataset("body_net_contact_forces_w", shape=(2, 2, 17, 3), dtype="f4")
        data.attrs["body_net_contact_force_schema"] = "per_primary_body_v1"
        data.attrs["body_net_contact_force_frame"] = "world"
        data.attrs["body_net_contact_force_units"] = "N"
        data.attrs["body_net_contact_force_timing"] = "post_solve_t_to_t_plus_1"

    assert not dataset_cache_complete(tmp_path, [dataset_path.name], experiment)

    with h5py.File(dataset_path, "a") as handle:
        handle["data"].attrs["body_net_contact_force_body_names"] = [f"body_{index}" for index in range(17)]

    assert dataset_cache_complete(tmp_path, [dataset_path.name], experiment)
