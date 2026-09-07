# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sanity checks for G1 OSMO mix training presets."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
PRESET_DIR = REPO_ROOT / "osmo_scripts" / "presets"
TRAIN_CFG_DIR = REPO_ROOT / "source" / "isaaclab_neural" / "isaaclab_neural" / "train" / "cfg" / "G1"
FORMAL_PRESET = "g1_newton_native_shared_per_body.yaml"
PERTURB_PRESET = "g1_newton_native_shared_per_body_perturb.yaml"
SMOKE_PRESET = "g1_newton_native_smoke.yaml"
G1_PRESETS = (FORMAL_PRESET, PERTURB_PRESET, SMOKE_PRESET)
G1_IMAGE = "docker.io/rowany28070/isaac-lab-nerd:g1-develop"


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _normalize_weights(entries: dict) -> dict[str, float]:
    total = sum(float(source["weight"]) for source in entries.values())
    return {name: float(source["weight"]) / total for name, source in entries.items()}


def test_g1_presets_use_public_dockerhub_image():
    for name in G1_PRESETS:
        workflow = _load_yaml(PRESET_DIR / name)["workflow"]
        assert workflow["image"] == G1_IMAGE
        assert "registry_credential" not in workflow


def test_g1_resolve_submit_exports_image_without_registry(monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "osmo_scripts"))
    from lib.preset import resolve_submit

    resolve_submit(
        Namespace(
            preset="g1_newton_native_smoke",
            preset_file=None,
            run_id="test",
            workflow_name=None,
            dataset_subdir=None,
            num_gpu=None,
            num_cpu=None,
            memory=None,
            storage=None,
            platform=None,
            image=None,
            registry_credential=None,
        )
    )
    output = capsys.readouterr().out
    assert f"OSMO_IMAGE={G1_IMAGE}" in output
    assert "OSMO_REGISTRY_CREDENTIAL=''" in output


def test_dockerhub_registry_login_prefers_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "osmo_scripts"))
    from lib.registry_credential import load_docker_hub_login, write_payload_files

    monkeypatch.setenv("DOCKERHUB_USERNAME", "hub-user")
    monkeypatch.setenv("DOCKERHUB_TOKEN", "hub-token")
    user, token = load_docker_hub_login(config_path=tmp_path / "missing.json")
    assert user == "hub-user"
    assert token == "hub-token"

    username_file = tmp_path / "user"
    auth_file = tmp_path / "auth"
    written_user, _ = write_payload_files(username_file, auth_file)
    assert written_user == "hub-user"
    assert username_file.read_text(encoding="utf-8") == "hub-user"
    assert auth_file.read_text(encoding="utf-8") == "hub-token"


def test_dockerhub_registry_login_reads_docker_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "osmo_scripts"))
    from lib.registry_credential import load_docker_hub_login

    monkeypatch.delenv("DOCKERHUB_USERNAME", raising=False)
    monkeypatch.delenv("DOCKERHUB_TOKEN", raising=False)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"auths":{"https://index.docker.io/v1/":{"auth":"Y29uZmlnLXVzZXI6Y29uZmlnLXRva2Vu"}}}',
        encoding="utf-8",
    )
    user, token = load_docker_hub_login(config_path=config_path)
    assert user == "config-user"
    assert token == "config-token"


def test_g1_formal_preset_uses_shared_per_body_contact_tokens():
    preset = _load_yaml(PRESET_DIR / FORMAL_PRESET)
    experiment = preset["experiment"]
    assert experiment["contact_representation"] == "contact_tokens"
    assert experiment["env_name"] == "G1-Native-ContactTokens"
    assert experiment["contact_packing_policy"] == "body_round_robin_pair_atomic"
    assert experiment["contact_filter"] == "solver_active"
    assert experiment["train_transitions"] == 20_000_000
    assert experiment["train_num_envs"] == 64
    assert experiment["trajectory_length"] == 1000
    assert experiment["dataset_mix"]["ratios"] == [2, 0, 0, 2, 2, 4]
    assert experiment["dataset_mix"]["policy_checkpoint_iters"] == [500, 1000, 1499]
    assert experiment["dataset_mix"]["perturbed_trajectory_length"] == 100
    assert experiment["dataset_mix"]["perturbed_random_fraction"] == 1.0
    assert experiment["dataset_mix"]["reset_trajectory_length"] == 500
    assert experiment["dataset_mix"]["valid"]["transitions"] == 250000
    assert experiment["dataset_mix"]["valid"]["always"] == ["policy", "random", "zero", "deployment"]
    assert "ckpt3-1499-stock" in preset["workflow"]["dataset_subdir"]
    train_cfg = _load_yaml(REPO_ROOT / experiment["train_cfg"])
    assert train_cfg["env"]["neural_solver_cfg"]["contact_representation"] == "contact_tokens"
    assert train_cfg["env"]["neural_solver_cfg"]["contact_filter"] == "solver_active"
    assert train_cfg["inputs"]["contact_set"]["encoder_type"] == "shared_per_body"
    assert train_cfg["algorithm"]["num_iters_per_epoch"] == 1250
    assert train_cfg["algorithm"]["batch_size"] == 1024
    assert float(train_cfg["algorithm"]["optimizer"]["lr_start"]) == 1.0e-3
    assert float(train_cfg["algorithm"]["optimizer"]["lr_end"]) == 1.0e-4
    assert train_cfg["network"]["model"]["mlp"]["layer_sizes"] == [128]
    assert train_cfg["algorithm"]["eval"]["dataset_path"].endswith("dataset_valid.hdf5")
    assert train_cfg["algorithm"]["eval"]["passive"] is False


def test_g1_perturb_preset_trains_only_pool_perturb(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "osmo_scripts"))
    from lib.g1_dataset_mix import mix_train_dataset_entries, mix_valid_dataset_entries
    from run_experiment import parse_dataset_specs

    preset = _load_yaml(PRESET_DIR / PERTURB_PRESET)
    experiment = preset["experiment"]
    assert experiment["dataset_mix"]["ratios"] == [0, 0, 0, 0, 0, 1]
    assert experiment["dataset_mix"]["perturbed_trajectory_length"] == 100
    assert experiment["dataset_mix"]["perturbed_random_fraction"] == 1.0
    assert "perturb-20m-pool20m-stock" in preset["workflow"]["dataset_subdir"]
    specs = parse_dataset_specs(experiment)
    by_name = {spec.filename: spec for spec in specs}
    train_names = [spec.filename for spec in specs if spec.split == "train"]
    assert train_names == ["dataset_pool_perturbed_random_train.hdf5"]
    assert by_name["dataset_pool_perturbed_random_train.hdf5"].num_transitions == 20_000_000
    assert by_name["dataset_pool_perturbed_random_train.hdf5"].trajectory_length == 100
    assert by_name["dataset_pool_perturbed_random_train.hdf5"].zero_actions is False
    assert by_name["dataset_pool_perturbed_random_train.hdf5"].resample_actions_every_frame is True
    assert "dataset_policy_noisy_train.hdf5" not in by_name
    assert "dataset_policy_train_ckpt_1499.hdf5" not in by_name
    assert "dataset_pool_perturbed_zero_train.hdf5" not in by_name
    assert by_name["dataset_policy_pool.hdf5"].split == "pool"
    assert by_name["dataset_policy_pool.hdf5"].num_transitions == 20_000_000
    valid_names = [spec.filename for spec in specs if spec.split == "valid"]
    assert valid_names == [
        "dataset_valid.hdf5",
        "dataset_random_action_valid.hdf5",
        "dataset_zero_action_valid.hdf5",
        "dataset_pool_perturbed_random_valid.hdf5",
        "dataset_deployment_policy_valid.hdf5",
    ]
    train_cfg = _load_yaml(REPO_ROOT / experiment["train_cfg"])
    expected = mix_train_dataset_entries(experiment)
    actual = train_cfg["algorithm"]["dataset"]["train_datasets"]
    assert set(actual) == {"pool_perturbed_random"}
    assert set(actual) == set(expected)
    assert actual["pool_perturbed_random"]["max_capacity"] == 20_000_000
    assert _normalize_weights(actual) == pytest.approx(_normalize_weights(expected))
    assert train_cfg["algorithm"]["dataset"]["valid_datasets"] == mix_valid_dataset_entries(experiment)
    assert set(train_cfg["algorithm"]["dataset"]["valid_datasets"]) >= {
        "random_action_trajectory",
        "zero_action_trajectory",
        "deployment_policy_trajectory",
    }
    assert float(train_cfg["algorithm"]["optimizer"]["lr_start"]) == 1.0e-3
    assert train_cfg["algorithm"]["num_iters_per_epoch"] == 2500
    assert train_cfg["algorithm"]["batch_size"] == 1024
    assert train_cfg["network"]["model"]["mlp"]["layer_sizes"] == [128]
    assert train_cfg["algorithm"]["eval"]["dataset_path"].endswith("dataset_valid.hdf5")
    assert train_cfg["algorithm"]["eval"]["passive"] is False


def test_g1_mix_expands_default_20m_recipe(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "osmo_scripts"))
    from lib.g1_dataset_mix import expand_g1_dataset_mix, mix_train_dataset_entries
    from run_experiment import parse_dataset_specs

    experiment = _load_yaml(PRESET_DIR / FORMAL_PRESET)["experiment"]
    specs = parse_dataset_specs(experiment)
    by_name = {spec.filename: spec for spec in specs}

    assert by_name["dataset_policy_train.hdf5"].num_transitions == 4_000_000
    assert by_name["dataset_policy_train.hdf5"].split == "train"
    assert by_name["dataset_policy_pool.hdf5"].split == "pool"
    assert by_name["dataset_policy_pool.hdf5"].num_transitions == 8_000_000
    assert by_name["dataset_policy_pool.hdf5"].trajectory_length == 1000
    assert "dataset_random_action_train.hdf5" not in by_name
    assert "dataset_zero_action_train.hdf5" not in by_name
    valid_names = [spec.filename for spec in specs if spec.split == "valid"]
    assert valid_names == [
        "dataset_valid.hdf5",
        "dataset_random_action_valid.hdf5",
        "dataset_zero_action_valid.hdf5",
        "dataset_policy_noisy_valid.hdf5",
        "dataset_pool_perturbed_random_valid.hdf5",
        "dataset_deployment_policy_valid.hdf5",
    ]
    assert "dataset_pool_perturbed_zero_valid.hdf5" not in by_name
    assert "dataset_pool_perturbed_zero_train.hdf5" not in by_name
    assert all(by_name[name].num_transitions == 250_000 for name in valid_names)
    assert by_name["dataset_random_action_valid.hdf5"].trajectory_length == 500
    assert by_name["dataset_zero_action_valid.hdf5"].trajectory_length == 500
    assert by_name["dataset_policy_noisy_train.hdf5"].num_transitions == 4_000_000
    assert by_name["dataset_policy_noisy_train.hdf5"].policy_action_noise_std == 0.2
    assert by_name["dataset_policy_noisy_train.hdf5"].clip_policy_action_noise is False
    assert by_name["dataset_policy_noisy_valid.hdf5"].clip_policy_action_noise is False
    ckpt_train = [spec for spec in specs if spec.filename.startswith("dataset_policy_train_ckpt_")]
    assert [spec.filename for spec in ckpt_train] == [
        "dataset_policy_train_ckpt_0500.hdf5",
        "dataset_policy_train_ckpt_1000.hdf5",
        "dataset_policy_train_ckpt_1499.hdf5",
    ]
    assert sum(spec.num_transitions for spec in ckpt_train) == 4_000_000
    assert [spec.num_transitions for spec in ckpt_train] == [1333334, 1333333, 1333333]
    assert by_name["dataset_pool_perturbed_random_train.hdf5"].num_transitions == 8_000_000
    assert by_name["dataset_pool_perturbed_random_train.hdf5"].trajectory_length == 100
    assert by_name["dataset_pool_perturbed_random_train.hdf5"].zero_actions is False
    assert by_name["dataset_pool_perturbed_random_train.hdf5"].initial_states_pool == "dataset_policy_pool.hdf5"
    assert by_name["dataset_pool_perturbed_random_train.hdf5"].resample_actions_every_frame is True
    train_transitions = sum(spec.num_transitions for spec in specs if spec.split == "train")
    assert train_transitions == 20_000_000
    assert specs[0].filename == "dataset_policy_pool.hdf5"
    pool_index = next(i for i, spec in enumerate(specs) if spec.filename == "dataset_policy_pool.hdf5")
    perturbed_index = next(
        i for i, spec in enumerate(specs) if spec.filename == "dataset_pool_perturbed_random_train.hdf5"
    )
    valid_index = next(i for i, spec in enumerate(specs) if spec.filename == "dataset_valid.hdf5")
    valid_perturbed_index = next(
        i for i, spec in enumerate(specs) if spec.filename == "dataset_pool_perturbed_random_valid.hdf5"
    )
    assert pool_index < perturbed_index
    assert valid_index < valid_perturbed_index
    assert expand_g1_dataset_mix(experiment)[0]["filename"] == "dataset_policy_pool.hdf5"
    entries = mix_train_dataset_entries(experiment)
    assert set(entries) == {
        "policy_trajectory",
        "policy_noisy_trajectory",
        *[f"policy_checkpoint_{iteration:04d}" for iteration in (500, 1000, 1499)],
        "pool_perturbed_random",
    }
    assert entries["policy_trajectory"]["max_capacity"] == 4_000_000
    assert entries["pool_perturbed_random"]["max_capacity"] == 8_000_000
    assert entries["pool_perturbed_random"]["weight"] == pytest.approx(4.0)


def test_g1_mix_expands_nonzero_policy_random_zero(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "osmo_scripts"))
    from lib.g1_dataset_mix import expand_g1_dataset_mix, mix_train_dataset_entries

    experiment = {
        "train_transitions": 20_000_000,
        "trajectory_length": 400,
        "dataset_mix": {
            "ratios": [2, 1, 1, 2, 2, 2],
            "perturbed_trajectory_length": 50,
            "perturbed_random_fraction": 0.5,
            "valid_transitions": 250_000,
            "policy_checkpoint_iters": [500, 2999],
            "policy_pool_transitions": 1,
        },
    }
    specs = expand_g1_dataset_mix(experiment)
    by_name = {spec["filename"]: spec for spec in specs}

    assert by_name["dataset_policy_pool.hdf5"]["split"] == "pool"
    assert by_name["dataset_policy_pool.hdf5"]["num_transitions"] == 4_000_000
    assert by_name["dataset_policy_pool.hdf5"]["trajectory_length"] == 400
    assert by_name["dataset_policy_train.hdf5"]["num_transitions"] == 4_000_000
    assert by_name["dataset_random_action_train.hdf5"]["num_transitions"] == 2_000_000
    assert by_name["dataset_zero_action_train.hdf5"]["num_transitions"] == 2_000_000
    assert by_name["dataset_policy_noisy_train.hdf5"]["num_transitions"] == 4_000_000
    assert by_name["dataset_policy_train_ckpt_0500.hdf5"]["num_transitions"] == 2_000_000
    assert by_name["dataset_policy_train_ckpt_2999.hdf5"]["num_transitions"] == 2_000_000
    assert by_name["dataset_pool_perturbed_random_train.hdf5"]["num_transitions"] == 2_000_000
    assert by_name["dataset_pool_perturbed_random_train.hdf5"]["initial_states_pool"] == (
        "dataset_policy_pool.hdf5"
    )
    assert by_name["dataset_pool_perturbed_random_train.hdf5"]["resample_actions_every_frame"] is True
    assert by_name["dataset_pool_perturbed_zero_train.hdf5"]["resample_actions_every_frame"] is True
    assert by_name["dataset_random_action_train.hdf5"]["trajectory_length"] == 500
    assert by_name["dataset_zero_action_train.hdf5"]["trajectory_length"] == 500
    assert by_name["dataset_pool_perturbed_random_train.hdf5"]["trajectory_length"] == 50
    assert by_name["dataset_random_action_valid.hdf5"]["split"] == "valid"
    assert by_name["dataset_zero_action_valid.hdf5"]["zero_actions"] is True
    entries = mix_train_dataset_entries(experiment)
    assert set(entries) == {
        "policy_trajectory",
        "random_action_trajectory",
        "zero_action_trajectory",
        "policy_noisy_trajectory",
        "policy_checkpoint_0500",
        "policy_checkpoint_2999",
        "pool_perturbed_random",
        "pool_perturbed_zero",
    }
    assert sum(source["max_capacity"] for source in entries.values()) == 20_000_000


def test_g1_train_cfg_matches_expanded_mix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "osmo_scripts"))
    from lib.g1_dataset_mix import mix_train_dataset_entries, mix_valid_dataset_entries

    experiment = _load_yaml(PRESET_DIR / FORMAL_PRESET)["experiment"]
    expected = mix_train_dataset_entries(experiment)
    train_cfg = _load_yaml(REPO_ROOT / experiment["train_cfg"])
    actual = train_cfg["algorithm"]["dataset"]["train_datasets"]
    assert set(actual) == set(expected)
    assert _normalize_weights(actual) == pytest.approx(_normalize_weights(expected))
    for name, source in expected.items():
        assert actual[name]["path"] == source["path"]
        assert actual[name]["max_capacity"] == source["max_capacity"]
    assert train_cfg["algorithm"]["dataset"]["valid_datasets"] == mix_valid_dataset_entries(experiment)
    assert train_cfg["algorithm"]["dataset"]["max_capacity"] == 20_000_000


def test_g1_smoke_mix_uses_single_checkpoint(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "osmo_scripts"))
    from lib.g1_dataset_mix import mix_train_dataset_entries, mix_valid_dataset_entries
    from run_experiment import parse_dataset_specs

    preset = _load_yaml(PRESET_DIR / SMOKE_PRESET)
    experiment = preset["experiment"]
    specs = parse_dataset_specs(experiment)
    assert experiment["dataset_mix"]["ratios"] == [2, 0, 0, 2, 2, 4]
    assert experiment["dataset_mix"]["policy_checkpoint_iters"] == [1499]
    assert "ckpt1499" in preset["workflow"]["dataset_subdir"]
    assert experiment["dataset_mix"]["valid"]["transitions"] == 3200
    assert experiment["dataset_mix"]["valid"]["always"] == ["policy", "random", "zero", "deployment"]
    train_specs = [spec for spec in specs if spec.split == "train"]
    assert sum(spec.num_transitions for spec in train_specs) == 3200
    assert {spec.filename for spec in train_specs} == {
        "dataset_policy_train.hdf5",
        "dataset_policy_noisy_train.hdf5",
        "dataset_policy_train_ckpt_1499.hdf5",
        "dataset_pool_perturbed_random_train.hdf5",
    }
    by_name = {spec.filename: spec for spec in specs}
    assert by_name["dataset_policy_pool.hdf5"].split == "pool"
    assert by_name["dataset_policy_pool.hdf5"].num_transitions == 1280
    assert by_name["dataset_policy_pool.hdf5"].trajectory_length == 100
    assert by_name["dataset_pool_perturbed_random_train.hdf5"].trajectory_length == 10
    assert by_name["dataset_pool_perturbed_random_train.hdf5"].resample_actions_every_frame is True
    assert experiment["dataset_mix"]["perturbed_trajectory_length"] == 10
    assert experiment["dataset_mix"]["perturbed_random_fraction"] == 1.0
    assert "stock" in preset["workflow"]["dataset_subdir"]
    train_cfg = _load_yaml(REPO_ROOT / experiment["train_cfg"])
    expected = mix_train_dataset_entries(experiment)
    actual = train_cfg["algorithm"]["dataset"]["train_datasets"]
    assert set(actual) == set(expected)
    assert _normalize_weights(actual) == pytest.approx(_normalize_weights(expected))
    assert train_cfg["algorithm"]["dataset"]["valid_datasets"] == mix_valid_dataset_entries(experiment)


def test_g1_presets_use_stock_deployment_policy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "osmo_scripts"))
    from run_experiment import parse_dataset_specs

    for name in G1_PRESETS:
        preset = _load_yaml(PRESET_DIR / name)
        experiment = preset["experiment"]
        assert experiment["deployment_task"] == "Isaac-Velocity-Flat-G1"
        assert experiment["policy_checkpoint"].endswith("g1_flat_terrain/model_1499.pt")
        assert experiment["require_policy_checkpoint"] is True
        specs = parse_dataset_specs(experiment)
        assert specs[-1].filename == "dataset_deployment_policy_valid.hdf5"
        assert specs[-1].use_policy is True
        assert experiment["kp_scale_range"] == [0.8, 1.2]
        assert experiment["kd_scale_range"] == [0.8, 1.2]


def test_g1_presets_parse_gain_ranges_with_osmo_yaml_loader(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "osmo_scripts"))
    from lib.simple_yaml import load_yaml

    for name in G1_PRESETS:
        preset = load_yaml(PRESET_DIR / name)
        experiment = preset["experiment"]
        workflow = preset["workflow"]
        assert experiment["kp_scale_range"] == [0.8, 1.2]
        assert experiment["kd_scale_range"] == [0.8, 1.2]
        assert experiment["dataset_mix"]["perturbed_random_fraction"] == 1.0
        assert experiment["dataset_mix"]["valid"]["always"] == ["policy", "random", "zero", "deployment"]
        assert workflow["image"] == G1_IMAGE
        assert "registry_credential" not in workflow
        if name == PERTURB_PRESET:
            assert experiment["dataset_mix"]["ratios"] == [0, 0, 0, 0, 0, 1]
            assert experiment["dataset_mix"]["perturbed_trajectory_length"] == 100
            assert experiment["dataset_mix"]["reset_trajectory_length"] == 500
            assert experiment["dataset_mix"]["valid"]["transitions"] == 250000
        else:
            assert experiment["dataset_mix"]["ratios"] == [2, 0, 0, 2, 2, 4]
            assert experiment["dataset_mix"]["perturbed_trajectory_length"] == (
                100 if name == FORMAL_PRESET else 10
            )
            assert experiment["dataset_mix"]["reset_trajectory_length"] == (
                500 if name == FORMAL_PRESET else 100
            )
            assert experiment["dataset_mix"]["valid"]["transitions"] == (
                250000 if name == FORMAL_PRESET else 3200
            )


def test_g1_nerd_training_does_not_apply_analytical_solver_preset():
    for name in G1_PRESETS:
        experiment = _load_yaml(PRESET_DIR / name)["experiment"]
        assert "train_preset" not in experiment


def test_g1_osmo_uses_relative_gain_randomization(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "osmo_scripts"))
    from run_experiment import DatasetSpec, build_dataset_args

    experiment = _load_yaml(PRESET_DIR / FORMAL_PRESET)["experiment"]
    spec = DatasetSpec(
        filename="dataset_train.hdf5",
        seed=0,
        split="train",
        task_kind="default",
        zero_actions=True,
        use_policy=False,
    )
    args = build_dataset_args(experiment, spec, REPO_ROOT / "data" / "datasets")
    assert "--zero-actions" in args
    assert "--randomize-pd-gains" in args
    assert "--kp-scale-min" in args
    assert "--kd-scale-max" in args
    assert "--kp-min" not in args


def test_g1_osmo_builds_env_reset_zero_valid_args(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "osmo_scripts"))
    from run_experiment import build_dataset_args, parse_dataset_specs

    experiment = _load_yaml(PRESET_DIR / FORMAL_PRESET)["experiment"]
    spec = next(
        item for item in parse_dataset_specs(experiment) if item.filename == "dataset_zero_action_valid.hdf5"
    )
    args = build_dataset_args(experiment, spec, REPO_ROOT / "data" / "datasets")
    assert args[args.index("--task") + 1] == "Isaac-Velocity-Flat-G1-Dataset-Gen-v0"
    assert args[args.index("--trajectory-length") + 1] == "500"
    assert args[args.index("--initial-states-source") + 1] == "env"
    assert "--zero-actions" in args
    assert "--randomize-pd-gains" in args
    assert "--initial-states-pool" not in args
    assert "--resample-actions-every-frame" not in args


def test_g1_osmo_builds_perturbed_dataset_args(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "osmo_scripts"))
    from run_experiment import build_dataset_args, parse_dataset_specs

    experiment = _load_yaml(PRESET_DIR / FORMAL_PRESET)["experiment"]
    spec = next(
        item
        for item in parse_dataset_specs(experiment)
        if item.filename == "dataset_pool_perturbed_random_train.hdf5"
    )
    args = build_dataset_args(experiment, spec, REPO_ROOT / "data" / "datasets")
    assert args[args.index("--task") + 1] == "Isaac-Velocity-Flat-G1-Dataset-Gen-v0"
    assert args[args.index("--trajectory-length") + 1] == "100"
    assert args[args.index("--initial-states-source") + 1] == "input_states_pool"
    assert args[args.index("--initial-states-pool") + 1].endswith("dataset_policy_pool.hdf5")
    assert "--resample-actions-every-frame" in args
    assert "--zero-actions" not in args
    assert "--randomize-pd-gains" not in args


def test_g1_osmo_builds_policy_noisy_dataset_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.syspath_prepend(str(REPO_ROOT / "osmo_scripts"))
    from run_experiment import build_dataset_args, parse_dataset_specs

    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    experiment = _load_yaml(PRESET_DIR / FORMAL_PRESET)["experiment"]
    experiment["policy_checkpoint"] = str(checkpoint)
    spec = next(
        item for item in parse_dataset_specs(experiment) if item.filename == "dataset_policy_noisy_train.hdf5"
    )
    args = build_dataset_args(experiment, spec, REPO_ROOT / "data" / "datasets")
    assert spec.policy_action_noise_std == 0.2
    assert spec.clip_policy_action_noise is False
    assert args[args.index("--task") + 1] == "Isaac-Velocity-Flat-G1"
    assert args[args.index("--sample-mode") + 1] == "policy"
    assert args[args.index("--policy-action-noise-std") + 1] == "0.2"
    assert args[args.index("--trajectory-length") + 1] == "1000"
    assert "--no-clip-policy-action-noise" in args
    assert "--randomize-pd-gains" not in args


def test_g1_osmo_rejects_policy_noise_without_policy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "osmo_scripts"))
    from run_experiment import parse_dataset_specs

    with pytest.raises(ValueError, match="use_policy is false"):
        parse_dataset_specs(
            {
                "dataset_specs": [
                    {
                        "filename": "dataset_bad.hdf5",
                        "seed": 0,
                        "split": "train",
                        "task_kind": "default",
                        "zero_actions": False,
                        "use_policy": False,
                        "policy_action_noise_std": 0.2,
                    }
                ]
            }
        )


def test_g1_osmo_prefers_per_spec_policy_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.syspath_prepend(str(REPO_ROOT / "osmo_scripts"))
    from run_experiment import build_dataset_args, parse_dataset_specs, required_dataset_files

    global_checkpoint = tmp_path / "global.pt"
    per_spec_checkpoint = tmp_path / "per_spec.pt"
    global_checkpoint.touch()
    per_spec_checkpoint.touch()
    experiment = _load_yaml(PRESET_DIR / FORMAL_PRESET)["experiment"]
    experiment["policy_checkpoint"] = str(global_checkpoint)
    spec = parse_dataset_specs(
        {
            "dataset_specs": [
                {
                    "filename": "dataset_policy_train.hdf5",
                    "seed": 100,
                    "split": "train",
                    "task_kind": "deployment",
                    "zero_actions": False,
                    "use_policy": True,
                    "policy_checkpoint": str(per_spec_checkpoint),
                }
            ]
        }
    )[0]
    args = build_dataset_args(experiment, spec, REPO_ROOT / "data" / "datasets")
    assert spec.policy_checkpoint == str(per_spec_checkpoint)
    assert args[args.index("--policy-checkpoint") + 1] == str(per_spec_checkpoint)
    assert required_dataset_files(experiment, [spec]) == [spec.filename]


def test_g1_osmo_requires_missing_per_spec_policy_checkpoint(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.syspath_prepend(str(REPO_ROOT / "osmo_scripts"))
    from run_experiment import DatasetSpec, build_dataset_args

    experiment = _load_yaml(PRESET_DIR / FORMAL_PRESET)["experiment"]
    spec = DatasetSpec(
        filename="dataset_policy_train.hdf5",
        seed=100,
        split="train",
        task_kind="deployment",
        zero_actions=False,
        use_policy=True,
        policy_checkpoint="/definitely/missing/intermediate-policy.pt",
    )
    with pytest.raises(FileNotFoundError, match="intermediate-policy.pt"):
        build_dataset_args(experiment, spec, REPO_ROOT / "data" / "datasets")


def test_g1_osmo_cache_checks_per_spec_train_transition_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.syspath_prepend(str(REPO_ROOT / "osmo_scripts"))
    from run_experiment import dataset_cache_complete

    experiment = {
        "env_name": "G1-Native-ContactTokens",
        "trajectory_length": 400,
        "contact_representation": "contact_tokens",
        "num_contacts_per_env": 64,
        "dataset_specs": [
            {
                "filename": "dataset_policy_noisy_train.hdf5",
                "seed": 3,
                "split": "train",
                "task_kind": "deployment",
                "zero_actions": False,
                "use_policy": True,
                "num_transitions": 4_000_000,
            }
        ],
    }
    dataset_path = tmp_path / "dataset_policy_noisy_train.hdf5"
    integer_fields = {
        "shape_ids_0",
        "shape_ids_1",
        "body_ids_0",
        "body_ids_1",
        "body_slots_0",
        "body_slots_1",
        "world_ids",
    }
    vector_fields = {
        "points_0_local",
        "points_1_local",
        "points_0_w",
        "points_1_w",
        "surface_points_0_w",
        "surface_points_1_w",
        "normal_w",
        "offsets_0",
        "offsets_1",
        "lever_arms_0_w",
        "lever_arms_1_w",
        "relative_velocities_0_w",
        "relative_velocities_1_w",
    }
    scalar_fields = {"separation", "thickness_0", "thickness_1"}

    def _write_native_pairs(handle: h5py.File, total_transitions: int) -> None:
        native = handle.create_group("contacts").create_group("native_pairs")
        native.create_dataset("frame_offsets", data=np.zeros(total_transitions + 1, dtype=np.int64))
        for name in integer_fields:
            native.create_dataset(name, data=np.zeros((0,), dtype=np.int64))
        for name in vector_fields:
            native.create_dataset(name, data=np.zeros((0, 3), dtype=np.float32))
        for name in scalar_fields:
            native.create_dataset(name, data=np.zeros((0,), dtype=np.float32))

    with h5py.File(dataset_path, "w") as handle:
        group = handle.create_group("data")
        group.attrs["mode"] = "trajectory"
        group.attrs["env"] = experiment["env_name"]
        group.attrs["total_transitions"] = 3_999_999
        group.attrs["dataset_schema_version"] = 3
        group.attrs["contact_storage_schema"] = "native_pairs_csr_v1"
        group.attrs["contact_frame"] = "world_v1"
        group.attrs["contact_representation"] = "contact_tokens"
        group.attrs["max_contact_tokens"] = 64
        group.create_dataset("states", data=np.zeros((1, 400, 1), dtype=np.float32))
        for field in {"root_body_q", "root_body_qd", "gravity_dir", "next_states", "joint_f"}:
            group.create_dataset(field, data=np.zeros((1, 400, 1), dtype=np.float32))
        _write_native_pairs(handle, 3_999_999)

    assert not dataset_cache_complete(tmp_path, [dataset_path.name], experiment)

    with h5py.File(dataset_path, "r+") as handle:
        handle["data"].attrs["total_transitions"] = 4_000_000
        del handle["contacts"]
        _write_native_pairs(handle, 4_000_000)
    assert dataset_cache_complete(tmp_path, [dataset_path.name], experiment)


def test_g1_osmo_cache_uses_per_spec_trajectory_length(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.syspath_prepend(str(REPO_ROOT / "osmo_scripts"))
    from run_experiment import dataset_cache_complete

    experiment = {
        "env_name": "G1-Native-ContactTokens",
        "trajectory_length": 400,
        "contact_representation": "contact_tokens",
        "num_contacts_per_env": 64,
        "dataset_specs": [
            {
                "filename": "dataset_pool_perturbed_random_train.hdf5",
                "seed": 5,
                "split": "train",
                "task_kind": "default",
                "zero_actions": False,
                "use_policy": False,
                "num_transitions": 100,
                "trajectory_length": 50,
            }
        ],
    }
    dataset_path = tmp_path / "dataset_pool_perturbed_random_train.hdf5"
    with h5py.File(dataset_path, "w") as handle:
        group = handle.create_group("data")
        group.attrs["mode"] = "trajectory"
        group.attrs["env"] = experiment["env_name"]
        group.attrs["total_transitions"] = 100
        group.attrs["dataset_schema_version"] = 3
        group.attrs["contact_storage_schema"] = "native_pairs_csr_v1"
        group.attrs["contact_frame"] = "world_v1"
        group.create_dataset("states", data=np.zeros((2, 50, 1), dtype=np.float32))
        for field in {"root_body_q", "root_body_qd", "gravity_dir", "next_states", "joint_f"}:
            group.create_dataset(field, data=np.zeros((2, 50, 1), dtype=np.float32))
        native = handle.create_group("contacts").create_group("native_pairs")
        native.create_dataset("frame_offsets", data=np.zeros(101, dtype=np.int64))
        for name in (
            "shape_ids_0",
            "shape_ids_1",
            "body_ids_0",
            "body_ids_1",
            "body_slots_0",
            "body_slots_1",
            "world_ids",
            "separation",
            "thickness_0",
            "thickness_1",
        ):
            native.create_dataset(name, data=np.zeros((0,), dtype=np.int64 if "ids" in name or "slots" in name or name == "world_ids" else np.float32))
        for name in (
            "points_0_local",
            "points_1_local",
            "points_0_w",
            "points_1_w",
            "surface_points_0_w",
            "surface_points_1_w",
            "normal_w",
            "offsets_0",
            "offsets_1",
            "lever_arms_0_w",
            "lever_arms_1_w",
            "relative_velocities_0_w",
            "relative_velocities_1_w",
        ):
            native.create_dataset(name, data=np.zeros((0, 3), dtype=np.float32))
    assert dataset_cache_complete(tmp_path, [dataset_path.name], experiment)


def test_g1_osmo_requires_stock_policy_checkpoint(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.syspath_prepend(str(REPO_ROOT / "osmo_scripts"))
    from run_experiment import DatasetSpec, build_dataset_args

    experiment = _load_yaml(PRESET_DIR / FORMAL_PRESET)["experiment"]
    spec = DatasetSpec(
        filename="dataset_deployment_policy_valid.hdf5",
        seed=40,
        split="valid",
        task_kind="deployment",
        zero_actions=False,
        use_policy=True,
    )
    experiment["policy_checkpoint"] = "/definitely/missing/g1-policy.pt"
    with pytest.raises(FileNotFoundError, match="Policy dataset generation is required"):
        build_dataset_args(experiment, spec, REPO_ROOT / "data" / "datasets")


def test_remaining_g1_train_cfgs_load_and_reference_matching_env_names():
    expected_names = {
        "transformer.yaml": ("G1", "G1"),
        "transformer_native_shared_per_body.yaml": ("G1-Native", "G1-Native-ContactTokens"),
        "transformer_native_shared_per_body_perturb.yaml": ("G1-Native", "G1-Native-ContactTokens"),
        "transformer_native_smoke.yaml": ("G1-Native", "G1-Native-ContactTokens"),
        "transformer_rough_native.yaml": ("G1-Rough-Native", "G1-Rough-Native"),
    }
    for name, (env_name, dataset_name) in expected_names.items():
        cfg = _load_yaml(TRAIN_CFG_DIR / name)
        assert cfg["env"]["env_name"] == env_name
        dataset_cfg = cfg["algorithm"]["dataset"]
        train_paths = (
            [source["path"] for source in dataset_cfg["train_datasets"].values()]
            if "train_datasets" in dataset_cfg
            else [dataset_cfg["train_dataset_path"]]
        )
        assert all(path.startswith(f"./data/datasets/{dataset_name}/") for path in train_paths)


def test_package_code_force_includes_gitignored_g1_assets(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "osmo_scripts"))
    from lib.package_code import collect_packaged_files, force_included_files

    checkpoint = tmp_path / "pretrained" / "control_policy" / "g1_flat_terrain" / "model_1499.pt"
    mix = (
        tmp_path
        / "pretrained"
        / "control_policy"
        / "g1_flat_terrain"
        / "checkpoint_mix"
        / "model_250.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    mix.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"pt")
    mix.write_bytes(b"pt")

    included = {path.as_posix() for path in force_included_files(tmp_path)}
    assert checkpoint.relative_to(tmp_path).as_posix() in included
    assert mix.relative_to(tmp_path).as_posix() in included

    monkeypatch.setattr("lib.package_code._git_listed_files", lambda _root: [])
    packaged = {path.as_posix() for path in collect_packaged_files(tmp_path)}
    assert included <= packaged
