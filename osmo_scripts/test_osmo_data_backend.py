"""Tests for the credential-backed OSMO data storage path."""

import argparse
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import yaml
from run_experiment import run_training, stage_generated_datasets


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_stage_generated_datasets_with_osmo_data(tmp_path: Path):
    dataset_dir = tmp_path / "Anymal-C-Rough-Native-ContactTokens"
    dataset_dir.mkdir()
    (dataset_dir / "dataset_train.hdf5").touch()

    with patch("run_experiment.subprocess.run") as run_mock:
        stage_generated_datasets(
            local_env_dir=dataset_dir,
            dataset_subdir="rough-token-20m",
            env_name=dataset_dir.name,
            storage_backend="osmo_data",
            swift_data_container="",
            nvdataset_data_dataset="",
            nvdataset_data_description="",
            osmo_data_dataset_url=(
                "swift://pdx.s8k.io/AUTH_team-nvr-srl/users/jiex/isaaclab-nerd-rowan/data/datasets/rough-token-20m/"
            ),
        )

    run_mock.assert_called_once_with(
        [
            "osmo",
            "data",
            "upload",
            ("swift://pdx.s8k.io/AUTH_team-nvr-srl/users/jiex/isaaclab-nerd-rowan/data/datasets/rough-token-20m/"),
            str(dataset_dir),
        ],
        check=True,
    )


def test_shared_per_body_lr_variant_only_changes_learning_rate():
    root = Path(__file__).resolve().parents[1]
    config_dir = root / "source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal"
    author = load_yaml(config_dir / "transformer_rough_native_shared_per_body.yaml")
    variant = load_yaml(config_dir / "transformer_rough_native_shared_per_body_lr_1e-3.yaml")

    assert variant["algorithm"]["optimizer"]["lr_start"] == "1e-3"
    assert variant["algorithm"]["optimizer"]["lr_end"] == "1e-4"

    normalized = deepcopy(variant)
    normalized["algorithm"]["optimizer"]["lr_start"] = author["algorithm"]["optimizer"]["lr_start"]
    normalized["algorithm"]["optimizer"]["lr_end"] = author["algorithm"]["optimizer"]["lr_end"]
    assert normalized == author


def test_shared_per_body_lr_preset_only_changes_names_and_config():
    preset_dir = Path(__file__).resolve().parent / "presets"
    author = load_yaml(preset_dir / "anymal_rough_newton_native_shared_per_body.yaml")
    variant = load_yaml(preset_dir / "anymal_rough_newton_native_shared_per_body_lr_1e-3.yaml")

    normalized = deepcopy(variant)
    normalized["workflow"]["base_name"] = author["workflow"]["base_name"]
    normalized["experiment"]["train_cfg"] = author["experiment"]["train_cfg"]
    assert normalized == author


def test_run_training_forwards_training_seed(tmp_path: Path):
    experiment = {
        "train_task": "test-task",
        "train_cfg": "test.yaml",
        "env_name": "test-env",
        "train_num_envs": 1,
        "num_gpus": 1,
    }
    args = argparse.Namespace(
        train_seed=2,
        enable_wandb=False,
    )

    with patch("run_experiment.subprocess.run") as run_mock:
        run_training(experiment, tmp_path, args)

    command = run_mock.call_args.args[0]
    seed_index = command.index("--seed")
    assert command[seed_index + 1] == "2"
