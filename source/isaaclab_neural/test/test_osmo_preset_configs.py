# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contract tests for the public OSMO preset surface."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
PRESET_DIR = REPOSITORY_ROOT / "osmo_scripts" / "presets"

CONTACT_FREE_ENV_NAMES = {"Cartpole", "Franka-Reach"}
NON_ANYMAL_PRESETS = {
    "cartpole_newton_native.yaml",
    "franka_reach_newton_native.yaml",
    "franka_reach_policy_train.yaml",
    "franka_reach_mixed.yaml",
}
FRANKA_OSMO_IMAGE = "docker.io/rowany28070/isaac-lab-nerd:g1-develop"
FRANKA_IMAGE_PRESETS = (
    "franka_reach_newton_native.yaml",
    "franka_reach_newton_native_smoke.yaml",
    "franka_reach_policy_train.yaml",
    "franka_reach_policy_train_smoke.yaml",
    "franka_reach_mixed.yaml",
    "franka_reach_mixed_smoke.yaml",
)

PUBLIC_PRESETS = {
    "cartpole_newton_native.yaml": "Cartpole",
    "franka_reach_newton_native.yaml": "Franka-Reach",
    "franka_reach_policy_train.yaml": "Franka-Reach",
    "franka_reach_mixed.yaml": "Franka-Reach",
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


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_public_preset_surface_is_intentional() -> None:
    on_disk = {path.name for path in PRESET_DIR.glob("*.yaml")}
    assert set(PUBLIC_PRESETS) <= on_disk


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
        if "train_datasets" in dataset_config:
            assert "train_dataset_path" not in dataset_config
            train_paths = [source["path"] for source in dataset_config["train_datasets"].values()]
            generated_train_files = {item["filename"] for item in dataset_specs if item["split"] == "train"}
            assert {Path(path).name for path in train_paths} == generated_train_files
            assert all(float(source["weight"]) > 0 for source in dataset_config["train_datasets"].values())
        else:
            train_paths = [dataset_config["train_dataset_path"]]
        configured_paths = [
            *train_paths,
            *dataset_config["valid_datasets"].values(),
            train_config["algorithm"]["eval"]["dataset_path"],
        ]
        assert all(Path(path).parent.name == env_name for path in configured_paths)
        generated_valid_files = {item["filename"] for item in dataset_specs if item["split"] == "valid"}
        configured_valid_files = {Path(path).name for path in dataset_config["valid_datasets"].values()}
        assert configured_valid_files == generated_valid_files
        if env_name not in CONTACT_FREE_ENV_NAMES:
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
        if env_name in CONTACT_FREE_ENV_NAMES:
            assert experiment["num_contacts_per_env"] == 0
        else:
            assert experiment["num_contacts_per_env"] > 0
        if env_name == "Franka-Reach":
            assert experiment["initial_states_source"] == "env"
            assert experiment["sample_mode"] == "action"
            assert Path(experiment["policy_checkpoint"]).name == "model_999.pt"
            train_spec = next(item for item in dataset_specs if item["split"] == "train")
            if filename == "franka_reach_mixed.yaml":
                assert experiment["action_hold_frames"] == [2, 5, 10, 20, 40]
                assert experiment["trajectory_length"] == 240
                train_specs = [item for item in dataset_specs if item["split"] == "train"]
                assert sum(item["num_transitions"] for item in train_specs) == experiment["train_transitions"]
                kinds = {
                    item["filename"]: (
                        item["use_policy"],
                        item.get("rl_action_contract", False),
                        float(item.get("policy_action_noise_std", 0.0)),
                    )
                    for item in train_specs
                }
                assert kinds == {
                    "dataset_random_train.hdf5": (False, False, 0.0),
                    "dataset_rl_contract_train.hdf5": (False, True, 0.0),
                    "dataset_policy_noisy_train.hdf5": (True, False, 0.3),
                    "dataset_policy_train.hdf5": (True, False, 0.0),
                }
                assert float(train_config["algorithm"]["optimizer"]["lr_start"]) == 2e-4
                continue
            assert experiment["action_hold_frames"] == 20
            if filename == "franka_reach_policy_train.yaml":
                assert experiment["trajectory_length"] == 240
                assert train_spec["use_policy"] is True
                policy_valid = [
                    item["filename"] for item in dataset_specs if item["split"] == "valid" and item["use_policy"]
                ]
                assert policy_valid == ["dataset_valid.hdf5", "dataset_policy_valid.hdf5"]
            else:
                assert train_spec["use_policy"] is False
                assert float(train_config["algorithm"]["optimizer"]["lr_start"]) == 2e-4
                assert float(train_config["algorithm"]["optimizer"]["lr_end"]) == 2e-5
                assert experiment["trajectory_length"] == 200
                assert workflow["dataset_subdir"] == "franka-reach-legacy-60hz-wide-reset-200"
                smoke = _load_yaml(PRESET_DIR / "franka_reach_newton_native_smoke.yaml")
                assert smoke["experiment"]["trajectory_length"] == experiment["trajectory_length"]
                assert smoke["workflow"]["dataset_subdir"] == "franka-reach-legacy-60hz-wide-reset-200-smoke"


def test_franka_legacy_dataset_paths_match_generation_and_training() -> None:
    for path in PRESET_DIR.glob("franka_reach*.yaml"):
        preset = _load_yaml(path)
        experiment = preset["experiment"]
        assert "legacy-60hz" in preset["workflow"]["dataset_subdir"]
        root = Path(experiment["dataset_dir"]) / experiment["env_name"]
        train = _load_yaml(REPOSITORY_ROOT / experiment["train_cfg"])
        dataset = train["algorithm"]["dataset"]
        train_paths = (
            [source["path"] for source in dataset["train_datasets"].values()]
            if "train_datasets" in dataset
            else [dataset["train_dataset_path"]]
        )
        paths = [*train_paths, *dataset["valid_datasets"].values()]
        paths.append(train["algorithm"]["eval"]["dataset_path"])
        assert all(Path(value).parent == root for value in paths)


def test_osmo_dataset_directory_prefers_cli_then_preset(tmp_path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    import run_experiment

    generated = []
    monkeypatch.setattr(run_experiment, "generate_all_datasets", lambda experiment, specs, root: generated.append(root))
    monkeypatch.setattr(run_experiment, "stage_generated_datasets", lambda **kwargs: None)
    monkeypatch.setattr(run_experiment, "start_tensorboard", lambda *args: None)
    monkeypatch.setattr(run_experiment, "run_training", lambda *args: None)
    args = Namespace(
        preset="franka_reach_newton_native_smoke",
        preset_file=None,
        output_root=str(tmp_path),
        dataset_dir=None,
        workflow_base_name="test",
        dataset_subdir="test-cache",
        dataset_cache_mode="off",
        dataset_input_path="",
        storage_backend="nvdataset",
        swift_data_container="",
        nvdataset_data_dataset="",
        nvdataset_data_description="test",
        tensorboard_port=6006,
    )
    run_experiment.run(args)
    assert generated[-1] == Path("data/datasets/legacy-60hz")
    args.dataset_dir = str(tmp_path / "explicit")
    run_experiment.run(args)
    assert generated[-1] == tmp_path / "explicit"


def test_franka_presets_use_public_g1_develop_image() -> None:
    for filename in FRANKA_IMAGE_PRESETS:
        workflow = _load_yaml(PRESET_DIR / filename)["workflow"]
        assert workflow["image"] == FRANKA_OSMO_IMAGE
        assert "registry_credential" not in workflow


def test_franka_resolve_submit_exports_g1_develop_image(monkeypatch, capsys) -> None:
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    from lib.preset import resolve_submit

    resolve_submit(
        Namespace(
            preset="franka_reach_policy_train",
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
        )
    )
    output = capsys.readouterr().out
    assert f"OSMO_IMAGE={FRANKA_OSMO_IMAGE}" in output


def test_compatible_anymal_presets_share_native_dataset_buckets() -> None:
    buckets = {
        filename: (
            preset["workflow"]["dataset_subdir"],
            preset["experiment"]["env_name"],
        )
        for filename in PUBLIC_PRESETS
        if filename not in NON_ANYMAL_PRESETS
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


def test_franka_reach_policy_train_smoke_matches_full_policy_recipe() -> None:
    full = _load_yaml(PRESET_DIR / "franka_reach_policy_train.yaml")
    smoke = _load_yaml(PRESET_DIR / "franka_reach_policy_train_smoke.yaml")
    assert smoke["experiment"]["train_cfg"] == full["experiment"]["train_cfg"]
    assert smoke["experiment"]["trajectory_length"] == full["experiment"]["trajectory_length"]
    assert smoke["experiment"]["policy_checkpoint"] == full["experiment"]["policy_checkpoint"]
    assert [item["use_policy"] for item in smoke["experiment"]["dataset_specs"]] == [
        item["use_policy"] for item in full["experiment"]["dataset_specs"]
    ]
    assert smoke["experiment"]["train_transitions"] == 3200
    assert smoke["workflow"]["image"] == full["workflow"]["image"] == FRANKA_OSMO_IMAGE


def test_franka_osmo_dataset_args_omit_removed_headless_flag(tmp_path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    from run_experiment import build_dataset_args, parse_dataset_specs

    experiment = _load_yaml(PRESET_DIR / "franka_reach_newton_native.yaml")["experiment"]
    spec = next(item for item in parse_dataset_specs(experiment) if item.filename == "dataset_train.hdf5")
    args = build_dataset_args(experiment, spec, tmp_path)
    assert args is not None
    assert "--headless" not in args
    assert "--force-overwrite" in args


def test_franka_mixed_specs_forward_per_spec_generation_flags(tmp_path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    from run_experiment import build_dataset_args, parse_dataset_specs, resolve_spec_transitions

    experiment = _load_yaml(PRESET_DIR / "franka_reach_mixed.yaml")["experiment"]
    experiment["policy_checkpoint"] = str(tmp_path / "model_999.pt")
    Path(experiment["policy_checkpoint"]).write_bytes(b"stub")
    specs = {spec.filename: spec for spec in parse_dataset_specs(experiment)}
    args = {name: build_dataset_args(experiment, spec, tmp_path) for name, spec in specs.items()}
    assert all(value is not None for value in args.values())

    random_args = args["dataset_random_train.hdf5"]
    hold_index = random_args.index("--action-hold-frames")
    assert random_args[hold_index + 1 : hold_index + 6] == ["2", "5", "10", "20", "40"]
    assert "--rl-action-contract" not in random_args
    assert resolve_spec_transitions(experiment, specs["dataset_random_train.hdf5"]) == 4_000_000

    contract_args = args["dataset_rl_contract_train.hdf5"]
    assert "--rl-action-contract" in contract_args
    hold_index = contract_args.index("--action-hold-frames")
    assert contract_args[hold_index + 1 : hold_index + 5] == ["2", "5", "10", "20"]

    noisy_args = args["dataset_policy_noisy_train.hdf5"]
    assert noisy_args[noisy_args.index("--sample-mode") + 1] == "policy"
    assert noisy_args[noisy_args.index("--policy-action-noise-std") + 1] == "0.3"
    assert "--no-clip-policy-action-noise" not in noisy_args
    assert "--action-hold-frames" not in noisy_args

    clean_args = args["dataset_policy_train.hdf5"]
    assert "--policy-action-noise-std" not in clean_args
    assert resolve_spec_transitions(experiment, specs["dataset_policy_valid.hdf5"]) == experiment["valid_transitions"]


def test_dataset_spec_rejects_noise_without_policy() -> None:
    import sys

    sys.path.insert(0, str(REPOSITORY_ROOT / "osmo_scripts"))
    from run_experiment import parse_dataset_specs

    experiment = {"dataset_specs": [{"filename": "x.hdf5", "seed": 0, "policy_action_noise_std": 0.1}]}
    with __import__("pytest").raises(ValueError, match="use_policy is false"):
        parse_dataset_specs(experiment)
