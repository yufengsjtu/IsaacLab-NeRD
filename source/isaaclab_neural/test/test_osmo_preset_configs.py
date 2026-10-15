# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contract tests for the public OSMO preset surface."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
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
    "franka_lift_policy_train.yaml",
    "franka_lift_mixed.yaml",
}
FRANKA_OSMO_IMAGE = "docker.io/rowany28070/isaac-lab-nerd:g1-develop"
FRANKA_IMAGE_PRESETS = (
    "franka_reach_newton_native.yaml",
    "franka_reach_policy_train.yaml",
    "franka_reach_mixed.yaml",
    "franka_lift_policy_train.yaml",
    "franka_lift_mixed.yaml",
)

PUBLIC_PRESETS = {
    "cartpole_newton_native.yaml": "Cartpole",
    "franka_reach_newton_native.yaml": "Franka-Reach",
    "franka_reach_policy_train.yaml": "Franka-Reach",
    "franka_reach_mixed.yaml": "Franka-Reach",
    "franka_lift_policy_train.yaml": "Franka-Lift",
    "franka_lift_mixed.yaml": "Franka-Lift",
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


def test_franka_policy_mixed_comparisons_share_training_contracts() -> None:
    for robot in ("reach", "lift"):
        policy = _load_yaml(PRESET_DIR / f"franka_{robot}_policy_train.yaml")
        mixed = _load_yaml(PRESET_DIR / f"franka_{robot}_mixed.yaml")
        p, m = policy["experiment"], mixed["experiment"]
        pc, mc = (_load_yaml(REPOSITORY_ROOT / e["train_cfg"]) for e in (p, m))
        assert pc["network"] == mc["network"]
        assert pc["env"] == mc["env"]
        assert pc["inputs"] == mc["inputs"]
        for key in ("optimizer", "loss", "batch_size", "num_epochs", "num_iters_per_epoch", "sample_sequence_length"):
            assert pc["algorithm"][key] == mc["algorithm"][key]
        assert p["train_transitions"] == m["train_transitions"] == 10_000_000
        assert p["valid_transitions"] == m["valid_transitions"]
        assert p["policy_checkpoint"] == m["policy_checkpoint"]
        assert policy["workflow"]["dataset_subdir"] != mixed["workflow"]["dataset_subdir"]
        common = (
            ("dataset_policy_valid.hdf5", "dataset_passive_valid.hdf5")
            if robot == "reach"
            else ("dataset_valid.hdf5", "dataset_zero_action_valid.hdf5")
        )
        for filename in common:
            assert next(s for s in p["dataset_specs"] if s["filename"] == filename) == next(
                s for s in m["dataset_specs"] if s["filename"] == filename
            )
        train = [s for s in m["dataset_specs"] if s["split"] == "train"]
        valid = [s for s in m["dataset_specs"] if s["split"] == "valid"]
        assert {s["seed"] for s in train}.isdisjoint(s["seed"] for s in valid)
        assert sum(s["num_transitions"] for s in train) == 10_000_000
        if robot == "lift":
            weights = [s["weight"] for s in mc["algorithm"]["dataset"]["train_datasets"].values()]
            assert weights == [0.6, 0.2, 0.2]
            assert p["require_finger_object_normal_forces"] and m["require_finger_object_normal_forces"]


@pytest.mark.parametrize("cache_mode", ["off", "auto"])
def test_training_only_cannot_fall_back_to_generation(tmp_path, monkeypatch, cache_mode):
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    import run_experiment

    exp = {
        "training_only": True,
        "env_name": "test",
        "data_gen_num_envs": 8,
        "train_num_envs": 8,
        "train_cfg": "unused",
        "dataset_specs": [{"filename": "missing.hdf5", "seed": 42}],
    }
    monkeypatch.setattr(run_experiment, "load_preset", lambda *args: {"experiment": exp})
    monkeypatch.setattr(run_experiment, "generate_all_datasets", lambda *args: pytest.fail("must not generate"))
    args = Namespace(
        preset="test",
        preset_file=None,
        output_root=str(tmp_path),
        dataset_dir=str(tmp_path / "cache"),
        workflow_base_name="test",
        dataset_subdir="test",
        dataset_cache_mode=cache_mode,
        dataset_input_path="",
    )
    with pytest.raises(ValueError, match="training-only"):
        run_experiment.run(args)


def test_keyboard_generator_rejects_generic_collector(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    from run_experiment import build_dataset_args, parse_dataset_specs

    exp = _load_yaml(PRESET_DIR / "so101_keyboard_resident_unique_smoke.yaml")["experiment"]
    with pytest.raises(ValueError, match="bounded Keyboard dataset profile"):
        build_dataset_args(exp, parse_dataset_specs(exp)[0], tmp_path)


@pytest.mark.parametrize("nucleus", [None, "test-nucleus-override"])
def test_keyboard_render_is_local_and_gpu_allocation_matches_runtime(tmp_path, monkeypatch, nucleus):
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    from lib.preset import render_workflow

    args = Namespace(
        preset="so101_keyboard_resident_unique_smoke",
        preset_file=None,
        run_id="test",
        workflow_name=None,
        dataset_subdir=None,
        num_gpu="4",
        num_cpu=None,
        memory=None,
        storage=None,
        platform=None,
        image=None,
        train_cfg_overrides="algorithm.num_epochs 3",
        nucleus_credential=nucleus,
        workflow_file=str(REPOSITORY_ROOT / "osmo_scripts/osmo_workflow.yaml"),
        output=str(tmp_path / "job.yaml"),
    )
    render_workflow(args)
    workflow = _load_yaml(Path(args.output))["workflow"]
    assert workflow["resources"]["default"]["gpu"] == 4
    task = workflow["tasks"][0]
    assert task["environment"]["NERD_NUM_GPUS"] == "4"
    assert task["environment"]["DATASET_CACHE_MODE"] == "auto"
    assert task["environment"]["TRAIN_CFG_OVERRIDES"] == "algorithm.num_epochs 3"
    assert "WANDB_API_KEY" not in task["environment"]
    preset = _load_yaml(PRESET_DIR / "so101_keyboard_resident_unique_smoke.yaml")
    expected_nucleus = nucleus or preset["workflow"]["nucleus_credential"]
    assert task["credentials"][expected_nucleus] == {"OMNI_USER": "omni_user", "OMNI_PASS": "omni_pass"}
    assert "OMNI_PASS" not in task["environment"]
    assert all(Path(entry["localpath"]).is_file() for entry in task["files"])
    with pytest.raises(FileExistsError):
        render_workflow(args)


def test_keyboard_training_command_uses_explicit_process_count(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    import run_experiment

    calls = []
    monkeypatch.setattr(run_experiment.subprocess, "run", lambda command, **kwargs: calls.append(command))
    exp = _load_yaml(PRESET_DIR / "so101_keyboard_resident_unique_smoke.yaml")["experiment"]
    exp["num_gpus"] = 4
    run_experiment.run_training(exp, tmp_path, Namespace(enable_wandb=False))
    command = calls[0]
    assert command[command.index("--nproc_per_node") + 1] == "4"
    assert command[command.index("--eval-interval") + 1] == "1"
    assert "--no-time-stamp" in command and "--skip-check-log-override" not in command


def test_keyboard_runtime_config_rebases_files_and_records_window_sampling(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    import isaaclab_neural.data.transition_index as transition_index
    import lib.keyboard_data
    import run_experiment

    monkeypatch.setattr(lib.keyboard_data, "validate_dataset_scale", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        transition_index,
        "build_transition_index",
        lambda path, *, window_sampling="canonical", **kwargs: SimpleNamespace(
            stats={"window_sampling": window_sampling}
        ),
    )
    exp = _load_yaml(PRESET_DIR / "so101_keyboard_resident_unique_smoke.yaml")["experiment"]
    source = REPOSITORY_ROOT / exp["train_cfg"]
    original = source.read_bytes()
    exp["train_cfg"] = str(source)
    cache = tmp_path / "cache"
    run_experiment.prepare_training_only_config(exp, cache, tmp_path, "algorithm.num_epochs 3")
    cfg = _load_yaml(Path(exp["train_cfg"]))
    assert cfg["algorithm"]["num_epochs"] == 3 and cfg["algorithm"]["num_iters_per_epoch"] == 2
    assert all(Path(item["path"]).parent == cache for item in cfg["algorithm"]["dataset"]["train_datasets"].values())
    assert source.read_bytes() == original
    contract = cfg["dataset_capacity_contract"]
    assert contract and all(stats["window_sampling"] == "sliding" for stats in contract.values())


@pytest.mark.parametrize("tamper", ["data", "receipt"])
def test_keyboard_bundle_rejects_changed_bytes_before_loading(tmp_path, monkeypatch, tamper):
    import json

    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    from lib.keyboard_data import MANIFEST, SCHEMA, file_sha256, validate_bundle

    data = tmp_path / "sample.hdf5"
    receipt = tmp_path / "sample.hdf5.replay.json"
    data.write_bytes(b"original")
    receipt.write_text("{}")
    manifest = {
        "schema": SCHEMA,
        "files": {data.name: {"sha256": file_sha256(data), "replay_sha256": file_sha256(receipt)}},
    }
    (tmp_path / MANIFEST).write_text(json.dumps(manifest))
    (data if tamper == "data" else receipt).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_bundle(tmp_path, {"dataset_specs": [{"filename": data.name, "split": "train", "seed": 42}]})


def test_keyboard_config_and_presets_are_packaged_without_logs(monkeypatch):
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    from lib.package_code import collect_packaged_files

    files = {path.as_posix() for path in collect_packaged_files(REPOSITORY_ROOT)}
    assert "osmo_scripts/presets/so101_keyboard_resident_unique.yaml" in files
    assert "osmo_scripts/presets/so101_keyboard_resident_unique_smoke.yaml" in files
    assert "source/isaaclab_neural/isaaclab_neural/train/cfg/SO101/transformer_keyboard_throughput.yaml" in files
    assert "osmo_scripts/lib/keyboard_data.py" in files
    assert not any(path.startswith("logs/") for path in files)


def test_keyboard_runtime_version_mismatch_is_not_silently_accepted(monkeypatch):
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    from lib import keyboard_data

    monkeypatch.setattr(keyboard_data, "version", lambda name: "wrong-version")
    with pytest.raises(RuntimeError, match="matching image"):
        keyboard_data.validate_runtime({"runtime_versions": {"newton": "1.6.0"}})


def test_keyboard_recipe_is_fixed_owner_frame_and_matches_franka_training_scale():
    preset = _load_yaml(PRESET_DIR / "so101_keyboard_resident_unique.yaml")
    reference = _load_yaml(PRESET_DIR / "franka_lift_mixed.yaml")
    exp = preset["experiment"]
    cfg = _load_yaml(REPOSITORY_ROOT / exp["train_cfg"])
    franka = _load_yaml(REPOSITORY_ROOT / reference["experiment"]["train_cfg"])
    assert exp["env_name"] == cfg["env"]["env_name"] == "SO101-Keyboard-Fixed"
    assert exp["train_task"] == "IsaacContrib-Keyboard-SO101-NeRD"
    assert cfg["env"]["keyboard_variant"] == 0
    assert cfg["inputs"]["low_dim"] == ["states_embedding", "joint_f", "gravity_dir"]
    solver = cfg["env"]["neural_solver_cfg"]
    assert not {"model_context_fields", "include_articulation_context"} & set(solver)
    assert solver["contact_token_frame"] == exp["contact_token_frame"] == "owner"
    assert exp["dataset_profile"] == "so101_keyboard" and not exp["training_only"]
    assert all("variant_ids" not in spec for spec in exp["dataset_specs"])
    for key in (
        "num_epochs",
        "num_iters_per_epoch",
        "batch_size",
        "sample_sequence_length",
        "num_valid_batches",
        "grad_norm",
    ):
        assert cfg["algorithm"][key] == franka["algorithm"][key]
    for key in ("lr_start", "lr_end"):
        assert float(cfg["algorithm"]["optimizer"][key]) == float(franka["algorithm"]["optimizer"][key])
    for key in ("type", "weight_decay", "lr_schedule", "betas"):
        assert cfg["algorithm"]["optimizer"][key] == franka["algorithm"]["optimizer"][key]
    # Same trunk as Franka except the streaming temporal cone (receptive field = history length).
    streaming = {"position_encoding", "attention_spans"}
    trunk = {key: value for key, value in cfg["network"]["transformer"].items() if key not in streaming}
    assert trunk == franka["network"]["transformer"]
    assert cfg["network"]["transformer"]["position_encoding"] == "relative_lag"
    spans = cfg["network"]["transformer"]["attention_spans"]
    assert len(spans) == trunk["n_layer"] and sum(span - 1 for span in spans) + 1 == solver["num_states_history"]
    assert solver["use_frame_cache"] and solver["inference_precision"] == "fp16" and solver["inference_compile"]
    for key in ("encoder_type", "body_latent_dim", "hidden_dim"):
        assert cfg["inputs"]["contact_set"][key] == franka["inputs"]["contact_set"][key]
    assert solver["num_states_history"] == cfg["algorithm"]["sample_sequence_length"] == 10
    assert cfg["env"]["num_envs"] == exp["train_num_envs"] == 64
    assert exp["eval_interval"] == cfg["algorithm"]["eval"]["interval"] == 5
    assert cfg["algorithm"]["eval"]["mode"] == "keyboard_policy"
    assert (REPOSITORY_ROOT / cfg["algorithm"]["eval"]["policy_checkpoint"]).is_file()
    dataset = cfg["algorithm"]["dataset"]
    assert (dataset["load_mode"], dataset["num_data_workers"]) == ("eager", 0)
    # The K losses and key head: the configuration that matched native closed-loop typing.
    assert cfg["network"]["key_head"]["enabled"]
    assert cfg["algorithm"]["loss"]["keyboard_state_group_weights"] == {"robot": 0.5, "key": 0.5}
    assert cfg["algorithm"]["loss"]["keyboard_key_position_scale_m"] == 0.0002
    assert cfg["algorithm"]["loss"]["keyboard_key_velocity_weighting"] == "pooled"
    assert exp["minimum_unique_transitions"] == {"train": 10_000_000, "valid": 3_000_000}
    for split, minimum in exp["minimum_unique_transitions"].items():
        specs = [item for item in exp["dataset_specs"] if item["split"] == split]
        assert sum(item["num_transitions"] for item in specs) == minimum


def test_keyboard_model_forward_backward_on_cpu():
    import copy

    import torch
    from isaaclab_neural.models.models import ModelMixedInput
    from isaaclab_neural.utils.running_mean_std import RunningMeanStd

    preset = _load_yaml(PRESET_DIR / "so101_keyboard_resident_unique.yaml")
    cfg = _load_yaml(REPOSITORY_ROOT / preset["experiment"]["train_cfg"])
    network = copy.deepcopy(cfg["network"])
    sample = {
        key: torch.zeros(2, 10, width)
        for key, width in {"states_embedding": 228, "joint_f": 114, "gravity_dir": 3}.items()
    }
    sample["contact_tokens"] = torch.zeros(2, 10, 128, 17)
    model = ModelMixedInput(sample, 228, cfg["inputs"], network, num_bodies=116, device="cpu")
    assert model.key_head is not None
    # The key head decodes with pooled output statistics, so it requires normalization buffers.
    model.set_input_rms({key: RunningMeanStd(shape=(value.shape[-1],), device="cpu") for key, value in sample.items()})
    model.set_output_rms(RunningMeanStd(shape=(228,), device="cpu"))
    model.set_robot_net_normal_force_rms(RunningMeanStd(shape=(7, 3), device="cpu"))
    output = model(copy.copy(sample), return_aux=True)
    assert output.state_prediction.shape == (2, 10, 228)
    assert output.auxiliary_predictions["robot_net_normal_forces_w"].shape == (2, 10, 7, 3)
    loss = output.state_prediction.square().mean() + sum(
        value.square().mean() for value in output.auxiliary_predictions.values()
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)


def test_keyboard_large_scale_gate_counts_unique_source_frames(tmp_path, monkeypatch):
    import h5py

    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    from lib.keyboard_data import unique_transition_count, validate_dataset_scale

    path = tmp_path / "overlapping.hdf5"
    with h5py.File(path, "w") as file:
        file.create_dataset("data/states", shape=(4, 64, 228))
        file.create_dataset("replay/frames/states", shape=(2, 128, 228))
        file.create_dataset("context/trajectories/source_env_id", data=[0, 0, 0, 1])
        file.create_dataset("context/trajectories/source_start_frame", data=[0, 32, 64, 0])
    assert unique_transition_count(path) == 192
    exp = {
        "minimum_unique_transitions": {"train": 193},
        "dataset_specs": [{"filename": path.name, "split": "train", "num_transitions": 0}],
    }
    with pytest.raises(ValueError, match="unique"):
        validate_dataset_scale(tmp_path, exp)
    exp["minimum_unique_transitions"]["train"] = 192
    assert validate_dataset_scale(tmp_path, exp) == {"train": 192}
    with pytest.raises(ValueError, match="unique"):
        validate_dataset_scale(tmp_path, exp, max_capacity=128)
    with pytest.raises(ValueError, match="unique"):
        validate_dataset_scale(tmp_path, exp, source_capacities={path.name: 128})
    exp["dataset_specs"][0]["num_transitions"] = 193
    with pytest.raises(ValueError, match="unique"):
        validate_dataset_scale(tmp_path, exp)


def test_keyboard_scale_gate_rejects_wrong_collection_mode(tmp_path, monkeypatch):
    import json

    import h5py

    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    from lib.keyboard_data import validate_dataset_scale

    path = tmp_path / "data.hdf5"
    with h5py.File(path, "w") as file:
        file.create_dataset("data/states", shape=(1, 64, 228))
        file.create_dataset("replay/frames/states", shape=(1, 64, 228))
        file.create_dataset("context/trajectories/source_env_id", data=[0])
        file.create_dataset("context/trajectories/source_start_frame", data=[0])
    exp = {
        "minimum_unique_transitions": {"train": 64},
        "collection_mode": "policy",
        "dataset_specs": [{"filename": path.name, "split": "train", "num_transitions": 64}],
    }
    for mode in ("script", "policy"):
        with h5py.File(path, "a") as file:
            file["data"].attrs["keyboard_contract"] = json.dumps({"collection_mode": mode})
        if mode == "script":
            with pytest.raises(ValueError, match="collection"):
                validate_dataset_scale(tmp_path, exp)
        else:
            assert validate_dataset_scale(tmp_path, exp) == {"train": 64}


def test_virtual_trajectory_view_preserves_rows_and_csr_offsets(tmp_path):
    import h5py
    import numpy as np
    from isaaclab_neural.data.hdf5 import create_virtual_trajectory_dataset

    sources = []
    for index in range(2):
        path = tmp_path / f"part{index}.hdf5"
        with h5py.File(path, "w") as file:
            for key in ("states", "next_states", "joint_f"):
                file.create_dataset("data/" + key, data=np.full((1, 2, 3), index, dtype="float32"))
            file["data"].attrs.update(mode="trajectory", env="test")
            file.create_dataset("context/trajectories/source_env_id", data=[index])
            file.create_dataset("contacts/native_pairs/frame_offsets", data=[0, 1, 1])
            file.create_dataset("contacts/native_pairs/world_ids", data=[index])
        sources.append(path)
    target = tmp_path / "view.hdf5"
    create_virtual_trajectory_dataset(target, sources)
    with h5py.File(target, "r") as file:
        states = file["data/states"]
        frame_offsets = file["contacts/native_pairs/frame_offsets"]
        world_ids = file["contacts/native_pairs/world_ids"]
        assert isinstance(states, h5py.Dataset)
        assert isinstance(frame_offsets, h5py.Dataset)
        assert isinstance(world_ids, h5py.Dataset)
        assert states.is_virtual
        assert states.shape == (2, 2, 3)
        assert states[:, 0, 0].tolist() == [0, 1]
        assert frame_offsets[()].tolist() == [0, 1, 1, 2, 2]
        assert world_ids[()].tolist() == [0, 1]
        assert file["data"].attrs["total_transitions"] == 4
        assert all(Path(link.file_name).name == link.file_name for link in states.virtual_sources())
    with pytest.raises(FileExistsError):
        create_virtual_trajectory_dataset(target, sources)
    import shutil

    relocated = tmp_path / "relocated"
    relocated.mkdir()
    for path in [target, *sources]:
        shutil.copy2(path, relocated / path.name)
    for path in sources:
        path.unlink()
    with h5py.File(relocated / target.name, "r") as file:
        states = file["data/states"]
        world_ids = file["contacts/native_pairs/world_ids"]
        assert isinstance(states, h5py.Dataset)
        assert isinstance(world_ids, h5py.Dataset)
        assert states[:, 0, 0].tolist() == [0, 1]
        assert world_ids[()].tolist() == [0, 1]


@pytest.mark.parametrize(
    "mode,cached,failure",
    [("auto", False, False), ("auto", True, False), ("require", False, False), ("auto", False, True)],
)
def test_keyboard_pipeline_stage_order_and_fail_closed(tmp_path, monkeypatch, mode, cached, failure):
    import json

    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    import run_experiment
    import torch
    from lib import keyboard_data

    preset = _load_yaml(PRESET_DIR / "so101_keyboard_resident_unique_smoke.yaml")
    calls = []
    monkeypatch.setattr(keyboard_data, "validate_runtime", lambda exp: {})
    monkeypatch.setattr(keyboard_data, "collection_plan", lambda exp: {"policy_sha256": "test"})
    monkeypatch.setattr(keyboard_data, "validate_bundle", lambda *args: {})
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(run_experiment, "dataset_cache_complete", lambda *args: True)
    local = tmp_path / "data" / preset["experiment"]["env_name"]
    if cached:
        local.mkdir(parents=True)
        (local / keyboard_data.MANIFEST).write_text("{}")

    def generate(*args):
        calls.append("collect/replay")
        if failure:
            raise RuntimeError("replay failed")

    monkeypatch.setattr(run_experiment, "generate_all_datasets", generate)
    monkeypatch.setattr(run_experiment, "prepare_training_only_config", lambda *args: calls.append("config"))
    monkeypatch.setattr(run_experiment, "stage_generated_datasets", lambda **kwargs: calls.append("publish"))
    monkeypatch.setattr(run_experiment, "run_training", lambda *args: calls.append("train"))
    monkeypatch.setattr(run_experiment, "evaluate_keyboard_checkpoint", lambda *args: calls.append("evaluate"))
    args = Namespace(
        output_root=str(tmp_path),
        dataset_dir=str(tmp_path / "data"),
        dataset_cache_mode=mode,
        dataset_input_path="",
        dataset_subdir="test",
        storage_backend="nvdataset",
        swift_data_container="",
        nvdataset_data_dataset="",
        nvdataset_data_description="test",
        no_tensorboard=True,
    )
    if failure or mode == "require":
        with pytest.raises(RuntimeError):
            run_experiment.run_keyboard_pipeline(args, preset)
        assert "publish" not in calls and "train" not in calls
    else:
        run_experiment.run_keyboard_pipeline(args, preset)
        assert calls == (
            ["config", "train", "evaluate"] if cached else ["collect/replay", "config", "publish", "train", "evaluate"]
        )
    status = json.loads((tmp_path / "pipeline_status.json").read_text())
    assert not status["task_fidelity_accepted"]
    events = status["stage_events"]
    assert events[0]["stage"] == "preflight" and events[0]["status"] == "running"
    assert [event["elapsed_seconds"] for event in events] == sorted(event["elapsed_seconds"] for event in events)
    assert all(event["unix_time"] > 0 for event in events)
    assert all(event["duration_seconds"] >= 0 for event in events if "duration_seconds" in event)


@pytest.mark.parametrize("failure", [False, True])
def test_keyboard_parallel_collection_isolates_devices_and_gates_manifest(tmp_path, monkeypatch, failure):
    import json
    import threading
    import time

    import h5py
    import isaaclab_neural.data.hdf5 as hdf5_data
    import isaaclab_neural.data.keyboard as keyboard_schema

    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    from lib import keyboard_data

    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"policy")
    exp = _load_yaml(PRESET_DIR / "so101_keyboard_resident_unique_smoke.yaml")["experiment"]
    exp["collection"].update(
        checkpoint=str(checkpoint), gpu_indices=[0, 1], max_chunks_per_group=8, persistent_workers=False
    )
    for spec in exp["dataset_specs"]:
        spec["num_transitions"] = 3
    monkeypatch.setattr(
        keyboard_data,
        "_collection_device_envs",
        lambda plan: [{"CUDA_VISIBLE_DEVICES": "GPU-a"}, {"CUDA_VISIBLE_DEVICES": "GPU-b"}],
    )
    lock, barrier = threading.Lock(), threading.Barrier(2, timeout=5)
    busy, calls, peaks = set(), [], []

    def collect(experiment, collection, spec, index, root, diagnostics, env):
        gpu = env["CUDA_VISIBLE_DEVICES"]
        with lock:
            assert gpu not in busy
            busy.add(gpu)
            peaks.append(len(busy))
            calls.append((spec["filename"], index, gpu))
        try:
            if index == 0:
                barrier.wait()
            if failure and spec["split"] == "valid":
                raise RuntimeError("replay failed")
            time.sleep(0.01)
            part = f"{Path(spec['filename']).stem}.part{index:05d}.hdf5"
            (root / part).write_bytes(b"verified")
            record = {
                "sha256": keyboard_data.file_sha256(root / part),
                "collect_sha256": "test",
                "validation_mode": "basic",
                "replay_performed": False,
            }
            return part, record, 1
        finally:
            with lock:
                busy.remove(gpu)

    def create_view(path, parts):
        with h5py.File(path, "w") as file:
            file.create_group("data")

    monkeypatch.setattr(keyboard_data, "_collect_part", collect)
    monkeypatch.setattr(hdf5_data, "create_virtual_trajectory_dataset", create_view)
    monkeypatch.setattr(keyboard_schema, "read_keyboard_metadata", lambda path: {})
    monkeypatch.setattr(keyboard_data, "validate_bundle", lambda *args: {})
    root = tmp_path / "data"
    if failure:
        with pytest.raises(RuntimeError, match="replay failed"):
            keyboard_data.collect_and_validate(exp, root, tmp_path / "output")
        assert not (root / keyboard_data.MANIFEST).exists()
    else:
        keyboard_data.collect_and_validate(exp, root, tmp_path / "output")
        manifest = json.loads((root / keyboard_data.MANIFEST).read_text())
        assert set(manifest["groups"]) == {s["filename"] for s in exp["dataset_specs"]}
        for name, parts in manifest["groups"].items():
            assert len(parts) >= 3
            assert parts == [f"{Path(name).stem}.part{i:05d}.hdf5" for i in range(len(parts))]
    assert max(peaks) == 2 and not busy
    assert len({(name, index) for name, index, gpu in calls}) == len(calls)


@pytest.mark.parametrize("changed", ["data.hdf5", "receipt.json", "keyboard_dataset_manifest.json"])
def test_bundle_validation_session_reuses_only_unchanged_verified_files(tmp_path, monkeypatch, changed):
    import os

    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    from lib import keyboard_data

    path = tmp_path / "data.hdf5"
    path.write_bytes(b"verified")
    (tmp_path / keyboard_data.MANIFEST).write_text("{}")
    (tmp_path / "receipt.json").write_text("{}")
    names = [path.name, "receipt.json", keyboard_data.MANIFEST]
    monkeypatch.setattr(keyboard_data, "bundle_files", lambda *args: names)
    expected = {name: keyboard_data.file_sha256(tmp_path / name) for name in names}
    calls = []

    def verify(root, experiment):
        calls.append(dict(experiment))
        if any(keyboard_data.file_sha256(root / name) != digest for name, digest in expected.items()):
            raise ValueError("Changed data hash")
        return {"files": dict(expected)}

    monkeypatch.setattr(keyboard_data, "_validate_bundle", verify)
    with keyboard_data.bundle_validation_session():
        first = keyboard_data.validate_bundle(tmp_path, {"train_cfg": "source.yaml", "minimum": 1})
        first["files"].clear()
        second = keyboard_data.validate_bundle(tmp_path, {"train_cfg": "resolved.yaml", "minimum": 1})
        assert second["files"] and len(calls) == 1
        keyboard_data.validate_bundle(tmp_path, {"minimum": 2})
        assert len(calls) == 2
        path = tmp_path / changed
        previous = path.stat()
        original = path.read_bytes()
        path.write_bytes(b"x" * len(original))
        os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        with pytest.raises(ValueError, match="Changed data hash"):
            keyboard_data.validate_bundle(tmp_path, {"minimum": 2})
    path.write_bytes(original)
    keyboard_data.validate_bundle(tmp_path, {"minimum": 2})
    assert len(calls) == 4


def test_bundle_validation_detects_mutation_during_full_check(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    from lib import keyboard_data

    path = tmp_path / keyboard_data.MANIFEST
    path.write_text("{}")
    monkeypatch.setattr(keyboard_data, "bundle_files", lambda *args: [keyboard_data.MANIFEST])

    def mutate(*args):
        path.write_text('{"changed": true}')
        return {}

    monkeypatch.setattr(keyboard_data, "_validate_bundle", mutate)
    with keyboard_data.bundle_validation_session(), pytest.raises(ValueError, match="changed during"):
        keyboard_data.validate_bundle(tmp_path, {})


@pytest.mark.parametrize("mode", ["full_replay", "basic"])
@pytest.mark.parametrize("tamper", [None, "data", "receipt", "acceptance", "plan", "path", "mode"])
def test_keyboard_partial_recovery_is_hash_bound_and_not_training_ready(tmp_path, monkeypatch, tamper, mode):
    import json

    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    from lib import keyboard_data

    exp = _load_yaml(PRESET_DIR / "so101_keyboard_resident_unique.yaml")["experiment"]
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"policy")
    exp["collection"]["checkpoint"] = str(checkpoint)
    if mode == "full_replay":
        exp["data_gen_num_envs"] = 8
        exp["collection"]["persistent_workers"] = False
    spec = exp["dataset_specs"][0]
    part = f"{Path(spec['filename']).stem}.part00000.hdf5"
    source = tmp_path / "source"
    source.mkdir()
    (source / part).write_bytes(b"immutable data")
    exp["collection"]["validation_mode"] = mode
    exp["require_keyboard_replay"] = mode == "full_replay"
    collection = keyboard_data.collection_plan(exp)
    suffix = ".collect.json" if mode == "basic" else ".replay.json"
    receipt_key = "collect_sha256" if mode == "basic" else "replay_sha256"
    (source / (part + suffix)).write_text('{"passed": true}')
    record = {
        "sha256": keyboard_data.file_sha256(source / part),
        receipt_key: keyboard_data.file_sha256(source / (part + suffix)),
        "collection": keyboard_data._part_signature(exp, collection, spec, 0),
    }
    (source / (part + ".accepted.json")).write_text(json.dumps(record))
    (source / part.replace("00000", "00001")).write_bytes(b"unverified in-flight chunk")
    validated = []
    monkeypatch.setattr(
        keyboard_data,
        "validate_recordings",
        lambda root, experiment, records, complete: validated.append((set(records), complete)),
    )
    export = tmp_path / "export"
    manifest = keyboard_data.export_collection_resume(source, export, exp)
    assert set(manifest["files"]) == {part}
    assert not manifest["training_ready"] and not (export / keyboard_data.MANIFEST).exists()
    assert not (export / part.replace("00000", "00001")).exists()
    assert validated and all(not complete for _, complete in validated)
    exp["collection"].update(gpu_indices=[0, 1], resume_from_input=True)
    if tamper in {"data", "receipt", "acceptance"}:
        suffix = {"data": "", "receipt": suffix, "acceptance": ".accepted.json"}[tamper]
        (export / (part + suffix)).write_bytes(b"tampered")
    elif tamper == "plan":
        exp["collection"]["chunk_control_steps"] += 4
    elif tamper == "path":
        manifest["files"]["../outside.hdf5"] = manifest["files"].pop(part)
        (export / keyboard_data.RESUME_MANIFEST).write_text(json.dumps(manifest))
    elif tamper == "mode":
        manifest["validation_mode"] = "basic" if mode == "full_replay" else "full_replay"
        (export / keyboard_data.RESUME_MANIFEST).write_text(json.dumps(manifest))
    restored = tmp_path / "restored"
    if tamper:
        with pytest.raises(ValueError, match="hash|provenance|Undeclared|mode"):
            keyboard_data.restore_collection_resume(export, restored, exp)
        assert not restored.exists()
    else:
        assert keyboard_data.restore_collection_resume(export, restored, exp) == 1
        assert (restored / part).read_bytes() == b"immutable data"
        assert not (restored / keyboard_data.MANIFEST).exists()
        with pytest.raises(FileExistsError):
            keyboard_data.restore_collection_resume(export, restored, exp)
        assert exp["minimum_unique_transitions"] == {"train": 10_000_000, "valid": 3_000_000}


@pytest.mark.parametrize("indices", [[], [0, 0], [-1], [True], ["0"]])
def test_keyboard_parallel_device_plan_rejects_invalid_indices(tmp_path, monkeypatch, indices):
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    from lib.keyboard_data import collection_plan

    exp = _load_yaml(PRESET_DIR / "so101_keyboard_resident_unique_smoke.yaml")["experiment"]
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"policy")
    exp["collection"].update(checkpoint=str(checkpoint), gpu_indices=indices)
    with pytest.raises(ValueError, match="gpu_indices"):
        collection_plan(exp)


def test_keyboard_parallel_devices_use_child_only_uuid_masks(monkeypatch):
    import os
    from types import SimpleNamespace

    import torch

    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    from lib.keyboard_data import _collection_device_envs

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "original-mask")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: SimpleNamespace(uuid=f"uuid-{index}"))
    envs = _collection_device_envs({"gpu_indices": [1, 0]})
    assert [env["CUDA_VISIBLE_DEVICES"] for env in envs] == ["GPU-uuid-1", "GPU-uuid-0"]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "original-mask"
    with pytest.raises(ValueError, match="visible CUDA"):
        _collection_device_envs({"gpu_indices": [2]})


@pytest.mark.parametrize("cache_mode", ["auto", "off", "require"])
def test_keyboard_missing_recovery_never_silently_recollects(tmp_path, monkeypatch, cache_mode):
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    import run_experiment
    from lib import keyboard_data

    monkeypatch.setattr(keyboard_data, "validate_runtime", lambda experiment: {})
    monkeypatch.setattr(keyboard_data, "collection_plan", lambda experiment: {"policy_sha256": "test"})
    monkeypatch.setattr(run_experiment, "load_dataset_cache_from_input", lambda *args: False)
    monkeypatch.setattr(run_experiment, "generate_all_datasets", lambda *args: pytest.fail("must not recollect"))
    preset = _load_yaml(PRESET_DIR / "so101_keyboard_resident_unique.yaml")
    preset["experiment"]["collection"]["resume_from_input"] = True
    args = Namespace(
        output_root=str(tmp_path),
        dataset_dir=str(tmp_path / "data"),
        dataset_cache_mode=cache_mode,
        dataset_input_path=str(tmp_path / "input"),
        dataset_subdir="test",
    )
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        run_experiment.run_keyboard_pipeline(args, preset)


def test_unique_resident_preset_has_global_capacity_and_isolated_cache(monkeypatch):
    import copy

    from isaaclab_neural.utils.python_utils import handle_cfg_overrides

    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    from lib.preset import load_preset
    from run_experiment import validate_training_capacity_config

    preset = load_preset("so101_keyboard_resident_unique")
    assert preset == _load_yaml(PRESET_DIR / "so101_keyboard_resident_unique.yaml")
    exp = preset["experiment"]
    cfg = _load_yaml(REPOSITORY_ROOT / exp["train_cfg"])
    handle_cfg_overrides(exp.get("train_cfg_overrides", ""), cfg)
    validate_training_capacity_config(exp, cfg)
    assert exp["dataset_capacity_unit"] == cfg["algorithm"]["dataset"]["capacity_unit"] == "unique_transition"
    assert cfg["algorithm"]["dataset"]["max_capacity"] == 10_000_000
    assert cfg["algorithm"]["dataset"]["valid_max_capacity"] == 3_000_000
    assert cfg["algorithm"]["dataset"]["window_sampling"] == exp["training_window_sampling"] == "sliding"
    assert cfg["env"]["neural_solver_cfg"]["contact_token_frame"] == exp["contact_token_frame"] == "owner"
    assert cfg["env"]["neural_solver_cfg"]["contact_filter"] == exp["contact_filter"] == "solver_active_movable"
    assert cfg["env"]["neural_solver_cfg"]["max_contact_tokens"] == exp["max_contact_tokens"] == 128
    assert exp["num_gpus"] == preset["resources"]["num_gpu"] == 8
    assert exp["data_gen_num_envs"] == 1024 and exp["collection"]["gpu_indices"] == list(range(8))
    assert exp["collection"]["persistent_workers"] and not exp["require_keyboard_replay"]
    assert exp["collection"]["validation_mode"] == "basic"
    assert exp["collection"]["deterministic_physics"] and exp["collection"]["bounded_deterministic_workspace"]
    bad = copy.deepcopy(cfg)
    bad["algorithm"]["dataset"]["max_capacity"] = 9_999_999
    with pytest.raises(ValueError, match="below"):
        validate_training_capacity_config(exp, bad)
    bad = copy.deepcopy(cfg)
    bad["algorithm"]["dataset"]["capacity_unit"] = "stored_frame"
    with pytest.raises(ValueError, match="units differ"):
        validate_training_capacity_config(exp, bad)
    bad = copy.deepcopy(cfg)
    bad["env"]["neural_solver_cfg"]["contact_filter"] = "none"
    with pytest.raises(ValueError, match="contact filters"):
        validate_training_capacity_config(exp, bad)
    bad = copy.deepcopy(cfg)
    bad["algorithm"]["sample_sequence_length"] = 65
    with pytest.raises(ValueError, match="sequence-length"):
        validate_training_capacity_config(exp, bad)


def test_unique_smoke_exercises_distributed_spawned_workers(monkeypatch):
    from isaaclab_neural.utils.python_utils import handle_cfg_overrides

    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    from lib.preset import load_preset
    from run_experiment import validate_training_capacity_config

    preset = load_preset("so101_keyboard_resident_unique_smoke")
    exp = preset["experiment"]
    cfg = _load_yaml(REPOSITORY_ROOT / exp["train_cfg"])
    handle_cfg_overrides(exp["train_cfg_overrides"], cfg)
    validate_training_capacity_config(exp, cfg)
    algo = cfg["algorithm"]
    assert exp["num_gpus"] == preset["resources"]["num_gpu"] == 2
    assert exp["collection"]["gpu_indices"] == [0, 1]
    assert algo["dataset"]["num_data_workers"] == 7 and algo["batch_size"] == 512
    assert algo["num_epochs"] == algo["num_iters_per_epoch"] == 2
    assert algo["dataset"]["max_capacity"] == exp["minimum_unique_transitions"]["train"] == 16384
    assert algo["dataset"]["valid_max_capacity"] == exp["minimum_unique_transitions"]["valid"] == 8192
    assert algo["dataset"]["window_sampling"] == exp["training_window_sampling"] == "sliding"
    assert cfg["env"]["neural_solver_cfg"]["contact_token_frame"] == "owner"
    assert cfg["env"]["neural_solver_cfg"]["contact_filter"] == exp["contact_filter"] == "solver_active_movable"
    assert cfg["env"]["neural_solver_cfg"]["max_contact_tokens"] == exp["max_contact_tokens"] == 128
    assert exp["require_keyboard_replay"] is False


def test_resident_keyboard_worker_reuses_process_and_isolates_device(tmp_path, monkeypatch):
    import json
    import subprocess
    import sys

    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    from lib import keyboard_data

    program = (
        "import json,os,sys; from pathlib import Path\n"
        "for line in sys.stdin:\n"
        " r=json.loads(line)\n"
        " Path(r['response']).write_text(json.dumps({**r,'ok':True,'pid':os.getpid(),'source_num_envs':1024}))\n"
    )
    original = subprocess.Popen
    starts = []

    def launch(command, **kwargs):
        starts.append((command, kwargs.get("env")))
        return original([sys.executable, "-u", "-c", program], **kwargs)

    monkeypatch.setattr(keyboard_data.subprocess, "Popen", launch)
    worker = keyboard_data.ResidentCollector(tmp_path / "worker", {"CUDA_VISIBLE_DEVICES": "GPU-test"}, timeout=10)
    command = [
        sys.executable,
        "-u",
        "-m",
        "isaaclab_neural.generate.keyboard_dataset",
        "collect",
        "--dataset",
        str(tmp_path / "first.hdf5"),
        "--report",
        str(tmp_path / "first.json"),
        "--seed",
        "42",
        "--num-envs",
        "1024",
    ]
    first = worker.collect(command, tmp_path / "first.log")
    command[command.index("--seed") + 1] = "43"
    command[command.index("--dataset") + 1] = str(tmp_path / "second.hdf5")
    command[command.index("--report") + 1] = str(tmp_path / "second.json")
    second = worker.collect(command, tmp_path / "second.log")
    worker.close()
    assert len(starts) == 1 and "worker" in starts[0][0]
    assert starts[0][1]["CUDA_VISIBLE_DEVICES"] == "GPU-test"
    assert first["pid"] == second["pid"]
    assert (first["seed"], second["seed"]) == (42, 43)
    assert (first["sequence"], second["sequence"]) == (0, 1)
    assert first["source_num_envs"] == second["source_num_envs"] == 1024
    assert json.loads((tmp_path / "second.log").read_text())["pid"] == second["pid"]


def test_keyboard_collection_only_stops_before_training(tmp_path, monkeypatch):
    import json

    import torch

    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    import run_experiment
    from lib import keyboard_data

    preset = _load_yaml(PRESET_DIR / "so101_keyboard_resident_unique_smoke.yaml")
    preset["experiment"]["collection_only"] = True
    calls = []
    monkeypatch.setattr(keyboard_data, "validate_runtime", lambda exp: {})
    monkeypatch.setattr(keyboard_data, "collection_plan", lambda exp: {"policy_sha256": "test"})
    monkeypatch.setattr(keyboard_data, "validate_bundle", lambda *args: {})
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(run_experiment, "dataset_cache_complete", lambda *args: True)
    monkeypatch.setattr(run_experiment, "generate_all_datasets", lambda *args: calls.append("collect"))
    monkeypatch.setattr(run_experiment, "prepare_training_only_config", lambda *args: None)
    monkeypatch.setattr(run_experiment, "stage_generated_datasets", lambda **kwargs: calls.append("publish"))
    monkeypatch.setattr(run_experiment, "run_training", lambda *args: pytest.fail("probe must not train"))
    args = Namespace(
        output_root=str(tmp_path),
        dataset_dir=str(tmp_path / "data"),
        dataset_cache_mode="off",
        dataset_input_path="",
        dataset_subdir="test",
        storage_backend="nvdataset",
        swift_data_container="",
        nvdataset_data_dataset="",
        nvdataset_data_description="test",
        no_tensorboard=True,
    )
    run_experiment.run_keyboard_pipeline(args, preset)
    assert calls == ["collect", "publish"]
    status = json.loads((tmp_path / "pipeline_status.json").read_text())
    assert status["data_validation_mode"] == "basic"
    assert status["stages"]["collection_and_basic_validation"] == "passed"
    assert "training" not in status["stages"] and "collection_and_replay" not in status["stages"]


def test_keyboard_basic_collection_allows_1024_without_weakening_replay_mode(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    from lib import keyboard_data

    exp = _load_yaml(PRESET_DIR / "so101_keyboard_resident_unique_smoke.yaml")["experiment"]
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"policy")
    exp["collection"]["checkpoint"] = str(checkpoint)
    exp["require_keyboard_replay"] = True
    exp["collection"].update(
        validation_mode="full_replay", persistent_workers=False, bounded_deterministic_workspace=False
    )
    exp["data_gen_num_envs"] = 1024
    with pytest.raises(ValueError, match="eight"):
        keyboard_data.collection_plan(exp)
    exp["collection"].update(validation_mode="basic", deterministic_physics=False)
    with pytest.raises(ValueError, match="require_keyboard_replay"):
        keyboard_data.collection_plan(exp)
    exp["require_keyboard_replay"] = False
    assert keyboard_data.collection_plan(exp)["validation_mode"] == "basic"
    exp["collection"]["bounded_deterministic_workspace"] = True
    with pytest.raises(ValueError, match="require deterministic physics"):
        keyboard_data.collection_plan(exp)
    exp["collection"]["deterministic_physics"] = True
    assert keyboard_data.collection_plan(exp)["bounded_deterministic_workspace"] is True


@pytest.mark.parametrize("bounded", [False, True])
def test_keyboard_basic_part_does_not_launch_or_claim_replay(tmp_path, monkeypatch, bounded):
    import json

    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    from lib import keyboard_data

    exp = _load_yaml(PRESET_DIR / "so101_keyboard_resident_unique_smoke.yaml")["experiment"]
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"policy")
    exp["require_keyboard_replay"] = False
    exp["collection"].update(
        checkpoint=str(checkpoint),
        validation_mode="basic",
        deterministic_physics=bounded,
        bounded_deterministic_workspace=bounded,
    )
    collection = keyboard_data.collection_plan(exp)
    calls = []

    def run(command, log_path, **kwargs):
        calls.append(command)
        Path(command[command.index("--dataset") + 1]).write_bytes(b"native states")
        Path(command[command.index("--report") + 1]).write_text('{"mode":"collect"}')

    monkeypatch.setattr(keyboard_data, "run_logged", run)
    monkeypatch.setattr(keyboard_data, "validate_recordings", lambda *args, **kwargs: {})
    monkeypatch.setattr(keyboard_data, "unique_transition_count", lambda path, **kwargs: 64)
    part, record, count = keyboard_data._collect_part(
        exp, collection, exp["dataset_specs"][0], 0, tmp_path, tmp_path, None
    )
    assert len(calls) == 1 and "collect" in calls[0] and "replay" not in calls[0]
    assert "--scene" not in calls[0] and "--keyboard-variant-offset" not in calls[0]
    assert ("--deterministic-physics" if bounded else "--no-deterministic-physics") in calls[0]
    assert ("--bounded-deterministic-workspace" in calls[0]) is bounded
    assert record["validation_mode"] == "basic" and record["replay_performed"] is False
    assert "replay_sha256" not in record and count == 64
    assert not (tmp_path / (part + ".replay.json")).exists()
    assert json.loads((tmp_path / (part + ".accepted.json")).read_text()) == record


def test_keyboard_collection_seed_schedule_cannot_leak(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    from lib.keyboard_data import collection_plan

    exp = _load_yaml(PRESET_DIR / "so101_keyboard_resident_unique_smoke.yaml")["experiment"]
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"test-policy")
    exp["collection"]["checkpoint"] = str(checkpoint)
    assert collection_plan(exp)["seed_stride"] == 1000
    exp["collection"]["seed_stride"] = 1
    with pytest.raises(ValueError, match="overlap"):
        collection_plan(exp)


def test_keyboard_cache_publication_includes_views_parts_receipts_and_manifest(tmp_path, monkeypatch):
    import json

    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    import run_experiment
    from lib import keyboard_data, nvdataset_io

    source = tmp_path / "source"
    source.mkdir()
    exp = {"require_keyboard_replay": True, "dataset_specs": [{"filename": "view.hdf5"}]}
    for name in ("view.hdf5", "part.hdf5", "part.hdf5.replay.json"):
        (source / name).write_text("test")
    (source / keyboard_data.MANIFEST).write_text(
        json.dumps({"schema": keyboard_data.CHUNKED_SCHEMA, "groups": {"view.hdf5": ["part.hdf5"]}})
    )
    calls = []
    monkeypatch.setattr(keyboard_data, "validate_bundle", lambda *args: calls.append("validated"))

    def upload(**kwargs):
        assert calls == ["validated"]
        uploaded = kwargs["source_dir"] / "bucket" / "env"
        assert {path.name for path in uploaded.iterdir()} == {
            "view.hdf5",
            "part.hdf5",
            "part.hdf5.replay.json",
            keyboard_data.MANIFEST,
        }
        calls.append("uploaded")

    monkeypatch.setattr(nvdataset_io, "upload_directory", upload)
    run_experiment.stage_generated_datasets(
        local_env_dir=source,
        dataset_subdir="bucket",
        env_name="env",
        storage_backend="nvdataset",
        swift_data_container="",
        nvdataset_data_dataset="mock-only",
        nvdataset_data_description="test",
        experiment=exp,
    )
    assert calls == ["validated", "uploaded"]
    assert (source / "part.hdf5").is_file()


@pytest.mark.parametrize("credential_env", ["NGC_API_KEY", "OMNI_PASS", "OMNI_PASSWORD"])
def test_keyboard_child_failure_reports_diagnostics_without_exposing_credentials(
    tmp_path, monkeypatch, capsys, credential_env
):
    import subprocess
    import sys

    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    from lib import keyboard_data

    secret = "synthetic-test-credential-not-real"
    monkeypatch.setenv(credential_env, secret)
    log = tmp_path / "collect.log"
    command = [
        sys.executable,
        "-c",
        "import os, sys; print(os.environ[sys.argv[1]]); print('collector failure', file=sys.stderr); sys.exit(7)",
        credential_env,
    ]
    with pytest.raises(subprocess.CalledProcessError) as failure:
        keyboard_data.run_logged(command, log)
    assert failure.value.returncode == 7
    assert "collector failure" in log.read_text()
    stderr = capsys.readouterr().err
    assert "collector failure" in stderr and "exit=7" in stderr
    assert secret not in stderr and "[REDACTED]" in stderr


def test_keyboard_child_failure_tail_is_bounded(tmp_path, monkeypatch, capsys):
    import subprocess
    import sys

    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    from lib import keyboard_data

    command = [
        sys.executable,
        "-c",
        "import sys; print('old prefix'); print('x'*100000); print('final error'); sys.exit(1)",
    ]
    log = tmp_path / "replay.log"
    with pytest.raises(subprocess.CalledProcessError):
        keyboard_data.run_logged(command, log)
    stderr = capsys.readouterr().err
    assert "final error" in stderr and "old prefix" not in stderr
    assert len(stderr) < 17000
    assert log.stat().st_size > 100000


def test_keyboard_success_keeps_log_and_existing_log_is_not_overwritten(tmp_path, monkeypatch, capsys):
    import sys

    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    from lib import keyboard_data

    log = tmp_path / "success.log"
    keyboard_data.run_logged([sys.executable, "-c", "print('success')"], log)
    assert log.read_text() == "success\n"
    assert capsys.readouterr().err == ""
    with pytest.raises(FileExistsError):
        keyboard_data.run_logged([sys.executable, "-c", "print('replaced')"], log)
    assert log.read_text() == "success\n"


def test_nucleus_template_attachment_preserves_paths_and_excludes_environment_secrets(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    from lib.nucleus_auth import attach_credential

    monkeypatch.setenv("OMNI_PASS", "synthetic-nucleus-secret")
    template = REPOSITORY_ROOT / "osmo_scripts/osmo_workflow.yaml"
    output = tmp_path / "attached.yaml"
    attach_credential(template, output, "test-nucleus")
    text = output.read_text()
    assert '"test-nucleus":\n        OMNI_USER: omni_user\n        OMNI_PASS: omni_pass' in text
    assert "synthetic-nucleus-secret" not in text
    import re

    defaults = yaml.safe_load(text.split("\ndefault-values:\n", 1)[1])
    parsed = yaml.safe_load(re.sub(r"{{[^}]+}}", "placeholder", text))
    assert set(parsed) == {"workflow", "default-values"}
    assert parsed["default-values"] == defaults
    for key in ("init_sh_localpath", "entrypoint_localpath", "nvdataset_io_localpath", "swift_io_localpath"):
        assert Path(defaults[key]).is_absolute() and Path(defaults[key]).is_file()
    with pytest.raises(FileExistsError):
        attach_credential(template, output, "test-nucleus")
    with pytest.raises(ValueError, match="Invalid"):
        attach_credential(template, tmp_path / "bad.yaml", "bad:\n  injected")
    with pytest.raises(ValueError, match="distinct"):
        attach_credential(template, tmp_path / "duplicate.yaml", "nvdataset_cred")


@pytest.mark.parametrize("service", ["nucleus", "wandb"])
@pytest.mark.parametrize("returncode", [0, 1])
def test_nucleus_registration_uses_private_files_and_does_not_print_payload(
    tmp_path, monkeypatch, capsys, returncode, service
):
    import subprocess

    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    from lib import nucleus_auth

    secret = "synthetic-nucleus-secret"
    monkeypatch.setenv("OMNI_USER", "$omni-api-token")
    monkeypatch.setenv("OMNI_PASS", secret)
    monkeypatch.setenv("WANDB_API_KEY", secret)
    paths = []

    def register(command, **kwargs):
        assert secret not in " ".join(command)
        assert "--payload-file" in command and "--payload" not in command
        for item in command[command.index("--payload-file") + 1 :]:
            key, name = item.split("=", 1)
            path = Path(name)
            paths.append(path)
            assert path.stat().st_mode & 0o777 == 0o600
            assert path.parent.stat().st_mode & 0o777 == 0o700
            assert path.read_text() == (secret if key in ("omni_pass", "api_key") else "$omni-api-token")
        return subprocess.CompletedProcess(command, returncode, stdout=secret, stderr=secret)

    monkeypatch.setattr(nucleus_auth.subprocess, "run", register)
    if returncode:
        with pytest.raises(RuntimeError) as failure:
            nucleus_auth.register_credential(service)
        assert secret not in str(failure.value)
    else:
        assert nucleus_auth.register_credential(service).startswith(f"so101-{service}-")
    assert all(not path.exists() for path in paths)
    assert secret not in str(capsys.readouterr())


def test_wandb_uses_only_managed_credential_references(tmp_path, monkeypatch):
    import re

    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    from lib.nucleus_auth import attach_credential

    template = REPOSITORY_ROOT / "osmo_scripts/osmo_workflow.yaml"
    nucleus = attach_credential(template, tmp_path / "nucleus.yaml", "test-nucleus")
    output = attach_credential(nucleus, tmp_path / "wandb.yaml", "test-wandb", service="wandb")
    parsed = yaml.safe_load(re.sub(r"{{[^}]+}}", "placeholder", output.read_text()))
    task = parsed["workflow"]["tasks"][0]
    assert task["credentials"]["test-wandb"] == {"WANDB_API_KEY": "api_key"}
    assert "test-nucleus" in task["credentials"]
    assert "WANDB_API_KEY" not in task["environment"]
    assert "wandb_api_key" not in parsed["default-values"]
    script = (REPOSITORY_ROOT / "osmo_scripts/start.sh").read_text()
    assert '"wandb_api_key=$WANDB_API_KEY"' not in script
    with pytest.raises(ValueError, match="distinct|already"):
        attach_credential(nucleus, tmp_path / "conflict.yaml", "test-nucleus", service="wandb")


def test_so101_presets_use_validated_image_and_single_gpu_cpu_budget():
    for path in PRESET_DIR.glob("so101_keyboard_*.yaml"):
        preset = _load_yaml(path)
        assert preset["workflow"]["image"] == "docker.io/rowany28070/isaac-lab-nerd:so101-newton160-cu130-sdk095"
        if preset["resources"]["num_gpu"] == 1:
            assert preset["resources"]["num_cpu"] == 12


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
            assert algorithm["batch_size"] == (512 if env_name == "Franka-Lift" else 1024)
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
                assert workflow["dataset_subdir"] == "franka-reach-menagerie-60hz-wide-reset-200"
        elif env_name == "Franka-Lift":
            assert experiment["sample_mode"] == "action"
            assert experiment["trajectory_length"] == 600
            assert experiment["train_num_envs"] == experiment["data_gen_num_envs"] == 1024
            assert train_config["algorithm"]["eval"]["preserve_source_origins"]
            assert not train_config["algorithm"]["eval"].get("rebase_env_origins", False)
            assert experiment["contact_representation"] == "contact_tokens"
            assert Path(experiment["policy_checkpoint"]).name.endswith("_rsl_rl.pt")


def test_franka_menagerie_dataset_paths_match_generation_and_training() -> None:
    for path in PRESET_DIR.glob("franka_reach*.yaml"):
        preset = _load_yaml(path)
        experiment = preset["experiment"]
        assert "menagerie-60hz" in preset["workflow"]["dataset_subdir"]
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
        preset="franka_reach_newton_native",
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
    assert generated[-1] == Path("data/datasets/menagerie-60hz")
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


def test_removed_franka_smoke_presets_are_rejected(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    from run_experiment import load_preset

    for name in (
        "franka_reach_newton_native_smoke",
        "franka_reach_policy_train_smoke",
        "franka_reach_mixed_smoke",
        "franka_reach_replay_smoke",
    ):
        with pytest.raises(FileNotFoundError, match=f"Preset {name!r} not found"):
            load_preset(name)


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
    args: dict[str, list[str]] = {}
    for name, spec in specs.items():
        dataset_args = build_dataset_args(experiment, spec, tmp_path)
        assert dataset_args is not None
        args[name] = dataset_args

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


def test_allegro_presets_split_default_teacher_from_packaged_nerd_pipeline() -> None:
    teacher_preset = _load_yaml(PRESET_DIR / "allegro_reorient_teacher.yaml")
    preset = _load_yaml(PRESET_DIR / "allegro_reorient_mixed.yaml")
    teacher = teacher_preset["experiment"]["teacher_policy"]
    assert teacher_preset["experiment"]["teacher_only"] is True
    assert teacher_preset["resources"]["num_gpu"] == teacher["num_gpus"] == 1
    # The core RSL-RL configuration and task world count are used unchanged.
    assert not {"num_envs", "max_iterations"} & set(teacher)
    assert "wandb_entity" not in teacher
    assert (teacher["wandb_project"], teacher["wandb_run_id_prefix"]) == ("rl", "allegro")

    experiment = preset["experiment"]
    assert "teacher_policy" not in experiment
    assert experiment["require_packaged_policy"] is True
    assert Path(experiment["policy_checkpoint"]).parent == Path("pretrained/control_policy/allegro_reorient")
    train_config = _load_yaml(REPOSITORY_ROOT / experiment["train_cfg"])
    dataset_config = train_config["algorithm"]["dataset"]
    specs = experiment["dataset_specs"]
    train_paths = [source["path"] for source in dataset_config["train_datasets"].values()]
    assert {Path(path).name for path in train_paths} == {item["filename"] for item in specs if item["split"] == "train"}
    assert {Path(path).name for path in dataset_config["valid_datasets"].values()} == {
        item["filename"] for item in specs if item["split"] == "valid"
    }
    for path in [*train_paths, *dataset_config["valid_datasets"].values()]:
        assert Path(path).parent == Path(experiment["dataset_dir"]) / experiment["env_name"]
    train = [item for item in specs if item["split"] == "train"]
    assert sum(item["num_transitions"] for item in train) == experiment["train_transitions"]
    assert {item["seed"] for item in train}.isdisjoint(item["seed"] for item in specs if item["split"] == "valid")
    solver_cfg = train_config["env"]["neural_solver_cfg"]
    for key in ("num_contacts_per_env", "max_contact_tokens", "contact_representation", "contact_packing_policy"):
        assert solver_cfg[key] == experiment[key], key
    eval_cfg = train_config["algorithm"]["eval"]
    assert eval_cfg["mode"] == "allegro_policy"
    assert eval_cfg["policy_checkpoint"] == experiment["policy_checkpoint"]
    keyboard = _load_yaml(
        REPOSITORY_ROOT / _load_yaml(PRESET_DIR / "so101_keyboard_resident_unique.yaml")["experiment"]["train_cfg"]
    )
    keyboard_solver = keyboard["env"]["neural_solver_cfg"]
    for key in ("use_frame_cache", "inference_precision", "inference_compile", "num_states_history"):
        assert solver_cfg[key] == keyboard_solver[key], key
    for key in ("position_encoding", "attention_spans"):
        assert train_config["network"]["transformer"][key] == keyboard["network"]["transformer"][key], key
    for item in (teacher_preset, preset):
        assert item["workflow"]["image"] == "docker.io/rowany28070/isaac-lab-nerd:so101-newton160-cu130-sdk095"


def _fake_policy_subprocesses(monkeypatch, tmp_path, mean_goals: float) -> list[tuple[list[str], dict]]:
    import json
    import subprocess

    calls: list[tuple[list[str], dict]] = []

    def fake_run(command, check, env=None):
        calls.append(([str(item) for item in command], env or {}))
        if "--report" in command:
            report = Path(command[command.index("--report") + 1])
            report.write_text(json.dumps({"mean_goals_per_episode": mean_goals, "drop_rate": 0.1}))
            return
        run_name = command[command.index("--run_name") + 1]
        run_dir = tmp_path / "logs" / "rsl_rl" / "teacher" / f"2026-01-01_00-00-00_{run_name}"
        (run_dir / "params").mkdir(parents=True)
        for iteration in (50, 4999, 9):
            (run_dir / f"model_{iteration}.pt").write_bytes(b"")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("WANDB_ENTITY", "")
    monkeypatch.delenv("WANDB_USERNAME", raising=False)
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    return calls


def _teacher_preset() -> dict:
    return {
        "teacher_only": True,
        "teacher_policy": {
            "task": "Isaac-Reorient-Cube-Allegro-Policy-v0",
            "experiment_name": "teacher",
            "wandb_project": "rl",
            "wandb_run_id_prefix": "allegro",
            "evaluation": {"module": "eval_module"},
        },
    }


def test_teacher_preset_trains_default_config_uploads_to_wandb_and_stops(tmp_path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    import run_experiment

    calls = _fake_policy_subprocesses(monkeypatch, tmp_path, mean_goals=3.0)
    monkeypatch.setattr(run_experiment, "load_preset", lambda *args: {"experiment": _teacher_preset()})
    monkeypatch.setattr(run_experiment, "generate_all_datasets", lambda *args: pytest.fail("must not generate"))
    monkeypatch.setattr(run_experiment, "run_training", lambda *args: pytest.fail("must not train NeRD"))
    args = Namespace(preset="teacher", preset_file=None, output_root=str(tmp_path / "output"), enable_wandb=True)
    args.wandb_entity = ""

    run_experiment.run(args)

    command, env = calls[0]
    assert command[:3] == [sys.executable, "-m", "isaaclab_neural.train.train_control_policy"]
    assert "--num_envs" not in command and "--max_iterations" not in command and "--distributed" not in command
    assert command[command.index("--logger") + 1] == "wandb"
    assert command[command.index("--log_project_name") + 1] == "rl"
    assert "WANDB_USERNAME" not in env and "WANDB_ENTITY" not in env
    assert env["WANDB_RUN_ID"].startswith("allegro-")
    checkpoint = Path(calls[1][0][calls[1][0].index("--checkpoint") + 1])
    assert checkpoint.name == "model_4999.pt"
    assert checkpoint.is_relative_to(tmp_path / "output" / "teacher_policy")
    assert (tmp_path / "output" / "teacher_policy" / "baseline.json").is_file()

    args.enable_wandb = False
    with pytest.raises(ValueError, match="--enable-wandb"):
        run_experiment.run(args)


def test_packaged_teacher_gate_runs_before_collection(tmp_path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    import run_experiment

    calls = _fake_policy_subprocesses(monkeypatch, tmp_path, mean_goals=0.5)
    staged = tmp_path / "model_4999.pt"
    staged.write_bytes(b"")
    experiment = dict(_load_yaml(PRESET_DIR / "allegro_reorient_mixed.yaml")["experiment"])
    experiment["policy_checkpoint"] = str(staged)
    monkeypatch.setattr(run_experiment, "load_preset", lambda *args: {"experiment": experiment})
    monkeypatch.setattr(run_experiment, "generate_all_datasets", lambda *args: pytest.fail("must not generate"))
    args = Namespace(
        preset="allegro_reorient_mixed",
        preset_file=None,
        output_root=str(tmp_path / "output"),
        num_gpus=None,
        dataset_cache_mode="off",
    )
    (tmp_path / "output").mkdir()

    with pytest.raises(RuntimeError, match="below the preset minimum"):
        run_experiment.run(args)

    assert len(calls) == 1 and calls[0][0][calls[0][0].index("--checkpoint") + 1] == str(staged)


def test_allegro_submit_requires_packaged_teacher(monkeypatch, capsys) -> None:
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts" / "lib"))
    import preset as preset_lib

    experiment = dict(_load_yaml(PRESET_DIR / "allegro_reorient_mixed.yaml")["experiment"])
    experiment["policy_checkpoint"] = "./pretrained/control_policy/allegro_reorient/missing.pt"
    monkeypatch.setattr(
        preset_lib, "load_preset", lambda *args: {"workflow": {}, "resources": {}, "experiment": experiment}
    )
    args = Namespace(preset="allegro_reorient_mixed", preset_file=None, run_id="1", workflow_name=None)
    for name in ("dataset_subdir", "num_gpu", "num_cpu", "memory", "storage", "platform", "image"):
        setattr(args, name, None)

    with pytest.raises(ValueError, match="allegro_reorient_mixed teacher"):
        preset_lib.resolve_submit(args)


def test_parallel_collection_isolates_gpus_and_stops_on_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    import run_experiment

    monkeypatch.setattr(run_experiment.time, "sleep", lambda _seconds: None)
    script = "import os, sys; print(os.environ['CUDA_VISIBLE_DEVICES']); sys.exit(int(sys.argv[1]))"
    jobs = [(f"job{index}", ["-c", script, "0"]) for index in range(3)]
    run_experiment.run_parallel_collection(jobs, 2, tmp_path / "logs")
    devices = {(tmp_path / "logs" / f"job{index}.log").read_text().strip() for index in range(3)}
    assert devices == {"0", "1"}

    sleeper = ("slow", ["-c", "import time; time.sleep(60)"])
    with pytest.raises(RuntimeError, match="bad failed with exit code 3"):
        run_experiment.run_parallel_collection([sleeper, ("bad", ["-c", script, "3"])], 2, tmp_path / "fail")


def test_generic_training_overrides_and_throughput_stage(tmp_path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    import run_experiment

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append([str(item) for item in command])
        if "--report" in command and "fail" not in command:
            Path(command[command.index("--report") + 1]).write_text('{"status": "ok"}')
        return SimpleNamespace(returncode=1 if "fail" in command else 0)

    monkeypatch.setattr(run_experiment.subprocess, "run", fake_run)
    experiment = dict(_load_yaml(PRESET_DIR / "allegro_reorient_mixed_smoke.yaml")["experiment"])
    run_experiment.run_training(experiment, tmp_path, Namespace(enable_wandb=False))
    command = calls[0]
    assert command[command.index("--cfg-overrides") + 1] == experiment["train_cfg_overrides"]
    keyboard = {**experiment, "dataset_profile": "so101_keyboard"}
    run_experiment.run_training(keyboard, tmp_path / "keyboard", Namespace(enable_wandb=False))
    assert "--cfg-overrides" not in calls[1]

    calls.clear()
    experiment["throughput_benchmark"]["variants"].append({"name": "broken", "args": ["fail"]})
    summary = run_experiment.run_throughput_benchmarks(experiment, tmp_path)
    names = set(summary["runs"])
    assert {"model_fp16", "mjwarp_newton_contacts_256", "nerd_optimized_256", "broken_256"} <= names
    assert summary["runs"]["broken_256"] == {"exit": 1}
    nerd = next(call for call in calls if call[call.index("--report") + 1].endswith("nerd_fp32_eager_256.json"))
    assert nerd[nerd.index("--nerd-checkpoint") + 1].endswith("Allegro-Reorient/nn/best_eval_model.pt")
    assert "--no-inference-compile" in nerd and nerd[nerd.index("--steps") + 1] == "50"
    mujoco = next(call for call in calls if "--mujoco-contacts" in call)
    assert "--nerd-checkpoint" not in mujoco


def test_allegro_smoke_presets_mirror_formal_pipeline() -> None:
    formal = _load_yaml(PRESET_DIR / "allegro_reorient_mixed.yaml")["experiment"]
    smoke_preset = _load_yaml(PRESET_DIR / "allegro_reorient_mixed_smoke.yaml")
    smoke = smoke_preset["experiment"]
    teacher = _load_yaml(PRESET_DIR / "allegro_reorient_teacher_smoke.yaml")["experiment"]["teacher_policy"]
    final_iteration = teacher["max_iterations"] - 1
    assert Path(smoke["policy_checkpoint"]).name == f"model_{final_iteration}.pt"
    assert smoke["policy_checkpoint"] in smoke["train_cfg_overrides"].split()
    assert [s["filename"] for s in smoke["dataset_specs"]] == [s["filename"] for s in formal["dataset_specs"]]
    assert smoke["train_cfg"] == formal["train_cfg"]
    assert smoke_preset["workflow"]["dataset_subdir"] != "allegro-reorient-newton-contacts-v1"
    assert smoke["parallel_collection_gpus"] == smoke["num_gpus"] == smoke_preset["resources"]["num_gpu"]
    assert formal["parallel_collection_gpus"] == len(formal["dataset_specs"]) == 8


def test_allegro_control_step_preset_follows_so101_control_step_recipe() -> None:
    preset = _load_yaml(PRESET_DIR / "allegro_reorient_control_step.yaml")
    experiment = preset["experiment"]
    config = _load_yaml(REPOSITORY_ROOT / experiment["train_cfg"])
    so101 = _load_yaml(
        REPOSITORY_ROOT
        / "source/isaaclab_neural/isaaclab_neural/train/cfg/SO101/transformer_keyboard_control_step.yaml"
    )
    solver, so101_solver = config["env"]["neural_solver_cfg"], so101["env"]["neural_solver_cfg"]
    for key in (
        "step_mode",
        "joint_target_input",
        "reset_history",
        "use_frame_cache",
        "inference_precision",
        "inference_compile",
        "contact_filter",
        "contact_token_frame",
        "num_states_history",
    ):
        assert solver[key] == so101_solver[key], key
    dataset, so101_dataset = config["algorithm"]["dataset"], so101["algorithm"]["dataset"]
    for key in ("capacity_unit", "capacity_scope", "window_sampling", "load_mode"):
        assert dataset[key] == so101_dataset[key], key
    assert config["algorithm"]["precision"] == so101["algorithm"]["precision"]
    assert config["inputs"]["low_dim"] == so101["inputs"]["low_dim"]
    assert experiment["record_control_steps"] is True and experiment["contact_filter"] == solver["contact_filter"]
    specs = experiment["dataset_specs"]
    root = Path(experiment["dataset_dir"]) / experiment["env_name"]
    assert {Path(source["path"]) for source in dataset["train_datasets"].values()} == {
        root / spec["filename"] for spec in specs if spec["split"] == "train"
    }
    assert {Path(path) for path in dataset["valid_datasets"].values()} == {
        root / spec["filename"] for spec in specs if spec["split"] == "valid"
    }
    train = sum(spec["num_transitions"] for spec in specs if spec["split"] == "train")
    assert train == experiment["train_transitions"] == dataset["max_capacity"]
    assert sum(source["max_capacity"] for source in dataset["train_datasets"].values()) == dataset["max_capacity"]
    assert experiment["parallel_collection_gpus"] == len(specs) == preset["resources"]["num_gpu"]


def test_allegro_collector_args_map_specs_to_collection_modes(tmp_path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    import run_experiment

    experiment = dict(_load_yaml(PRESET_DIR / "allegro_reorient_control_step.yaml")["experiment"])
    teacher = tmp_path / "model.pt"
    teacher.write_bytes(b"")
    experiment["policy_checkpoint"] = str(teacher)
    specs = run_experiment.parse_dataset_specs(experiment)
    commands = {spec.filename: run_experiment.build_dataset_args(experiment, spec, tmp_path) for spec in specs}
    for spec in specs:
        command = commands[spec.filename]
        assert command[:2] == ["-m", "isaaclab_neural.generate.allegro_dataset"]
        assert "--record-control-steps" in command
        assert command[command.index("--dataset") + 1] == str(tmp_path / "Allegro-Reorient" / spec.filename)
        assert command[command.index("--contact-filter") + 1] == "solver_active_movable"
        assert command[command.index("--horizon") + 1] == str(experiment["trajectory_length"])
        mode = command[command.index("--collection-mode") + 1]
        assert mode == ("policy" if spec.use_policy else "zero" if spec.zero_actions else "random")
        assert ("--checkpoint" in command) == spec.use_policy
        noisy = "--policy-action-noise-std" in command
        assert noisy == (spec.policy_action_noise_std > 0.0)


CONTROL_STEP_TASK_PRESETS = ("anymal_flat_control_step", "franka_lift_control_step")


@pytest.mark.parametrize("name", CONTROL_STEP_TASK_PRESETS)
def test_control_step_task_presets_follow_allegro_recipe(name) -> None:
    preset = _load_yaml(PRESET_DIR / f"{name}.yaml")
    smoke = _load_yaml(PRESET_DIR / f"{name}_smoke.yaml")
    allegro = _load_yaml(PRESET_DIR / "allegro_reorient_control_step.yaml")["experiment"]
    experiment = preset["experiment"]
    config = _load_yaml(REPOSITORY_ROOT / experiment["train_cfg"])
    allegro_cfg = _load_yaml(REPOSITORY_ROOT / allegro["train_cfg"])
    solver, reference_solver = config["env"]["neural_solver_cfg"], allegro_cfg["env"]["neural_solver_cfg"]
    for key in (
        "step_mode",
        "joint_target_input",
        "reset_history",
        "use_frame_cache",
        "inference_precision",
        "inference_compile",
        "contact_filter",
        "contact_token_frame",
        "num_states_history",
    ):
        assert solver[key] == reference_solver[key], key
    dataset, reference_dataset = config["algorithm"]["dataset"], allegro_cfg["algorithm"]["dataset"]
    for key in ("capacity_unit", "capacity_scope", "window_sampling", "load_mode", "resident_float16_keys"):
        assert dataset[key] == reference_dataset[key], key
    assert config["algorithm"]["precision"] == allegro_cfg["algorithm"]["precision"]
    assert config["algorithm"]["optimizer"]["fused"] is True
    assert config["inputs"]["low_dim"] == allegro_cfg["inputs"]["low_dim"]
    assert config["network"]["transformer"] == allegro_cfg["network"]["transformer"]
    assert config["env"]["env_name"] == experiment["env_name"]
    assert experiment["dataset_collector"] == "env_driven" and experiment["record_control_steps"] is True
    assert experiment["contact_filter"] == solver["contact_filter"]
    assert experiment["max_contact_tokens"] == solver["max_contact_tokens"]
    specs = experiment["dataset_specs"]
    root = Path(experiment["dataset_dir"]) / experiment["env_name"]
    assert {Path(source["path"]) for source in dataset["train_datasets"].values()} == {
        root / spec["filename"] for spec in specs if spec["split"] == "train"
    }
    assert {Path(path) for path in dataset["valid_datasets"].values()} == {
        root / spec["filename"] for spec in specs if spec["split"] == "valid"
    }
    train = sum(spec["num_transitions"] for spec in specs if spec["split"] == "train")
    assert train == experiment["train_transitions"] == dataset["max_capacity"]
    assert sum(source["max_capacity"] for source in dataset["train_datasets"].values()) == dataset["max_capacity"]
    assert experiment["parallel_collection_gpus"] == preset["resources"]["num_gpu"] <= len(specs)
    # Closed-loop selection is anchored to the native metric reported by the teacher gate.
    eval_cfg = config["algorithm"]["eval"]
    evaluation = experiment["policy_evaluation"]
    assert eval_cfg["mode"] == "policy_rollout" and eval_cfg["selection"]["reference"] is None
    assert eval_cfg["policy_checkpoint"] == experiment["policy_checkpoint"]
    assert evaluation["reference"] == {
        "metric": eval_cfg["selection"]["metric"],
        "cfg_key": "algorithm.eval.selection.reference",
    }
    assert evaluation["args"] == ["--task", experiment["data_gen_task"], "--metric", eval_cfg["selection"]["metric"]]
    assert [metric["name"] for metric in eval_cfg["metrics"]] == [eval_cfg["selection"]["metric"]]
    bench = experiment["throughput_benchmark"]
    assert bench["env_args"] == ["--task", experiment["data_gen_task"], "--nerd-task", experiment["train_task"]]
    # The smoke preset runs the same pipeline at small sizes with its own dataset cache.
    smoke_experiment = smoke["experiment"]
    assert smoke["workflow"]["dataset_subdir"] != preset["workflow"]["dataset_subdir"]
    assert [spec["filename"] for spec in smoke_experiment["dataset_specs"]] == [spec["filename"] for spec in specs]
    for key in ("train_cfg", "train_task", "data_gen_task", "dataset_collector", "policy_checkpoint"):
        assert smoke_experiment[key] == experiment[key], key


@pytest.mark.parametrize("name", CONTROL_STEP_TASK_PRESETS)
def test_control_step_task_collector_args_and_reference(name, tmp_path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT / "osmo_scripts"))
    import run_experiment

    experiment = dict(_load_yaml(PRESET_DIR / f"{name}.yaml")["experiment"])
    teacher = tmp_path / "model.pt"
    teacher.write_bytes(b"")
    experiment["policy_checkpoint"] = str(teacher)
    for spec in run_experiment.parse_dataset_specs(experiment):
        command = run_experiment.build_dataset_args(experiment, spec, tmp_path)
        assert command[:2] == ["-m", "isaaclab_neural.generate.allegro_dataset"]
        assert command[command.index("--task") + 1] == experiment["data_gen_task"]
        assert command[command.index("--env-name") + 1] == experiment["env_name"]
        assert command[command.index("--dataset") + 1] == str(tmp_path / experiment["env_name"] / spec.filename)
        mode = command[command.index("--collection-mode") + 1]
        assert mode == ("policy" if spec.use_policy else "zero" if spec.zero_actions else "random")
        assert ("--checkpoint" in command) == spec.use_policy
    evaluation = experiment["policy_evaluation"]
    run_experiment.apply_policy_reference(experiment, evaluation, {evaluation["reference"]["metric"]: 0.25})
    assert experiment["train_cfg_overrides"].endswith("algorithm.eval.selection.reference 0.25")
    with pytest.raises(RuntimeError, match="did not report"):
        run_experiment.apply_policy_reference(experiment, evaluation, {})
