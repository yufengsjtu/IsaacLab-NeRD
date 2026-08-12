# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run an OSMO experiment from a declarative preset."""

from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import TextIO, cast

import h5py
from lib.simple_yaml import load_yaml


@dataclass
class DatasetSpec:
    filename: str
    seed: int
    split: str
    task_kind: str
    zero_actions: bool
    use_policy: bool


class Tee(io.TextIOBase):
    """Write output to multiple streams."""

    def __init__(self, *streams: TextIO):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def load_preset(name: str, preset_file: str | None = None) -> dict:
    path = Path(preset_file) if preset_file else Path(__file__).resolve().parent / "presets" / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Preset {name!r} not found at {path}")

    preset = load_yaml(path)
    if not isinstance(preset, dict):
        raise ValueError(f"Preset file {path} did not contain a mapping.")
    return preset


def parse_dataset_specs(experiment: dict) -> list[DatasetSpec]:
    specs = []
    for item in experiment.get("dataset_specs", []):
        specs.append(
            DatasetSpec(
                filename=str(item["filename"]),
                seed=int(item["seed"]),
                split=str(item.get("split", "valid")),
                task_kind=str(item.get("task_kind", "default")),
                zero_actions=bool(item.get("zero_actions", False)),
                use_policy=bool(item.get("use_policy", False)),
            )
        )
    if not specs:
        raise ValueError("Preset must define at least one dataset spec.")
    return specs


def required_dataset_files(experiment: dict, specs: list[DatasetSpec]) -> list[str]:
    policy_checkpoint = experiment.get("policy_checkpoint", "")
    required_files = []
    for spec in specs:
        if spec.use_policy and policy_checkpoint and not Path(policy_checkpoint).is_file():
            continue
        required_files.append(spec.filename)
    return required_files


def dataset_cache_complete(
    candidate_dir: Path,
    required_files: list[str],
    experiment: dict,
) -> bool:
    specs_by_filename = {spec.filename: spec for spec in parse_dataset_specs(experiment)}
    for filename in required_files:
        dataset_path = candidate_dir / filename
        if not dataset_path.is_file():
            return False
        if not h5py.is_hdf5(dataset_path):
            print(f"Rejected malformed cached dataset: {dataset_path}")
            return False
        require_tokens = experiment.get("contact_representation", "flat") == "contact_tokens"
        require_terrain_context = bool(experiment.get("require_terrain_context", False))
        with h5py.File(dataset_path, "r") as handle:
            if "data" not in handle:
                print(f"Rejected cached dataset without a data group: {dataset_path}")
                return False
            data_group = cast(h5py.Group, handle["data"])
            spec = specs_by_filename[filename]
            states = cast(h5py.Dataset, data_group["states"]) if "states" in data_group else None
            expected_transitions = int(
                experiment["train_transitions"] if spec.split == "train" else experiment["valid_transitions"]
            )
            expected_trajectory_length = int(experiment["trajectory_length"])
            if (
                data_group.attrs.get("mode") != "trajectory"
                or data_group.attrs.get("env") != experiment["env_name"]
                or states is None
                or states.ndim != 3
                or states.shape[1] != expected_trajectory_length
                or int(data_group.attrs.get("total_transitions", -1)) < expected_transitions
            ):
                print(f"Rejected cached dataset with incompatible trajectory metadata: {dataset_path}")
                return False
            step_shape = states.shape[:2]
            inconsistent_fields = [
                key
                for key, value in data_group.items()
                if isinstance(value, h5py.Dataset) and value.ndim >= 2 and value.shape[:2] != step_shape
            ]
            if inconsistent_fields:
                print(
                    f"Rejected cached dataset with inconsistent trajectory fields "
                    f"{sorted(inconsistent_fields)}: {dataset_path}"
                )
                return False
            if require_tokens:
                expected_capacity = int(experiment.get("max_contact_tokens", 64))
                representation = str(data_group.attrs.get("contact_representation", ""))
                capacity = int(data_group.attrs.get("max_contact_tokens", -1))
                identity_schema = str(data_group.attrs.get("contact_identity_schema", ""))
                token_frame = str(data_group.attrs.get("contact_token_frame", ""))
                required_token_fields = {
                    "contact_tokens",
                    "contact_token_body_ids",
                    "contact_token_world_ids",
                    "root_body_q",
                    "root_body_qd",
                    "gravity_dir",
                    "states",
                    "next_states",
                    "joint_f",
                }
                missing_token_fields = sorted(required_token_fields - set(data_group.keys()))
                if (
                    representation != "contact_tokens"
                    or capacity != expected_capacity
                    or identity_schema != "world_owner_v1"
                    or token_frame != "world_v1"
                    or missing_token_fields
                ):
                    print(
                        f"Rejected cached dataset {dataset_path}: expected contact_tokens capacity "
                        f"{expected_capacity} with root kinematics, found representation={representation!r}, "
                        f"capacity={capacity}, identity_schema={identity_schema!r}, token_frame={token_frame!r}, "
                        f"missing_fields={missing_token_fields}."
                    )
                    return False
            if require_terrain_context:
                data_group = cast(h5py.Group, handle["data"])
                missing_root_kinematics = sorted({"root_body_q", "root_body_qd"} - set(data_group.keys()))
                if missing_root_kinematics:
                    print(
                        f"Rejected cached dataset without replay root kinematics "
                        f"{missing_root_kinematics}: {dataset_path}"
                    )
                    return False
                trajectory_keys = {"source_env_id", "terrain_level", "terrain_type", "env_origin"}
                if require_tokens and experiment.get("diagnostic_only", False):
                    trajectory_keys.update({"state_world_id", "root_world_id", "contact_world_id"})
                context_group = cast(h5py.Group, handle["context"]) if "context" in handle else None
                trajectory_group = (
                    cast(h5py.Group, context_group["trajectories"])
                    if context_group is not None and "trajectories" in context_group
                    else None
                )
                has_context = (
                    context_group is not None
                    and "terrain" in context_group
                    and trajectory_group is not None
                    and trajectory_keys.issubset(trajectory_group.keys())
                )
                terrain_attrs = (
                    set(cast(h5py.Group, context_group["terrain"]).attrs.keys())
                    if context_group is not None and "terrain" in context_group
                    else set()
                )
                required_terrain_attrs = {
                    "schema_version",
                    "seed",
                    "config_sha256",
                    "mesh_sha256",
                    "origins_sha256",
                    "num_rows",
                    "num_cols",
                }
                has_context = has_context and required_terrain_attrs.issubset(terrain_attrs)
                if has_context and context_group is not None:
                    terrain_group = cast(h5py.Group, context_group["terrain"])
                    terrain_seed = terrain_group.attrs["seed"]
                    has_context = isinstance(terrain_seed, Integral) and int(terrain_seed) == spec.seed
                if not has_context:
                    print(f"Rejected cached dataset without required terrain context: {dataset_path}")
                    return False
    return True


def load_dataset_cache_from_input(
    input_root: Path,
    dataset_subdir: str,
    env_name: str,
    local_env_dir: Path,
    required_files: list[str],
    experiment: dict,
) -> bool:
    if not input_root.is_dir():
        print(f"Dataset input path does not exist: {input_root}")
        return False

    candidates = (
        input_root / dataset_subdir / env_name,
        input_root / dataset_subdir,
        input_root / env_name,
        input_root,
    )
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        files = sorted(candidate.glob("*.hdf5"))
        if not files:
            continue

        if local_env_dir.exists():
            shutil.rmtree(local_env_dir)
        local_env_dir.mkdir(parents=True, exist_ok=True)
        for file in files:
            shutil.copy2(file, local_env_dir / file.name)
        if dataset_cache_complete(local_env_dir, required_files, experiment):
            print(f"Loaded datasets for {env_name} from storage input: {candidate}")
            return True

    print(f"Storage input did not contain the required datasets for {env_name} under {dataset_subdir}.")
    return False


def stage_generated_datasets(
    *,
    local_env_dir: Path,
    dataset_subdir: str,
    env_name: str,
    storage_backend: str,
    swift_data_container: str,
    nvdataset_data_dataset: str,
    nvdataset_data_description: str,
):
    files = sorted(local_env_dir.glob("*.hdf5"))
    if not files:
        print(f"No generated HDF5 datasets found under {local_env_dir} to stage.")
        return

    if storage_backend == "swift" and swift_data_container:
        from lib import swift_io

        prefix = f"{dataset_subdir}/{env_name}"
        print(f"Uploading generated datasets to Swift {swift_data_container}/{prefix}.")
        swift_io.upload_directory(
            container=swift_data_container,
            source_dir=local_env_dir,
            prefix=prefix,
            resume=True,
        )
    elif storage_backend == "nvdataset" and nvdataset_data_dataset:
        from lib import nvdataset_io

        upload_root = Path("/tmp/nvdatasets/generated_dataset_upload")
        upload_dir = upload_root / dataset_subdir / env_name
        if upload_root.exists():
            shutil.rmtree(upload_root)
        upload_dir.mkdir(parents=True, exist_ok=True)
        for file in files:
            shutil.copy2(file, upload_dir / file.name)
        print(f"Uploading generated datasets to NV-Datasets dataset {nvdataset_data_dataset}/{dataset_subdir}.")
        nvdataset_io.upload_directory(
            name=nvdataset_data_dataset,
            source_dir=upload_root,
            description=nvdataset_data_description,
        )
    else:
        print("No data storage target configured; generated HDF5 files remain only in the pod-local dataset dir.")


def contact_args(experiment: dict) -> list[str]:
    args = ["--contact-mode", str(experiment.get("contact_mode", "fixed_ground"))]
    if experiment.get("contact_mode") == "newton_native":
        representation = str(experiment.get("contact_representation", "flat"))
        args += [
            "--num-contacts-per-env",
            str(experiment.get("num_contacts_per_env", 64)),
            "--contact-packing-policy",
            str(experiment.get("contact_packing_policy", "penetration_priority")),
            "--contact-representation",
            representation,
        ]
        if representation == "contact_tokens":
            args += [
                "--max-contact-tokens",
                str(experiment.get("max_contact_tokens", 64)),
            ]
    return args


def build_dataset_args(experiment: dict, spec: DatasetSpec, dataset_dir: Path) -> list[str] | None:
    policy_checkpoint = str(experiment.get("policy_checkpoint", ""))
    if spec.use_policy:
        if not policy_checkpoint or not Path(policy_checkpoint).is_file():
            print(f"Skipping policy validation dataset; checkpoint not found: {policy_checkpoint}")
            return None
        sample_mode = "policy"
    else:
        sample_mode = str(experiment.get("sample_mode", "action"))

    transitions = experiment["train_transitions"] if spec.split == "train" else experiment["valid_transitions"]
    task = experiment.get("deployment_task") if spec.task_kind == "deployment" else experiment.get("data_gen_task")

    args = [
        "-m",
        "isaaclab_neural.generate.generate_dataset",
        "--task",
        str(task),
        "--dataset-dir",
        str(dataset_dir),
        "--dataset-name",
        spec.filename,
        "--env-name",
        str(experiment["env_name"]),
        "--robot-name",
        str(experiment["robot_name"]),
        "--sample-mode",
        sample_mode,
        "--initial-states-source",
        str(experiment.get("initial_states_source", "env")),
        *contact_args(experiment),
        "--num-envs",
        str(experiment["data_gen_num_envs"]),
        "--num-transitions",
        str(transitions),
        "--trajectory-length",
        str(experiment["trajectory_length"]),
        "--seed",
        str(spec.seed),
        "--headless",
        "--force-overwrite",
    ]

    if experiment.get("states_frame"):
        args += ["--states-frame", str(experiment["states_frame"])]
    if experiment.get("write_chunk_transitions"):
        args += ["--write-chunk-transitions", str(experiment["write_chunk_transitions"])]
    if spec.zero_actions:
        args.append("--zero-actions")
    if spec.use_policy:
        args += ["--policy-checkpoint", policy_checkpoint]
    if experiment.get("randomize_pd_gains") and spec.task_kind == "default" and not spec.zero_actions:
        args += ["--randomize-pd-gains", "--kp-min", "30.0", "--kp-max", "200.0", "--kd-min", "0.0", "--kd-max", "4.0"]
    return args


def generate_all_datasets(experiment: dict, specs: list[DatasetSpec], dataset_dir: Path):
    for spec in specs:
        args = build_dataset_args(experiment, spec, dataset_dir)
        if args is not None:
            subprocess.run([sys.executable, *args], check=True)


def run_training(experiment: dict, output_root: Path, wandb_args: argparse.Namespace):
    train_args = [
        "--task",
        str(experiment["train_task"]),
        "--cfg",
        str(experiment["train_cfg"]),
        "--logdir",
        str(output_root / str(experiment["env_name"])),
        "--num-envs",
        str(experiment["train_num_envs"]),
        "--seed",
        "0",
        "--headless",
        "--skip-check-log-override",
    ]
    if experiment.get("update_dataset_statistics"):
        train_args.append("--update-dataset-statistics")
    if experiment.get("train_preset"):
        train_args.append(f"presets={experiment['train_preset']}")
    if wandb_args.enable_wandb:
        train_args.append("--enable-wandb")
        train_args += ["--wandb-project-name", wandb_args.wandb_project_name]
        exp_name = wandb_args.wandb_exp_name
        if not exp_name:
            exp_name = wandb_args.workflow_base_name
            if wandb_args.dataset_subdir:
                exp_name = f"{exp_name}/{wandb_args.dataset_subdir}"
        train_args += ["--wandb-exp-name", exp_name]
        if wandb_args.wandb_entity:
            train_args += ["--wandb-entity", wandb_args.wandb_entity]
        if not wandb_args.wandb_save_checkpoints:
            train_args.append("--no-wandb-save-checkpoints")

    num_gpus = int(experiment.get("num_gpus", 1))
    if num_gpus > 1:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(num_gpus),
            "-m",
            "isaaclab_neural.train.train",
            *train_args,
        ]
    else:
        command = [sys.executable, "-m", "isaaclab_neural.train.train", *train_args]
    subprocess.run(command, check=True)


def run_context_diagnostic(experiment: dict, specs: list[DatasetSpec], dataset_dir: Path):
    """Validate terrain and contact reconstruction for a generated dataset."""
    diagnostic_filename = str(experiment.get("diagnostic_dataset", specs[0].filename))
    dataset_path = dataset_dir / str(experiment["env_name"]) / diagnostic_filename
    command = [
        sys.executable,
        "-m",
        "isaaclab_neural.eval.contact_reconstruction_diagnostic",
        "--task",
        str(experiment["train_task"]),
        "--dataset",
        str(dataset_path),
        "--num-envs",
        str(experiment.get("diagnostic_num_envs", experiment["data_gen_num_envs"])),
        "--step",
        str(experiment.get("diagnostic_step", 0)),
        "--require-terrain-context",
        "--headless",
        *contact_args(experiment),
    ]
    if experiment.get("states_frame"):
        command += ["--states-frame", str(experiment["states_frame"])]
    if experiment.get("diagnostic_contact_tolerance") is not None:
        command += ["--contact-tolerance", str(experiment["diagnostic_contact_tolerance"])]
    if experiment.get("diagnostic_contact_normal_tolerance") is not None:
        command += ["--contact-normal-tolerance", str(experiment["diagnostic_contact_normal_tolerance"])]
    if experiment.get("diagnostic_contact_velocity_tolerance") is not None:
        command += ["--contact-velocity-tolerance", str(experiment["diagnostic_contact_velocity_tolerance"])]
    if experiment.get("diagnostic_contact_context_validation") is not None:
        command += [
            "--contact-context-validation",
            str(experiment["diagnostic_contact_context_validation"]),
        ]
    if experiment.get("diagnostic_random_samples"):
        command += ["--random-samples", str(experiment["diagnostic_random_samples"])]
        command += ["--random-seed", str(experiment.get("diagnostic_random_seed", 0))]
        command += ["--eval-horizon", str(experiment.get("diagnostic_eval_horizon", 10))]
    if experiment.get("train_preset"):
        command.append(f"presets={experiment['train_preset']}")
    subprocess.run(command, check=True)
    if experiment.get("diagnostic_negative_seed_test", False):
        dataset_spec = next(spec for spec in specs if spec.filename == diagnostic_filename)
        negative_command = [*command, "--terrain-seed-override", str(dataset_spec.seed + 1)]
        result = subprocess.run(
            negative_command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(result.stdout, end="")
        if result.returncode == 0:
            raise RuntimeError("Negative terrain-seed diagnostic unexpectedly passed.")
        if "Terrain context mismatch" not in result.stdout:
            raise RuntimeError(
                f"Negative terrain-seed diagnostic failed for an unexpected reason (exit code {result.returncode})."
            )
        print(f"Negative terrain-seed diagnostic failed as expected with exit code {result.returncode}.")


def start_tensorboard(output_root: Path, port: int):
    log_file = (output_root / "tensorboard.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tensorboard.main",
            "--logdir",
            str(output_root),
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    (output_root / "tensorboard.pid").write_text(f"{process.pid}\n", encoding="utf-8")
    print(f"TensorBoard started on port {port} with logdir {output_root}")


def run(args: argparse.Namespace):
    preset = load_preset(args.preset, args.preset_file)
    experiment = preset["experiment"]
    specs = parse_dataset_specs(experiment)
    output_root = Path(args.output_root)
    dataset_dir = Path(args.dataset_dir)
    local_env_dir = dataset_dir / str(experiment["env_name"])
    required_files = required_dataset_files(experiment, specs)

    print(f"Workflow base name: {args.workflow_base_name}")
    print(f"Dataset subdirectory: {args.dataset_subdir}")
    print(f"Environment: {experiment['env_name']}")
    print(f"Data generation envs: {experiment['data_gen_num_envs']}")
    print(f"Training envs per rank: {experiment['train_num_envs']}")
    print(f"Training config: {experiment['train_cfg']}")

    datasets_available = False
    if args.dataset_cache_mode != "off" and args.dataset_input_path:
        datasets_available = load_dataset_cache_from_input(
            input_root=Path(args.dataset_input_path),
            dataset_subdir=args.dataset_subdir,
            env_name=str(experiment["env_name"]),
            local_env_dir=local_env_dir,
            required_files=required_files,
            experiment=experiment,
        )
        if not datasets_available:
            print(f"Storage input cache unavailable for {experiment['env_name']}; generating datasets locally.")
            if args.dataset_cache_mode == "require":
                raise RuntimeError("DATASET_CACHE_MODE=require but required datasets were not found in storage input.")
    elif args.dataset_cache_mode == "require":
        raise RuntimeError("DATASET_CACHE_MODE=require but DATASET_INPUT_PATH is empty.")

    if not datasets_available:
        generate_all_datasets(experiment, specs, dataset_dir)
        stage_generated_datasets(
            local_env_dir=local_env_dir,
            dataset_subdir=args.dataset_subdir,
            env_name=str(experiment["env_name"]),
            storage_backend=args.storage_backend,
            swift_data_container=args.swift_data_container,
            nvdataset_data_dataset=args.nvdataset_data_dataset,
            nvdataset_data_description=args.nvdataset_data_description,
        )

    if experiment.get("diagnostic_only", False):
        run_context_diagnostic(experiment, specs, dataset_dir)
        return

    start_tensorboard(output_root, args.tensorboard_port)
    run_training(experiment, output_root, args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--preset-file")
    parser.add_argument("--workflow-base-name", required=True)
    parser.add_argument("--dataset-subdir", required=True)
    parser.add_argument("--dataset-cache-mode", default="auto", choices=("auto", "require", "off"))
    parser.add_argument("--dataset-input-path", default="")
    parser.add_argument("--storage-backend", default="nvdataset", choices=("nvdataset", "swift"))
    parser.add_argument("--swift-data-container", default="")
    parser.add_argument("--nvdataset-data-dataset", default="")
    parser.add_argument("--nvdataset-data-description", default="IsaacLab-NeRD generated HDF5 datasets.")
    parser.add_argument("--output-root", default="/tmp/runs/output")
    parser.add_argument("--dataset-dir", default="./data/datasets")
    parser.add_argument("--tensorboard-port", type=int, default=6006)
    parser.add_argument("--enable-wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project-name", default="nerd-newton")
    parser.add_argument("--wandb-exp-name", default="")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument(
        "--wandb-save-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upload best checkpoints to the active W&B run.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "main_scripts.log"
    with log_path.open("a", encoding="utf-8") as log_file:
        tee_stdout = Tee(sys.stdout, log_file)
        tee_stderr = Tee(sys.stderr, log_file)
        with redirect_stdout(cast(TextIO, tee_stdout)), redirect_stderr(cast(TextIO, tee_stderr)):
            run(args)


if __name__ == "__main__":
    main()
