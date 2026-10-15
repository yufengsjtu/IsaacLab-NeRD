# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run an OSMO experiment from a declarative preset."""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import warnings
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import TextIO, cast

import h5py
import numpy as np
from lib.keyboard_data import requires_keyboard_bundle
from lib.simple_yaml import load_yaml

NATIVE_CONTACT_DATASET_SCHEMA_VERSION = 3


@dataclass
class DatasetSpec:
    filename: str
    seed: int
    split: str
    task_kind: str
    zero_actions: bool
    use_policy: bool
    num_transitions: int | None = None
    policy_checkpoint: str | None = None
    policy_action_noise_std: float = 0.0
    clip_policy_action_noise: bool = True
    action_hold_frames: list[int] | None = None
    rl_action_contract: bool = False


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
    """Load a preset and fill ``experiment.task_spec`` defaults, as the submit-time loader does."""
    from lib.preset import load_preset as load_submit_preset

    return load_submit_preset(name, preset_file)


def parse_dataset_specs(experiment: dict) -> list[DatasetSpec]:
    specs = []
    for item in experiment.get("dataset_specs", []):
        use_policy = bool(item.get("use_policy", False))
        noise_std = float(item.get("policy_action_noise_std", 0.0) or 0.0)
        if noise_std < 0.0:
            raise ValueError(f"Dataset {item.get('filename')!r} policy_action_noise_std must be non-negative.")
        if noise_std > 0.0 and not use_policy:
            raise ValueError(f"Dataset {item.get('filename')!r} sets policy_action_noise_std but use_policy is false.")
        hold = item.get("action_hold_frames")
        if hold is not None:
            hold = [int(value) for value in (hold if isinstance(hold, (list, tuple)) else [hold])]
            if not hold or any(value < 0 for value in hold):
                raise ValueError(f"Dataset {item.get('filename')!r} action_hold_frames must be non-negative.")
        specs.append(
            DatasetSpec(
                filename=str(item["filename"]),
                seed=int(item["seed"]),
                split=str(item.get("split", "valid")),
                task_kind=str(item.get("task_kind", "default")),
                zero_actions=bool(item.get("zero_actions", False)),
                use_policy=use_policy,
                num_transitions=int(item["num_transitions"]) if item.get("num_transitions") is not None else None,
                policy_checkpoint=str(item["policy_checkpoint"]) if item.get("policy_checkpoint") else None,
                policy_action_noise_std=noise_std,
                clip_policy_action_noise=bool(item.get("clip_policy_action_noise", True)),
                action_hold_frames=hold,
                rl_action_contract=bool(item.get("rl_action_contract", False)),
            )
        )
    if not specs:
        raise ValueError("Preset must define at least one dataset spec.")
    return specs


def resolve_spec_transitions(experiment: dict, spec: DatasetSpec) -> int:
    """Return the transition count requested for one dataset spec."""
    if spec.num_transitions is not None:
        transitions = spec.num_transitions
    else:
        transitions = experiment["train_transitions"] if spec.split == "train" else experiment["valid_transitions"]
    transitions = int(transitions)
    if transitions <= 0:
        raise ValueError(f"Dataset {spec.filename!r} must request a positive number of transitions.")
    return transitions


def resolve_spec_policy_checkpoint(experiment: dict, spec: DatasetSpec) -> str:
    """Return the policy checkpoint path for one spec, falling back to the experiment default."""
    return spec.policy_checkpoint or str(experiment.get("policy_checkpoint", "") or "")


def required_dataset_files(experiment: dict, specs: list[DatasetSpec]) -> list[str]:
    required_files = []
    for spec in specs:
        policy_checkpoint = resolve_spec_policy_checkpoint(experiment, spec)
        if spec.use_policy and policy_checkpoint and not Path(policy_checkpoint).is_file():
            continue
        required_files.append(spec.filename)
    return required_files


def _native_contact_cache_complete(native_group: h5py.Group, total_transitions: int, dataset_path: Path) -> bool:
    """Validate native-pair CSR structure without importing the training package."""
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
    required = {"frame_offsets", *integer_fields, *vector_fields, *scalar_fields}
    if set(native_group.keys()) != required:
        print(f"Rejected cached native-contact dataset with field mismatch: {dataset_path}")
        return False
    offsets = cast(h5py.Dataset, native_group["frame_offsets"])
    pair_count = int(cast(h5py.Dataset, native_group["world_ids"]).shape[0])
    if (
        offsets.ndim != 1
        or offsets.shape != (total_transitions + 1,)
        or offsets.dtype != np.dtype("int64")
        or int(offsets[0]) != 0
        or int(offsets[-1]) != pair_count
    ):
        print(f"Rejected cached native-contact dataset with invalid CSR offsets: {dataset_path}")
        return False
    previous = 0
    for start in range(0, offsets.shape[0], 1_000_000):
        chunk = np.asarray(offsets[start : min(start + 1_000_000, offsets.shape[0])], dtype=np.int64)
        if chunk.size and (chunk[0] < previous or np.any(chunk[1:] < chunk[:-1])):
            print(f"Rejected cached native-contact dataset with non-monotonic offsets: {dataset_path}")
            return False
        if chunk.size:
            previous = int(chunk[-1])
    precision = native_group.attrs.get("contact_value_precision", "full")
    precision = precision.decode() if isinstance(precision, bytes) else precision
    if precision not in ("full", "compact_v1"):
        print(f"Rejected cached native-contact dataset with unknown precision {precision!r}: {dataset_path}")
        return False
    for name in integer_fields | vector_fields | scalar_fields:
        dataset = cast(h5py.Dataset, native_group[name])
        expected_tail = (3,) if name in vector_fields else ()
        # compact_v1 narrows fields (the training package validates the exact per-field dtypes).
        if name in integer_fields:
            expected_dtypes = {np.dtype("int64")} if precision == "full" else {np.dtype("int16"), np.dtype("int32")}
        else:
            expected_dtypes = (
                {np.dtype("float32")} if precision == "full" else {np.dtype("float16"), np.dtype("float32")}
            )
        if dataset.shape != (pair_count, *expected_tail) or dataset.dtype not in expected_dtypes:
            print(f"Rejected cached native-contact dataset with invalid field {name!r}: {dataset_path}")
            return False
    return True


def dataset_cache_complete(
    candidate_dir: Path,
    required_files: list[str],
    experiment: dict,
) -> bool:
    if requires_keyboard_bundle(experiment):
        from lib.keyboard_data import validate_bundle

        try:
            validate_bundle(candidate_dir, experiment)
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"Rejected Keyboard bundle: {error}")
            return False
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
            expected_transitions = resolve_spec_transitions(experiment, spec)
            expected_trajectory_length = int(experiment["trajectory_length"])
            total_transitions = data_group.attrs.get("total_transitions", -1)
            if (
                data_group.attrs.get("mode") != "trajectory"
                or data_group.attrs.get("env") != experiment["env_name"]
                or states is None
                or states.ndim != 3
                or states.shape[1] != expected_trajectory_length
                or not isinstance(total_transitions, (int, np.integer))
                or total_transitions < expected_transitions
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
            contacts_group = handle.get("contacts")
            if contacts_group is not None and not isinstance(contacts_group, h5py.Group):
                print(f"Rejected cached dataset with a non-group contacts node: {dataset_path}")
                return False
            native_group = contacts_group.get("native_pairs") if contacts_group is not None else None
            if native_group is not None:
                if not isinstance(native_group, h5py.Group):
                    print(f"Rejected cached dataset with a non-group native_pairs node: {dataset_path}")
                    return False
                native_metadata_matches = (
                    int(data_group.attrs.get("dataset_schema_version", -1)) == NATIVE_CONTACT_DATASET_SCHEMA_VERSION
                    and data_group.attrs.get("contact_storage_schema") == "native_pairs_csr_v1"
                    and data_group.attrs.get("contact_frame") == "world_v1"
                )
                if not native_metadata_matches or not _native_contact_cache_complete(
                    native_group,
                    int(total_transitions),
                    dataset_path,
                ):
                    print(f"Rejected cached native-contact dataset with incompatible metadata: {dataset_path}")
                    return False
            has_contacts = int(experiment.get("num_contacts_per_env", 0)) > 0
            if has_contacts and native_group is None:
                print(f"Rejected cached dataset without native contact pairs: {dataset_path}")
                return False
            if has_contacts:
                required_projection_context = {
                    "root_body_q",
                    "root_body_qd",
                    "gravity_dir",
                    "states",
                    "next_states",
                    "joint_f",
                }
                missing_projection_context = sorted(required_projection_context - set(data_group.keys()))
                if missing_projection_context:
                    print(
                        f"Rejected cached dataset {dataset_path}: contact projection requires "
                        f"{missing_projection_context}."
                    )
                    return False
                if str(experiment.get("contact_token_frame", "root")) == "owner":
                    pose_key = "primary_body_q"
                    expected_primary_bodies = int(experiment.get("num_primary_bodies", 17))
                    pose_dataset = cast(h5py.Dataset, data_group[pose_key]) if pose_key in data_group else None
                    if (
                        pose_dataset is None
                        or pose_dataset.ndim != 4
                        or pose_dataset.shape[:2] != step_shape
                        or pose_dataset.shape[-2:] != (expected_primary_bodies, 7)
                    ):
                        print(f"Rejected cached dataset without owner-frame primary_body_q poses: {dataset_path}")
                        return False
            if experiment.get("require_finger_object_normal_forces", False):
                from isaaclab_neural.contacts.finger_object_force import LABEL_KEY, validate_force_metadata

                if LABEL_KEY not in data_group:
                    print(f"Rejected cached dataset without filtered finger normal-force labels: {dataset_path}")
                    return False
                validate_force_metadata(data_group)
            if experiment.get("record_body_net_contact_forces", False):
                force_key = "body_net_contact_forces_w"
                expected_primary_bodies = int(experiment.get("num_primary_bodies", 17))
                force_dataset = cast(h5py.Dataset, data_group[force_key]) if force_key in data_group else None
                force_metadata_matches = (
                    data_group.attrs.get("body_net_contact_force_schema") == "per_primary_body_v1"
                    and data_group.attrs.get("body_net_contact_force_frame") == "world"
                    and data_group.attrs.get("body_net_contact_force_units") == "N"
                    and data_group.attrs.get("body_net_contact_force_timing") == "post_solve_t_to_t_plus_1"
                )
                body_names = tuple(
                    name.decode() if isinstance(name, bytes) else str(name)
                    for name in data_group.attrs.get("body_net_contact_force_body_names", ())
                )
                force_body_names_match = (
                    len(body_names) == expected_primary_bodies
                    and len(set(body_names)) == expected_primary_bodies
                    and all(body_names)
                )
                if (
                    force_dataset is None
                    or force_dataset.ndim != 4
                    or force_dataset.shape[:2] != step_shape
                    or force_dataset.shape[-2:] != (expected_primary_bodies, 3)
                    or not force_metadata_matches
                    or not force_body_names_match
                ):
                    print(f"Rejected cached dataset without compatible per-body force labels: {dataset_path}")
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

        if requires_keyboard_bundle(experiment):
            from lib.keyboard_data import MANIFEST, file_sha256

            if not dataset_cache_complete(candidate, required_files, experiment):
                continue
            if local_env_dir.exists():
                if dataset_cache_complete(local_env_dir, required_files, experiment) and file_sha256(
                    local_env_dir / MANIFEST
                ) == file_sha256(candidate / MANIFEST):
                    return True
                raise FileExistsError(f"Refusing to replace existing unqualified cache: {local_env_dir}")
            local_env_dir.parent.mkdir(parents=True, exist_ok=True)
            local_env_dir.symlink_to(candidate.resolve(), target_is_directory=True)
            return dataset_cache_complete(local_env_dir, required_files, experiment)
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
    experiment: dict | None = None,
):
    if experiment and requires_keyboard_bundle(experiment):
        from lib.keyboard_data import bundle_files, validate_bundle

        validate_bundle(local_env_dir, experiment)
        if storage_backend == "swift" and swift_data_container:
            from lib import swift_io

            swift_io.upload_directory(
                container=swift_data_container,
                source_dir=local_env_dir,
                prefix=f"{dataset_subdir}/{env_name}",
                resume=True,
            )
        elif storage_backend == "nvdataset" and nvdataset_data_dataset:
            from lib import nvdataset_io

            with tempfile.TemporaryDirectory(prefix="keyboard-publish-") as temporary:
                directory = Path(temporary) / dataset_subdir / env_name
                directory.mkdir(parents=True)
                for name in bundle_files(experiment, local_env_dir):
                    try:
                        os.link(local_env_dir / name, directory / name)
                    except OSError:
                        shutil.copy2(local_env_dir / name, directory / name)
                nvdataset_io.upload_directory(
                    name=nvdataset_data_dataset, source_dir=Path(temporary), description=nvdataset_data_description
                )
        else:
            print("Qualified Keyboard cache retained locally; no remote publication target configured.")
        return
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
        try:
            nvdataset_io.upload_directory(
                name=nvdataset_data_dataset,
                source_dir=upload_root,
                description=nvdataset_data_description,
            )
        except Exception as error:  # noqa: BLE001 - the cache is optional; keep the run alive
            # Generic recordings are a reusable cache, not a run output: a failed publication must not discard
            # the collected data or stop training. Later runs recollect when the cache is incomplete.
            print(f"WARNING: generated-dataset cache upload failed; continuing without it: {error!r}")
    else:
        print("No data storage target configured; generated HDF5 files remain only in the pod-local dataset dir.")


def contact_args(experiment: dict) -> list[str]:
    contact_mode = str(experiment.get("contact_mode", "newton_native"))
    if contact_mode == "fixed_ground":
        warnings.warn(
            "OSMO preset contact_mode='fixed_ground' is deprecated and unsupported.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise ValueError(
            "fixed_ground and newton_native use incompatible contact representations. "
            "Update the preset to contact_mode='newton_native' and use matching data."
        )
    args = ["--contact-mode", contact_mode]
    if contact_mode == "newton_native":
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


ENV_DRIVEN_COLLECTORS = ("allegro_reorient", "env_driven")
"""``dataset_collector`` values that select the env-driven collector (``allegro_reorient`` is the historical name)."""


def build_allegro_collector_args(
    experiment: dict, spec: DatasetSpec, dataset_dir: Path, task: str, transitions: int, policy_checkpoint: str
) -> list[str]:
    """Arguments for the env-driven collector, which records reset-safe clips with a recording contract.

    Policy specs drive the teacher (optionally noisy), ``zero_actions`` specs hold zero actions and other specs hold
    uniform random actions. ``policy_task`` names the task supplying the teacher's fallback runner config.
    """
    mode = "policy" if spec.use_policy else "zero" if spec.zero_actions else "random"
    args = [
        "-m",
        "isaaclab_neural.generate.allegro_dataset",
        "--task",
        task,
        "--env-name",
        str(experiment["env_name"]),
        "--dataset",
        str(dataset_dir / str(experiment["env_name"]) / spec.filename),
        "--collection-mode",
        mode,
        "--num-envs",
        str(experiment["data_gen_num_envs"]),
        "--num-transitions",
        str(transitions),
        "--horizon",
        str(experiment["trajectory_length"]),
        "--chunk-control-steps",
        str(experiment.get("collection_chunk_control_steps", 64)),
        "--seed",
        str(spec.seed),
        "--max-contact-tokens",
        str(experiment.get("max_contact_tokens", 128)),
        "--contact-filter",
        str(experiment.get("contact_filter", "none")),
    ]
    if experiment.get("record_control_steps"):
        args.append("--record-control-steps")
    if experiment.get("record_body_net_contact_forces"):
        args.append("--record-body-net-contact-forces")
    if spec.use_policy:
        args += ["--checkpoint", policy_checkpoint]
        if experiment.get("policy_task"):
            args += ["--policy-task", str(experiment["policy_task"])]
        if spec.policy_action_noise_std > 0.0:
            args += ["--policy-action-noise-std", str(spec.policy_action_noise_std)]
    return args


def build_dataset_args(experiment: dict, spec: DatasetSpec, dataset_dir: Path) -> list[str] | None:
    if experiment.get("training_only"):
        raise ValueError("Keyboard training-only presets cannot use the generic dataset generator.")
    if experiment.get("dataset_profile") == "so101_keyboard":
        raise ValueError("Use the bounded Keyboard dataset profile through generate_all_datasets.")
    policy_checkpoint = resolve_spec_policy_checkpoint(experiment, spec)
    if spec.use_policy:
        if not policy_checkpoint or not Path(policy_checkpoint).is_file():
            print(f"Skipping policy dataset {spec.filename}; checkpoint not found: {policy_checkpoint}")
            return None
        sample_mode = "policy"
        initial_states_source = "env"
    else:
        sample_mode = str(experiment.get("sample_mode", "action"))
        initial_states_source = str(experiment.get("initial_states_source", "env"))

    transitions = resolve_spec_transitions(experiment, spec)
    task = experiment.get("deployment_task") if spec.task_kind == "deployment" else experiment.get("data_gen_task")
    if experiment.get("dataset_collector") in ENV_DRIVEN_COLLECTORS:
        return build_allegro_collector_args(experiment, spec, dataset_dir, str(task), transitions, policy_checkpoint)

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
        initial_states_source,
        *contact_args(experiment),
        "--num-envs",
        str(experiment["data_gen_num_envs"]),
        "--num-transitions",
        str(transitions),
        "--trajectory-length",
        str(experiment["trajectory_length"]),
        "--seed",
        str(spec.seed),
        "--force-overwrite",
    ]

    if experiment.get("states_frame"):
        args += ["--states-frame", str(experiment["states_frame"])]
    if experiment.get("write_chunk_transitions"):
        args += ["--write-chunk-transitions", str(experiment["write_chunk_transitions"])]
    hold_frames = spec.action_hold_frames
    if hold_frames is None:
        hold_frames = experiment.get("action_hold_frames")
    if hold_frames and not spec.use_policy:
        hold_values = hold_frames if isinstance(hold_frames, (list, tuple)) else [hold_frames]
        args += ["--action-hold-frames", *[str(int(value)) for value in hold_values]]
    if spec.rl_action_contract and not spec.use_policy and not spec.zero_actions:
        args.append("--rl-action-contract")
    if experiment.get("record_body_net_contact_forces", False):
        args.append("--record-body-net-contact-forces")
    if spec.zero_actions:
        args.append("--zero-actions")
    if spec.use_policy:
        args += ["--policy-checkpoint", policy_checkpoint]
        if spec.policy_action_noise_std > 0.0:
            args += ["--policy-action-noise-std", str(spec.policy_action_noise_std)]
            if not spec.clip_policy_action_noise:
                args.append("--no-clip-policy-action-noise")
    if experiment.get("randomize_pd_gains") and spec.task_kind == "default" and not spec.zero_actions:
        args += ["--randomize-pd-gains", "--kp-min", "30.0", "--kp-max", "200.0", "--kd-min", "0.0", "--kd-max", "4.0"]
    return args


def run_parallel_collection(jobs: list[tuple[str, list[str]]], num_gpus: int, log_dir: Path) -> None:
    """Run dataset-generation commands concurrently, one process per visible GPU.

    Each job sees a single GPU through ``CUDA_VISIBLE_DEVICES`` and logs to ``log_dir/<name>.log``. The first
    failure terminates the remaining jobs and raises with the tail of the failed log.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(jobs)
    running: dict[int, tuple[str, subprocess.Popen, TextIO]] = {}

    def stop_all() -> None:
        for _, process, stream in running.values():
            process.terminate()
            process.wait()
            stream.close()

    try:
        while pending or running:
            for gpu in range(num_gpus):
                if gpu not in running and pending:
                    name, args = pending.pop(0)
                    stream = (log_dir / f"{name}.log").open("w", encoding="utf-8")
                    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
                    print(f"[collection] gpu={gpu} start {name}", flush=True)
                    process = subprocess.Popen(
                        [sys.executable, *args], stdout=stream, stderr=subprocess.STDOUT, env=env
                    )
                    running[gpu] = (name, process, stream)
            for gpu, (name, process, stream) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                del running[gpu]
                if code != 0:
                    tail = (log_dir / f"{name}.log").read_text(encoding="utf-8", errors="replace")[-4000:]
                    raise RuntimeError(f"Dataset generation {name} failed with exit code {code}:\n{tail}")
                print(f"[collection] gpu={gpu} done {name}", flush=True)
            time.sleep(5)
    except BaseException:
        stop_all()
        raise


def generate_all_datasets(experiment: dict, specs: list[DatasetSpec], dataset_dir: Path, log_dir: Path | None = None):
    if experiment.get("dataset_profile") == "so101_keyboard":
        from lib.keyboard_data import collect_and_validate

        collect_and_validate(
            experiment, dataset_dir / experiment["env_name"], Path(experiment["collection_output_root"])
        )
        return
    jobs = []
    for spec in specs:
        args = build_dataset_args(experiment, spec, dataset_dir)
        if args is not None:
            jobs.append((Path(spec.filename).stem, args))
    num_gpus = int(experiment.get("parallel_collection_gpus", 0) or 0)
    if num_gpus > 1:
        run_parallel_collection(jobs, num_gpus, log_dir or dataset_dir / "collection_logs")
        return
    for _, args in jobs:
        subprocess.run([sys.executable, *args], check=True)


def latest_rsl_rl_checkpoint(run_dir: Path) -> Path:
    """Return the highest-iteration ``model_<iteration>.pt`` in an RSL-RL run directory."""
    checkpoints = [path for path in run_dir.glob("model_*.pt") if path.stem.split("_", 1)[1].isdigit()]
    if not checkpoints:
        raise FileNotFoundError(f"No RSL-RL checkpoint found in {run_dir}.")
    return max(checkpoints, key=lambda path: int(path.stem.split("_", 1)[1]))


def evaluate_policy(evaluation: dict, checkpoint: Path, report: Path) -> dict:
    """Run a closed-loop policy evaluation and enforce its optional quality gates.

    ``args`` are passed to the evaluation module. ``min_mean_goals_per_episode``, ``min_metrics`` and ``max_metrics``
    reject a teacher before any NeRD data is collected with it.
    """
    command = [
        sys.executable,
        "-m",
        str(evaluation["module"]),
        "--checkpoint",
        str(checkpoint),
        "--report",
        str(report),
        "--num-envs",
        str(evaluation.get("num_envs", 256)),
        "--control-steps",
        str(evaluation.get("control_steps", 1500)),
        "--seed",
        str(evaluation.get("seed", 100)),
        *[str(item) for item in evaluation.get("args", [])],
    ]
    subprocess.run(command, check=True)
    result = json.loads(report.read_text())
    minimum = evaluation.get("min_mean_goals_per_episode")
    mean_goals = result.get("mean_goals_per_episode")
    print(f"Policy evaluation: { {k: v for k, v in result.items() if not isinstance(v, (list, dict))} }")
    if minimum is not None and (mean_goals is None or mean_goals < float(minimum)):
        raise RuntimeError(
            f"Policy reached {mean_goals} goals per episode, below the preset minimum {minimum}; "
            "refusing to collect NeRD data with it."
        )
    for name, bound in evaluation.get("min_metrics", {}).items():
        if result.get(name) is None or result[name] < float(bound):
            raise RuntimeError(f"Policy {name}={result.get(name)} is below the preset minimum {bound}.")
    for name, bound in evaluation.get("max_metrics", {}).items():
        if result.get(name) is None or result[name] > float(bound):
            raise RuntimeError(f"Policy {name}={result.get(name)} exceeds the preset maximum {bound}.")
    return result


def apply_policy_reference(experiment: dict, evaluation: dict, result: dict) -> None:
    """Pass a native evaluation metric to training as the closed-loop checkpoint-selection reference.

    ``evaluation["reference"]`` names the report ``metric`` and the training ``cfg_key`` it overrides.
    """
    reference = evaluation.get("reference")
    if not reference:
        return
    value = result.get(reference["metric"])
    if value is None:
        raise RuntimeError(f"Native policy evaluation did not report {reference['metric']!r}.")
    overrides = str(experiment.get("train_cfg_overrides", "") or "").strip()
    experiment["train_cfg_overrides"] = f"{overrides} {reference['cfg_key']} {float(value)!r}".strip()
    print(f"Closed-loop selection reference: {reference['cfg_key']}={float(value)!r}")


def run_teacher_training(experiment: dict, output_root: Path, wandb_args: argparse.Namespace) -> Path:
    """Train the preset's RSL-RL teacher, upload its checkpoints to W&B and evaluate it.

    Only the options present in ``teacher_policy`` override the task's registered RSL-RL
    configuration. RSL-RL's W&B writer uploads every saved checkpoint; the run is also
    copied under ``output_root/teacher_policy`` with its saved agent and environment config.

    Returns:
        The final teacher checkpoint inside ``output_root``.
    """
    teacher = experiment["teacher_policy"]
    wandb_project = teacher.get("wandb_project")
    if wandb_project and not wandb_args.enable_wandb:
        raise ValueError("This teacher preset uploads its checkpoints to W&B; submit it with --enable-wandb.")
    experiment_name = str(teacher.get("experiment_name", "teacher_policy"))
    run_name = f"osmo_{int(time.time())}"
    args = [
        "-m",
        "isaaclab_neural.train.train_control_policy",
        "--task",
        str(teacher["task"]),
        "--experiment_name",
        experiment_name,
        "--run_name",
        run_name,
    ]
    for key, flag in (("num_envs", "--num_envs"), ("max_iterations", "--max_iterations"), ("seed", "--seed")):
        if teacher.get(key) is not None:
            args += [flag, str(teacher[key])]
    env = os.environ.copy()
    if wandb_project:
        args += ["--logger", "wandb", "--log_project_name", str(wandb_project)]
        # Without an explicit entity the API key's default entity is used. RSL-RL's W&B writer reads
        # WANDB_USERNAME; Isaac Lab's printed run reference prefers WANDB_ENTITY.
        entity = teacher.get("wandb_entity") or wandb_args.wandb_entity
        if entity:
            env["WANDB_USERNAME"] = env["WANDB_ENTITY"] = str(entity)
        for key in ("WANDB_ENTITY", "WANDB_USERNAME"):
            # The workflow template exports empty values; drop them so W&B falls back to the default entity.
            if not env.get(key):
                env.pop(key, None)
        # Isaac Lab otherwise draws the id from wandb.util.generate_id(), which wandb 0.30 (init.sh) removed.
        prefix = str(teacher.get("wandb_run_id_prefix", "") or "")
        env.setdefault("WANDB_RUN_ID", f"{prefix}-{uuid.uuid4().hex[:8]}" if prefix else uuid.uuid4().hex[:8])
        print(f"Teacher W&B run: wandb:{entity or '<default-entity>'}/{wandb_project}/{env['WANDB_RUN_ID']}")
    num_gpus = int(teacher.get("num_gpus", 1))
    if num_gpus > 1:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(num_gpus),
            *args,
            "--distributed",
        ]
    else:
        command = [sys.executable, *args]
    subprocess.run(command, check=True, env=env)
    runs = sorted((Path.cwd() / "logs" / "rsl_rl" / experiment_name).glob(f"*_{run_name}"))
    if len(runs) != 1:
        raise RuntimeError(f"Expected one teacher run named {run_name!r}, found {[str(run) for run in runs]}.")
    teacher_root = output_root / "teacher_policy"
    run_dir = teacher_root / runs[0].name
    shutil.copytree(runs[0], run_dir)
    checkpoint = latest_rsl_rl_checkpoint(run_dir)
    print(f"Trained teacher policy: {checkpoint}")
    if teacher.get("evaluation"):
        evaluate_policy(teacher["evaluation"], checkpoint, teacher_root / "baseline.json")
    return checkpoint


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
        str(experiment.get("train_seed", 0)),
        "--skip-check-log-override",
    ]
    if requires_keyboard_bundle(experiment):
        if (output_root / experiment["env_name"]).exists():
            raise FileExistsError("Refusing to overwrite a previous Keyboard training run.")
        train_args.remove("--skip-check-log-override")
    if experiment.get("headless"):
        train_args += ["--visualizer", "none"]
    if experiment.get("no_time_stamp"):
        train_args.append("--no-time-stamp")
    for key, flag in (("eval_interval", "--eval-interval"), ("save_interval", "--save-interval")):
        if key in experiment:
            train_args += [flag, str(experiment[key])]
    if experiment.get("update_dataset_statistics"):
        train_args.append("--update-dataset-statistics")
    # Keyboard presets write their overrides into the prepared runtime config instead.
    generic_overrides = str(experiment.get("train_cfg_overrides", "") or "").strip()
    if generic_overrides and not experiment.get("training_only") and not experiment.get("dataset_profile"):
        train_args += ["--cfg-overrides", generic_overrides]
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


def run_throughput_benchmarks(experiment: dict, output_root: Path) -> dict:
    """Time the trained NeRD against native MJWarp on one otherwise idle GPU after training.

    ``throughput_benchmark`` lists ``variants`` (``name``, optional ``nerd`` and extra ``args``), the
    ``num_envs`` sweep, step counts, model-only ``model_autocast`` settings and the ``checkpoint_name`` under
    ``<output_root>/<env_name>/nn``. Each run writes ``throughput/<variant>_<num_envs>.json``; a failed variant is
    recorded in ``throughput/summary.json`` instead of failing the workflow, so the trained model is still kept.
    """
    bench = experiment["throughput_benchmark"]
    root = output_root / "throughput"
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = output_root / str(experiment["env_name"]) / "nn" / str(bench.get("checkpoint_name", "final_model.pt"))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(bench.get("gpu", 0))}
    module = str(bench["module"])
    runs = []
    model_args = [str(item) for item in bench.get("model_args", [])]
    env_args = [str(item) for item in bench.get("env_args", [])]
    for autocast in bench.get("model_autocast", []):
        runs.append(
            (
                f"model_{autocast}",
                ["model", "--nerd-checkpoint", str(checkpoint), "--autocast", str(autocast), *model_args],
            )
        )
    for variant in bench.get("variants", []):
        for num_envs in bench.get("num_envs", [8192]):
            args = ["env", "--checkpoint", str(experiment["policy_checkpoint"]), "--num-envs", str(num_envs), *env_args]
            for key in ("warmup", "steps", "breakdown_steps", "profile_steps"):
                if bench.get(key) is not None:
                    args += [f"--{key.replace('_', '-')}", str(bench[key])]
            if variant.get("nerd"):
                args += ["--nerd-checkpoint", str(checkpoint)]
            runs.append((f"{variant['name']}_{num_envs}", [*args, *[str(item) for item in variant.get("args", [])]]))
    summary = {"checkpoint": str(checkpoint), "runs": {}}
    for name, args in runs:
        report = root / f"{name}.json"
        with (root / f"{name}.log").open("w", encoding="utf-8") as log:
            code = subprocess.run(
                [sys.executable, "-m", module, *args, "--report", str(report)],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            ).returncode
        summary["runs"][name] = json.loads(report.read_text()) if code == 0 and report.is_file() else {"exit": code}
        print(f"[throughput] {name}: exit={code}", flush=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


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


def validate_training_capacity_config(experiment: dict, cfg: dict) -> None:
    """Reject inconsistent capacity units or insufficient source budgets before collection."""
    data = cfg["algorithm"]["dataset"]
    if experiment.get("dataset_profile") == "so101_keyboard":
        solver_cfg = cfg["env"]["neural_solver_cfg"]
        if solver_cfg.get("contact_filter", "none") != experiment.get("contact_filter", "none"):
            raise ValueError("Training and collection contact filters differ.")
        for name in ("num_contacts_per_env", "max_contact_tokens"):
            if name in experiment and solver_cfg.get(name) != experiment[name]:
                raise ValueError(f"Training and collection contact capacities differ: {name}.")
    unit = data.get("capacity_unit", "stored_frame")
    if unit != experiment.get("dataset_capacity_unit", "stored_frame"):
        raise ValueError("Training and collection capacity units differ.")
    if unit == "stored_frame":
        return
    if unit != "unique_transition" or data.get("capacity_scope", "global") != "global":
        raise ValueError("Unique-transition capacity must use the global scope.")
    sampling = data.get("window_sampling", "canonical")
    if sampling not in ("canonical", "sliding") or sampling != experiment.get("training_window_sampling", sampling):
        raise ValueError("Training window sampling differs from the requested preset.")
    if cfg["algorithm"]["name"] != "SequenceModelTrainer":
        raise ValueError("Unique-transition capacity requires sequence training.")
    sequence = int(cfg["algorithm"]["sample_sequence_length"])
    if sequence != experiment.get("training_sequence_length") or not 0 < sequence <= experiment["trajectory_length"]:
        raise ValueError("Training and collection sequence-length contracts differ.")
    sources = data["train_datasets"]
    if len(sources) > 1 and any("max_capacity" not in source for source in sources.values()):
        raise ValueError("Mixed unique datasets require explicit source budgets.")
    budgets = {
        Path(source["path"]).name: int(source.get("max_capacity", data["max_capacity"])) for source in sources.values()
    }
    if len(budgets) != len(sources):
        raise ValueError("Unique training sources must use distinct cache filenames.")
    if any(value <= 0 for value in budgets.values()) or sum(budgets.values()) > int(data["max_capacity"]):
        raise ValueError("Unique source budgets exceed the global capacity or are non-positive.")
    if sum(budgets.values()) < int(experiment.get("minimum_unique_transitions", {}).get("train", 0)):
        raise ValueError("Unique source budgets are below the global training data requirement.")
    for spec in experiment["dataset_specs"]:
        capacity = (
            budgets.get(spec["filename"], 0)
            if spec["split"] == "train"
            else int(data.get("valid_max_capacity", 100_000_000))
        )
        if capacity < int(spec["num_transitions"]):
            raise ValueError(f"Unique capacity is below the required data target: {spec['filename']}.")


def prepare_training_only_config(experiment: dict, dataset_dir: Path, output_root: Path, overrides: str = "") -> None:
    """Materialize verified cache paths and explicit training overrides without altering source configs."""
    import yaml
    from isaaclab_neural.utils.python_utils import handle_cfg_overrides

    cfg = load_yaml(Path(experiment["train_cfg"]))
    if not isinstance(cfg, dict):
        raise ValueError("Training configuration must be a mapping.")
    handle_cfg_overrides(" ".join((experiment.get("train_cfg_overrides", ""), overrides)), cfg)
    validate_training_capacity_config(experiment, cfg)
    data = cfg["algorithm"]["dataset"]
    paths = [(source, "path", "train") for source in data["train_datasets"].values()]
    paths += [(data["valid_datasets"], key, "valid") for key in data["valid_datasets"]]
    used: dict[str, set[str]] = {"train": set(), "valid": set()}
    for owner, key, split in paths:
        filename = Path(owner[key]).name
        owner[key] = str((dataset_dir / filename).resolve())
        used[split].add(filename)
    for split in used:
        if used[split] != {item["filename"] for item in experiment["dataset_specs"] if item["split"] == split}:
            raise ValueError(f"Training configuration does not match the verified {split} bundle files.")
    if cfg["env"]["env_name"] != experiment["env_name"]:
        raise ValueError("Training config environment differs from the preset.")
    if experiment.get("dataset_profile") == "so101_keyboard":
        solver_cfg = cfg["env"]["neural_solver_cfg"]
        for name in ("num_contacts_per_env", "max_contact_tokens"):
            if solver_cfg.get(name) != experiment.get(name):
                raise ValueError(f"Keyboard training and collection contact capacities differ: {name}.")
    if experiment.get("minimum_unique_transitions"):
        from lib.keyboard_data import validate_dataset_scale

        capacities = {
            Path(source["path"]).name: int(source.get("max_capacity", data["max_capacity"]))
            for source in data["train_datasets"].values()
        }
        if data.get("capacity_unit") == "unique_transition":
            capacities.update(
                {
                    Path(path).name: int(data.get("valid_max_capacity", 100_000_000))
                    for path in data["valid_datasets"].values()
                }
            )
        validate_dataset_scale(dataset_dir, experiment, source_capacities=capacities)
        if data.get("capacity_unit") == "unique_transition":
            from isaaclab_neural.data.transition_index import build_transition_index

            cfg["dataset_capacity_contract"] = {
                name: build_transition_index(
                    dataset_dir / name,
                    max_capacity=capacity,
                    sequence_length=int(cfg["algorithm"]["sample_sequence_length"]),
                    window_sampling=data.get("window_sampling", "canonical"),
                ).stats
                for name, capacity in capacities.items()
            }
    path = output_root / "resolved_train_cfg.yaml"
    with path.open("x") as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)
    experiment["train_cfg"] = str(path.resolve())


def evaluate_keyboard_checkpoint(experiment: dict, dataset_dir: Path, output_root: Path) -> None:
    """Run held-out synchronized-window diagnostics; execution success is not fidelity acceptance."""
    options = experiment.get("checkpoint_eval")
    if not options:
        return
    checkpoint = output_root / experiment["env_name"] / "nn/final_model.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    from lib.keyboard_data import MANIFEST

    manifest = json.loads((dataset_dir / MANIFEST).read_text())
    filenames = []
    for spec in parse_dataset_specs(experiment):
        if spec.split == "valid":
            parts = manifest.get("groups", {}).get(spec.filename, [spec.filename])
            count = min(len(parts), int(options.get("parts_per_group", 1)))
            if count <= 0:
                raise ValueError("Checkpoint evaluation requires positive parts_per_group.")
            filenames.extend(parts[index] for index in np.linspace(0, len(parts) - 1, count, dtype=int))
    reports = []
    for filename in filenames:
        report = output_root / (filename + ".nerd_eval.json")
        command = [
            sys.executable,
            "-m",
            "isaaclab_neural.eval.keyboard_checkpoint",
            "--checkpoint",
            str(checkpoint),
            "--dataset",
            str(dataset_dir / filename),
            "--report",
            str(report),
            "--horizon",
            str(options.get("horizon", 16)),
            "--device",
            "cuda:0",
            "--visualizer",
            "none",
        ]
        if options.get("max_batches") is not None:
            command += ["--max-batches", str(options["max_batches"])]
        subprocess.run(command, check=True)
        reports.append(str(report))
    (output_root / "keyboard_run_status.json").write_text(
        json.dumps(
            {
                "training_completed": True,
                "checkpoint": str(checkpoint),
                "evaluation_reports": reports,
                "execution_passed": all(json.loads(Path(path).read_text())["execution_passed"] for path in reports),
                "task_fidelity_accepted": False,
            },
            indent=2,
        )
        + "\n"
    )


def run_keyboard_pipeline(args: argparse.Namespace, preset: dict) -> None:
    """Use the common storage/training runner with Keyboard collection and acceptance hooks."""
    from lib.keyboard_data import bundle_validation_session

    with bundle_validation_session():
        _run_keyboard_pipeline(args, preset)


def _run_keyboard_pipeline(args: argparse.Namespace, preset: dict) -> None:
    """Execute one pipeline without retaining verified-file identities across runs."""
    from lib.keyboard_data import MANIFEST, collection_plan, validate_bundle, validate_runtime, validation_mode

    experiment = dict(preset["experiment"])
    output_root = Path(args.output_root)
    state = {"stages": {}, "task_fidelity_accepted": False, "data_validation_mode": validation_mode(experiment)}
    stage = "preflight"
    pipeline_started = time.perf_counter()
    stage_started = {}
    events = []
    state["stage_events"] = events

    def record(name, status):
        now = time.perf_counter()
        event = {"stage": name, "status": status, "unix_time": time.time(), "elapsed_seconds": now - pipeline_started}
        if status == "running":
            stage_started[name] = now
        elif name in stage_started:
            event["duration_seconds"] = now - stage_started.pop(name)
        events.append(event)
        state["stages"][name] = status
        (output_root / "pipeline_status.json").write_text(json.dumps(state, indent=2) + "\n")
        print("[Pipeline stage] " + json.dumps(event), flush=True)

    try:
        record(stage, "running")
        validate_runtime(experiment)
        if experiment.get("dataset_capacity_unit") == "unique_transition":
            from isaaclab_neural.utils.python_utils import handle_cfg_overrides

            capacity_cfg = load_yaml(Path(experiment["train_cfg"]))
            if not isinstance(capacity_cfg, dict):
                raise ValueError("Training configuration must be a mapping.")
            handle_cfg_overrides(
                " ".join((experiment.get("train_cfg_overrides", ""), getattr(args, "train_cfg_overrides", ""))),
                capacity_cfg,
            )
            validate_training_capacity_config(experiment, capacity_cfg)
        plan = collection_plan(experiment)
        experiment["policy_sha256"] = plan["policy_sha256"]
        experiment["collection_output_root"] = str(output_root.resolve())
        if getattr(args, "num_gpus", None) is not None:
            experiment["num_gpus"] = args.num_gpus
        if int(experiment["num_gpus"]) <= 0:
            raise ValueError("Training process count must be positive.")
        specs = parse_dataset_specs(experiment)
        dataset_dir = Path(args.dataset_dir or experiment["dataset_dir"])
        local_env_dir = dataset_dir / experiment["env_name"]
        required = required_dataset_files(experiment, specs)
        (output_root / "collection_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        record(stage, "passed")
        stage = "cache_lookup"
        record(stage, "running")
        cached = False
        if args.dataset_cache_mode != "off":
            if (local_env_dir / MANIFEST).exists():
                cached = dataset_cache_complete(local_env_dir, required, experiment)
                if not cached:
                    raise ValueError("Existing local Keyboard cache is invalid; it will not be replaced.")
            elif args.dataset_input_path:
                cached = load_dataset_cache_from_input(
                    Path(args.dataset_input_path),
                    args.dataset_subdir,
                    experiment["env_name"],
                    local_env_dir,
                    required,
                    experiment,
                )
        elif local_env_dir.exists():
            raise FileExistsError("Cache mode off requires a fresh dataset directory.")
        record(stage, "hit" if cached else "miss")
        if not cached and args.dataset_cache_mode == "require":
            raise RuntimeError("Required replay-qualified Keyboard cache is unavailable.")
        if not cached and experiment["collection"].get("resume_from_input", False):
            from lib.keyboard_data import RESUME_MANIFEST, restore_collection_resume, validate_collection_resume

            stage = "collection_resume"
            if args.dataset_cache_mode != "auto" or not args.dataset_input_path:
                raise ValueError("Collection recovery requires cache mode auto and an explicit input checkpoint.")
            input_root = Path(args.dataset_input_path)
            candidates = (
                input_root / args.dataset_subdir / experiment["env_name"],
                input_root / args.dataset_subdir,
                input_root / experiment["env_name"],
                input_root,
            )
            source = next((path for path in candidates if (path / RESUME_MANIFEST).is_file()), None)
            if source is None:
                raise FileNotFoundError("Required collection recovery checkpoint is missing; refusing to recollect.")
            if getattr(args, "preflight_only", False):
                count = len(validate_collection_resume(source, experiment)["files"])
                record(stage, f"validated {count} accepted chunks; not restored in preflight-only mode")
            else:
                count = restore_collection_resume(source, local_env_dir, experiment)
                record(stage, f"restored {count} accepted chunks")
        if getattr(args, "preflight_only", False):
            state["would_collect"] = not cached
            if cached:
                prepare_training_only_config(
                    experiment, local_env_dir, output_root, getattr(args, "train_cfg_overrides", "")
                )
            record("preflight_only", "complete; no collection, upload, training or submission")
            return
        import torch

        if torch.cuda.device_count() < int(experiment["num_gpus"]):
            raise RuntimeError("Visible CUDA devices are fewer than requested training processes.")
        if not cached:
            stage = (
                "collection_and_basic_validation" if validation_mode(experiment) == "basic" else "collection_and_replay"
            )
            record(stage, "running")
            generate_all_datasets(experiment, specs, dataset_dir)
            record(stage, "passed")
        stage = "data_validation"
        record(stage, "running")
        if not dataset_cache_complete(local_env_dir, required, experiment):
            raise RuntimeError("Keyboard data validation failed; no cache publication or training.")
        validate_bundle(local_env_dir, experiment)
        prepare_training_only_config(experiment, local_env_dir, output_root, getattr(args, "train_cfg_overrides", ""))
        (output_root / "resolved_experiment.json").write_text(json.dumps(experiment, indent=2) + "\n")
        record(stage, "passed")
        stage = "cache_publication"
        record(stage, "running")
        if not cached:
            stage_generated_datasets(
                local_env_dir=local_env_dir,
                dataset_subdir=args.dataset_subdir,
                env_name=experiment["env_name"],
                storage_backend=args.storage_backend,
                swift_data_container=args.swift_data_container,
                nvdataset_data_dataset=args.nvdataset_data_dataset,
                nvdataset_data_description=args.nvdataset_data_description,
                experiment=experiment,
            )
            record(stage, "published" if args.swift_data_container or args.nvdataset_data_dataset else "local_only")
        else:
            record(stage, "reused")
        if experiment.get("collection_only", False):
            record("collection_only", "complete; no training or checkpoint evaluation")
            return
        stage = "training"
        record(stage, "running")
        if not getattr(args, "no_tensorboard", False):
            start_tensorboard(output_root, args.tensorboard_port)
        run_training(experiment, output_root, args)
        record(stage, "passed")
        stage = "checkpoint_evaluation"
        record(stage, "running")
        evaluate_keyboard_checkpoint(experiment, local_env_dir, output_root)
        record(stage, "passed; fidelity requires separate review")
    except Exception as error:
        state["error"] = str(error)
        record(stage, "failed")
        raise


def run(args: argparse.Namespace):
    preset = load_preset(args.preset, args.preset_file)
    if preset["experiment"].get("dataset_profile") == "so101_keyboard":
        run_keyboard_pipeline(args, preset)
        return
    experiment = dict(preset["experiment"])
    if experiment.get("teacher_only"):
        run_teacher_training(experiment, Path(args.output_root), args)
        return
    if getattr(args, "preflight_only", False) and not experiment.get("training_only"):
        raise ValueError("preflight-only is supported only for training-only presets.")
    if experiment.get("training_only") and args.dataset_cache_mode != "require":
        raise ValueError("Keyboard training-only presets require --dataset-cache-mode require; generation is disabled.")
    if getattr(args, "num_gpus", None) is not None:
        if args.num_gpus <= 0:
            raise ValueError("num-gpus must be positive.")
        experiment["num_gpus"] = args.num_gpus
    if experiment.get("training_only"):
        from lib.keyboard_data import validate_runtime

        print(f"Validated runtime: {validate_runtime(experiment)}")
    specs = parse_dataset_specs(experiment)
    output_root = Path(args.output_root)
    if experiment.get("policy_evaluation"):
        policy_checkpoint = Path(str(experiment.get("policy_checkpoint", "")))
        if not policy_checkpoint.is_file():
            raise FileNotFoundError(f"Policy checkpoint not found: {policy_checkpoint}")
        report = output_root / "policy_evaluation.json"
        result = evaluate_policy(experiment["policy_evaluation"], policy_checkpoint, report)
        apply_policy_reference(experiment, experiment["policy_evaluation"], result)
    dataset_dir = Path(args.dataset_dir or experiment.get("dataset_dir", "./data/datasets"))
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
            print(f"Storage input cache unavailable for {experiment['env_name']}.")
            if args.dataset_cache_mode == "require":
                raise RuntimeError("DATASET_CACHE_MODE=require but required datasets were not found in storage input.")
    elif args.dataset_cache_mode == "require":
        raise RuntimeError("DATASET_CACHE_MODE=require but DATASET_INPUT_PATH is empty.")

    if not datasets_available:
        if experiment.get("training_only"):
            raise RuntimeError(
                "Keyboard training-only cache is unavailable; refusing to generate or upload replacements."
            )
        # Concurrent collection keeps one log per shard with the workflow output.
        log_dir = [output_root / "collection_logs"] if experiment.get("parallel_collection_gpus") else []
        generate_all_datasets(experiment, specs, dataset_dir, *log_dir)
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

    if experiment.get("training_only"):
        prepare_training_only_config(experiment, local_env_dir, output_root, getattr(args, "train_cfg_overrides", ""))
        (output_root / "resolved_experiment.json").write_text(json.dumps(experiment, indent=2) + "\n")
    if getattr(args, "preflight_only", False):
        print("Preflight passed; no training, evaluation, upload or submission was performed.")
        return
    if experiment.get("training_only"):
        import torch

        if torch.cuda.device_count() < int(experiment.get("num_gpus", 1)):
            raise RuntimeError("Allocated visible CUDA devices do not satisfy the requested training process count.")
    if not getattr(args, "no_tensorboard", False):
        start_tensorboard(output_root, args.tensorboard_port)
    run_training(experiment, output_root, args)
    if experiment.get("training_only"):
        evaluate_keyboard_checkpoint(experiment, local_env_dir, output_root)
    if experiment.get("throughput_benchmark"):
        run_throughput_benchmarks(experiment, output_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--preset-file")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument(
        "--num-gpus", type=int, default=int(os.environ["NERD_NUM_GPUS"]) if os.environ.get("NERD_NUM_GPUS") else None
    )
    parser.add_argument("--train-cfg-overrides", default=os.environ.get("TRAIN_CFG_OVERRIDES", ""))
    parser.add_argument("--workflow-base-name", required=True)
    parser.add_argument("--dataset-subdir", required=True)
    parser.add_argument("--dataset-cache-mode", default="auto", choices=("auto", "require", "off"))
    parser.add_argument("--dataset-input-path", default="")
    parser.add_argument("--storage-backend", default="nvdataset", choices=("nvdataset", "swift"))
    parser.add_argument("--swift-data-container", default="")
    parser.add_argument("--nvdataset-data-dataset", default="")
    parser.add_argument("--nvdataset-data-description", default="IsaacLab-NeRD generated HDF5 datasets.")
    parser.add_argument("--output-root", default="/tmp/runs/output")
    parser.add_argument("--dataset-dir", default=None, help="Override the preset's local dataset root.")
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
