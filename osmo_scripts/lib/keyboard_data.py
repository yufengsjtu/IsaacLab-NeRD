# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare and verify immutable, replay-qualified Keyboard OSMO inputs locally."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack, suppress
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

MANIFEST = "keyboard_dataset_manifest.json"
SCHEMA = "keyboard_replay_bundle_v1"
CHUNKED_SCHEMA = "keyboard_replay_bundle_v2"
BASIC_SCHEMA = "keyboard_basic_collection_bundle_v1"
RESUME_MANIFEST = "keyboard_collection_resume.json"
RESUME_SCHEMA = "keyboard_collection_resume_v1"
LIMITS = {
    "robot_q_error_rad": ("robot_q_atol_rad", 1e-3),
    "key_q_error_m": ("key_q_atol_m", 5e-5),
    "robot_qd_error_rad_s": ("robot_qd_atol_rad_s", 1e-2),
    "key_qd_error_m_s": ("key_qd_atol_m_s", 1e-3),
    "force_error_n": ("force_atol_n", 0.1),
}


def run_logged(
    command: list[str], log_path: Path, *, env: dict[str, str] | None = None, timeout: float | None = None
) -> None:
    """Keep the complete child log and report a bounded, redacted failure tail before re-raising."""
    with log_path.open("xb+") as stream:
        try:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=timeout)
        except subprocess.CalledProcessError as error:
            stream.flush()
            start = max(0, stream.seek(0, os.SEEK_END) - 16384)
            stream.seek(start)
            data = stream.read(16384)
            if start:
                data = data.partition(b"\n")[2]
            tail = data.decode("utf-8", errors="replace")
            for key in (
                "NGC_API_KEY",
                "NVDATASET_NGC_API_KEY",
                "SWIFT_AUTH_KEY",
                "WANDB_API_KEY",
                "OMNI_PASS",
                "OMNI_PASSWORD",
            ):
                if value := os.environ.get(key):
                    tail = tail.replace(value, "[REDACTED]")
            tail = re.sub(r"nvapi-[A-Za-z0-9_-]+", "[REDACTED]", tail)
            tail = re.sub(r"(?i)(enter code:\s*)\S+", r"\1[REDACTED]", tail)
            print(f"[failed] {log_path}: exit={error.returncode}; log tail:\n{tail}", file=sys.stderr, flush=True)
            raise


def file_sha256(path: Path) -> str:
    """Hash a file without loading its contents into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_runtime(experiment: dict[str, Any]) -> dict[str, str]:
    """Reject unvalidated binary dependency versions without installing or changing packages."""
    actual = {name: version(name) for name in experiment.get("runtime_versions", {})}
    mismatch = {
        name: (expected, actual[name])
        for name, expected in experiment.get("runtime_versions", {}).items()
        if expected != actual[name]
    }
    if mismatch:
        raise RuntimeError(
            f"Keyboard runtime differs from the validated stack (expected, actual): {mismatch}. Use a matching image."
        )
    return actual


def validation_mode(experiment: dict[str, Any]) -> str:
    """Resolve an explicit basic-only mode without weakening replay-qualified presets."""
    mode = experiment.get("collection", {}).get("validation_mode", "full_replay")
    if mode not in ("full_replay", "basic"):
        raise ValueError(f"Unsupported Keyboard collection validation mode: {mode}.")
    if mode == "basic" and experiment.get("require_keyboard_replay", True):
        raise ValueError("Basic collection requires explicit require_keyboard_replay=false.")
    return mode


def requires_keyboard_bundle(experiment: dict[str, Any]) -> bool:
    """Identify both replay-qualified and explicitly basic-checked Keyboard bundles."""
    return experiment.get("dataset_profile") == "so101_keyboard" or bool(experiment.get("require_keyboard_replay"))


def bundle_files(experiment: dict[str, Any], root: Path | None = None) -> list[str]:
    """Enumerate only explicitly declared input files and their replay receipts."""
    names = [spec["filename"] for spec in experiment["dataset_specs"]]
    if (
        not names
        or len(set(names)) != len(names)
        or any(Path(name).name != name or not name.endswith(".hdf5") for name in names)
    ):
        raise ValueError("Keyboard dataset specs require unique HDF5 basenames.")
    if root is not None:
        manifest = json.loads((root / MANIFEST).read_text())
        if manifest.get("schema") in (CHUNKED_SCHEMA, BASIC_SCHEMA):
            parts = [part for name in names for part in manifest["groups"][name]]
            if len(set([*parts, *names])) != len(parts) + len(names) or any(
                Path(part).name != part or not part.endswith(".hdf5") for part in parts
            ):
                raise ValueError("Chunked bundle requires distinct sibling HDF5 files.")
            suffix = ".collect.json" if manifest["schema"] == BASIC_SCHEMA else ".replay.json"
            return [*names, *parts, *(part + suffix for part in parts), MANIFEST]
    return [*names, *(name + ".replay.json" for name in names), MANIFEST]


def _dataset(group: h5py.Group, name: str) -> h5py.Dataset:
    value = group[name]
    if not isinstance(value, h5py.Dataset):
        raise ValueError(f"Expected an HDF5 dataset at {name}.")
    return value


def _group(parent: h5py.Group, name: str) -> h5py.Group:
    value = parent[name]
    if not isinstance(value, h5py.Group):
        raise ValueError(f"Expected an HDF5 group at {name}.")
    return value


def _contract(file: h5py.File, *, required: bool = True) -> dict[str, Any]:
    attrs = _group(file, "data").attrs
    encoded = attrs["keyboard_contract"] if required else attrs.get("keyboard_contract", "{}")
    if not isinstance(encoded, (str, bytes, bytearray)):
        raise ValueError("Keyboard contract must be a JSON string.")
    metadata = json.loads(encoded)
    if not isinstance(metadata, dict):
        raise ValueError("Keyboard contract must be a JSON object.")
    return metadata


def unique_transition_count(
    path: Path, *, max_capacity: int | None = None, capacity_unit: str = "stored_frame", sequence_length: int = 1
) -> int:
    """Count usable source transitions under the explicitly selected loader capacity unit."""
    if capacity_unit == "unique_transition":
        from isaaclab_neural.data.transition_index import build_transition_index

        return build_transition_index(
            path, max_capacity=max_capacity, sequence_length=sequence_length, build_windows=False
        ).unique_selected
    if capacity_unit != "stored_frame":
        raise ValueError(f"Unsupported capacity unit: {capacity_unit}.")
    with h5py.File(path, "r") as file:
        metadata = _contract(file, required=False)
        states = _dataset(file, "data/states")
        parts = metadata.get("training_parts")
        if parts:
            if len(set(parts)) != len(parts) or any(Path(part).name != part or part == path.name for part in parts):
                raise ValueError("Invalid training-view source paths.")
            horizon = states.shape[1]
            remaining = states.shape[0] if max_capacity is None else (max_capacity + horizon - 1) // horizon
            total = 0
            for part in parts:
                with h5py.File(path.parent / part, "r") as source:
                    source_states = _dataset(source, "data/states")
                    if "replay/frames/states" not in source or source_states.is_virtual:
                        raise ValueError("Training views must refer directly to recorded timelines.")
                    count = min(remaining, source_states.shape[0])
                if count:
                    total += unique_transition_count(path.parent / part, max_capacity=count * horizon)
                remaining -= count
            return total
        timeline = _dataset(file, "replay/frames/states")
        if states.ndim != 3 or timeline.ndim != 3 or states.shape[1] <= 0:
            raise ValueError(f"Invalid timeline dimensions: {path}.")
        clips, horizon = states.shape[:2]
        worlds, frames = timeline.shape[:2]
        if max_capacity is not None:
            if max_capacity <= 0:
                raise ValueError("Dataset max_capacity must be positive.")
            clips = min(clips, (max_capacity + horizon - 1) // horizon)
        ids = _dataset(file, "context/trajectories/source_env_id")[:clips]
        starts = _dataset(file, "context/trajectories/source_start_frame")[:clips]
    if ids.shape != (clips,) or starts.shape != (clips,) or ids.dtype.kind not in "iu" or starts.dtype.kind not in "iu":
        raise ValueError(f"Invalid source-world interval indices: {path}.")
    total, previous_world, previous_end = 0, -1, 0
    for world, start in sorted(zip(ids.tolist(), starts.tolist())):
        if not 0 <= world < worlds or not 0 <= start <= frames - horizon:
            raise ValueError(f"Training clip lies outside the recorded timeline: {path}.")
        if world != previous_world:
            previous_world, previous_end = world, 0
        end = start + horizon
        total += max(0, end - max(start, previous_end))
        previous_end = max(previous_end, end)
    return total


def validate_dataset_scale(
    root: Path,
    experiment: dict[str, Any],
    *,
    max_capacity: int | None = None,
    source_capacities: dict[str, int] | None = None,
) -> dict[str, int]:
    """Reject undersized large-run inputs, including overlapping clips and loader truncation."""
    minimums = experiment.get("minimum_unique_transitions")
    if not minimums:
        return {}
    totals = {split: 0 for split in minimums}
    for spec in experiment["dataset_specs"]:
        path = root / spec["filename"]
        capacity = (source_capacities or {}).get(spec["filename"], max_capacity)
        count = unique_transition_count(
            path,
            max_capacity=capacity,
            capacity_unit=experiment.get("dataset_capacity_unit", "stored_frame"),
            sequence_length=int(experiment.get("training_sequence_length", 1)),
        )
        minimum = int(spec.get("num_transitions", 0))
        if count < minimum:
            raise ValueError(f"Insufficient unique transitions in {path.name}: {count:,} < {minimum:,}.")
        if experiment.get("collection_mode"):
            with h5py.File(path, "r") as file:
                metadata = _contract(file)
            if metadata.get("collection_mode") != experiment["collection_mode"]:
                raise ValueError(f"Unexpected collection mode in {path.name}.")
        totals[spec["split"]] += count
    for split, minimum in minimums.items():
        if minimum <= 0 or totals[split] < minimum:
            raise ValueError(f"Insufficient unique {split} transitions: {totals[split]:,} < {minimum:,}.")
    return totals


def validate_basic_recordings(root: Path, experiment: dict[str, Any], records: dict, *, complete: bool = True) -> dict:
    """Check recorded data directly without re-simulating it or claiming replay equivalence."""
    from isaaclab_neural.contacts.keyboard_normal_force import validate_force_metadata
    from isaaclab_neural.data.keyboard import (
        KEYBOARD_MJW_MODEL_FIELDS,
        KEYBOARD_SCHEMA,
        read_keyboard_metadata,
        validate_keyboard_training_contracts,
    )
    from isaaclab_neural.generate.keyboard_determinism import workspace_contract

    expected_workspace = (
        workspace_contract() if experiment["collection"].get("bounded_deterministic_workspace", False) else None
    )
    contracts: dict[str, dict] = {"train": {}, "valid": {}}

    def check_finite(name: str, value: h5py.Group | h5py.Dataset) -> None:
        if not isinstance(value, h5py.Dataset) or value.dtype.kind not in "fc":
            return
        if not value.shape:
            finite = bool(np.isfinite(value[()]).all())
        else:
            row_bytes = max(1, int(np.prod(value.shape[1:])) * value.dtype.itemsize)
            step = max(1, (8 * 1024 * 1024) // row_bytes)
            finite = all(
                bool(np.isfinite(value[start : start + step]).all()) for start in range(0, value.shape[0], step)
            )
        if not finite:
            raise ValueError(f"Non-finite Keyboard dataset field: {name}.")

    for spec in experiment["dataset_specs"]:
        name, split = spec["filename"], spec["split"]
        record = records[name]
        path, receipt = root / name, root / (name + ".collect.json")
        if record.get("validation_mode") != "basic" or record.get("replay_performed") is not False:
            raise ValueError(f"Basic validation provenance is missing: {name}.")
        if record["sha256"] != file_sha256(path) or record["collect_sha256"] != file_sha256(receipt):
            raise ValueError(f"Keyboard data/collection report hash mismatch: {name}.")
        metadata = read_keyboard_metadata(path)
        if (
            metadata["schema"] != KEYBOARD_SCHEMA
            or metadata["keyboard_variant"] != 0
            or metadata["source_seed"] != spec["seed"]
            or metadata["source_num_envs"] != experiment["data_gen_num_envs"]
            or metadata.get("collection_mode") != experiment.get("collection_mode", "policy")
            or metadata.get("deterministic") != experiment["collection"].get("deterministic_physics", False)
            or metadata.get("deterministic_workspace") != expected_workspace
            or metadata.get("max_contact_tokens", 256) != experiment.get("max_contact_tokens", 256)
            or metadata.get("contact_filter", "none") != experiment.get("contact_filter", "none")
            or (
                expected_workspace is not None
                and metadata.get("mjwarp_model_context_fields") != list(KEYBOARD_MJW_MODEL_FIELDS)
            )
            or (experiment.get("policy_sha256") and metadata.get("policy_sha256") != experiment["policy_sha256"])
        ):
            raise ValueError(f"Keyboard basic collection contract mismatch: {name}.")
        report = json.loads(receipt.read_text())
        if experiment["collection"].get("persistent_workers", False):
            worker = report.get("resident_worker", {})
            if (
                worker != metadata.get("resident_worker")
                or worker.get("environment_creations") != 1
                or worker.get("policy_loads") != 1
                or worker.get("source_num_envs") != experiment["data_gen_num_envs"]
                or not isinstance(worker.get("pid"), int)
                or not isinstance(worker.get("sequence"), int)
            ):
                raise ValueError(f"Resident worker provenance is missing or inconsistent: {name}.")
        packing = report.get("native_contact_packing", {})
        if (
            report.get("mode") != "collect"
            or report.get("basic_validation_passed") is not True
            or any(
                packing.get(key, -1) != 0
                for key in ("dropped_contacts", "truncated_frames", "overflow_tokens_total", "overflow_frames")
            )
        ):
            raise ValueError(f"Keyboard basic collection report or contact capacity check failed: {name}.")
        with h5py.File(path, "r") as file:
            if expected_workspace is not None:
                for field in KEYBOARD_MJW_MODEL_FIELDS:
                    value = _dataset(file, "replay/contexts/mjwarp_model_" + field.replace(".", "_"))
                    if value.shape[:2] != (
                        experiment["data_gen_num_envs"],
                        experiment["collection"]["chunk_control_steps"],
                    ):
                        raise ValueError(f"Invalid MJWarp derived-model context layout: {name}/{field}.")
            data = _group(file, "data")
            validate_force_metadata(data)
            states = _dataset(file, "data/states")
            if (
                states.ndim != 3
                or states.shape[0] <= 0
                or states.shape[1] != experiment["trajectory_length"]
                or states.shape[2] != metadata["num_q"] + metadata["num_qd"]
                or _dataset(file, "data/next_states").shape != states.shape
                or report.get("trajectory_count") != states.shape[0]
                or report.get("total_recorded_frames_per_env") != _dataset(file, "replay/frames/states").shape[1]
            ):
                raise ValueError(f"Keyboard basic data shape mismatch: {name}.")
            done = _dataset(file, "data/terminated")[:].astype(bool) | _dataset(file, "data/truncated")[:].astype(bool)
            if done.shape != states.shape[:2] or done[:, :-1].any():
                raise ValueError(f"Keyboard training clip crosses a reset: {name}.")
            file.visititems(check_finite)
        unique_transition_count(
            path,
            capacity_unit=experiment.get("dataset_capacity_unit", "stored_frame"),
            sequence_length=int(experiment.get("training_sequence_length", 1)),
        )
        contracts[split][name] = metadata
    if complete:
        validate_keyboard_training_contracts(contracts["train"], contracts["valid"])
    return records


def validate_recordings(root: Path, experiment: dict[str, Any], records: dict, *, complete: bool = True) -> dict:
    """Validate data hashes, full native replay receipts, labels and independent validation seeds."""
    if validation_mode(experiment) == "basic":
        return validate_basic_recordings(root, experiment, records, complete=complete)
    from isaaclab_neural.contacts.keyboard_normal_force import validate_force_metadata
    from isaaclab_neural.data.keyboard import (
        KEYBOARD_REPLAY_LAYOUT,
        KEYBOARD_SCHEMA,
        read_keyboard_metadata,
        validate_keyboard_training_contracts,
    )

    specs = experiment["dataset_specs"]
    contracts: dict[str, dict] = {"train": {}, "valid": {}}
    for spec in specs:
        name, split = spec["filename"], spec["split"]
        path, receipt = root / name, root / (name + ".replay.json")
        entry = records[name]
        if entry["sha256"] != file_sha256(path) or entry["replay_sha256"] != file_sha256(receipt):
            raise ValueError(f"Keyboard data/receipt hash mismatch: {name}.")
        metadata = read_keyboard_metadata(path)
        if (
            metadata["schema"] != KEYBOARD_SCHEMA
            or metadata["keyboard_variant"] != 0
            or metadata.get("replay_layout") != KEYBOARD_REPLAY_LAYOUT
            or metadata["source_seed"] != spec["seed"]
            or metadata.get("deterministic") is not True
            or metadata.get("use_cuda_graph") is not False
            or metadata.get("contact_filter", "none") != experiment.get("contact_filter", "none")
            or (experiment.get("collection_mode") and metadata.get("collection_mode") != experiment["collection_mode"])
            or (experiment.get("policy_sha256") and metadata.get("policy_sha256") != experiment["policy_sha256"])
        ):
            raise ValueError(f"Keyboard collection contract mismatch: {name}.")
        with h5py.File(path, "r") as file:
            validate_force_metadata(_group(file, "data"))
            frames = _dataset(file, "replay/frames/states").shape[1]
            clips, horizon = _dataset(file, "data/states").shape[:2]
        report = json.loads(receipt.read_text())
        if (
            report.get("passed") is not True
            or report.get("mode") != "synchronous_open_loop_action_replay"
            or report.get("replay_horizon") != frames
            or report.get("trajectory_count") != clips
            or report.get("stored_horizon") != horizon
            or report.get("source_num_envs") != metadata["source_num_envs"]
            or not report.get("batches")
        ):
            raise ValueError(f"Keyboard replay receipt is incomplete or unsuccessful: {name}.")
        for batch in report["batches"]:
            if (
                batch.get("state_values_finite") is not True
                or batch.get("ongoing_episode_state_corrections") != 0
                or not 0 <= batch.get("articulation_root_pose_error", float("inf")) <= 1e-6
            ):
                raise ValueError(f"Keyboard replay state validation failed: {name}.")
            for metric, (threshold, maximum) in LIMITS.items():
                if (
                    not 0 <= report.get(threshold, float("inf")) <= maximum
                    or not 0 <= batch.get(metric, float("inf")) <= report[threshold]
                ):
                    raise ValueError(f"Keyboard replay tolerance failed: {name}/{metric}.")
            if any(
                batch.get(key) != 0 for key in ("key_pressed_mismatches", "typed_mismatches", "termination_mismatches")
            ):
                raise ValueError(f"Keyboard replay event mismatch: {name}.")
        contracts[split][name] = metadata
    if complete:
        validate_keyboard_training_contracts(contracts["train"], contracts["valid"])
    return records


def validate_bundle(root: Path, experiment: dict[str, Any]) -> dict[str, Any]:
    """Verify legacy bundles or portable chunked bundles, including their virtual training views."""
    from isaaclab_neural.data.keyboard import read_keyboard_metadata

    manifest = json.loads((root / MANIFEST).read_text())
    basic = validation_mode(experiment) == "basic"
    if basic != (manifest.get("schema") == BASIC_SCHEMA):
        raise ValueError("Keyboard bundle validation mode differs from the requested preset.")
    names = {spec["filename"] for spec in experiment["dataset_specs"]}
    bundle_files(experiment, root)
    if manifest.get("schema") == SCHEMA:
        if set(manifest.get("files", {})) != names:
            raise ValueError("Keyboard bundle manifest does not match the preset.")
        validate_dataset_scale(root, experiment)
        validate_recordings(root, experiment, manifest["files"])
    elif manifest.get("schema") in (CHUNKED_SCHEMA, BASIC_SCHEMA):
        stride = manifest.get("seed_stride", 0)
        if stride <= 0 or stride != experiment.get("collection", {}).get("seed_stride", stride):
            raise ValueError("Chunked cache collection seed schedule differs from the preset.")
        if set(manifest["groups"]) != names:
            raise ValueError("Chunked bundle groups differ from the preset.")
        specs = []
        for spec in experiment["dataset_specs"]:
            name = spec["filename"]
            parts = manifest["groups"][name]
            if not parts or file_sha256(root / name) != manifest["files"][name]["sha256"]:
                raise ValueError(f"Invalid or modified training view: {name}.")
            metadata = read_keyboard_metadata(root / name)
            seeds = [spec["seed"] + index * manifest["seed_stride"] for index in range(len(parts))]
            if metadata.get("training_parts") != parts or metadata.get("collection_seeds") != seeds:
                raise ValueError(f"Training view provenance mismatch: {name}.")
            with h5py.File(root / name, "r") as file:
                for group in ("data", "context/trajectories", "contacts/native_pairs"):
                    for dataset in _group(file, group).values():
                        if not isinstance(dataset, h5py.Dataset):
                            raise ValueError(f"Expected only datasets inside training-view group {group}.")
                        dataset_name = dataset.name
                        if dataset_name is None:
                            raise ValueError("Training-view datasets must have a name.")
                        if dataset.is_virtual and any(
                            link.file_name not in parts or link.dset_name.lstrip("/") != dataset_name.lstrip("/")
                            for link in dataset.virtual_sources()
                        ):
                            raise ValueError("Training view references an undeclared file or field.")
            specs.extend({**spec, "filename": part, "seed": seed} for part, seed in zip(parts, seeds))
        if set(manifest["files"]) != names | {spec["filename"] for spec in specs}:
            raise ValueError("Chunked manifest has missing or undeclared data.")
        validate_recordings(root, {**experiment, "dataset_specs": specs}, manifest["files"])
        validate_dataset_scale(root, experiment)
    else:
        raise ValueError("Unsupported Keyboard bundle schema.")
    return manifest


def collection_plan(experiment: dict[str, Any]) -> dict[str, Any]:
    """Validate a bounded collection plan and pin the explicitly supplied policy bytes."""
    bundle_files(experiment)
    collection = dict(experiment["collection"])
    contact_filter = experiment.get("contact_filter", "none")
    if contact_filter not in ("none", "solver_active", "solver_active_movable"):
        raise ValueError("Unsupported collection contact filter.")
    if contact_filter != "none":
        collection["contact_filter"] = contact_filter
    tokens = int(experiment.get("max_contact_tokens", 256))
    if tokens <= 0:
        raise ValueError("Keyboard contact-token capacity must be positive.")
    if tokens != 256:
        collection["max_contact_tokens"] = tokens
    checkpoint = Path(collection["checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Stage the SO101 policy checkpoint before collection: {checkpoint}")
    collection["checkpoint"] = str(checkpoint)
    collection["policy_sha256"] = file_sha256(checkpoint)
    indices = collection.get("gpu_indices")
    if indices is not None and (
        not isinstance(indices, list)
        or not indices
        or any(type(index) is not int or index < 0 for index in indices)
        or len(set(indices)) != len(indices)
        or collection.get("device", "cuda:0") != "cuda:0"
    ):
        raise ValueError("Collection gpu_indices must be distinct non-negative indices with logical device cuda:0.")
    basic = validation_mode(experiment) == "basic"
    worlds = int(experiment["data_gen_num_envs"])
    if worlds <= 0 or (not basic and worlds > 8):
        raise ValueError("Replay-qualified Keyboard collection is bounded to at most eight worlds per process.")
    if collection.get("persistent_workers", False) and not basic:
        raise ValueError("Resident collection requires basic validation without cloud replay.")
    if not basic and collection.get("deterministic_physics") is False:
        raise ValueError("Replay-qualified collection requires deterministic physics.")
    if collection.get("bounded_deterministic_workspace", False) and not collection.get(
        "deterministic_physics", not basic
    ):
        raise ValueError("Bounded deterministic workspaces require deterministic physics.")
    if (
        collection["chunk_control_steps"] < experiment["trajectory_length"] // 4
        or collection["max_chunks_per_group"] <= 0
        or collection["seed_stride"] <= 0
    ):
        raise ValueError("Invalid bounded Keyboard collection budget.")
    seeds = {
        split: {
            spec["seed"] + index * collection["seed_stride"]
            for spec in experiment["dataset_specs"]
            if spec["split"] == split
            for index in range(collection["max_chunks_per_group"])
        }
        for split in ("train", "valid")
    }
    if seeds["train"] & seeds["valid"]:
        raise ValueError("Collection seed schedules overlap between train and validation.")
    return collection


def _part_signature(experiment: dict, collection: dict, spec: dict, index: int) -> dict:
    signature = {
        **collection,
        "seed": spec["seed"] + index * collection["seed_stride"],
        "horizon": experiment["trajectory_length"],
    }
    if experiment.get("dataset_capacity_unit") == "unique_transition":
        signature["capacity_unit"] = "unique_transition"
        signature["training_sequence_length"] = experiment["training_sequence_length"]
    for name in ("max_chunks_per_group", "gpu_indices", "resume_from_input", "timeout_seconds"):
        signature.pop(name, None)
    return signature


def _collection_device_envs(collection: dict) -> list[dict[str, str] | None]:
    indices = collection.get("gpu_indices")
    if indices is None:
        return [None]
    import torch

    if max(indices) >= torch.cuda.device_count():
        raise ValueError("Collection GPU index exceeds visible CUDA devices.")
    environments = []
    for index in indices:
        gpu = str(torch.cuda.get_device_properties(index).uuid)
        if not gpu.startswith("GPU-"):
            gpu = "GPU-" + gpu
        environments.append({**os.environ, "CUDA_VISIBLE_DEVICES": gpu})
    if len({environment["CUDA_VISIBLE_DEVICES"] for environment in environments}) != len(environments):
        raise ValueError("Collection GPU indices do not resolve to distinct devices.")
    return environments


class ResidentCollector:
    """Keep one isolated simulator/policy process alive across sequential chunk requests."""

    def __init__(self, directory: Path, env: dict[str, str] | None, *, timeout: float = 900):
        self.directory = directory
        self.env = env
        self.timeout = timeout
        self.process: subprocess.Popen | None = None
        self.log = None
        self.sequence = 0
        self.settings: dict[str, str | None] | None = None

    def collect(self, command: list[str], log_path: Path) -> dict:
        """Collect one chunk with an exclusive resident process and return its acknowledgement."""
        settings = {
            key: command[command.index(key) + 1] if key in command else None
            for key in (
                "--num-envs",
                "--checkpoint",
                "--control-steps",
                "--horizon",
                "--device",
                "--contact-filter",
                "--max-contact-tokens",
            )
        }
        if self.settings is not None and settings != self.settings:
            raise ValueError("A resident worker cannot change its environment or policy configuration.")
        self.settings = settings
        if self.process is None:
            self.directory.mkdir(parents=True, exist_ok=False)
            self.log = (self.directory / "worker.log").open("x")
            launch = list(command)
            launch[launch.index("collect")] = "worker"
            launch += ["--worker-status", str((self.directory / "status.json").resolve())]
            self.process = subprocess.Popen(
                launch, stdin=subprocess.PIPE, stdout=self.log, stderr=subprocess.STDOUT, text=True, env=self.env
            )
        if self.process.poll() is not None or self.process.stdin is None:
            raise RuntimeError(f"Resident collector exited; see {self.directory / 'worker.log'}.")
        response = self.directory / f"response{self.sequence:06d}.json"
        request = {
            "sequence": self.sequence,
            "dataset": command[command.index("--dataset") + 1],
            "report": command[command.index("--report") + 1],
            "seed": int(command[command.index("--seed") + 1]),
            "response": str(response.resolve()),
        }
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        deadline = time.monotonic() + self.timeout
        while not response.exists():
            if self.process.poll() is not None:
                raise RuntimeError(f"Resident collector failed; see {self.directory / 'worker.log'}.")
            if time.monotonic() >= deadline:
                self.close()
                raise TimeoutError(f"Resident collection timed out; see {self.directory / 'worker.log'}.")
            time.sleep(0.1)
        result = json.loads(response.read_text())
        if (
            result.get("ok") is not True
            or any(result.get(key) != request[key] for key in ("sequence", "dataset", "seed"))
            or result.get("pid") != self.process.pid
            or result.get("source_num_envs") != int(command[command.index("--num-envs") + 1])
        ):
            raise RuntimeError(f"Invalid resident collection response; see {self.directory / 'worker.log'}.")
        with log_path.open("x") as stream:
            json.dump({**result, "worker_log": str(self.directory / "worker.log")}, stream, indent=2)
        self.sequence += 1
        return result

    def close(self) -> None:
        """Release simulator resources before training, including after worker failure."""
        if self.process is not None:
            if self.process.stdin is not None and not self.process.stdin.closed:
                with suppress(BrokenPipeError):
                    self.process.stdin.close()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        if self.log is not None:
            self.log.close()


def _collect_part(
    experiment: dict,
    collection: dict,
    spec: dict,
    index: int,
    root: Path,
    diagnostics: Path,
    env: dict[str, str] | None,
    worker: ResidentCollector | None = None,
) -> tuple[str, dict, int]:
    part = f"{Path(spec['filename']).stem}.part{index:05d}.hdf5"
    basic = validation_mode(experiment) == "basic"
    suffix = ".collect.json" if basic else ".replay.json"
    data, receipt, accepted = root / part, root / (part + suffix), root / (part + ".accepted.json")
    signature = _part_signature(experiment, collection, spec, index)
    if accepted.exists():
        record = json.loads(accepted.read_text())
        previous = dict(record["collection"])
        for key in ("max_chunks_per_group", "gpu_indices", "resume_from_input", "timeout_seconds"):
            previous.pop(key, None)
        if previous != signature:
            raise ValueError(f"Cannot resume a changed collection plan: {part}.")
    else:
        if data.exists() or receipt.exists():
            raise FileExistsError(f"Unverified partial collection retained at {data}; use a fresh dataset directory.")
        command = [
            sys.executable,
            "-u",
            "-m",
            "isaaclab_neural.generate.keyboard_dataset",
            "collect",
            "--collection-mode",
            "policy",
            "--checkpoint",
            collection["checkpoint"],
            "--num-envs",
            str(experiment["data_gen_num_envs"]),
            "--seed",
            str(signature["seed"]),
            "--deterministic-physics"
            if collection.get("deterministic_physics", not basic)
            else "--no-deterministic-physics",
            "--max-contact-tokens",
            str(experiment.get("max_contact_tokens", 256)),
            "--contact-filter",
            str(experiment.get("contact_filter", "none")),
            "--control-steps",
            str(collection["chunk_control_steps"]),
            "--horizon",
            str(experiment["trajectory_length"]),
            "--dataset",
            str(data.resolve()),
            "--report",
            str((receipt if basic else diagnostics / (part + ".collect.json")).resolve()),
            "--device",
            collection.get("device", "cuda:0"),
            "--visualizer",
            "none",
        ]
        if collection.get("bounded_deterministic_workspace", False):
            command.append("--bounded-deterministic-workspace")
        if worker is None:
            run_logged(
                command, diagnostics / (part + ".collect.log"), env=env, timeout=collection.get("timeout_seconds")
            )
        else:
            worker.collect(command, diagnostics / (part + ".collect.log"))
        before = file_sha256(data)
        if basic:
            record = {
                "sha256": before,
                "collect_sha256": file_sha256(receipt),
                "collection": signature,
                "validation_mode": "basic",
                "replay_performed": False,
            }
        else:
            replay = [
                sys.executable,
                "-u",
                "-m",
                "isaaclab_neural.generate.keyboard_dataset",
                "replay",
                "--dataset",
                str(data.resolve()),
                "--report",
                str(receipt.resolve()),
                "--device",
                collection.get("device", "cuda:0"),
                "--visualizer",
                "none",
            ]
            run_logged(replay, diagnostics / (part + ".replay.log"), env=env)
            if file_sha256(data) != before:
                raise ValueError("Recorded data changed during replay.")
            record = {"sha256": before, "replay_sha256": file_sha256(receipt), "collection": signature}
        if env is not None:
            record["execution"] = {"gpu_uuid": env["CUDA_VISIBLE_DEVICES"], "logical_device": "cuda:0"}
    part_spec = {**spec, "filename": part, "seed": signature["seed"]}
    validate_recordings(root, {**experiment, "dataset_specs": [part_spec]}, {part: record}, complete=False)
    if not accepted.exists():
        temporary = accepted.with_suffix(".pending")
        with temporary.open("x") as stream:
            json.dump(record, stream, indent=2)
        os.replace(temporary, accepted)
    count = (
        unique_transition_count(
            data, capacity_unit="unique_transition", sequence_length=int(experiment.get("training_sequence_length", 1))
        )
        if experiment.get("dataset_capacity_unit") == "unique_transition"
        else unique_transition_count(data)
    )
    return part, record, count


def collect_and_validate(experiment: dict[str, Any], root: Path, output_root: Path) -> None:
    """Schedule isolated collect/replay pairs, then publish only a fully qualified bundle."""
    from isaaclab_neural.data.hdf5 import create_virtual_trajectory_dataset
    from isaaclab_neural.data.keyboard import read_keyboard_metadata

    collection = collection_plan(experiment)
    experiment["policy_sha256"] = collection["policy_sha256"]
    if (root / MANIFEST).exists():
        validate_bundle(root, experiment)
        return
    environments = _collection_device_envs(collection)
    root.mkdir(parents=True, exist_ok=True)
    diagnostics = output_root / "collection"
    diagnostics.mkdir(parents=True, exist_ok=True)
    specs = experiment["dataset_specs"]
    targets = [int(s.get("num_transitions", experiment.get(s["split"] + "_transitions", 0))) for s in specs]
    if any(target <= 0 for target in targets):
        raise ValueError("Collection requires a positive per-group usable transition target.")
    totals, next_indices = [0] * len(specs), [0] * len(specs)
    records: list[dict[int, tuple[str, dict]]] = [{} for _ in specs]
    budget = collection["max_chunks_per_group"]

    def accept(group: int, index: int, result: tuple[str, dict, int]) -> None:
        part, record, count = result
        records[group][index] = (part, record)
        totals[group] += count
        tag = "basic-checked" if validation_mode(experiment) == "basic" else "qualified"
        print(
            f"[{tag}] {specs[group]['filename']}: {totals[group]:,}/{targets[group]:,}"
            f" unique transitions in {len(records[group])} chunks",
            flush=True,
        )

    for group, spec in enumerate(specs):
        while next_indices[group] < budget:
            index = next_indices[group]
            if not (root / f"{Path(spec['filename']).stem}.part{index:05d}.hdf5.accepted.json").exists():
                break
            accept(group, index, _collect_part(experiment, collection, spec, index, root, diagnostics, None))
            next_indices[group] += 1
    available = deque(range(len(environments)))
    pending = {}
    cursor = 0
    with ExitStack() as cleanup, ThreadPoolExecutor(max_workers=len(environments)) as pool:
        workers = []
        if collection.get("persistent_workers", False):
            for device, environment in enumerate(environments):
                worker = ResidentCollector(
                    diagnostics / f"worker{device:02d}", environment, timeout=collection.get("timeout_seconds", 900)
                )
                cleanup.callback(worker.close)
                workers.append(worker)
        while pending or any(total < target for total, target in zip(totals, targets)):
            while available:
                order = (
                    range(len(specs))
                    if len(environments) == 1
                    else ((cursor + offset) % len(specs) for offset in range(len(specs)))
                )
                group = next((i for i in order if totals[i] < targets[i] and next_indices[i] < budget), None)
                if group is None:
                    break
                index, device = next_indices[group], available.popleft()
                next_indices[group] += 1
                cursor = (group + 1) % len(specs)
                future = pool.submit(
                    _collect_part,
                    experiment,
                    collection,
                    specs[group],
                    index,
                    root,
                    diagnostics,
                    environments[device],
                    **({"worker": workers[device]} if workers else {}),
                )
                pending[future] = (group, index, device)
            if not pending:
                raise RuntimeError("Collection budget exhausted; no training or cache publication.")
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                group, index, device = pending.pop(future)
                accept(group, index, future.result())
                available.append(device)
    manifest: dict[str, Any] = {
        "schema": BASIC_SCHEMA if validation_mode(experiment) == "basic" else CHUNKED_SCHEMA,
        "seed_stride": collection["seed_stride"],
        "policy_sha256": collection["policy_sha256"],
        "groups": {},
        "files": {},
    }
    if collection.get("persistent_workers", False):
        if any(worker.process is not None and worker.process.returncode != 0 for worker in workers):
            raise RuntimeError("A resident collector did not exit cleanly; no cache publication or training.")
        manifest["workers"] = [
            json.loads((worker.directory / "status.json").read_text())
            for worker in workers
            if worker.process is not None
        ]
        if any(
            status.get("environment_creations") != 1
            or status.get("policy_loads") != 1
            or status.get("source_num_envs") != experiment["data_gen_num_envs"]
            for status in manifest["workers"]
        ):
            raise ValueError("Resident worker lifetime or environment count validation failed.")
    for group, spec in enumerate(specs):
        name = spec["filename"]
        if sorted(records[group]) != list(range(len(records[group]))):
            raise ValueError(f"Non-contiguous accepted chunk sequence: {name}.")
        parts = []
        for index in sorted(records[group]):
            part, record = records[group][index]
            parts.append(part)
            keys = (
                ("sha256", "collect_sha256", "validation_mode", "replay_performed")
                if validation_mode(experiment) == "basic"
                else ("sha256", "replay_sha256")
            )
            manifest["files"][part] = {key: record[key] for key in keys}
        view = root / name
        if not view.exists():
            create_virtual_trajectory_dataset(view, [root / part for part in parts])
            metadata = read_keyboard_metadata(root / parts[0])
            metadata.pop("replay_layout", None)
            metadata["training_parts"] = parts
            metadata["collection_seeds"] = [
                spec["seed"] + index * collection["seed_stride"] for index in range(len(parts))
            ]
            with h5py.File(view, "a") as file:
                _group(file, "data").attrs["keyboard_contract"] = json.dumps(metadata, sort_keys=True)
        manifest["groups"][name] = parts
        manifest["files"][name] = {"sha256": file_sha256(view)}
    with (root / MANIFEST).open("x") as stream:
        json.dump(manifest, stream, indent=2)
    validate_bundle(root, experiment)


def _recovery_receipt(experiment: dict[str, Any]) -> tuple[str, str]:
    return (
        (".collect.json", "collect_sha256")
        if validation_mode(experiment) == "basic"
        else (".replay.json", "replay_sha256")
    )


def validate_collection_resume(root: Path, experiment: dict[str, Any], manifest: dict | None = None) -> dict:
    """Verify a partial recovery checkpoint without treating it as a training-ready bundle."""
    collection = collection_plan(experiment)
    if manifest is None:
        manifest = json.loads((root / RESUME_MANIFEST).read_text())
    if not isinstance(manifest, dict) or manifest.get("schema") != RESUME_SCHEMA or not manifest.get("files"):
        raise ValueError("Invalid or empty collection recovery checkpoint.")
    if manifest.get("validation_mode", "full_replay") != validation_mode(experiment):
        raise ValueError("Collection recovery validation mode differs from the requested preset.")
    receipt_suffix, receipt_hash = _recovery_receipt(experiment)
    allowed = {
        f"{Path(spec['filename']).stem}.part{index:05d}.hdf5": (spec, index)
        for spec in experiment["dataset_specs"]
        for index in range(collection["max_chunks_per_group"])
    }
    records, specs = {}, []
    for part, hashes in manifest["files"].items():
        if part not in allowed:
            raise ValueError(f"Undeclared recovery part: {part}.")
        for suffix, key in (("", "sha256"), (receipt_suffix, receipt_hash), (".accepted.json", "accepted_sha256")):
            path = root / (part + suffix)
            if path.is_symlink() or file_sha256(path) != hashes[key]:
                raise ValueError(f"Collection recovery hash mismatch: {path.name}.")
        record = json.loads((root / (part + ".accepted.json")).read_text())
        spec, index = allowed[part]
        previous = dict(record["collection"])
        for key in ("max_chunks_per_group", "gpu_indices", "resume_from_input", "timeout_seconds"):
            previous.pop(key, None)
        signature = _part_signature(experiment, collection, spec, index)
        if previous != signature or any(record[key] != hashes[key] for key in ("sha256", receipt_hash)):
            raise ValueError(f"Collection recovery provenance mismatch: {part}.")
        records[part] = record
        specs.append({**spec, "filename": part, "seed": signature["seed"]})
    validate_recordings(
        root,
        {**experiment, "dataset_specs": specs, "policy_sha256": collection["policy_sha256"]},
        records,
        complete=False,
    )
    return manifest


def _copy_immutable(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def export_collection_resume(source: Path, output: Path, experiment: dict[str, Any]) -> dict:
    """Snapshot only accepted chunks into a fresh recovery directory; never copy an in-flight chunk."""
    collection = collection_plan(experiment)
    receipt_suffix, receipt_hash = _recovery_receipt(experiment)
    files = {}
    for spec in experiment["dataset_specs"]:
        for index in range(collection["max_chunks_per_group"]):
            part = f"{Path(spec['filename']).stem}.part{index:05d}.hdf5"
            accepted = source / (part + ".accepted.json")
            if accepted.is_file():
                record = json.loads(accepted.read_text())
                files[part] = {key: record[key] for key in ("sha256", receipt_hash)}
                files[part]["accepted_sha256"] = file_sha256(accepted)
    manifest = {
        "schema": RESUME_SCHEMA,
        "training_ready": False,
        "validation_mode": validation_mode(experiment),
        "files": files,
    }
    validate_collection_resume(source, experiment, manifest)
    output.mkdir(parents=True, exist_ok=False)
    for part in files:
        for suffix in ("", receipt_suffix, ".accepted.json"):
            _copy_immutable(source / (part + suffix), output / (part + suffix))
    with (output / RESUME_MANIFEST).open("x") as stream:
        json.dump(manifest, stream, indent=2)
    validate_collection_resume(output, experiment)
    return manifest


def restore_collection_resume(source: Path, output: Path, experiment: dict[str, Any]) -> int:
    """Restore verified partial chunks into a fresh local root, retaining all collection acceptance gates."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite a local collection directory: {output}")
    manifest = validate_collection_resume(source, experiment)
    receipt_suffix, _ = _recovery_receipt(experiment)
    output.mkdir(parents=True, exist_ok=False)
    for part in manifest["files"]:
        for suffix in ("", receipt_suffix, ".accepted.json"):
            _copy_immutable(source / (part + suffix), output / (part + suffix))
    with (output / RESUME_MANIFEST).open("x") as stream:
        json.dump(manifest, stream, indent=2)
    validate_collection_resume(output, experiment)
    return len(manifest["files"])


def prepare_bundle(source: Path, output: Path, preset: dict[str, Any], device: str) -> Path:
    """Copy explicit datasets, freshly replay each copy, and publish hash-bound receipts; never upload."""
    experiment = preset["experiment"]
    if not experiment.get("require_keyboard_replay"):
        raise ValueError("Data preparation requires a replay-qualified Keyboard preset.")
    versions = validate_runtime(experiment)
    bundle_files(experiment)
    validate_dataset_scale(source, experiment)
    root = output.resolve() / preset["workflow"]["dataset_subdir"] / experiment["env_name"]
    root.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {"schema": SCHEMA, "runtime_versions": versions, "files": {}}
    for spec in experiment["dataset_specs"]:
        name = spec["filename"]
        src, dst = source / name, root / name
        before = file_sha256(src)
        shutil.copy2(src, dst)
        if file_sha256(dst) != before:
            raise ValueError(f"Source changed while preparing {name}.")
        receipt = root / (name + ".replay.json")
        command = [
            sys.executable,
            "-u",
            "-m",
            "isaaclab_neural.generate.keyboard_dataset",
            "replay",
            "--dataset",
            str(dst),
            "--report",
            str(receipt),
            "--device",
            device,
            "--visualizer",
            "none",
        ]
        print(f"Verifying full replay: {name}", flush=True)
        run_logged(command, root / (name + ".replay.log"))
        if file_sha256(dst) != before:
            raise ValueError(f"Dataset changed during replay: {name}.")
        manifest["files"][name] = {"sha256": before, "replay_sha256": file_sha256(receipt)}
    manifest_path = root / MANIFEST
    with manifest_path.open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    validate_bundle(root, experiment)
    print(f"Replay-qualified bundle: {root}", flush=True)
    return root


def main() -> None:
    if TYPE_CHECKING:
        from .local_isaaclab import apply_sys_path
        from .preset import load_preset
    else:
        try:
            from .local_isaaclab import apply_sys_path
            from .preset import load_preset
        except ImportError:
            from local_isaaclab import apply_sys_path
            from preset import load_preset

    apply_sys_path(Path(__file__).resolve().parents[2])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "verify", "stage-policy"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--preset", default="so101_keyboard_resident_unique_smoke")
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    preset = load_preset(args.preset)
    if args.mode == "stage-policy":
        if args.checkpoint is None:
            parser.error("stage-policy requires --checkpoint")
        destination = Path(preset["experiment"]["collection"]["checkpoint"])
        digest = file_sha256(args.checkpoint)
        if destination.exists():
            if file_sha256(destination) != digest:
                raise FileExistsError(f"A different policy is already staged at {destination}.")
            print(f"Policy already staged and verified: {destination}; SHA-256: {digest}.")
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        with args.checkpoint.open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target)
        if file_sha256(destination) != digest:
            raise ValueError("Policy copy verification failed.")
        print(f"Staged policy: {destination}; SHA-256: {digest}. No upload was performed.")
        return
    if args.source_dir is None:
        parser.error("prepare and verify require --source-dir")
    if args.mode == "prepare":
        if args.output_dir is None:
            parser.error("prepare requires --output-dir")
        prepare_bundle(args.source_dir.resolve(), args.output_dir, preset, args.device)
    else:
        validate_bundle(args.source_dir.resolve(), preset["experiment"])
        print("Keyboard bundle verification passed.")


if __name__ == "__main__":
    main()
