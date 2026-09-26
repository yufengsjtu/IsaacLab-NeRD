# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-safe fixed SO101 keyboard datasets using the native-contact HDF5 schema."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import torch

from .hdf5 import write_rollouts_to_hdf5

KEYBOARD_SCHEMA = "so101_keyboard_fixed_v1"
KEYBOARD_TASK = "IsaacContrib-Keyboard-SO101"
KEYBOARD_REPLAY_LAYOUT = "synchronous_world_timeline_v1"
KEYBOARD_MJW_MODEL_FIELDS = ("actuator_acc0", "body_invweight0", "dof_invweight0", "jnt_solref", "stat.meaninertia")


def keyboard_clip_indices(done: torch.Tensor, *, horizon: int, decimation: int) -> list[tuple[int, int]]:
    """Select full physics-frame clips without an internal episode reset.

    Args:
        done: Post-control terminal flags, shape ``[worlds, physics_frames]``.
        horizon: Physics frames per clip. The final frame may be terminal.
        decimation: Physics frames per action/MDP update.

    Returns:
        Source ``(world, start_frame)`` pairs, each beginning at a control boundary.
    """
    if done.ndim != 2 or done.dtype != torch.bool:
        raise ValueError("done must be a two-dimensional boolean tensor.")
    if decimation <= 0 or horizon <= 0 or horizon % decimation:
        raise ValueError("horizon must be a positive multiple of decimation.")
    frames = done.shape[1]
    boundaries = torch.arange(frames, device=done.device).remainder(decimation) == decimation - 1
    if bool(done[:, ~boundaries].any()):
        raise ValueError("Terminal frames must coincide with control boundaries.")
    clips = []
    for world in range(done.shape[0]):
        start = 0
        for terminal in [*torch.nonzero(done[world], as_tuple=False).flatten().tolist(), frames - 1]:
            end = int(terminal) + 1
            segment_start = start
            while start + horizon <= end:
                clips.append((world, start))
                start += horizon
            if bool(done[world, terminal]) and end - segment_start >= horizon and start != end:
                clips.append((world, end - horizon))
            start = end
    return clips


def keyboard_replay_batches(
    source_worlds: torch.Tensor, selected: list[int], num_worlds: int
) -> list[tuple[list[int], list[int]]]:
    """Schedule clips in their original world slots; filler rows are not scored.

    Fixed-base Newton/MJWarp scenes retain world-dependent transforms. Reordering
    clips into arbitrary worlds is not a validated replay operation.
    """
    if source_worlds.ndim != 1 or source_worlds.dtype not in (torch.int32, torch.int64):
        raise ValueError("Source worlds must be an integer vector.")
    if num_worlds <= 0 or not selected or len(set(selected)) != len(selected):
        raise ValueError("Replay selection must be non-empty and unique.")
    worlds = source_worlds.tolist()
    if min(selected) < 0 or max(selected) >= len(worlds):
        raise ValueError("Replay trajectory index outside the dataset.")
    if set(worlds) != set(range(num_worlds)):
        raise ValueError("Replay requires at least one clip for every source world.")
    queues = [[index for index in selected if worlds[index] == world] for world in range(num_worlds)]
    fillers = [worlds.index(world) for world in range(num_worlds)]
    result = []
    for step in range(max(map(len, queues))):
        active = [world for world, queue in enumerate(queues) if step < len(queue)]
        batch = [queue[step] if step < len(queue) else fillers[world] for world, queue in enumerate(queues)]
        result.append((batch, active))
    return result


def validate_keyboard_metadata(metadata: Mapping[str, Any]) -> None:
    """Reject incompatible timing, task, coordinate, and force-label conventions."""
    if metadata.get("schema") != KEYBOARD_SCHEMA:
        raise ValueError("Incompatible keyboard dataset schema.")
    contact_filter = metadata.get("contact_filter", "none")
    if contact_filter not in ("none", "solver_active", "solver_active_movable"):
        raise ValueError("Unsupported recorded contact filter.")
    if contact_filter == "solver_active_movable":
        slots = metadata.get("contact_fixed_body_slots")
        body_count = sum(len(names) for names in metadata["body_names"].values())
        if (
            not isinstance(slots, list)
            or any(type(slot) is not int or not 0 <= slot < body_count for slot in slots)
            or len(set(slots)) != len(slots)
        ):
            raise ValueError("Movable-filter recordings require valid fixed-body mobility slots.")
    expected = {
        "task": KEYBOARD_TASK,
        "force_frame": "world",
        "force_units": "N",
        "force_timing": "post_solve_before_reset",
    }
    if any(metadata.get(name) != value for name, value in expected.items()):
        raise ValueError("Incompatible keyboard dataset contract.")
    if float(metadata["physics_dt"]) <= 0 or int(metadata["decimation"]) <= 0:
        raise ValueError("Keyboard timing must be positive.")
    names = metadata["force_body_names"]
    if not names or len(set(names)) != len(names):
        raise ValueError("Force body names must be non-empty and unique.")
    indices = metadata["key_q_indices"]
    if len(indices) != 108 or len(set(indices)) != 108:
        raise ValueError("Expected 108 unique key position coordinates.")
    if min(indices) < 0 or max(indices) >= int(metadata["num_q"]):
        raise ValueError("Key position coordinate is outside the state layout.")
    json.dumps(dict(metadata), allow_nan=False)


def validate_keyboard_replay(
    replay: Mapping[str, Any], rollouts: Mapping[str, Any], metadata: Mapping[str, Any]
) -> None:
    """Validate the synchronous world timeline and its exact correspondence to training clips."""
    frames, contexts = replay["frames"], replay["contexts"]
    worlds, length = frames["states"].shape[:2]
    decimation = int(metadata["decimation"])
    if worlds != metadata["source_num_envs"] or length <= 0 or length % decimation:
        raise ValueError("Invalid synchronous replay timeline dimensions.")
    required_context = {
        "reset_state",
        "episode_length",
        "last_action",
        "previous_action",
        "env_origin",
        "source_env_id",
    }
    if "mjwarp_model_context_fields" in metadata:
        if metadata["mjwarp_model_context_fields"] != list(KEYBOARD_MJW_MODEL_FIELDS):
            raise ValueError("Unsupported MJWarp derived-model context contract.")
        required_context.update("mjwarp_model_" + name.replace(".", "_") for name in KEYBOARD_MJW_MODEL_FIELDS)
    if not required_context <= contexts.keys() or not {"control_boundary", "typed_after"} <= frames.keys():
        raise ValueError("Incomplete synchronous replay timeline.")
    for values, prefix in ((frames, (worlds, length)), (contexts, (worlds, length // decimation))):
        for name, value in values.items():
            if tuple(value.shape[:2]) != prefix or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Invalid synchronous replay timeline field: {name}.")
    source_ids = torch.arange(worlds)[:, None].expand(worlds, length // decimation)
    if not torch.equal(contexts["source_env_id"], source_ids):
        raise ValueError("Synchronous replay timeline must preserve world identities.")
    boundary = torch.arange(length).remainder(decimation) == decimation - 1
    if not torch.equal(frames["control_boundary"], boundary.expand(worlds, length)):
        raise ValueError("Invalid synchronous replay timeline control boundaries.")
    actions = frames["actions"].unflatten(1, (length // decimation, decimation))
    if not torch.equal(actions, actions[:, :, :1].expand_as(actions)):
        raise ValueError("Synchronous replay timeline actions changed inside a control interval.")
    clip_context = rollouts["trajectory_context"]
    horizon = rollouts["states"].shape[1]
    source_worlds, starts = clip_context["source_env_id"], clip_context["source_start_frame"]
    if bool(((source_worlds < 0) | (source_worlds >= worlds)).any()) or bool(
        ((starts < 0) | (starts + horizon > length) | (starts % decimation != 0)).any()
    ):
        raise ValueError("Training clip lies outside the synchronous replay timeline.")
    for name, value in rollouts.items():
        if isinstance(value, torch.Tensor):
            if name not in frames:
                raise ValueError(f"Synchronous replay timeline is missing {name}.")
            selected = torch.stack(
                [frames[name][world, start : start + horizon] for world, start in zip(source_worlds, starts)]
            )
            if not torch.equal(value, selected):
                raise ValueError(f"Training clips differ from the synchronous replay timeline: {name}.")
    for name in clip_context.keys() & contexts.keys():
        selected = contexts[name][source_worlds, starts // decimation]
        if not torch.equal(clip_context[name], selected):
            raise ValueError(f"Training context differs from the synchronous replay timeline: {name}.")


def write_keyboard_dataset(
    path: str | Path,
    rollouts: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    replay: Mapping[str, Any] | None = None,
) -> None:
    """Publish a validated dataset atomically without overwriting an existing file.

    State, force, action and key labels use the common ``[clips, frames, ...]``
    layout. Contact pairs use the existing untruncated native CSR representation.
    Normal-force labels cover the robot sensor bodies, not all articulated bodies.
    """
    path = Path(path).expanduser()
    if path.exists():
        raise FileExistsError(path)
    validate_keyboard_metadata(metadata)
    required = {
        "states",
        "next_states",
        "joint_f",
        "actions",
        "robot_net_normal_forces_w",
        "key_pressed",
        "terminated",
        "truncated",
        "native_contacts",
        "trajectory_context",
    }
    if missing := required - rollouts.keys():
        raise ValueError(f"Missing keyboard rollout fields: {sorted(missing)}.")
    states = rollouts["states"]
    if states.ndim != 3 or states.shape[0] == 0 or states.shape[-1] != metadata["num_q"] + metadata["num_qd"]:
        raise ValueError("Keyboard state dimensions do not match metadata.")
    prefix = states.shape[:2]
    expected_shapes = {
        "next_states": states.shape,
        "joint_f": (*prefix, metadata["num_qd"]),
        "actions": (*prefix, 6),
        "robot_net_normal_forces_w": (*prefix, len(metadata["force_body_names"]), 3),
        "key_pressed": (*prefix, 108),
        "terminated": prefix,
        "truncated": prefix,
    }
    for name, shape in expected_shapes.items():
        if tuple(rollouts[name].shape) != tuple(shape):
            raise ValueError(f"Invalid {name} shape: expected {shape}.")
    for name in ("key_pressed", "terminated", "truncated"):
        if rollouts[name].dtype != torch.bool:
            raise ValueError(f"{name} must be boolean.")
    for name, value in rollouts.items():
        values = value.values() if isinstance(value, Mapping) else (value,)
        if any(not bool(torch.isfinite(tensor).all()) for tensor in values):
            raise ValueError(f"Non-finite keyboard field: {name}.")
    done = rollouts["terminated"] | rollouts["truncated"]
    if bool(done[:, :-1].any()):
        raise ValueError("A keyboard training clip must not cross an episode reset.")
    keyboard_clip_indices(done, horizon=prefix[1], decimation=int(metadata["decimation"]))
    if metadata.get("replay_layout") is not None or replay is not None:
        if metadata.get("replay_layout") != KEYBOARD_REPLAY_LAYOUT or replay is None:
            raise ValueError("Synchronous replay timeline and metadata must be provided together.")
        validate_keyboard_replay(replay, rollouts, metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".hdf5", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        write_rollouts_to_hdf5(temporary, rollouts, "SO101-Keyboard-Fixed")
        with h5py.File(temporary, "a") as file:
            file["data"].attrs["keyboard_contract"] = json.dumps(dict(metadata), sort_keys=True, allow_nan=False)
            if replay is not None:
                for group_name in ("frames", "contexts"):
                    group = file.create_group("replay/" + group_name)
                    for name, value in replay[group_name].items():
                        group.create_dataset(name, data=value.detach().cpu().numpy(), chunks=True)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_keyboard_training_contracts(
    train: Mapping[str, dict[str, Any]], valid: Mapping[str, dict[str, Any]]
) -> None:
    """Validate shared tensor semantics, independent validation seeds, and the fixed keyboard identity."""
    if not train:
        raise ValueError("Keyboard training requires at least one source.")
    reference = next(iter(train.values()))
    seeds = {
        seed for metadata in train.values() for seed in metadata.get("collection_seeds", [metadata["source_seed"]])
    }
    if any(
        seeds.intersection(metadata.get("collection_seeds", [metadata["source_seed"]])) for metadata in valid.values()
    ):
        raise ValueError("Keyboard validation requires independently collected seeds.")
    for metadata in (*train.values(), *valid.values()):
        validate_keyboard_metadata(metadata)
        if metadata.get("contact_filter", "none") != reference.get("contact_filter", "none") or metadata.get(
            "contact_fixed_body_slots"
        ) != reference.get("contact_fixed_body_slots"):
            raise ValueError("Keyboard dataset contact filter or mobility contract mismatch.")
        for key in (
            "schema",
            "task",
            "physics_dt",
            "decimation",
            "num_q",
            "num_qd",
            "key_q_indices",
            "joint_names",
            "body_names",
            "asset_order",
            "force_body_names",
            "force_frame",
            "force_units",
            "force_timing",
        ):
            if metadata.get(key) != reference.get(key):
                raise ValueError(f"Keyboard dataset contract mismatch: {key}.")
        if (
            metadata["keyboard_variant"] != reference["keyboard_variant"]
            or metadata["physics_parameters"] != reference["physics_parameters"]
        ):
            raise ValueError("Fixed keyboard training requires matching variants and physics parameters.")


def read_keyboard_metadata(path: str | Path) -> dict[str, Any]:
    """Read and validate fixed-keyboard metadata without unpickling Python objects."""
    with h5py.File(Path(path).expanduser(), "r") as file:
        data = file.get("data")
        if not isinstance(data, h5py.Group) or "keyboard_contract" not in data.attrs:
            raise ValueError("Dataset has no keyboard replay contract.")
        encoded = data.attrs["keyboard_contract"]
        if not isinstance(encoded, (str, bytes, bytearray)):
            raise ValueError("Keyboard replay contract must be a JSON string.")
        metadata = json.loads(encoded)
    if not isinstance(metadata, dict):
        raise ValueError("Keyboard replay contract must be a JSON object.")
    validate_keyboard_metadata(metadata)
    if metadata.get("training_parts"):
        for name in metadata["training_parts"]:
            if Path(name).name != name or not (Path(path).expanduser().parent / name).is_file():
                raise ValueError(f"Training view source is missing or outside its bundle: {name}.")
    return metadata
