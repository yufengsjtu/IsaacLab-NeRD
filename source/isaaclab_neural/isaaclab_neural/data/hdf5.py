# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HDF5 serialization helpers for NeRD rollout datasets."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import torch


def _to_numpy(value: torch.Tensor):
    """Convert a tensor-like rollout value to a CPU numpy array."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    raise TypeError(f"Expected rollout values to be torch.Tensor, got {type(value).__name__}.")


def write_rollouts_to_hdf5(
    dataset_path: str | Path,
    rollouts: Mapping,
    env_name: str,
    terrain_context: Mapping[str, Any] | None = None,
) -> None:
    """Serialize rollout tensors to a NeRD HDF5 trajectory dataset.

    Args:
        dataset_path: Output HDF5 path.
        rollouts: Rollout dictionary containing ``states``, ``next_states``,
            ``joint_f``, and a nested ``contacts`` dictionary.
        env_name: Environment name stored in dataset metadata.
        terrain_context: Optional deterministic terrain provenance.
    """
    dataset_path = Path(dataset_path).expanduser()
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _flatten_rollouts(rollouts)
    trajectory_context = _trajectory_context_arrays(rollouts, arrays["states"].shape[0])
    _validate_rollout_arrays(arrays)
    _validate_data_context_keys(arrays, trajectory_context)
    _validate_contact_token_identity(arrays, trajectory_context)
    _validate_replay_context(arrays, terrain_context)

    with h5py.File(dataset_path, "w") as dataset_file:
        data_group = dataset_file.create_group("data")
        for name, data in arrays.items():
            data_group.create_dataset(
                name=name,
                data=data,
                maxshape=(None, *data.shape[1:]),
                chunks=True,
            )
        _create_context_groups(dataset_file, trajectory_context, terrain_context)
        data_group.attrs["env"] = env_name
        data_group.attrs["mode"] = "trajectory"
        _update_metadata(data_group)


def append_rollouts_to_hdf5(
    dataset_path: str | Path,
    rollouts: Mapping,
    env_name: str,
    terrain_context: Mapping[str, Any] | None = None,
) -> None:
    """Append rollout tensors to a NeRD HDF5 trajectory dataset.

    The appended file has the same schema as :func:`write_rollouts_to_hdf5`,
    but datasets are resizable along the trajectory dimension. This avoids
    keeping very large data-generation runs in memory until the end.
    """
    dataset_path = Path(dataset_path).expanduser()
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _flatten_rollouts(rollouts)
    trajectory_context = _trajectory_context_arrays(rollouts, arrays["states"].shape[0])
    _validate_rollout_arrays(arrays)
    _validate_data_context_keys(arrays, trajectory_context)
    _validate_contact_token_identity(arrays, trajectory_context)
    _validate_replay_context(arrays, terrain_context)

    with h5py.File(dataset_path, "a") as dataset_file:
        data_group = dataset_file.require_group("data")
        is_new = len(data_group) == 0
        _validate_append_target(
            data_group,
            arrays,
            env_name=env_name,
            is_new=is_new,
        )
        _validate_context_target(
            dataset_file,
            trajectory_context,
            terrain_context,
            is_new=is_new,
        )
        if is_new:
            _create_appendable_datasets(data_group, arrays)
            _create_context_groups(dataset_file, trajectory_context, terrain_context)
            data_group.attrs["env"] = env_name
            data_group.attrs["mode"] = "trajectory"
        else:
            trajectory_group = (
                cast(h5py.Group, dataset_file["context"]["trajectories"]) if "context" in dataset_file else None
            )
            _append_arrays_atomically(data_group, arrays, trajectory_group, trajectory_context)
        _update_metadata(data_group)


def _flatten_rollouts(rollouts: Mapping) -> dict[str, Any]:
    """Flatten rollout tensors and nested contact tensors into numpy arrays."""
    arrays = {}
    for key, value in rollouts.items():
        if key == "trajectory_context":
            continue
        if isinstance(value, Mapping):
            for sub_key, sub_value in value.items():
                if sub_key in arrays:
                    raise ValueError(f"Duplicate rollout dataset key: {sub_key!r}.")
                arrays[sub_key] = _to_numpy(sub_value)
        else:
            if key in arrays:
                raise ValueError(f"Duplicate rollout dataset key: {key!r}.")
            arrays[key] = _to_numpy(value)
    return arrays


def _trajectory_context_arrays(rollouts: Mapping, trajectory_count: int) -> dict[str, Any]:
    """Convert optional per-trajectory context tensors to numpy arrays."""
    context = rollouts.get("trajectory_context")
    if context is None:
        return {}
    if not isinstance(context, Mapping):
        raise TypeError("trajectory_context must be a mapping.")
    arrays = {str(key): _to_numpy(value) for key, value in context.items()}
    for name, data in arrays.items():
        if data.ndim < 1 or data.shape[0] != trajectory_count:
            raise ValueError(
                f"Trajectory context {name!r} must start with trajectory count {trajectory_count}, got {data.shape}."
            )
    return arrays


def _validate_data_context_keys(arrays: Mapping[str, Any], trajectory_context: Mapping[str, Any]) -> None:
    """Reject ambiguous fields duplicated between rollout data and trajectory context."""
    collisions = sorted(set(arrays).intersection(trajectory_context))
    if collisions:
        raise ValueError(f"Rollout data and trajectory context keys overlap: {collisions}.")


def _validate_contact_token_identity(
    arrays: Mapping[str, Any],
    trajectory_context: Mapping[str, Any],
) -> None:
    """Validate packed token identity shapes, padding, and row-world ownership."""
    if "contact_tokens" not in arrays:
        return
    tokens = arrays["contact_tokens"]
    expected_shape = tokens.shape[:-1]
    body_ids = arrays["contact_token_body_ids"]
    world_ids = arrays["contact_token_world_ids"]
    if body_ids.shape != expected_shape or world_ids.shape != expected_shape:
        raise ValueError(
            "Contact-token identity fields must have shape "
            f"{expected_shape}, got body_ids={body_ids.shape}, world_ids={world_ids.shape}."
        )
    valid = tokens[..., 0] > 0.5
    if np.any(valid & ((body_ids < 0) | (world_ids < 0))):
        raise ValueError("Every valid contact token must have nonnegative owner body and world ids.")
    if np.any(~valid & ((body_ids != -1) | (world_ids != -1))):
        raise ValueError("Every padded contact token must use owner body and world id -1.")

    required_context = {"state_world_id", "root_world_id", "contact_world_id"}
    missing_context = sorted(required_context - set(trajectory_context))
    if missing_context:
        raise ValueError(f"Contact-token rollouts are missing trajectory world context: {missing_context}.")
    state_world_ids = trajectory_context["state_world_id"]
    root_world_ids = trajectory_context["root_world_id"]
    contact_world_ids = trajectory_context["contact_world_id"]
    if not np.array_equal(state_world_ids, root_world_ids) or not np.array_equal(
        state_world_ids, contact_world_ids
    ):
        raise ValueError("State, root, and contact trajectory rows must map to the same Newton worlds.")
    expected_world_ids = np.broadcast_to(contact_world_ids[:, None, None], world_ids.shape)
    if np.any(valid & (world_ids != expected_world_ids)):
        raise ValueError("Contact-token owner world ids do not match their trajectory rows.")


def _validate_replay_context(
    arrays: Mapping[str, Any],
    terrain_context: Mapping[str, Any] | None,
) -> None:
    """Require maximal root kinematics for terrain-context replay."""
    if terrain_context is None:
        return
    missing = sorted({"root_body_q", "root_body_qd"} - set(arrays))
    if missing:
        raise ValueError(f"Terrain-context rollouts are missing root kinematics: {missing}.")


def _validate_rollout_arrays(arrays: Mapping[str, Any]) -> None:
    """Validate required fields and common trajectory dimensions."""
    required = {"states", "next_states", "joint_f"}
    if "contact_tokens" in arrays:
        # Replay needs maximal root kinematics in addition to generalized states.
        required.update(
            {
                "contact_tokens",
                "contact_token_body_ids",
                "contact_token_world_ids",
                "root_body_q",
                "root_body_qd",
                "gravity_dir",
            }
        )
    else:
        required.update({"contact_depths", "contact_masks"})
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"Rollout data is missing required fields: {missing}.")
    for name in ("contact_token_body_ids", "contact_token_world_ids"):
        if name in arrays and not np.issubdtype(arrays[name].dtype, np.integer):
            raise ValueError(f"Rollout field {name!r} must use an integer dtype, got {arrays[name].dtype}.")

    states = arrays["states"]
    if states.ndim < 3:
        raise ValueError(f"states must have shape [trajectories, steps, features], got {states.shape}.")
    trajectory_shape = states.shape[:2]
    for name, data in arrays.items():
        # Per-step scalars (e.g. contact_token_overflow) are [N, T]; all other fields are [N, T, ...].
        matches_trajectory = data.shape[:2] == trajectory_shape and (
            data.ndim >= 3 or data.shape == trajectory_shape
        )
        if not matches_trajectory:
            raise ValueError(
                f"Rollout field {name!r} must start with trajectory shape {trajectory_shape} "
                f"(ndim>=3, or exactly {trajectory_shape} for per-step scalars); got shape={data.shape}."
            )


def _validate_append_target(
    data_group: h5py.Group,
    arrays: Mapping[str, Any],
    *,
    env_name: str,
    is_new: bool,
) -> None:
    """Validate append compatibility before mutating any HDF5 dataset."""
    if is_new:
        return
    if data_group.attrs.get("env") != env_name:
        raise ValueError(f"Dataset env mismatch: existing={data_group.attrs.get('env')!r}, new={env_name!r}.")
    if data_group.attrs.get("mode") != "trajectory":
        raise ValueError(f"Cannot append trajectories to dataset mode {data_group.attrs.get('mode')!r}.")
    if "contact_tokens" in arrays:
        token_frame = data_group.attrs.get("contact_token_frame")
        identity_schema = data_group.attrs.get("contact_identity_schema")
        if token_frame != "world_v1" or identity_schema != "world_owner_v1":
            raise ValueError(
                "Cannot append contact tokens unless contact_token_frame='world_v1' and "
                "contact_identity_schema='world_owner_v1'."
            )
    if set(data_group.keys()) != set(arrays):
        missing = sorted(set(data_group.keys()) - set(arrays))
        extra = sorted(set(arrays) - set(data_group.keys()))
        raise ValueError(f"Dataset key mismatch: missing={missing}, extra={extra}.")

    for name, data in arrays.items():
        dataset = cast(h5py.Dataset, data_group[name])
        if dataset.chunks is None or dataset.maxshape is None or dataset.maxshape[0] is not None:
            raise ValueError(f"Dataset field {name!r} is not appendable.")
        if dataset.shape[1:] != data.shape[1:]:
            raise ValueError(
                f"Dataset shape mismatch for {name!r}: existing={dataset.shape[1:]}, new={data.shape[1:]}."
            )
        if dataset.dtype != data.dtype:
            raise ValueError(f"Dataset dtype mismatch for {name!r}: existing={dataset.dtype}, new={data.dtype}.")


def _validate_context_target(
    dataset_file: h5py.File,
    trajectory_context: Mapping[str, Any],
    terrain_context: Mapping[str, Any] | None,
    *,
    is_new: bool,
) -> None:
    """Validate global and per-trajectory context before appending."""
    if is_new:
        return
    if "context" not in dataset_file or "trajectories" not in dataset_file["context"]:
        if trajectory_context or terrain_context is not None:
            raise ValueError("Existing dataset has no context groups.")
        return

    context_group = cast(h5py.Group, dataset_file["context"])
    trajectory_group = cast(h5py.Group, context_group["trajectories"])
    if set(trajectory_group.keys()) != set(trajectory_context):
        raise ValueError("Trajectory context keys do not match the existing dataset.")
    for name, data in trajectory_context.items():
        dataset = cast(h5py.Dataset, trajectory_group[name])
        if dataset.shape[1:] != data.shape[1:] or dataset.dtype != data.dtype:
            raise ValueError(f"Trajectory context {name!r} is incompatible with the existing dataset.")

    if terrain_context is None:
        if "terrain" in context_group:
            raise ValueError("Terrain context is required when appending to this dataset.")
        return
    if "terrain" not in context_group:
        raise ValueError("Existing dataset has no terrain context.")
    terrain_group = cast(h5py.Group, context_group["terrain"])
    existing = {
        key: (
            value.decode("utf-8")
            if isinstance(value, bytes)
            else value.item()
            if isinstance(value, np.generic)
            else value
        )
        for key, value in terrain_group.attrs.items()
    }
    if existing != dict(terrain_context):
        raise ValueError("Terrain context does not match the existing dataset.")


def _create_context_groups(
    dataset_file: h5py.File,
    trajectory_context: Mapping[str, Any],
    terrain_context: Mapping[str, Any] | None,
) -> None:
    """Create global and per-trajectory context groups."""
    context_group = dataset_file.create_group("context")
    trajectory_group = context_group.create_group("trajectories")
    for name, data in trajectory_context.items():
        trajectory_group.create_dataset(name=name, data=data, maxshape=(None, *data.shape[1:]), chunks=True)
    if terrain_context is not None:
        terrain_group = context_group.create_group("terrain")
        for name, value in terrain_context.items():
            terrain_group.attrs[name] = value


def _create_appendable_datasets(data_group: h5py.Group, arrays: Mapping[str, Any]) -> None:
    """Create a new set of appendable datasets, cleaning up on failure."""
    created = []
    try:
        for name, data in arrays.items():
            data_group.create_dataset(
                name=name,
                data=data,
                maxshape=(None, *data.shape[1:]),
                chunks=True,
            )
            created.append(name)
    except Exception:
        for name in created:
            del data_group[name]
        raise


def _append_arrays_atomically(
    data_group: h5py.Group,
    arrays: Mapping[str, Any],
    trajectory_group: h5py.Group | None,
    trajectory_context: Mapping[str, Any],
) -> None:
    """Append all fields and roll back their sizes if any write fails."""
    old_count = cast(h5py.Dataset, data_group["states"]).shape[0]
    append_count = arrays["states"].shape[0]
    resized: list[h5py.Dataset] = []
    try:
        datasets_and_arrays = [
            *((cast(h5py.Dataset, data_group[name]), data) for name, data in arrays.items()),
            *(
                (cast(h5py.Dataset, trajectory_group[name]), data)
                for name, data in trajectory_context.items()
                if trajectory_group is not None
            ),
        ]
        for dataset, _data in datasets_and_arrays:
            dataset.resize((old_count + append_count, *dataset.shape[1:]))
            resized.append(dataset)
        for dataset, data in datasets_and_arrays:
            dataset[old_count : old_count + append_count] = data
    except Exception:
        for dataset in resized:
            dataset.resize((old_count, *dataset.shape[1:]))
        raise


def _update_metadata(data_group: h5py.Group) -> None:
    """Update aggregate metadata from successfully written datasets."""
    states = cast(h5py.Dataset, data_group["states"])
    data_group.attrs["total_trajectories"] = states.shape[0]
    data_group.attrs["total_transitions"] = states.shape[0] * states.shape[1]
    data_group.attrs["state_dim"] = states.shape[-1]
    data_group.attrs["next_state_dim"] = cast(h5py.Dataset, data_group["next_states"]).shape[-1]
    if "contact_depths" in data_group:
        data_group.attrs["num_contacts_per_env"] = cast(h5py.Dataset, data_group["contact_depths"]).shape[-1]
    if "contact_tokens" in data_group:
        tokens = cast(h5py.Dataset, data_group["contact_tokens"])
        # Token capacity is max_contact_tokens; keep num_contacts_per_env only if already set
        # (e.g. from the generator config) so it is not silently overwritten by K.
        data_group.attrs["max_contact_tokens"] = tokens.shape[-2]
        data_group.attrs["contact_token_dim"] = tokens.shape[-1]
        data_group.attrs["contact_representation"] = "contact_tokens"
        data_group.attrs["contact_identity_schema"] = "world_owner_v1"
        data_group.attrs["contact_token_frame"] = "world_v1"
        if "num_contacts_per_env" not in data_group.attrs:
            data_group.attrs["num_contacts_per_env"] = tokens.shape[-2]
    data_group.attrs["joint_f_dim"] = cast(h5py.Dataset, data_group["joint_f"]).shape[-1]
    if "actions" in data_group:
        data_group.attrs["action_dim"] = cast(h5py.Dataset, data_group["actions"]).shape[-1]
