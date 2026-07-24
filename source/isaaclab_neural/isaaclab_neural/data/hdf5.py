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


def _validate_rollout_arrays(arrays: Mapping[str, Any]) -> None:
    """Validate required fields and common trajectory dimensions."""
    required = {"states", "next_states", "joint_f", "contact_depths", "contact_masks"}
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"Rollout data is missing required fields: {missing}.")

    states = arrays["states"]
    if states.ndim < 3:
        raise ValueError(f"states must have shape [trajectories, steps, features], got {states.shape}.")
    trajectory_shape = states.shape[:2]
    for name, data in arrays.items():
        if data.ndim < 3 or data.shape[:2] != trajectory_shape:
            raise ValueError(
                f"Rollout field {name!r} must start with trajectory shape {trajectory_shape}, got {data.shape}."
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
    data_group.attrs["num_contacts_per_env"] = cast(h5py.Dataset, data_group["contact_depths"]).shape[-1]
    data_group.attrs["joint_f_dim"] = cast(h5py.Dataset, data_group["joint_f"]).shape[-1]
    if "actions" in data_group:
        data_group.attrs["action_dim"] = cast(h5py.Dataset, data_group["actions"]).shape[-1]
