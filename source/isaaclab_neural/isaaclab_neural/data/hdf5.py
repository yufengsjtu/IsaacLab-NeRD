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
) -> None:
    """Serialize rollout tensors to a NeRD HDF5 trajectory dataset.

    Args:
        dataset_path: Output HDF5 path.
        rollouts: Rollout dictionary containing ``states``, ``next_states``,
            ``joint_f``, and a nested ``contacts`` dictionary.
        env_name: Environment name stored in dataset metadata.
    """
    dataset_path = Path(dataset_path).expanduser()
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _flatten_rollouts(rollouts)
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
        data_group.attrs["env"] = env_name
        data_group.attrs["mode"] = "trajectory"
        _update_metadata(data_group)


def append_rollouts_to_hdf5(
    dataset_path: str | Path,
    rollouts: Mapping,
    env_name: str,
) -> None:
    """Append rollout tensors to a NeRD HDF5 trajectory dataset.

    The appended file has the same schema as :func:`write_rollouts_to_hdf5`,
    but datasets are resizable along the trajectory dimension. This avoids
    keeping very large data-generation runs in memory until the end.
    """
    dataset_path = Path(dataset_path).expanduser()
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _flatten_rollouts(rollouts)
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
        if is_new:
            _create_appendable_datasets(data_group, arrays)
            data_group.attrs["env"] = env_name
            data_group.attrs["mode"] = "trajectory"
        else:
            _append_arrays_atomically(data_group, arrays)
        _update_metadata(data_group)


def _flatten_rollouts(rollouts: Mapping) -> dict[str, Any]:
    """Flatten rollout tensors and nested contact tensors into numpy arrays."""
    arrays = {}
    for key, value in rollouts.items():
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


def _validate_rollout_arrays(arrays: Mapping[str, Any]) -> None:
    """Validate required fields and common trajectory dimensions."""
    required = {"states", "next_states", "joint_f"}
    if "contact_tokens" in arrays:
        # Body-frame training needs the root pose used by process_neural_model_inputs.
        required.update({"contact_tokens", "root_body_q", "gravity_dir"})
    else:
        required.update({"contact_depths", "contact_masks"})
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"Rollout data is missing required fields: {missing}.")

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


def _append_arrays_atomically(data_group: h5py.Group, arrays: Mapping[str, Any]) -> None:
    """Append all fields and roll back their sizes if any write fails."""
    old_count = cast(h5py.Dataset, data_group["states"]).shape[0]
    append_count = arrays["states"].shape[0]
    resized = []
    try:
        for name in arrays:
            dataset = cast(h5py.Dataset, data_group[name])
            dataset.resize((old_count + append_count, *dataset.shape[1:]))
            resized.append(name)
        for name, data in arrays.items():
            cast(h5py.Dataset, data_group[name])[old_count : old_count + append_count] = data
    except Exception:
        for name in resized:
            dataset = cast(h5py.Dataset, data_group[name])
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
        if "num_contacts_per_env" not in data_group.attrs:
            data_group.attrs["num_contacts_per_env"] = tokens.shape[-2]
    data_group.attrs["joint_f_dim"] = cast(h5py.Dataset, data_group["joint_f"]).shape[-1]
    if "actions" in data_group:
        data_group.attrs["action_dim"] = cast(h5py.Dataset, data_group["actions"]).shape[-1]
