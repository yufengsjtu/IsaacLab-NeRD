# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HDF5 serialization helpers for NeRD rollout datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, cast

import h5py
import torch


def _to_numpy(value: torch.Tensor):
    """Convert a tensor-like rollout value to a CPU numpy array."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    raise TypeError(f"Expected rollout values to be torch.Tensor, got {type(value).__name__}.")


def write_rollouts_to_hdf5(dataset_path: str | Path, rollouts: Mapping, env_name: str) -> None:
    """Serialize rollout tensors to a NeRD HDF5 trajectory dataset.

    Args:
        dataset_path: Output HDF5 path.
        rollouts: Rollout dictionary containing ``states``, ``next_states``,
            ``joint_f``, and a nested ``contacts`` dictionary.
        env_name: Environment name stored in dataset metadata.
    """
    dataset_path = Path(dataset_path).expanduser()
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(dataset_path, "w") as dataset_file:
        data_group = dataset_file.create_group("data")
        data_group.attrs["env"] = env_name
        data_group.attrs["mode"] = "trajectory"
        data_group.attrs["total_trajectories"] = rollouts["states"].shape[0]
        data_group.attrs["total_transitions"] = rollouts["states"].shape[0] * rollouts["states"].shape[1]

        for key, value in rollouts.items():
            if isinstance(value, Mapping):
                for sub_key, sub_value in value.items():
                    data_group.create_dataset(name=sub_key, data=_to_numpy(sub_value))
            else:
                data_group.create_dataset(name=key, data=_to_numpy(value))

        data_group.attrs["state_dim"] = rollouts["states"].shape[-1]
        data_group.attrs["next_state_dim"] = rollouts["next_states"].shape[-1]
        data_group.attrs["num_contacts_per_env"] = rollouts["contacts"]["contact_depths"].shape[-1]
        data_group.attrs["joint_f_dim"] = rollouts["joint_f"].shape[-1]
        if "actions" in rollouts:
            data_group.attrs["action_dim"] = rollouts["actions"].shape[-1]


def append_rollouts_to_hdf5(dataset_path: str | Path, rollouts: Mapping, env_name: str) -> None:
    """Append rollout tensors to a NeRD HDF5 trajectory dataset.

    The appended file has the same schema as :func:`write_rollouts_to_hdf5`,
    but datasets are resizable along the trajectory dimension. This avoids
    keeping very large data-generation runs in memory until the end.
    """
    dataset_path = Path(dataset_path).expanduser()
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(dataset_path, "a") as dataset_file:
        data_group = dataset_file.require_group("data")
        if "mode" not in data_group.attrs:
            data_group.attrs["env"] = env_name
            data_group.attrs["mode"] = "trajectory"

        for key, value in rollouts.items():
            if isinstance(value, Mapping):
                for sub_key, sub_value in value.items():
                    _append_dataset(data_group, sub_key, sub_value)
            else:
                _append_dataset(data_group, key, value)

        states = cast(h5py.Dataset, data_group["states"])
        data_group.attrs["total_trajectories"] = states.shape[0]
        data_group.attrs["total_transitions"] = states.shape[0] * states.shape[1]
        data_group.attrs["state_dim"] = states.shape[-1]
        data_group.attrs["next_state_dim"] = cast(h5py.Dataset, data_group["next_states"]).shape[-1]
        data_group.attrs["num_contacts_per_env"] = cast(h5py.Dataset, data_group["contact_depths"]).shape[-1]
        data_group.attrs["joint_f_dim"] = cast(h5py.Dataset, data_group["joint_f"]).shape[-1]
        if "actions" in data_group:
            data_group.attrs["action_dim"] = cast(h5py.Dataset, data_group["actions"]).shape[-1]


def _append_dataset(data_group: h5py.Group, name: str, value: torch.Tensor) -> None:
    data = _to_numpy(value)
    if name not in data_group:
        maxshape = (None, *data.shape[1:])
        data_group.create_dataset(name=name, data=data, maxshape=maxshape, chunks=True)
        return

    dataset = cast(h5py.Dataset, data_group[name])
    old_count = dataset.shape[0]
    new_count = old_count + data.shape[0]
    dataset.resize((new_count, *dataset.shape[1:]))
    dataset[old_count:new_count] = data
