# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dataset loaders for NeRD HDF5 trajectory and transition datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from isaaclab_neural.utils.commons import DATASET_MODES

BOOL_DATASET_KEYS = {"contact_masks"}


def _decode_attr(value: Any) -> Any:
    """Decode byte-string HDF5 attributes while leaving other values unchanged."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _read_dataset_array(data_group: h5py.Group, key: str, selection: Any = None) -> np.ndarray:
    """Read one HDF5 dataset while preserving boolean mask fields."""
    dataset = cast(h5py.Dataset, data_group[key])
    data = np.asarray(dataset[selection] if selection is not None else dataset[()])
    if key in BOOL_DATASET_KEYS:
        return data.astype(bool)
    return data.astype("float32")


def _torch_tensor(data: np.ndarray, device: str | torch.device | None = None) -> torch.Tensor:
    """Convert loaded numpy data to torch with boolean masks preserved."""
    if data.dtype == np.bool_:
        return torch.as_tensor(data, dtype=torch.bool, device=device)
    return torch.as_tensor(data, device=device)


class BatchTransitionDataset:
    """Batched transition loader for NeRD training datasets.

    The loader materializes the HDF5 datasets into torch tensors with shape
    ``[num_batches, batch_size, feature_dim]``.
    """

    def __init__(
        self,
        batch_size: int,
        hdf5_dataset_path: str | Path,
        max_capacity: int = 100_000_000,
        device: str | torch.device = "cpu",
    ):
        self.batch_size = batch_size
        self.max_capacity = max_capacity
        self.device = device
        self.traj_lengths: np.ndarray | None = None
        self.dataset: dict[str, torch.Tensor] = {}
        self.dataset_length = 0

        self.load_dataset(hdf5_dataset_path)

    def load_dataset(self, hdf5_dataset_path: str | Path) -> None:
        """Load a transition or trajectory HDF5 dataset into batched tensors."""
        dataset_path = Path(hdf5_dataset_path).expanduser()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file {dataset_path.resolve()} not found.")

        with h5py.File(dataset_path, "r", swmr=True, libver="latest") as dataset_file:
            data_group = cast(h5py.Group, dataset_file["data"])
            mode = _decode_attr(data_group.attrs["mode"])
            if mode not in DATASET_MODES:
                raise ValueError(f"Unsupported dataset mode: {mode!r}. Expected one of {DATASET_MODES}.")

            total_transitions = int(cast(Any, data_group.attrs["total_transitions"]))
            self.dataset_length = min(self.max_capacity, total_transitions) // self.batch_size
            self.dataset = {}

            if mode == "trajectory":
                self.traj_lengths = None

            for key in data_group.keys():
                if mode == "trajectory" and key == "traj_lengths":
                    self.traj_lengths = np.asarray(cast(h5py.Dataset, data_group[key])[()]).astype("int32")
                    continue

                data = _read_dataset_array(data_group, key)
                data = data.reshape(total_transitions, -1)
                self.dataset[key] = _torch_tensor(
                    data[: self.dataset_length * self.batch_size].reshape(self.dataset_length, self.batch_size, -1),
                    device=self.device,
                )

    def __len__(self) -> int:
        """Return the number of batches."""
        return self.dataset_length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one pre-batched sample."""
        if index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of length {len(self)}.")
        return {key: value[index] for key, value in self.dataset.items()}

    def shuffle(self) -> None:
        """Shuffle batches independently for each slot in the batch dimension."""
        for batch_index in range(self.batch_size):
            permutation = torch.randperm(self.dataset_length, device=self.device)
            for value in self.dataset.values():
                value[:, batch_index, :] = value[permutation, batch_index, :]


def collate_fn_batch_transition_dataset(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate :class:`BatchTransitionDataset` samples into ``[B, 1, dim]`` tensors."""
    return {key: value.unsqueeze(1) for key, value in batch[0].items()}


# Backwards-compatible name used by the original training code.
collate_fn_BatchTransitionDataset = collate_fn_batch_transition_dataset


class TrajectoryDataset(Dataset):
    """Sequence sampler over fixed-length windows from trajectory HDF5 datasets."""

    def __init__(
        self,
        hdf5_dataset_path: str | Path,
        sample_sequence_length: int = 10,
        max_capacity: int = 100_000_000,
    ):
        self.max_capacity = max_capacity
        self.dataset: dict[str, np.ndarray] = {}
        self.traj_lengths: np.ndarray = np.array([], dtype="int32")
        self.sample_sequence_length = sample_sequence_length
        self.mapping_index2traj = np.zeros((0, 2), dtype=int)
        self.length = 0

        self.load_dataset(hdf5_dataset_path)
        self.update_sample_sequence_length(sample_sequence_length)

    def load_dataset(self, hdf5_dataset_path: str | Path) -> None:
        """Load a trajectory HDF5 dataset into memory."""
        with h5py.File(Path(hdf5_dataset_path).expanduser(), "r", swmr=True, libver="latest") as dataset_file:
            data_group = cast(h5py.Group, dataset_file["data"])
            mode = _decode_attr(data_group.attrs["mode"])
            if mode != "trajectory":
                raise ValueError(f"TrajectoryDataset requires dataset mode 'trajectory', got {mode!r}.")

            states_dataset = cast(h5py.Dataset, data_group["states"])
            num_transitions_per_trajectory = int(states_dataset.shape[1])
            num_trajectories = min(
                int(np.ceil(self.max_capacity / num_transitions_per_trajectory)),
                int(states_dataset.shape[0]),
            )

            self.dataset = {}
            traj_lengths = None
            for key in data_group.keys():
                if key == "traj_lengths":
                    traj_lengths = np.asarray(cast(h5py.Dataset, data_group[key])[:num_trajectories]).astype("int32")
                    continue

                data = _read_dataset_array(data_group, key, slice(None, num_trajectories))
                self.dataset[key] = data.reshape(data.shape[0], data.shape[1], -1)

            if traj_lengths is None:
                traj_lengths = np.full(num_trajectories, num_transitions_per_trajectory, dtype="int32")
            self.traj_lengths = traj_lengths

    def update_sample_sequence_length(self, sample_sequence_length: int) -> None:
        """Update sequence length and rebuild the trajectory-window index."""
        self.sample_sequence_length = sample_sequence_length
        self.build_index()

    def build_index(self) -> None:
        """Build a flat index mapping samples to ``(trajectory, start_step)``."""
        self.length = int(
            sum(max(0, int(traj_length) - self.sample_sequence_length + 1) for traj_length in self.traj_lengths)
        )
        self.mapping_index2traj = np.zeros((self.length, 2), dtype=int)

        index = 0
        for traj_index, traj_length in enumerate(self.traj_lengths):
            num_windows = max(0, int(traj_length) - self.sample_sequence_length + 1)
            for step_index in range(num_windows):
                self.mapping_index2traj[index] = (traj_index, step_index)
                index += 1

    def __len__(self) -> int:
        """Return the number of available trajectory windows."""
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one trajectory window as torch tensors."""
        if index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of length {len(self)}.")

        traj_index, traj_step_index = self.mapping_index2traj[index]
        return {
            key: _torch_tensor(value[traj_index, traj_step_index : traj_step_index + self.sample_sequence_length])
            for key, value in self.dataset.items()
        }

    def shuffle(self) -> None:
        """No-op kept for API symmetry with :class:`BatchTransitionDataset`."""
        pass
