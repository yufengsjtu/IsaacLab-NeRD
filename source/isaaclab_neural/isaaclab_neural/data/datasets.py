# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dataset loaders for NeRD HDF5 trajectory and transition datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from isaaclab_neural.utils.commons import DATASET_MODES

BOOL_DATASET_KEYS = {"contact_masks"}
INTEGER_DATASET_KEYS = {
    "contact_body_ids",
    "contact_token_body_ids",
    "contact_token_overflow",
    "contact_token_world_ids",
}
DatasetLoadMode = Literal["eager", "lazy"]


def _decode_attr(value: Any) -> Any:
    """Decode byte-string HDF5 attributes while leaving other values unchanged."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _read_dataset_array(data_group: h5py.Group, key: str, selection: Any = None) -> np.ndarray:
    """Read one HDF5 dataset while preserving categorical field dtypes."""
    dataset = cast(h5py.Dataset, data_group[key])
    data = np.asarray(dataset[selection] if selection is not None else dataset[()])
    if key in BOOL_DATASET_KEYS:
        return data.astype(bool)
    if key in INTEGER_DATASET_KEYS:
        return data.astype("int64")
    return data.astype("float32")


def _torch_tensor(data: np.ndarray, device: str | torch.device | None = None) -> torch.Tensor:
    """Convert loaded numpy data to torch with categorical dtypes preserved."""
    if data.dtype == np.bool_:
        return torch.as_tensor(data, dtype=torch.bool, device=device)
    if np.issubdtype(data.dtype, np.integer):
        return torch.as_tensor(data, dtype=torch.long, device=device)
    return torch.as_tensor(data, device=device)


def _resolve_dataset_path(hdf5_dataset_path: str | Path) -> Path:
    """Return an existing dataset path or raise."""
    dataset_path = Path(hdf5_dataset_path).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file {dataset_path.resolve()} not found.")
    return dataset_path


def validate_contact_token_metadata(data_group: h5py.Group) -> None:
    """Reject contact-token datasets with ambiguous frame or identity metadata."""
    if "contact_tokens" not in data_group:
        return
    token_frame = _decode_attr(data_group.attrs.get("contact_token_frame", ""))
    identity_schema = _decode_attr(data_group.attrs.get("contact_identity_schema", ""))
    if token_frame != "world_v1" or identity_schema != "world_owner_v1":
        raise ValueError(
            "Contact-token dataset requires contact_token_frame='world_v1' and "
            "contact_identity_schema='world_owner_v1'; regenerate this dataset."
        )
    required_ids = {"contact_token_body_ids", "contact_token_world_ids"}
    missing_ids = sorted(required_ids - set(data_group.keys()))
    if missing_ids:
        raise ValueError(f"Contact-token dataset is missing identity fields: {missing_ids}.")
    invalid_dtypes = {
        key: str(cast(h5py.Dataset, data_group[key]).dtype)
        for key in required_ids
        if not np.issubdtype(cast(h5py.Dataset, data_group[key]).dtype, np.integer)
    }
    if invalid_dtypes:
        raise ValueError(f"Contact-token identity fields must use integer dtypes: {invalid_dtypes}.")


def _read_hdf5_metadata(
    hdf5_dataset_path: str | Path,
    *,
    max_capacity: int,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Read dataset metadata without materializing rollout arrays."""
    dataset_path = _resolve_dataset_path(hdf5_dataset_path)
    with h5py.File(dataset_path, "r", swmr=True, libver="latest") as dataset_file:
        data_group = cast(h5py.Group, dataset_file["data"])
        validate_contact_token_metadata(data_group)
        mode = _decode_attr(data_group.attrs["mode"])
        if mode not in DATASET_MODES:
            raise ValueError(f"Unsupported dataset mode: {mode!r}. Expected one of {DATASET_MODES}.")

        total_transitions = int(cast(Any, data_group.attrs["total_transitions"]))
        data_keys = [key for key in data_group.keys() if key != "traj_lengths"]
        states_dataset = cast(h5py.Dataset, data_group["states"])
        num_transitions_per_trajectory = int(states_dataset.shape[1]) if states_dataset.ndim == 3 else 1
        num_trajectories = int(states_dataset.shape[0]) if states_dataset.ndim >= 2 else total_transitions

        metadata: dict[str, Any] = {
            "dataset_path": dataset_path,
            "mode": mode,
            "total_transitions": total_transitions,
            "data_keys": data_keys,
            "num_transitions_per_trajectory": num_transitions_per_trajectory,
            "num_trajectories": num_trajectories,
            "traj_lengths": None,
        }

        if mode == "trajectory":
            if batch_size is not None:
                metadata["dataset_length"] = min(max_capacity, total_transitions) // batch_size
            else:
                num_trajectories = min(
                    int(np.ceil(max_capacity / num_transitions_per_trajectory)),
                    num_trajectories,
                )
                metadata["num_trajectories"] = num_trajectories
                if "traj_lengths" in data_group:
                    metadata["traj_lengths"] = np.asarray(
                        cast(h5py.Dataset, data_group["traj_lengths"])[:num_trajectories]
                    ).astype("int32")
                else:
                    metadata["traj_lengths"] = np.full(num_trajectories, num_transitions_per_trajectory, dtype="int32")
        else:
            if batch_size is None:
                raise ValueError("batch_size is required for transition-mode metadata.")
            metadata["dataset_length"] = min(max_capacity, total_transitions) // batch_size

        return metadata


class _LazyHdf5Accessor:
    """Open one HDF5 file per process for lazy dataset reads."""

    def __init__(self, dataset_path: str | Path):
        self.dataset_path = Path(dataset_path)
        self._file: h5py.File | None = None
        self._data_group: h5py.Group | None = None

    def _ensure_open(self) -> h5py.Group:
        if self._data_group is None:
            self._file = h5py.File(self.dataset_path, "r", swmr=True, libver="latest")
            self._data_group = cast(h5py.Group, self._file["data"])
        return self._data_group

    def dataset(self, key: str) -> h5py.Dataset:
        return cast(h5py.Dataset, self._ensure_open()[key])

    def trajectory_context_dataset(self, key: str) -> h5py.Dataset:
        """Return one per-trajectory context dataset from the open file."""
        self._ensure_open()
        return cast(h5py.Dataset, cast(h5py.File, self._file)["context"]["trajectories"][key])

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._data_group = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_file"] = None
        state["_data_group"] = None
        return state


def _read_flat_transition_range(
    dataset: h5py.Dataset,
    flat_start: int,
    flat_end: int,
    num_transitions_per_trajectory: int,
    *,
    preserve_tail_shape: bool = False,
) -> np.ndarray:
    """Read a flat transition range while optionally preserving trailing axes."""
    if flat_end <= flat_start:
        raise ValueError("flat_end must be greater than flat_start.")

    if dataset.ndim == 2:
        return np.asarray(dataset[flat_start:flat_end])

    if dataset.ndim < 3:
        raise ValueError(f"Unsupported rollout dataset rank: {dataset.ndim}.")

    tail_shape = tuple(dataset.shape[2:]) if preserve_tail_shape else (int(np.prod(dataset.shape[2:])),)
    output = np.empty((flat_end - flat_start, *tail_shape), dtype=dataset.dtype)
    write_offset = 0
    position = flat_start
    while position < flat_end:
        traj_index, step_index = divmod(position, num_transitions_per_trajectory)
        remaining_in_trajectory = num_transitions_per_trajectory - step_index
        remaining_in_range = flat_end - position
        count = min(remaining_in_trajectory, remaining_in_range)
        chunk = np.asarray(dataset[traj_index, step_index : step_index + count])
        output[write_offset : write_offset + count] = chunk.reshape(count, *tail_shape)
        write_offset += count
        position += count
    return output


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
        dataset_path = _resolve_dataset_path(hdf5_dataset_path)

        with h5py.File(dataset_path, "r", swmr=True, libver="latest") as dataset_file:
            data_group = cast(h5py.Group, dataset_file["data"])
            validate_contact_token_metadata(data_group)
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
                usable = self.dataset_length * self.batch_size
                if key == "contact_tokens" and data.ndim >= 4:
                    # Preserve ``[..., K, 17]`` for set encoding.
                    token_shape = data.shape[-2:]
                    flat = data.reshape(-1, *token_shape)[:usable]
                    self.dataset[key] = _torch_tensor(
                        flat.reshape(self.dataset_length, self.batch_size, *token_shape),
                        device=self.device,
                    )
                else:
                    data = data.reshape(total_transitions, -1)
                    self.dataset[key] = _torch_tensor(
                        data[:usable].reshape(self.dataset_length, self.batch_size, -1),
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
                value[:, batch_index] = value[permutation, batch_index]


class LazyBatchTransitionDataset:
    """Lazy batched transition loader that reads rollout slices from HDF5 on demand."""

    def __init__(
        self,
        batch_size: int,
        hdf5_dataset_path: str | Path,
        max_capacity: int = 100_000_000,
        device: str | torch.device = "cpu",
    ):
        self.batch_size = batch_size
        self.max_capacity = max_capacity
        # Lazy loading keeps HDF5 storage on CPU; trainer preprocessing moves batches to the requested device.
        self.device = torch.device("cpu")
        self.requested_device = torch.device(device)
        self.traj_lengths: np.ndarray | None = None
        self._batch_permutation: np.ndarray | None = None

        metadata = _read_hdf5_metadata(
            hdf5_dataset_path,
            max_capacity=max_capacity,
            batch_size=batch_size,
        )
        self.dataset_path = metadata["dataset_path"]
        self.dataset_length = int(metadata["dataset_length"])
        self.total_transitions = int(metadata["total_transitions"])
        self.data_keys = list(metadata["data_keys"])
        self.num_transitions_per_trajectory = int(metadata["num_transitions_per_trajectory"])
        self._hdf5 = _LazyHdf5Accessor(self.dataset_path)

    def __len__(self) -> int:
        """Return the number of batches."""
        return self.dataset_length

    def _resolve_batch_indices(self, index: int) -> np.ndarray:
        if index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of length {len(self)}.")
        if self._batch_permutation is None:
            return np.full(self.batch_size, index, dtype=np.int64)
        return self._batch_permutation[index]

    def _read_batch_key(self, key: str, batch_indices: np.ndarray) -> torch.Tensor:
        dataset = self._hdf5.dataset(key)
        if np.all(batch_indices == batch_indices[0]):
            flat_start = int(batch_indices[0]) * self.batch_size
            data = _read_flat_transition_range(
                dataset,
                flat_start,
                flat_start + self.batch_size,
                self.num_transitions_per_trajectory,
                preserve_tail_shape=key == "contact_tokens",
            )
        else:
            rows = [
                _read_flat_transition_range(
                    dataset,
                    int(batch_index) * self.batch_size + batch_slot,
                    int(batch_index) * self.batch_size + batch_slot + 1,
                    self.num_transitions_per_trajectory,
                    preserve_tail_shape=key == "contact_tokens",
                )[0]
                for batch_slot, batch_index in enumerate(batch_indices)
            ]
            data = np.stack(rows, axis=0)

        if key in BOOL_DATASET_KEYS:
            return torch.as_tensor(data, dtype=torch.bool)
        if key in INTEGER_DATASET_KEYS:
            return torch.as_tensor(data, dtype=torch.long)
        return torch.as_tensor(data, dtype=torch.float32)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one pre-batched sample read from HDF5."""
        batch_indices = self._resolve_batch_indices(index)
        sample: dict[str, torch.Tensor] = {}
        for key in self.data_keys:
            sample[key] = self._read_batch_key(key, batch_indices)
        return sample

    def shuffle(self) -> None:
        """Shuffle batches independently for each slot in the batch dimension."""
        self._batch_permutation = np.stack(
            [np.random.permutation(self.dataset_length) for _ in range(self.batch_size)],
            axis=1,
        )

    def close(self) -> None:
        """Close the lazily opened HDF5 file handle."""
        if hasattr(self, "_hdf5"):
            self._hdf5.close()

    def __del__(self) -> None:
        self.close()


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
        rank: int = 0,
        world_size: int = 1,
    ):
        if world_size <= 0:
            raise ValueError("world_size must be positive.")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}.")
        self.max_capacity = max_capacity
        self.rank = rank
        self.world_size = world_size
        self.is_rank_sharded = world_size > 1
        self.dataset: dict[str, np.ndarray] = {}
        self.trajectory_context: dict[str, np.ndarray] = {}
        self.traj_lengths: np.ndarray = np.array([], dtype="int32")
        self.global_trajectory_indices = np.array([], dtype=np.int64)
        self.sample_sequence_length = sample_sequence_length
        self.mapping_index2traj = np.zeros((0, 2), dtype=int)
        self.length = 0

        self.load_dataset(hdf5_dataset_path)
        self.update_sample_sequence_length(sample_sequence_length)

    def load_dataset(self, hdf5_dataset_path: str | Path) -> None:
        """Load a trajectory HDF5 dataset into memory."""
        with h5py.File(_resolve_dataset_path(hdf5_dataset_path), "r", swmr=True, libver="latest") as dataset_file:
            data_group = cast(h5py.Group, dataset_file["data"])
            validate_contact_token_metadata(data_group)
            mode = _decode_attr(data_group.attrs["mode"])
            if mode != "trajectory":
                raise ValueError(f"TrajectoryDataset requires dataset mode 'trajectory', got {mode!r}.")

            states_dataset = cast(h5py.Dataset, data_group["states"])
            num_transitions_per_trajectory = int(states_dataset.shape[1])
            num_trajectories = min(
                int(np.ceil(self.max_capacity / num_transitions_per_trajectory)),
                int(states_dataset.shape[0]),
            )
            self.global_trajectory_indices = np.arange(self.rank, num_trajectories, self.world_size, dtype=np.int64)
            if self.global_trajectory_indices.size == 0:
                raise ValueError(
                    f"Dataset has {num_trajectories} usable trajectories, which leaves rank {self.rank} "
                    f"empty for world_size={self.world_size}."
                )
            trajectory_selection = slice(self.rank, num_trajectories, self.world_size)

            self.dataset = {}
            self.trajectory_context = {}
            traj_lengths = None
            for key in data_group.keys():
                if key == "traj_lengths":
                    traj_lengths = np.asarray(cast(h5py.Dataset, data_group[key])[trajectory_selection]).astype("int32")
                    continue

                data = _read_dataset_array(data_group, key, trajectory_selection)
                # Keep contact token set axes ``[..., K, 17]``; flatten only rank-3 fields.
                if key == "contact_tokens" and data.ndim >= 4:
                    self.dataset[key] = data
                else:
                    self.dataset[key] = data.reshape(data.shape[0], data.shape[1], -1)
            if "context" in dataset_file and "trajectories" in dataset_file["context"]:
                trajectory_group = cast(h5py.Group, dataset_file["context"]["trajectories"])
                collisions = set(self.dataset).intersection(trajectory_group.keys())
                if collisions:
                    raise ValueError(f"Data and trajectory context keys overlap: {sorted(collisions)}.")
                for key in trajectory_group.keys():
                    self.trajectory_context[key] = np.asarray(
                        cast(h5py.Dataset, trajectory_group[key])[trajectory_selection]
                    )

            if traj_lengths is None:
                traj_lengths = np.full(
                    self.global_trajectory_indices.size,
                    num_transitions_per_trajectory,
                    dtype="int32",
                )
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
        sample = {
            key: _torch_tensor(value[traj_index, traj_step_index : traj_step_index + self.sample_sequence_length])
            for key, value in self.dataset.items()
        }
        sample.update({key: _torch_tensor(value[traj_index]) for key, value in self.trajectory_context.items()})
        return sample

    def shuffle(self) -> None:
        """No-op kept for API symmetry with :class:`BatchTransitionDataset`."""
        pass


class LazyTrajectoryDataset(Dataset):
    """Lazy sequence sampler that reads trajectory windows from HDF5 on demand."""

    def __init__(
        self,
        hdf5_dataset_path: str | Path,
        sample_sequence_length: int = 10,
        max_capacity: int = 100_000_000,
    ):
        self.max_capacity = max_capacity
        self.sample_sequence_length = sample_sequence_length
        self.traj_lengths: np.ndarray = np.array([], dtype="int32")
        self.mapping_index2traj = np.zeros((0, 2), dtype=int)
        self.length = 0
        self._sample_permutation: np.ndarray | None = None

        metadata = _read_hdf5_metadata(hdf5_dataset_path, max_capacity=max_capacity)
        if metadata["mode"] != "trajectory":
            raise ValueError(f"LazyTrajectoryDataset requires dataset mode 'trajectory', got {metadata['mode']!r}.")

        self.dataset_path = metadata["dataset_path"]
        self.data_keys = list(metadata["data_keys"])
        self.traj_lengths = cast(np.ndarray, metadata["traj_lengths"])
        self._hdf5 = _LazyHdf5Accessor(self.dataset_path)
        self.trajectory_context_keys: list[str] = []
        with h5py.File(self.dataset_path, "r", swmr=True, libver="latest") as dataset_file:
            if "context" in dataset_file and "trajectories" in dataset_file["context"]:
                trajectory_group = cast(h5py.Group, dataset_file["context"]["trajectories"])
                collisions = set(self.data_keys).intersection(trajectory_group.keys())
                if collisions:
                    raise ValueError(f"Data and trajectory context keys overlap: {sorted(collisions)}.")
                self.trajectory_context_keys = list(trajectory_group.keys())
        self.update_sample_sequence_length(sample_sequence_length)

    def update_sample_sequence_length(self, sample_sequence_length: int) -> None:
        """Update sequence length and rebuild the trajectory-window index."""
        self.sample_sequence_length = sample_sequence_length
        self.build_index()
        self._sample_permutation = None

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

    def _resolve_sample_index(self, index: int) -> int:
        if index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of length {len(self)}.")
        if self._sample_permutation is None:
            return index
        return int(self._sample_permutation[index])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one trajectory window read from HDF5."""
        sample_index = self._resolve_sample_index(index)
        traj_index, traj_step_index = self.mapping_index2traj[sample_index]
        step_slice = slice(traj_step_index, traj_step_index + self.sample_sequence_length)
        sample: dict[str, torch.Tensor] = {}
        for key in self.data_keys:
            data = np.asarray(self._hdf5.dataset(key)[traj_index, step_slice])
            if key in BOOL_DATASET_KEYS:
                sample[key] = torch.as_tensor(data.astype(bool), dtype=torch.bool)
            elif key in INTEGER_DATASET_KEYS:
                sample[key] = torch.as_tensor(data.astype("int64"), dtype=torch.long)
            else:
                sample[key] = torch.as_tensor(data.astype("float32"), dtype=torch.float32)
        for key in self.trajectory_context_keys:
            sample[key] = _torch_tensor(np.asarray(self._hdf5.trajectory_context_dataset(key)[traj_index]))
        return sample

    def shuffle(self) -> None:
        """Shuffle sample indices without moving rollout tensors."""
        self._sample_permutation = np.random.permutation(self.length)

    def close(self) -> None:
        """Close the lazily opened HDF5 file handle."""
        self._hdf5.close()

    def __del__(self) -> None:
        self.close()


def create_batch_transition_dataset(
    *,
    load_mode: DatasetLoadMode = "eager",
    batch_size: int,
    hdf5_dataset_path: str | Path,
    max_capacity: int = 100_000_000,
    device: str | torch.device = "cpu",
) -> BatchTransitionDataset | LazyBatchTransitionDataset:
    """Create an eager or lazy batched transition dataset."""
    if load_mode == "lazy":
        return LazyBatchTransitionDataset(
            batch_size=batch_size,
            hdf5_dataset_path=hdf5_dataset_path,
            max_capacity=max_capacity,
            device=device,
        )
    if load_mode == "eager":
        return BatchTransitionDataset(
            batch_size=batch_size,
            hdf5_dataset_path=hdf5_dataset_path,
            max_capacity=max_capacity,
            device=device,
        )
    raise ValueError(f"Unsupported dataset load_mode: {load_mode!r}. Expected 'eager' or 'lazy'.")


def create_trajectory_dataset(
    *,
    load_mode: DatasetLoadMode = "eager",
    hdf5_dataset_path: str | Path,
    sample_sequence_length: int = 10,
    max_capacity: int = 100_000_000,
    rank: int = 0,
    world_size: int = 1,
) -> TrajectoryDataset | LazyTrajectoryDataset:
    """Create an eager or lazy trajectory-window dataset."""
    if load_mode == "lazy":
        return LazyTrajectoryDataset(
            hdf5_dataset_path=hdf5_dataset_path,
            sample_sequence_length=sample_sequence_length,
            max_capacity=max_capacity,
        )
    if load_mode == "eager":
        return TrajectoryDataset(
            hdf5_dataset_path=hdf5_dataset_path,
            sample_sequence_length=sample_sequence_length,
            max_capacity=max_capacity,
            rank=rank,
            world_size=world_size,
        )
    raise ValueError(f"Unsupported dataset load_mode: {load_mode!r}. Expected 'eager' or 'lazy'.")
