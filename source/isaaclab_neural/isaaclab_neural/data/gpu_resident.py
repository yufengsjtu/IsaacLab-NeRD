# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Device-resident unique-transition windows that convert native contacts to tokens per batch."""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import torch

from isaaclab_neural.data.datasets import (
    BOOL_DATASET_KEYS,
    INTEGER_DATASET_KEYS,
    UniqueTrajectoryDataset,
    WeightedMixedTrajectoryDataset,
    _read_dataset_array,
    _trajectory_context_group,
)
from isaaclab_neural.data.native_contacts import NATIVE_INTEGER_FIELDS, canonical_native_dtype
from isaaclab_neural.data.transition_index import (
    SUPERVISION_MASK,
    SUPERVISION_SOURCE,
    SUPERVISION_TARGET_IDS,
    SUPERVISION_WINDOW,
)

RESIDENT_PAIR_DTYPES: dict[str, torch.dtype] = {
    "body_ids_0": torch.int32,
    "body_ids_1": torch.int32,
    "body_slots_0": torch.int16,
    "body_slots_1": torch.int16,
    "world_ids": torch.int32,
    "surface_points_0_w": torch.float32,
    "surface_points_1_w": torch.float32,
    "separation": torch.float32,
    "relative_velocities_0_w": torch.float32,
    "relative_velocities_1_w": torch.float32,
    "normal_w": torch.float16,
    "lever_arms_0_w": torch.float16,
    "lever_arms_1_w": torch.float16,
}
"""Device storage of retained native pairs; bounded unit normals and lever arms [m] are kept in FP16."""


class GpuResidentTrajectoryDataset(UniqueTrajectoryDataset):
    """Unique-transition windows stored on the training device.

    The rank's recordings are copied to ``device`` once. Native contact pairs stay raw: pairs rejected by the
    configured contact filter (a per-pair predicate) are dropped while loading, and the remaining pairs are kept in
    CSR form. :meth:`get_batch` gathers a batch of windows on the device and converts its frames' native pairs to
    contact tokens with the shared native-pair projector, so the token contract equals the eager loader's.

    Args:
        hdf5_dataset_path: Unique-transition HDF5 dataset.
        device: Training device holding the resident tensors.
        float16_keys: Stored floating fields kept in FP16 on the device and widened to FP32 per batch.
        compact_pairs: Store retained native-pair fields with :data:`RESIDENT_PAIR_DTYPES` instead of FP32/int64.
        **kwargs: Arguments of :class:`UniqueTrajectoryDataset` except ``load_mode``.
    """

    def __init__(
        self,
        hdf5_dataset_path: str | Path,
        *,
        device: str | torch.device,
        float16_keys: Sequence[str] = (),
        compact_pairs: bool = True,
        **kwargs: Any,
    ):
        super().__init__(hdf5_dataset_path, load_mode="lazy", **kwargs)
        self.load_mode = "gpu"
        self.device = torch.device(device)
        projection = self.native_projection
        if projection is not None and (
            projection.representation != "contact_tokens" or projection.packing_policy != "body_round_robin_pair_atomic"
        ):
            raise ValueError("GPU-resident datasets support deterministic contact-token projection only.")
        self.float16_keys = frozenset(float16_keys)
        self.compact_pairs = bool(compact_pairs)
        self.horizon = int(self.num_transitions_per_trajectory)
        self.resident: dict[str, torch.Tensor] = {}
        self.resident_context: dict[str, torch.Tensor] = {}
        self.pair_offsets: torch.Tensor | None = None
        self.pairs: dict[str, torch.Tensor] = {}
        self._load(int(kwargs.get("eager_preload_frames", 4096)))
        self._blocks = torch.as_tensor(self.sliding_blocks, device=self.device)
        self._windows = torch.as_tensor(self.windows, device=self.device)
        self._window_ids = torch.as_tensor(self.window_indices, device=self.device)
        self._sliding_ends = torch.as_tensor(self.sliding_ends, device=self.device)
        self._sliding_offsets = torch.as_tensor(self.sliding_offsets, device=self.device)
        self._rows = torch.as_tensor(self.global_trajectory_indices, device=self.device)

    def _load(self, frame_budget: int) -> None:
        started = time.perf_counter()
        rows = self.global_trajectory_indices
        horizon = self.horizon
        chunk_rows = max(1, frame_budget // horizon)
        projection = self.native_projection
        with h5py.File(self.dataset_path, "r") as file:
            data_group = cast(h5py.Group, file["data"])
            for key in self.storage_data_keys:
                dataset = cast(h5py.Dataset, data_group[key])
                dtype = (
                    torch.bool
                    if key in BOOL_DATASET_KEYS
                    else torch.int64
                    if key in INTEGER_DATASET_KEYS
                    else torch.float16
                    if key in self.float16_keys
                    else torch.float32
                )
                self.resident[key] = torch.empty((len(rows), *dataset.shape[1:]), dtype=dtype, device=self.device)
            context = _trajectory_context_group(file)
            if context is not None:
                for key in self.trajectory_context_keys:
                    value = np.asarray(cast(h5py.Dataset, context[key])[np.asarray(rows)])
                    self.resident_context[key] = torch.as_tensor(value, device=self.device)
            native = cast(h5py.Group, file["contacts/native_pairs"]) if projection is not None else None
            counts = []
            pair_chunks: dict[str, list[torch.Tensor]] = {name: [] for name in self._pair_fields()}
            for start in range(0, len(rows), chunk_rows):
                selected = rows[start : start + chunk_rows]
                selection = (
                    slice(int(selected[0]), int(selected[-1]) + 1)
                    if selected[-1] - selected[0] + 1 == len(selected)
                    else selected
                )
                for key in self.storage_data_keys:
                    value = torch.from_numpy(np.ascontiguousarray(_read_dataset_array(data_group, key, selection)))
                    self.resident[key][start : start + len(selected)] = value.to(self.resident[key].dtype)
                if native is not None:
                    breaks = np.flatnonzero(np.diff(selected) != 1) + 1
                    for run in np.split(selected, breaks):
                        count, kept = self._read_pairs(native, int(run[0]), int(run[-1]) + 1)
                        counts.append(count)
                        for name, value in kept.items():
                            pair_chunks[name].append(value)
        if projection is not None:
            frame_counts = torch.cat(counts) if counts else torch.zeros(0, dtype=torch.long)
            offsets = torch.zeros(frame_counts.numel() + 1, dtype=torch.long)
            torch.cumsum(frame_counts, 0, out=offsets[1:])
            self.pair_offsets = offsets.to(self.device)
            self.pairs = {name: torch.cat(values).to(self.device) for name, values in pair_chunks.items()}
        resident_bytes = sum(
            value.numel() * value.element_size()
            for value in (*self.resident.values(), *self.resident_context.values(), *self.pairs.values())
        )
        print(
            f"[GPU dataset] rank={self.capacity_stats['rank']} dataset={self.dataset_path.name} "
            f"rows={len(rows)} retained_pairs={int(self.pair_offsets[-1]) if self.pair_offsets is not None else 0} "
            f"resident_bytes={resident_bytes} seconds={time.perf_counter() - started:.3f}",
            flush=True,
        )

    def _pair_fields(self) -> tuple[str, ...]:
        return () if self.native_projection is None else self.native_projection.input_keys

    def _read_pairs(
        self, native: h5py.Group, row_start: int, row_stop: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Read the native pairs of contiguous recording rows and drop those the contact filter rejects."""
        from isaaclab_neural.contacts.native_pair_projector import native_pair_filter_mask

        projection = self.native_projection
        assert projection is not None
        frame_count = (row_stop - row_start) * self.horizon
        offsets = np.asarray(
            cast(h5py.Dataset, native["frame_offsets"])[row_start * self.horizon : row_stop * self.horizon + 1],
            dtype=np.int64,
        )
        begin, end = int(offsets[0]), int(offsets[-1])
        values = {
            name: torch.from_numpy(
                np.asarray(cast(h5py.Dataset, native[name])[begin:end]).astype(canonical_native_dtype(name), copy=False)
            )
            for name in self._pair_fields()
        }
        frames = torch.repeat_interleave(torch.arange(frame_count), torch.from_numpy(np.diff(offsets)))
        keep = native_pair_filter_mask(values, projection.contact_filter, projection.fixed_body_slots)
        counts = torch.bincount(frames[keep], minlength=frame_count)
        dtypes = RESIDENT_PAIR_DTYPES if self.compact_pairs else {}
        return counts, {name: value[keep].to(dtypes.get(name, value.dtype)) for name, value in values.items()}

    def _resolve_windows(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        """Vectorized :meth:`UniqueTrajectoryDataset._window` on device indices."""
        if self.window_sampling == "sliding":
            block = torch.searchsorted(self._sliding_ends, indices, right=True)
            previous = torch.where(block > 0, self._sliding_ends[(block - 1).clamp_min(0)], 0)
            offset = indices - previous
            selected = self._blocks[block]
            return {
                "row": selected[:, 0],
                "start": selected[:, 1] + offset,
                "begin": torch.zeros_like(offset),
                "end": selected[:, 3],
                "window": self._sliding_offsets[block] + offset,
                "target": selected[:, 4] + offset,
            }
        selected = self._windows[indices]
        return {
            "row": selected[:, 0],
            "start": selected[:, 1],
            "begin": selected[:, 2],
            "end": selected[:, 3],
            "window": self._window_ids[indices],
        }

    def get_batch(self, indices: torch.Tensor | np.ndarray | Sequence[int]) -> dict[str, torch.Tensor]:
        """Return a batch of windows ``[B, T, ...]`` on the device, including projected contact tokens."""
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.device).reshape(-1)
        windows = self._resolve_windows(indices)
        length = self.sample_sequence_length
        local_rows = torch.searchsorted(self._rows, windows["row"])
        steps = torch.arange(length, device=self.device)
        frames = local_rows[:, None] * self.horizon + windows["start"][:, None] + steps
        batch: dict[str, torch.Tensor] = {}
        for key, value in self.resident.items():
            gathered = value.flatten(0, 1).index_select(0, frames.reshape(-1))
            if gathered.dtype == torch.float16:
                gathered = gathered.float()
            batch[key] = gathered.reshape(*frames.shape, *value.shape[2:])
        for key, value in self.resident_context.items():
            batch[key] = value.index_select(0, local_rows)
        if self.native_projection is not None:
            batch.update(self._project(frames.reshape(-1), frames.shape))
        mask = (steps >= windows["begin"][:, None]) & (steps < windows["end"][:, None])
        batch[SUPERVISION_MASK] = mask
        batch[SUPERVISION_SOURCE] = torch.full_like(indices, self.supervision_source_index)
        batch[SUPERVISION_WINDOW] = windows["window"]
        if "target" in windows:
            batch[SUPERVISION_TARGET_IDS] = torch.where(mask, windows["target"][:, None] + steps, -1)
        return batch

    def _project(self, frames: torch.Tensor, shape: torch.Size) -> dict[str, torch.Tensor]:
        """Convert the retained native pairs of ``frames`` into contact tokens."""
        from isaaclab_neural.contacts.native_pair_projector import project_native_pairs

        projection = self.native_projection
        assert projection is not None and self.pair_offsets is not None
        starts = self.pair_offsets.index_select(0, frames)
        counts = self.pair_offsets.index_select(0, frames + 1) - starts
        total = int(counts.sum())
        frame_ids = torch.repeat_interleave(torch.arange(frames.numel(), device=self.device), counts, output_size=total)
        first = torch.repeat_interleave(starts - (torch.cumsum(counts, 0) - counts), counts, output_size=total)
        pair_ids = first + torch.arange(total, device=self.device)
        native = {
            name: value.index_select(0, pair_ids).to(torch.int64 if name in NATIVE_INTEGER_FIELDS else torch.float32)
            for name, value in self.pairs.items()
        }
        projected = project_native_pairs(
            native,
            num_frames=frames.numel(),
            representation="contact_tokens",
            packing_policy="body_round_robin_pair_atomic",
            frame_ids=frame_ids,
            max_contact_tokens=projection.capacity,
        )
        return {key: projected[key].reshape(*shape, *projected[key].shape[1:]) for key in projection.output_keys}

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[0] for key, value in self.get_batch([index]).items()}

    def __getitems__(self, indices: list[int]) -> list[dict[str, torch.Tensor]]:
        batch = self.get_batch(indices)
        return [{key: value[i] for key, value in batch.items()} for i in range(len(indices))]


def mixed_get_batch(dataset: WeightedMixedTrajectoryDataset, indices: torch.Tensor) -> dict[str, torch.Tensor]:
    """Batch a weighted mixture of GPU-resident sources, keeping the virtual sample order."""
    boundaries = torch.as_tensor(dataset.source_boundaries, device=indices.device)
    source = torch.searchsorted(boundaries, indices, right=True)
    batch: dict[str, torch.Tensor] = {}
    for source_index, name in enumerate(dataset.source_names):
        positions = torch.nonzero(source == source_index, as_tuple=False).reshape(-1)
        if positions.numel() == 0:
            continue
        start = 0 if source_index == 0 else int(dataset.source_boundaries[source_index - 1])
        local = (indices.index_select(0, positions) - start) % len(dataset.datasets[name])
        part = cast(GpuResidentTrajectoryDataset, dataset.datasets[name]).get_batch(local)
        for key, value in part.items():
            if key not in batch:
                batch[key] = value.new_empty((indices.numel(), *value.shape[1:]))
            batch[key].index_copy_(0, positions, value)
    return batch


class GpuBatchLoader:
    """DataLoader replacement for GPU-resident datasets: shuffled index batches gathered on the device.

    Args:
        dataset: A :class:`GpuResidentTrajectoryDataset` or a weighted mixture of them.
        batch_size: Windows per batch.
        shuffle: Draw a new permutation every epoch.
        drop_last: Drop the final partial batch.
        seed: Base seed of the per-epoch permutations (offset by the epoch count).
    """

    def __init__(self, dataset: Any, batch_size: int, *, shuffle: bool, drop_last: bool = False, seed: int = 0):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = int(seed)
        self.epoch = 0
        mixed = isinstance(dataset, WeightedMixedTrajectoryDataset)
        sources = dataset.datasets.values() if mixed else (dataset,)
        if not all(isinstance(source, GpuResidentTrajectoryDataset) for source in sources):
            raise TypeError("GpuBatchLoader requires GPU-resident datasets.")
        self.device = next(iter(sources)).device
        self._get_batch = (lambda indices: mixed_get_batch(dataset, indices)) if mixed else dataset.get_batch

    def __len__(self) -> int:
        count = len(self.dataset)
        return count // self.batch_size if self.drop_last else -(-count // self.batch_size)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        count = len(self.dataset)
        if self.shuffle:
            generator = torch.Generator(device="cpu").manual_seed(self.seed + self.epoch)
            order = torch.randperm(count, generator=generator).to(self.device)
        else:
            order = torch.arange(count, device=self.device)
        self.epoch += 1
        for start in range(0, len(self) * self.batch_size, self.batch_size):
            yield self._get_batch(order[start : start + self.batch_size])
