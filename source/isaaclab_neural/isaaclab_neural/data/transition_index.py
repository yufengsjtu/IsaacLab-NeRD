# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Source-identity capacity selection shared by collection gates and sequence loaders."""

from __future__ import annotations

import hashlib
import json
from array import array
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

SUPERVISION_MASK = "supervision_mask"
SUPERVISION_SOURCE = "supervision_source_index"
SUPERVISION_WINDOW = "supervision_window_index"
SUPERVISION_TARGET_IDS = "supervision_target_ids"
WINDOW_SAMPLING = ("canonical", "sliding")
CAPACITY_UNITS = ("stored_frame", "unique_transition")


@dataclass
class TransitionIndex:
    """A unique source pool with independently configured window sampling."""

    windows: np.ndarray
    intervals: np.ndarray
    recording_ids: tuple[str, ...]
    unique_available: int
    unique_selected: int
    raw_frame_slots: int
    sequence_length: int
    window_sampling: str = "canonical"
    sliding_blocks: np.ndarray | None = None
    group_target_counts: np.ndarray | None = None

    @property
    def stats(self) -> dict[str, int | str]:
        digest = hashlib.sha256(json.dumps([self.recording_ids, self.sequence_length]).encode())
        digest.update(self.intervals.astype("<i8", copy=False).tobytes())
        return {
            "index_schema": "sliding_unique_pool_v1"
            if self.window_sampling == "sliding"
            else "canonical_disjoint_blocks_v1",
            "window_sampling": self.window_sampling,
            "selection_sha256": digest.hexdigest(),
            "capacity_unit": "unique_transition",
            "capacity_scope": "global",
            "unique_available": self.unique_available,
            "unique_selected": self.unique_selected,
            "raw_frame_slots": self.raw_frame_slots,
            "num_windows": int(self.sliding_blocks[:, 2].sum())
            if self.sliding_blocks is not None
            else len(self.windows),
            "sequence_length": self.sequence_length,
        }


def _dataset(file: h5py.File, name: str) -> h5py.Dataset:
    value = file.get(name)
    if not isinstance(value, h5py.Dataset):
        raise ValueError(f"Missing transition-index dataset: {name}.")
    return value


def _metadata(file: h5py.File) -> dict:
    # Keyboard recordings use ``keyboard_contract``; other env-driven recordings use ``recording_contract``.
    attrs = file["data"].attrs
    value = json.loads(attrs.get("keyboard_contract", attrs.get("recording_contract", "{}")))
    if not isinstance(value, dict):
        raise ValueError("Recording metadata must be a mapping.")
    return value


def _recording_id(path: Path, file: h5py.File, metadata: dict) -> str:
    identity = metadata.get("recording_id", file["data"].attrs.get("recording_id"))
    if identity is not None:
        if not isinstance(identity, str) or not identity:
            raise ValueError("Recording identity must be a non-empty string.")
        return "recording:" + identity
    manifest = path.parent / "keyboard_dataset_manifest.json"
    if manifest.is_file():
        digest = json.loads(manifest.read_text()).get("files", {}).get(path.name, {}).get("sha256")
        if isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest):
            return "sha256:" + digest
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def _sliding_blocks(clips: np.ndarray, intervals: np.ndarray, sequence: int) -> tuple[np.ndarray, np.ndarray]:
    """Encode stride-one window ranges without expanding millions of window records."""
    groups: dict[tuple[int, int], tuple[list[int], list[int], list[int]]] = {}
    offset = 0
    for domain, world, begin, end in intervals:
        starts, stops, offsets = groups.setdefault((int(domain), int(world)), ([], [], []))
        starts.append(int(begin))
        stops.append(int(end))
        offsets.append(offset)
        offset += int(end - begin)
    group_ids = {key: index for index, key in enumerate(groups)}
    counts = np.asarray(
        [sum(end - begin for begin, end in zip(starts, stops, strict=True)) for starts, stops, _ in groups.values()],
        dtype=np.int64,
    )
    blocks = array("q")
    previous = None
    window_end = target_end = 0
    for domain, world, start, length, row in clips:
        domain, world, start, length, row = map(int, (domain, world, start, length, row))
        key = domain, world
        if key not in groups:
            continue
        if key != previous:
            previous, window_end, target_end = key, 0, 0
        starts, stops, offsets = groups[key]
        stop = min(start + length, stops[-1])
        if stop <= start:
            continue
        if stop - start >= sequence:
            first, last = max(start, window_end), stop - sequence + 1
            if first >= last:
                continue
            number, supervised = last - first, sequence
        elif stop > target_end:
            first, last, number, supervised = start, start + 1, 1, stop - start
        else:
            continue
        selected = bisect_right(starts, first) - 1
        if selected < 0 or first >= stops[selected]:
            raise ValueError("Window is outside the selected unique pool.")
        target = offsets[selected] + first - starts[selected]
        blocks.extend((row, first - start, number, supervised, target, group_ids[key]))
        window_end, target_end = max(window_end, last), max(target_end, stop)
    return np.asarray(blocks, dtype=np.int64).reshape(-1, 6), counts


def build_transition_index(
    path: str | Path,
    *,
    max_capacity: int | None = None,
    sequence_length: int = 1,
    build_windows: bool = True,
    window_sampling: str = "canonical",
) -> TransitionIndex:
    """Select a global unique source pool, then construct canonical or sliding windows.

    Source identity is recording/world/frame, never floating-point state equality.
    Sliding windows repeat supervision as in sequence training; only capacity-boundary
    context is masked. Canonical mode retains the historical single-owner objective.
    Existing accepted recordings can use their immutable manifest SHA256 identity.
    """
    if window_sampling not in WINDOW_SAMPLING:
        raise ValueError(f"Unsupported window sampling: {window_sampling}.")
    if sequence_length <= 0 or (max_capacity is not None and max_capacity <= 0):
        raise ValueError("Sequence length and unique-transition capacity must be positive.")
    path = Path(path).resolve()
    with h5py.File(path, "r") as file:
        metadata = _metadata(file)
        shape = _dataset(file, "data/states").shape
        if len(shape) != 3:
            raise ValueError("Unique capacity requires trajectory data.")
        parts = metadata.get("training_parts")
    if parts is not None:
        if not isinstance(parts, list) or not parts or any(Path(p).name != p or p == path.name for p in parts):
            raise ValueError("Invalid unique-capacity source manifest.")
        sources = [path.parent / part for part in parts]
    else:
        sources = [path]
    identities: dict[str, int] = {}
    identity_paths: dict[str, Path] = {}
    chunks = []
    row_offset = 0
    raw_frames = 0
    for source in sources:
        with h5py.File(source, "r") as file:
            meta = _metadata(file)
            if source != path and meta.get("training_parts"):
                raise ValueError("Unique capacity requires direct recording sources, not nested views.")
            states = _dataset(file, "data/states")
            rows, horizon = states.shape[:2]
            if states.ndim != 3 or horizon != shape[1]:
                raise ValueError("Inconsistent source trajectory shape.")
            required = ("source_env_id", "source_start_frame")
            context = file.get("context/trajectories")
            if not isinstance(context, h5py.Group) or any(key not in context for key in required):
                raise ValueError("Unique capacity requires source environment and frame identities.")
            worlds, starts = (np.asarray(context[key]) for key in required)
            lengths = np.asarray(file["data/traj_lengths"]) if "data/traj_lengths" in file else np.full(rows, horizon)
            if any(
                value.shape != (rows,) or not np.issubdtype(value.dtype, np.integer)
                for value in (worlds, starts, lengths)
            ):
                raise ValueError("Invalid source identity or trajectory-length arrays.")
            if np.any(worlds < 0) or np.any(starts < 0) or np.any(lengths <= 0) or np.any(lengths > horizon):
                raise ValueError("Source identities or trajectory lengths are out of bounds.")
            if "source_num_envs" in meta and np.any(worlds >= meta["source_num_envs"]):
                raise ValueError("Source world is outside its recording.")
            if "replay/frames/states" in file:
                timeline = _dataset(file, "replay/frames/states")
                if (
                    timeline.ndim != 3
                    or np.any(worlds >= timeline.shape[0])
                    or np.any(starts + lengths > timeline.shape[1])
                ):
                    raise ValueError("Source interval lies outside its recorded timeline.")
            done = np.zeros((rows, horizon), dtype=bool)
            for key in ("terminated", "truncated"):
                if "data/" + key not in file:
                    raise ValueError("Unique capacity requires recorded reset boundaries.")
                value = np.asarray(file["data/" + key])
                if value.shape != done.shape:
                    raise ValueError("Invalid reset-boundary layout.")
                done |= value.astype(bool)
            if np.any(done & (np.arange(horizon)[None, :] < lengths[:, None] - 1)):
                raise ValueError("A training trajectory crosses a reset boundary.")
            identity = _recording_id(source, file, meta)
            previous_path = identity_paths.setdefault(identity, source)
            if previous_path != source:
                with previous_path.open("rb") as first, source.open("rb") as second:
                    if hashlib.file_digest(first, "sha256").digest() != hashlib.file_digest(second, "sha256").digest():
                        raise ValueError("Recording identity collision between different source payloads.")
            domain = identities.setdefault(identity, len(identities))
            valid = lengths >= sequence_length
            chunks.append(
                np.column_stack((np.full(rows, domain), worlds, starts, lengths, np.arange(rows) + row_offset))[valid]
            )
            row_offset += rows
            raw_frames += int(lengths.sum())
    if row_offset != shape[0]:
        raise ValueError("Training view rows do not match the source recordings.")
    clips = np.concatenate(chunks).astype(np.int64, copy=False)
    order = np.lexsort((clips[:, 4], clips[:, 2], clips[:, 1], clips[:, 0]))
    windows, intervals = array("q"), array("q")
    available = selected = 0
    previous = None
    covered_end = 0
    for domain, world, start, length, row in clips[order]:
        domain, world, start, length, row = map(int, (domain, world, start, length, row))
        key = domain, world
        if key != previous:
            previous, covered_end = key, 0
        end = start + length
        begin = max(start, covered_end)
        count = max(0, end - begin)
        available += count
        take = count if max_capacity is None else min(count, max(0, max_capacity - selected))
        if take:
            intervals.extend((domain, world, begin, begin + take))
            selected += take
            if build_windows and window_sampling == "canonical":
                local_begin, local_end = begin - start, begin + take - start
                for target in range(local_begin, local_end, sequence_length):
                    stop = min(target + sequence_length, local_end)
                    window = min(target, length - sequence_length)
                    windows.extend((row, window, target - window, stop - window))
        covered_end = max(covered_end, end)
    selected_intervals = np.asarray(intervals, dtype=np.int64).reshape(-1, 4)
    sliding, group_counts = (
        _sliding_blocks(clips[order], selected_intervals, sequence_length)
        if build_windows and window_sampling == "sliding"
        else (None, None)
    )
    return TransitionIndex(
        windows=np.asarray(windows, dtype=np.int64).reshape(-1, 4),
        intervals=selected_intervals,
        window_sampling=window_sampling,
        sliding_blocks=sliding,
        group_target_counts=group_counts,
        recording_ids=tuple(identities),
        unique_available=available,
        unique_selected=selected,
        raw_frame_slots=raw_frames,
        sequence_length=sequence_length,
    )
