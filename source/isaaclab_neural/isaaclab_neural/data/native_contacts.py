# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validation and in-memory projection for native contact-pair datasets."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import h5py
import numpy as np
import torch

from isaaclab_neural.contacts.contact_set_schema import (
    CONTACT_TOKEN_DIM,
    NATIVE_CONTACT_COMPACT_DTYPES,
    NATIVE_CONTACT_DATASET_SCHEMA_VERSION,
    NATIVE_CONTACT_FRAME,
    NATIVE_CONTACT_INTEGER_FIELDS,
    NATIVE_CONTACT_PRECISION_ATTR,
    NATIVE_CONTACT_PRECISION_COMPACT,
    NATIVE_CONTACT_PRECISION_FULL,
    NATIVE_CONTACT_STORAGE_SCHEMA,
    NATIVE_CONTACT_VALUE_FIELDS,
    NATIVE_CONTACT_VECTOR_FIELDS,
)
from isaaclab_neural.contacts.packing import ContactPackingPolicy

NATIVE_PAIR_FIELDS = NATIVE_CONTACT_VALUE_FIELDS
NATIVE_INTEGER_FIELDS = frozenset(NATIVE_CONTACT_INTEGER_FIELDS)
ProjectionRepresentation = Literal["flat", "contact_tokens"]


def canonical_native_dtype(name: str) -> np.dtype:
    """Return the in-memory dtype of a native-pair field (int64 identifiers, float32 values)."""
    return np.dtype("int64") if name in NATIVE_INTEGER_FIELDS else np.dtype("float32")


def stored_native_dtype(name: str, precision: str) -> np.dtype:
    """Return the on-disk dtype of a native-pair field for a storage precision."""
    if precision == NATIVE_CONTACT_PRECISION_FULL:
        return canonical_native_dtype(name)
    if precision == NATIVE_CONTACT_PRECISION_COMPACT:
        return np.dtype(NATIVE_CONTACT_COMPACT_DTYPES[name])
    raise ValueError(f"Unsupported native-contact precision: {precision!r}.")


def native_group_precision(group: h5py.Group | Mapping[str, Any]) -> str:
    """Return the stored precision of a native-pair group (full precision when unspecified)."""
    attrs = getattr(group, "attrs", {})
    return str(_decode_attr(attrs.get(NATIVE_CONTACT_PRECISION_ATTR, NATIVE_CONTACT_PRECISION_FULL)))


def compact_native_arrays(arrays: Mapping[str, np.ndarray], precision: str) -> dict[str, np.ndarray]:
    """Cast canonical native-pair arrays to their stored dtypes, rejecting values the cast would change.

    Identifiers must fit the narrower integer types exactly, and every narrowed float must stay finite.
    """
    result = {}
    for name, value in arrays.items():
        dtype = stored_native_dtype(name, precision)
        if value.dtype == dtype:
            result[name] = value
            continue
        if dtype.kind == "i":
            limits = np.iinfo(dtype)
            if value.size and (value.min() < limits.min or value.max() > limits.max):
                raise ValueError(f"Native-contact field {name!r} exceeds its compact {dtype} range.")
            result[name] = value.astype(dtype)
        else:
            narrowed = value.astype(dtype)
            if not np.isfinite(narrowed).all():
                raise ValueError(f"Native-contact field {name!r} is not finite after narrowing to {dtype}.")
            result[name] = narrowed
    return result


def _to_numpy(value: Any) -> np.ndarray:
    """Convert one native-contact value to a CPU numpy array."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    raise TypeError(f"Native-contact values must be torch.Tensor or numpy.ndarray, got {type(value).__name__}.")


def prepare_native_contacts(
    native_contacts: Mapping[str, Any],
    trajectory_shape: tuple[int, int],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Validate native pair arrays and construct flattened CSR frame offsets."""
    if "frame_counts" not in native_contacts:
        raise ValueError("native_contacts is missing required field 'frame_counts'.")
    missing = sorted(set(NATIVE_PAIR_FIELDS) - set(native_contacts))
    extra = sorted(set(native_contacts) - {"frame_counts", *NATIVE_PAIR_FIELDS})
    if missing or extra:
        raise ValueError(f"Native-contact field mismatch: missing={missing}, extra={extra}.")

    counts = _to_numpy(native_contacts["frame_counts"])
    if counts.dtype != np.dtype("int64"):
        raise ValueError(f"native_contacts.frame_counts must use int64, got {counts.dtype}.")
    if counts.shape != trajectory_shape:
        raise ValueError(
            f"native_contacts.frame_counts must have trajectory shape {trajectory_shape}, got {counts.shape}."
        )
    if np.any(counts < 0):
        raise ValueError("native_contacts.frame_counts cannot contain negative counts.")

    pair_count = int(counts.sum(dtype=np.int64))
    arrays: dict[str, np.ndarray] = {}
    for name in NATIVE_PAIR_FIELDS:
        data = _to_numpy(native_contacts[name])
        expected_shape = (pair_count, 3) if name in NATIVE_CONTACT_VECTOR_FIELDS else (pair_count,)
        if data.shape != expected_shape:
            raise ValueError(f"Native-contact field {name!r} must have shape {expected_shape}, got {data.shape}.")
        expected_dtype = np.dtype("int64") if name in NATIVE_INTEGER_FIELDS else np.dtype("float32")
        if data.dtype != expected_dtype:
            raise ValueError(f"Native-contact field {name!r} must use {expected_dtype}, got {data.dtype}.")
        arrays[name] = data

    offsets = np.empty(counts.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts.reshape(-1), dtype=np.int64, out=offsets[1:])
    validate_native_offsets(offsets, pair_count)
    return offsets, arrays


def validate_native_offsets(offsets: np.ndarray, pair_count: int, expected_frames: int | None = None) -> None:
    """Validate native-contact CSR offsets."""
    if offsets.dtype != np.dtype("int64") or offsets.ndim != 1:
        raise ValueError(
            f"Native-contact frame_offsets must be one-dimensional int64, got {offsets.dtype} {offsets.shape}."
        )
    if expected_frames is not None and offsets.shape != (expected_frames + 1,):
        raise ValueError(
            f"Native-contact frame_offsets must contain {expected_frames + 1} entries, got {offsets.shape[0]}."
        )
    if offsets.size == 0 or offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("Native-contact frame_offsets must start at zero and be monotonic.")
    if int(offsets[-1]) != pair_count:
        raise ValueError(f"Native-contact final frame offset must equal P={pair_count}, got {int(offsets[-1])}.")


def validate_native_group(group: h5py.Group, expected_frames: int) -> int:
    """Validate a stored native-contact CSR group and return its pair count."""
    missing = sorted({"frame_offsets", *NATIVE_PAIR_FIELDS} - set(group))
    extra = sorted(set(group) - {"frame_offsets", *NATIVE_PAIR_FIELDS})
    if missing or extra:
        raise ValueError(f"Stored native-contact field mismatch: missing={missing}, extra={extra}.")
    offsets_dataset = cast(h5py.Dataset, group["frame_offsets"])
    if offsets_dataset.dtype != np.dtype("int64"):
        raise ValueError(f"Native-contact frame_offsets must use int64, got {offsets_dataset.dtype}.")
    offsets = np.asarray(offsets_dataset[()], dtype=np.int64)
    pair_count = int(cast(h5py.Dataset, group[NATIVE_PAIR_FIELDS[0]]).shape[0])
    validate_native_offsets(offsets, pair_count, expected_frames)
    precision = native_group_precision(group)
    for name in NATIVE_PAIR_FIELDS:
        dataset = cast(h5py.Dataset, group[name])
        expected_shape = (pair_count, 3) if name in NATIVE_CONTACT_VECTOR_FIELDS else (pair_count,)
        if dataset.shape != expected_shape:
            raise ValueError(
                f"Stored native-contact field {name!r} must have shape {expected_shape}, got {dataset.shape}."
            )
        expected_dtype = stored_native_dtype(name, precision)
        if dataset.dtype != expected_dtype:
            raise ValueError(f"Stored native-contact field {name!r} must use {expected_dtype}, got {dataset.dtype}.")
    return pair_count


def validate_native_storage(dataset_file: h5py.File, expected_frames: int) -> h5py.Group | None:
    """Validate native-contact groups and metadata in an HDF5 dataset."""
    if "contacts" not in dataset_file:
        return None
    contacts_group = cast(h5py.Group, dataset_file["contacts"])
    if set(contacts_group) != {"native_pairs"}:
        raise ValueError(f"The /contacts group must contain only native_pairs, got {sorted(contacts_group.keys())}.")
    data_group = cast(h5py.Group, dataset_file["data"])
    native_group = cast(h5py.Group, contacts_group["native_pairs"])
    expected_attrs = {
        "dataset_schema_version": NATIVE_CONTACT_DATASET_SCHEMA_VERSION,
        "contact_storage_schema": NATIVE_CONTACT_STORAGE_SCHEMA,
        "contact_frame": NATIVE_CONTACT_FRAME,
    }
    metadata_owners = {
        "/": dataset_file.attrs,
        "/data": data_group.attrs,
        "/contacts/native_pairs": native_group.attrs,
    }
    mismatched = {
        f"{owner}:{name}": _decode_attr(attrs.get(name))
        for owner, attrs in metadata_owners.items()
        for name, expected in expected_attrs.items()
        if _decode_attr(attrs.get(name)) != expected
    }
    if mismatched:
        raise ValueError(
            "Native-contact dataset must use /contacts/native_pairs schema "
            f"v{NATIVE_CONTACT_DATASET_SCHEMA_VERSION}: {mismatched}."
        )
    validate_native_group(native_group, expected_frames)
    return native_group


def _decode_attr(value: Any) -> Any:
    """Decode byte-string HDF5 attributes while leaving other values unchanged."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


@dataclass(frozen=True)
class NativeContactProjection:
    """Projection configuration for variable native pairs."""

    representation: ProjectionRepresentation
    capacity: int
    packing_policy: str
    contact_filter: str = "none"
    fixed_body_slots: tuple[int, ...] | None = None

    @property
    def output_keys(self) -> tuple[str, ...]:
        """Return fields generated by this projection."""
        if self.representation == "flat":
            return (
                "contact_masks",
                "contact_normals",
                "contact_depths",
                "contact_thicknesses_0",
                "contact_thicknesses_1",
                "contact_points_0",
                "contact_points_1",
            )
        return (
            "contact_tokens",
            "contact_token_overflow",
            "contact_token_body_ids",
            "contact_token_world_ids",
        )

    @property
    def input_keys(self) -> tuple[str, ...]:
        """Return native fields required by this projection."""
        if self.representation == "flat":
            return (
                "world_ids",
                "shape_ids_0",
                "shape_ids_1",
                "points_0_w",
                "points_1_w",
                "normal_w",
                "separation",
                "thickness_0",
                "thickness_1",
            )
        return (
            "body_ids_0",
            "body_ids_1",
            "body_slots_0",
            "body_slots_1",
            "world_ids",
            "surface_points_0_w",
            "surface_points_1_w",
            "normal_w",
            "separation",
            "lever_arms_0_w",
            "lever_arms_1_w",
            "relative_velocities_0_w",
            "relative_velocities_1_w",
        )


def read_contact_mobility(data_group: h5py.Group, contact_filter: str) -> tuple[int, ...] | None:
    """Read the verified recording contract required for movable-only token filtering."""
    if contact_filter != "solver_active_movable":
        return None
    attrs = data_group.attrs
    metadata = json.loads(attrs.get("keyboard_contract", attrs.get("recording_contract", "{}")))
    values = metadata.get("contact_fixed_body_slots")
    if not isinstance(values, list) or any(type(value) is not int or value < 0 for value in values):
        raise ValueError("Movable filtering requires recorded fixed body slots (mobility metadata).")
    if metadata.get("contact_filter") != contact_filter:
        raise ValueError("Stored contact filter differs from the requested training projection.")
    return tuple(values)


def resolve_native_projection(
    *,
    has_native_contacts: bool,
    native_pair_count: int | None = None,
    contact_representation: str | None,
    num_contacts_per_env: int | None,
    max_contact_tokens: int | None,
    contact_packing_policy: str | None,
    contact_filter: str = "none",
    fixed_body_slots: tuple[int, ...] | None = None,
) -> NativeContactProjection | None:
    """Resolve requested projection settings and require native storage."""
    del native_pair_count
    flat_capacity = int(num_contacts_per_env or 0)
    token_capacity = int(max_contact_tokens or 0)
    if contact_representation == "flat" and flat_capacity == 0:
        return None
    if contact_representation == "contact_tokens" and token_capacity == 0:
        return None
    requests_contacts = flat_capacity > 0 or token_capacity > 0
    if contact_representation is None and not requests_contacts:
        return None
    if not has_native_contacts:
        raise ValueError(
            "A positive contact capacity requires /contacts/native_pairs with native-contact dataset "
            f"schema v{NATIVE_CONTACT_DATASET_SCHEMA_VERSION}."
        )
    if contact_representation is None:
        raise ValueError("contact_representation is required when requesting a positive contact capacity.")
    if contact_filter not in {"none", "solver_active", "solver_active_movable"}:
        raise ValueError(f"Unsupported contact_filter: {contact_filter!r}.")
    if contact_filter == "solver_active_movable":
        if contact_representation != "contact_tokens" or fixed_body_slots is None:
            raise ValueError("Movable token projection requires recorded fixed body slots and token representation.")
    if contact_representation == "flat":
        if flat_capacity <= 0:
            raise ValueError("Native flat projection requires a positive num_contacts_per_env.")
        policy = contact_packing_policy or "penetration_priority"
        if policy not in {"stable_index", "penetration_priority", "force_priority", "random"}:
            raise ValueError(f"Unsupported flat contact_packing_policy: {policy!r}.")
        return NativeContactProjection("flat", flat_capacity, policy, contact_filter)
    if contact_representation == "contact_tokens":
        if token_capacity <= 0:
            raise ValueError("Native contact-token projection requires a positive max_contact_tokens.")
        policy = contact_packing_policy or "body_round_robin_pair_atomic"
        if policy != "body_round_robin_pair_atomic":
            raise ValueError(
                "Native contact-token projection requires contact_packing_policy='body_round_robin_pair_atomic'."
            )
        return NativeContactProjection("contact_tokens", token_capacity, policy, contact_filter, fixed_body_slots)
    raise ValueError(f"Unsupported contact_representation: {contact_representation!r}.")


def read_and_project_frames(
    group: h5py.Group | Mapping[str, h5py.Dataset],
    frame_indices: np.ndarray,
    projection: NativeContactProjection,
) -> dict[str, np.ndarray]:
    """Read selected CSR frames and project them through the shared Torch projector."""
    raw_indices = np.asarray(frame_indices)
    if raw_indices.dtype != np.dtype("int64"):
        raise ValueError(f"Native-contact frame indices must use int64, got {raw_indices.dtype}.")
    indices = raw_indices.reshape(-1)
    if indices.size == 0:
        return _project_native_pairs(
            np.zeros(1, dtype=np.int64),
            _empty_native_arrays(projection.input_keys),
            projection,
        )
    frame_count = cast(h5py.Dataset, group["frame_offsets"]).shape[0] - 1
    if np.any(indices < 0) or np.any(indices >= frame_count):
        raise IndexError(f"Native-contact frame indices must be in [0, {frame_count}), got {indices}.")
    order = np.argsort(indices, kind="stable")
    sorted_indices = indices[order]
    if projection.packing_policy == "body_round_robin_pair_atomic" and np.any(
        sorted_indices[1:] == sorted_indices[:-1]
    ):
        unique, inverse = np.unique(indices, return_inverse=True)
        projected = read_and_project_frames(group, unique, projection)
        return {key: value[inverse] for key, value in projected.items()}
    frames: list[dict[str, np.ndarray] | None] = [None] * indices.size
    run_start = 0
    while run_start < sorted_indices.size:
        run_end = run_start + 1
        while run_end < sorted_indices.size and sorted_indices[run_end] == sorted_indices[run_end - 1] + 1:
            run_end += 1
        run_indices = sorted_indices[run_start:run_end]
        run_frames = _read_contiguous_frames(
            group,
            int(run_indices[0]),
            int(run_indices[-1]) + 1,
            projection.input_keys,
            projection,
        )
        for sorted_position, frame in enumerate(run_frames, start=run_start):
            frames[int(order[sorted_position])] = frame
        run_start = run_end
    return _project_native_pairs(
        *_frames_to_csr(cast(list[dict[str, np.ndarray]], frames), projection.input_keys),
        projection,
    )


def _read_contiguous_frames(
    group: h5py.Group | Mapping[str, h5py.Dataset],
    frame_start: int,
    frame_end: int,
    field_names: tuple[str, ...],
    projection: NativeContactProjection | None = None,
) -> list[dict[str, np.ndarray]]:
    """Read contiguous pairs, skipping geometry when no pair survives the configured filter."""
    offsets_dataset = cast(h5py.Dataset, group["frame_offsets"])
    offsets = np.asarray(offsets_dataset[frame_start : frame_end + 1], dtype=np.int64)
    value_start, value_end = int(offsets[0]), int(offsets[-1])
    if value_start == value_end:
        return [_empty_native_arrays(field_names) for _ in range(frame_end - frame_start)]
    values = {}
    if projection is not None and projection.contact_filter != "none":
        from isaaclab_neural.contacts.native_pair_projector import native_pair_filter_mask

        filter_fields = ("separation",)
        if projection.contact_filter == "solver_active_movable":
            filter_fields += ("body_ids_0", "body_ids_1", "body_slots_0", "body_slots_1")
        values = {name: _read_native_field(group, name, value_start, value_end) for name in filter_fields}
        keep = native_pair_filter_mask(
            {name: torch.from_numpy(value) for name, value in values.items()},
            projection.contact_filter,
            projection.fixed_body_slots,
        )
        if not bool(keep.any()):
            return [_empty_native_arrays(field_names) for _ in range(frame_end - frame_start)]
    values.update(
        {name: _read_native_field(group, name, value_start, value_end) for name in field_names if name not in values}
    )
    relative_offsets = offsets - value_start
    return [
        {name: value[int(relative_offsets[index]) : int(relative_offsets[index + 1])] for name, value in values.items()}
        for index in range(frame_end - frame_start)
    ]


def _read_native_field(group: h5py.Group | Mapping[str, h5py.Dataset], name: str, start: int, end: int) -> np.ndarray:
    """Read one native-pair field range, widening compact storage to the canonical dtype."""
    return np.asarray(cast(h5py.Dataset, group[name])[start:end]).astype(canonical_native_dtype(name), copy=False)


def _empty_native_arrays(field_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Create correctly shaped empty native-pair arrays."""
    return {
        name: np.empty(
            (0, 3) if name in NATIVE_CONTACT_VECTOR_FIELDS else (0,),
            dtype=np.int64 if name in NATIVE_INTEGER_FIELDS else np.float32,
        )
        for name in field_names
    }


def _frames_to_csr(
    frames: list[dict[str, np.ndarray]],
    field_names: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Combine selected frames into one CSR batch in requested order."""
    offsets = np.empty(len(frames) + 1, dtype=np.int64)
    offsets[0] = 0
    counts = np.asarray([frame["separation"].shape[0] for frame in frames], dtype=np.int64)
    np.cumsum(counts, dtype=np.int64, out=offsets[1:])
    arrays = {
        name: np.concatenate([frame[name] for frame in frames], axis=0)
        if int(offsets[-1]) > 0
        else _empty_native_arrays(field_names)[name]
        for name in field_names
    }
    return offsets, arrays


def _project_native_pairs(
    frame_offsets: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    projection: NativeContactProjection,
) -> dict[str, np.ndarray]:
    """Call the shared pure-Torch native-pair projector and validate its outputs."""
    from isaaclab_neural.contacts.native_pair_projector import project_native_pairs

    tensor_arrays = {name: torch.from_numpy(value) for name, value in arrays.items()}
    frame_counts = torch.from_numpy(np.diff(frame_offsets))
    frame_ids = torch.repeat_interleave(
        torch.arange(frame_counts.shape[0], dtype=torch.long),
        frame_counts,
    )
    projected = project_native_pairs(
        tensor_arrays,
        num_frames=frame_counts.shape[0],
        representation=projection.representation,
        packing_policy=cast(ContactPackingPolicy, projection.packing_policy),
        frame_ids=frame_ids,
        flat_capacity=projection.capacity if projection.representation == "flat" else None,
        max_contact_tokens=projection.capacity if projection.representation == "contact_tokens" else None,
        contact_filter=projection.contact_filter,
        fixed_body_slots=projection.fixed_body_slots,
    )
    if not isinstance(projected, Mapping):
        raise TypeError(f"Native-pair projector must return a mapping, got {type(projected).__name__}.")
    missing = sorted(set(projection.output_keys) - set(projected))
    extra = sorted(set(projected) - set(projection.output_keys))
    if missing or extra:
        raise ValueError(f"Native-pair projector output mismatch: missing={missing}, extra={extra}.")
    frame_count = frame_offsets.shape[0] - 1
    if projection.representation == "flat":
        expected_shapes = {
            "contact_masks": (frame_count, projection.capacity),
            "contact_normals": (frame_count, projection.capacity * 3),
            "contact_depths": (frame_count, projection.capacity),
            "contact_thicknesses_0": (frame_count, projection.capacity),
            "contact_thicknesses_1": (frame_count, projection.capacity),
            "contact_points_0": (frame_count, projection.capacity * 3),
            "contact_points_1": (frame_count, projection.capacity * 3),
        }
    else:
        expected_shapes = {
            "contact_tokens": (frame_count, projection.capacity, CONTACT_TOKEN_DIM),
            "contact_token_overflow": (frame_count,),
            "contact_token_body_ids": (frame_count, projection.capacity),
            "contact_token_world_ids": (frame_count, projection.capacity),
        }
    integer_outputs = {
        "contact_token_overflow",
        "contact_token_body_ids",
        "contact_token_world_ids",
    }
    result: dict[str, np.ndarray] = {}
    for name in projection.output_keys:
        value = projected[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Native-pair projector field {name!r} must be a torch.Tensor.")
        if tuple(value.shape) != expected_shapes[name]:
            raise ValueError(
                f"Native-pair projector field {name!r} must have shape {expected_shapes[name]}, "
                f"got {tuple(value.shape)}."
            )
        if name == "contact_masks":
            expected_dtype = torch.bool
        elif name in integer_outputs:
            expected_dtype = torch.int64
        else:
            expected_dtype = torch.float32
        if value.dtype != expected_dtype:
            raise ValueError(f"Native-pair projector field {name!r} must use {expected_dtype}, got {value.dtype}.")
        result[name] = value.detach().cpu().numpy()
    return result
