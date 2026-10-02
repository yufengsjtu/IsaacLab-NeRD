# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HDF5 serialization helpers for NeRD rollout datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import torch

from isaaclab_neural.contacts.contact_set_schema import (
    NATIVE_CONTACT_DATASET_SCHEMA_VERSION,
    NATIVE_CONTACT_FRAME,
    NATIVE_CONTACT_PRECISION_ATTR,
    NATIVE_CONTACT_PRECISION_FULL,
    NATIVE_CONTACT_STORAGE_SCHEMA,
)
from isaaclab_neural.data.native_contacts import (
    NATIVE_PAIR_FIELDS,
    compact_native_arrays,
    prepare_native_contacts,
    validate_native_group,
)


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
    body_net_contact_force_body_names: Sequence[str] | None = None,
    native_contact_precision: str = NATIVE_CONTACT_PRECISION_FULL,
) -> None:
    """Serialize rollout tensors to a NeRD HDF5 trajectory dataset.

    Args:
        dataset_path: Output HDF5 path.
        rollouts: Rollout dictionary containing ``states``, ``next_states``,
            ``joint_f``, and a nested ``contacts`` dictionary.
        env_name: Environment name stored in dataset metadata.
        terrain_context: Optional deterministic terrain provenance.
        body_net_contact_force_body_names: Primary body names in force-label slot order.
        native_contact_precision: Stored native-pair precision, ``"full"`` (int64/float32) or ``"compact_v1"``
            (see ``NATIVE_CONTACT_COMPACT_DTYPES``). Readers widen compact values to the full dtypes.
    """
    dataset_path = Path(dataset_path).expanduser()
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _flatten_rollouts(rollouts)
    native_contacts = _prepare_native_rollouts(rollouts, arrays)
    trajectory_context = _trajectory_context_arrays(rollouts, arrays["states"].shape[0])
    _validate_rollout_arrays(arrays, has_native_contacts=native_contacts is not None)
    _validate_data_context_keys(arrays, trajectory_context)
    _validate_native_contact_identity(native_contacts, trajectory_context, arrays["states"].shape[:2])
    _validate_replay_context(arrays, terrain_context)
    _validate_requested_body_net_contact_force_body_names(arrays, body_net_contact_force_body_names)

    with h5py.File(dataset_path, "w") as dataset_file:
        data_group = dataset_file.create_group("data")
        for name, data in arrays.items():
            data_group.create_dataset(
                name=name,
                data=data,
                maxshape=(None, *data.shape[1:]),
                chunks=True,
            )
        if native_contacts is not None:
            _create_native_contact_group(dataset_file, *native_contacts, precision=native_contact_precision)
        _create_context_groups(dataset_file, trajectory_context, terrain_context)
        data_group.attrs["env"] = env_name
        data_group.attrs["mode"] = "trajectory"
        _update_metadata(data_group)
        _update_native_metadata(dataset_file, native_contacts is not None)
        _write_body_net_contact_force_body_names(data_group, body_net_contact_force_body_names)


def create_virtual_trajectory_dataset(dataset_path: str | Path, sources: Sequence[str | Path]) -> None:
    """Concatenate bounded trajectory files without copying dense tensors or native contact pairs.

    Sources must be immutable sibling files. Relative virtual links remain portable
    when the complete bundle is relocated. CSR offsets are rebased in bounded chunks.
    Replay timelines are intentionally not combined; replay each original recording.

    Args:
        dataset_path: New output path; existing files are never overwritten.
        sources: Immutable sibling HDF5 trajectory files, in concatenation order.
    """
    target = Path(dataset_path).resolve()
    paths = [Path(source).resolve() for source in sources]
    if target.exists():
        raise FileExistsError(target)
    if (
        not paths
        or len(set(paths)) != len(paths)
        or any(path.parent != target.parent or path == target for path in paths)
    ):
        raise ValueError("Virtual trajectory sources must be distinct sibling files.")
    groups = ("data", "context/trajectories", "contacts/native_pairs")
    schemas: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
    sizes: dict[str, list[int]] = {}
    attributes: dict[str, dict] = {}
    frame_counts, pair_counts = [], []
    for index, path in enumerate(paths):
        with h5py.File(path, "r") as file:
            names = set()
            for group_name in groups:
                group = file[group_name]
                if not isinstance(group, h5py.Group):
                    raise ValueError(f"Expected an HDF5 group at {group_name}.")
                if index == 0:
                    attributes[group_name] = dict(group.attrs)
                for name, data in group.items():
                    full_name = f"{group_name}/{name}"
                    if full_name == "contacts/native_pairs/frame_offsets":
                        continue
                    if not isinstance(data, h5py.Dataset) or data.ndim == 0:
                        raise ValueError(f"Expected a leading-dimension dataset at {full_name}.")
                    signature = (data.shape[1:], data.dtype)
                    if index == 0:
                        schemas[full_name] = signature
                        sizes[full_name] = []
                    if schemas.get(full_name) != signature:
                        raise ValueError(f"Incompatible virtual source field: {full_name}.")
                    names.add(full_name)
                    sizes[full_name].append(data.shape[0])
            if names != schemas.keys():
                raise ValueError("Virtual source fields differ.")
            states = file["data/states"]
            offsets = file["contacts/native_pairs/frame_offsets"]
            world_ids = file["contacts/native_pairs/world_ids"]
            if not isinstance(states, h5py.Dataset) or states.ndim != 3:
                raise ValueError("Expected a three-dimensional states dataset.")
            if not isinstance(offsets, h5py.Dataset) or not isinstance(world_ids, h5py.Dataset):
                raise ValueError("Expected native CSR offsets and world ID datasets.")
            frames = states.shape[0] * states.shape[1]
            pairs = world_ids.shape[0]
            if (
                offsets.shape != (frames + 1,)
                or offsets.dtype != np.dtype("int64")
                or offsets[0] != 0
                or offsets[-1] != pairs
            ):
                raise ValueError("Invalid native CSR source offsets.")
            frame_counts.append(frames)
            pair_counts.append(pairs)
    with h5py.File(target, "x", libver="latest") as output:
        for name, attrs in attributes.items():
            output.require_group(name).attrs.update(attrs)
        for name, (tail, dtype) in schemas.items():
            layout = h5py.VirtualLayout(shape=(sum(sizes[name]), *tail), dtype=dtype)
            start = 0
            for path, size in zip(paths, sizes[name]):
                if size:
                    layout[start : start + size] = h5py.VirtualSource(path.name, name, shape=(size, *tail))
                start += size
            output.create_virtual_dataset(name, layout)
        offsets = output.require_group("contacts/native_pairs").create_dataset(
            "frame_offsets", shape=(sum(frame_counts) + 1,), dtype="int64", chunks=True
        )
        frame_base = pair_base = 0
        for path, frames, pairs in zip(paths, frame_counts, pair_counts):
            with h5py.File(path, "r") as source:
                original = cast(h5py.Dataset, source["contacts/native_pairs/frame_offsets"])
                for start in range(0, frames, 65536):
                    stop = min(start + 65536, frames)
                    offsets[frame_base + start : frame_base + stop] = original[start:stop] + pair_base
            frame_base += frames
            pair_base += pairs
        offsets[-1] = pair_base
        _update_metadata(cast(h5py.Group, output["data"]))
        _update_native_metadata(output, True)


def append_rollouts_to_hdf5(
    dataset_path: str | Path,
    rollouts: Mapping,
    env_name: str,
    terrain_context: Mapping[str, Any] | None = None,
    body_net_contact_force_body_names: Sequence[str] | None = None,
) -> None:
    """Append rollout tensors to a NeRD HDF5 trajectory dataset.

    The appended file has the same schema as :func:`write_rollouts_to_hdf5`,
    but datasets are resizable along the trajectory dimension. This avoids
    keeping very large data-generation runs in memory until the end.
    """
    dataset_path = Path(dataset_path).expanduser()
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _flatten_rollouts(rollouts)
    native_contacts = _prepare_native_rollouts(rollouts, arrays)
    trajectory_context = _trajectory_context_arrays(rollouts, arrays["states"].shape[0])
    _validate_rollout_arrays(arrays, has_native_contacts=native_contacts is not None)
    _validate_data_context_keys(arrays, trajectory_context)
    _validate_native_contact_identity(native_contacts, trajectory_context, arrays["states"].shape[:2])
    _validate_replay_context(arrays, terrain_context)
    _validate_requested_body_net_contact_force_body_names(arrays, body_net_contact_force_body_names)

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
        _validate_native_append_target(dataset_file, native_contacts, is_new=is_new)
        if is_new:
            try:
                _create_appendable_datasets(data_group, arrays)
                if native_contacts is not None:
                    _create_native_contact_group(dataset_file, *native_contacts)
                _create_context_groups(dataset_file, trajectory_context, terrain_context)
                data_group.attrs["env"] = env_name
                data_group.attrs["mode"] = "trajectory"
            except Exception:
                for name in tuple(data_group.keys()):
                    del data_group[name]
                for name in ("contacts", "context"):
                    if name in dataset_file:
                        del dataset_file[name]
                raise
        else:
            _validate_body_net_contact_force_body_names(data_group, body_net_contact_force_body_names)
            trajectory_group = (
                cast(h5py.Group, dataset_file["context/trajectories"]) if "context" in dataset_file else None
            )
            _append_arrays_atomically(
                data_group,
                arrays,
                trajectory_group,
                trajectory_context,
                dataset_file=dataset_file,
                native_contacts=native_contacts,
            )
        _update_metadata(data_group)
        _update_native_metadata(dataset_file, native_contacts is not None)
        _write_body_net_contact_force_body_names(data_group, body_net_contact_force_body_names)


def _flatten_rollouts(rollouts: Mapping) -> dict[str, Any]:
    """Flatten rollout tensors and nested contact tensors into numpy arrays."""
    arrays = {}
    for key, value in rollouts.items():
        if key in {"trajectory_context", "native_contacts"}:
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


def _prepare_native_rollouts(
    rollouts: Mapping,
    arrays: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    """Validate the optional top-level native-contact mapping."""
    native_contacts = rollouts.get("native_contacts")
    if native_contacts is None:
        return None
    if not isinstance(native_contacts, Mapping):
        raise TypeError("native_contacts must be a mapping.")
    states = arrays["states"]
    return prepare_native_contacts(native_contacts, states.shape[:2])


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


def _validate_native_contact_identity(
    native_contacts: tuple[np.ndarray, dict[str, np.ndarray]] | None,
    trajectory_context: Mapping[str, Any],
    trajectory_shape: tuple[int, int],
) -> None:
    """Validate that each ragged contact pair belongs to its trajectory world."""
    if native_contacts is None:
        return
    required_context = {"state_world_id", "root_world_id", "contact_world_id"}
    missing_context = sorted(required_context - set(trajectory_context))
    if missing_context:
        raise ValueError(f"Native-contact rollouts are missing trajectory world context: {missing_context}.")
    state_world_ids = trajectory_context["state_world_id"]
    root_world_ids = trajectory_context["root_world_id"]
    contact_world_ids = trajectory_context["contact_world_id"]
    if not np.array_equal(state_world_ids, root_world_ids) or not np.array_equal(state_world_ids, contact_world_ids):
        raise ValueError("State, root, and contact trajectory rows must map to the same Newton worlds.")
    offsets, native_arrays = native_contacts
    frame_world_ids = np.broadcast_to(contact_world_ids[:, None], trajectory_shape).reshape(-1)
    expected_pair_world_ids = np.repeat(frame_world_ids, np.diff(offsets))
    if not np.array_equal(native_arrays["world_ids"], expected_pair_world_ids):
        raise ValueError("Native contact-pair world ids do not match their trajectory rows.")


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


def _validate_rollout_arrays(arrays: Mapping[str, Any], *, has_native_contacts: bool = False) -> None:
    """Validate required fields and common trajectory dimensions."""
    required = {"states", "next_states", "joint_f"}
    baked_contact_fields = sorted(name for name in arrays if name.startswith("contact_"))
    if baked_contact_fields:
        raise ValueError(
            "Contact-enabled rollouts must store only top-level native_contacts CSR data; "
            f"baked contact fields are incompatible: {baked_contact_fields}."
        )
    if has_native_contacts:
        # Replay needs maximal root kinematics in addition to generalized states.
        required.update(
            {
                "root_body_q",
                "root_body_qd",
                "gravity_dir",
            }
        )
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"Rollout data is missing required fields: {missing}.")

    finger_key = "finger_object_normal_forces_w"
    if finger_key in arrays:
        forces = arrays[finger_key]
        if not has_native_contacts or forces.ndim != 4 or forces.shape[-2:] != (2, 3):
            raise ValueError(f"{finger_key} requires native contacts and shape [B, T, 2, 3].")
        if not np.issubdtype(forces.dtype, np.floating) or not np.isfinite(forces).all():
            raise ValueError(f"{finger_key} requires finite floating-point labels.")

    force_key = "body_net_contact_forces_w"
    if force_key in arrays:
        if not has_native_contacts:
            raise ValueError(f"Rollout field {force_key!r} requires top-level native_contacts.")
        forces = arrays[force_key]
        if forces.ndim != 4 or forces.shape[-1] != 3 or forces.shape[-2] <= 0:
            raise ValueError(
                f"Rollout field {force_key!r} must have shape "
                f"[trajectories, steps, num_primary_bodies, 3], got {forces.shape}."
            )
        if not np.issubdtype(forces.dtype, np.floating):
            raise ValueError(f"Rollout field {force_key!r} must use a floating dtype, got {forces.dtype}.")

    pose_key = "primary_body_q"
    if pose_key in arrays:
        if not has_native_contacts:
            raise ValueError(f"Rollout field {pose_key!r} requires top-level native_contacts.")
        poses = arrays[pose_key]
        if poses.ndim != 4 or poses.shape[-1] != 7 or poses.shape[-2] <= 0:
            raise ValueError(
                f"Rollout field {pose_key!r} must have shape "
                f"[trajectories, steps, num_primary_bodies, 7], got {poses.shape}."
            )
        if not np.issubdtype(poses.dtype, np.floating):
            raise ValueError(f"Rollout field {pose_key!r} must use a floating dtype, got {poses.dtype}.")

    states = arrays["states"]
    if states.ndim < 3:
        raise ValueError(f"states must have shape [trajectories, steps, features], got {states.shape}.")
    trajectory_shape = states.shape[:2]
    for name, data in arrays.items():
        # Per-step scalars (e.g. contact_token_overflow) are [N, T]; all other fields are [N, T, ...].
        matches_trajectory = data.shape[:2] == trajectory_shape and (data.ndim >= 3 or data.shape == trajectory_shape)
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
    if "finger_object_normal_forces_w" in arrays:
        from isaaclab_neural.contacts.finger_object_force import validate_force_metadata

        validate_force_metadata(data_group)
    if "body_net_contact_forces_w" in arrays:
        expected_attrs = {
            "body_net_contact_force_schema": "per_primary_body_v1",
            "body_net_contact_force_frame": "world",
            "body_net_contact_force_units": "N",
            "body_net_contact_force_timing": "post_solve_t_to_t_plus_1",
        }
        mismatched = {
            name: data_group.attrs.get(name)
            for name, expected in expected_attrs.items()
            if data_group.attrs.get(name) != expected
        }
        if mismatched:
            raise ValueError(f"Cannot append body net contact-force labels with incompatible metadata: {mismatched}.")
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
    if "context/trajectories" not in dataset_file:
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


def _create_native_contact_group(
    dataset_file: h5py.File,
    frame_offsets: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    precision: str = NATIVE_CONTACT_PRECISION_FULL,
) -> None:
    """Create appendable native-pair CSR datasets."""
    arrays = compact_native_arrays(arrays, precision)
    contacts_group = dataset_file.create_group("contacts")
    native_group = contacts_group.create_group("native_pairs")
    if precision != NATIVE_CONTACT_PRECISION_FULL:
        native_group.attrs[NATIVE_CONTACT_PRECISION_ATTR] = precision
    try:
        native_group.create_dataset(
            "frame_offsets",
            data=frame_offsets,
            maxshape=(None,),
            chunks=True,
            compression="lzf",
            shuffle=True,
        )
        for name in NATIVE_PAIR_FIELDS:
            data = arrays[name]
            native_group.create_dataset(
                name,
                data=data,
                maxshape=(None, *data.shape[1:]),
                chunks=True,
                compression="lzf",
                shuffle=True,
            )
    except Exception:
        del dataset_file["contacts"]
        raise


def _validate_native_append_target(
    dataset_file: h5py.File,
    native_contacts: tuple[np.ndarray, dict[str, np.ndarray]] | None,
    *,
    is_new: bool,
) -> None:
    """Validate native-pair schema compatibility before append mutation."""
    if is_new:
        return
    has_native_group = "contacts/native_pairs" in dataset_file
    if has_native_group != (native_contacts is not None):
        raise ValueError("Native-contact storage presence does not match the existing dataset.")
    if not has_native_group:
        return
    data_group = cast(h5py.Group, dataset_file["data"])
    expected_attrs = {
        "dataset_schema_version": NATIVE_CONTACT_DATASET_SCHEMA_VERSION,
        "contact_storage_schema": NATIVE_CONTACT_STORAGE_SCHEMA,
        "contact_frame": NATIVE_CONTACT_FRAME,
    }
    native_group = cast(h5py.Group, dataset_file["contacts/native_pairs"])
    metadata_owners = {
        "/": dataset_file.attrs,
        "/data": data_group.attrs,
        "/contacts/native_pairs": native_group.attrs,
    }
    mismatched = {
        f"{owner}:{name}": attrs.get(name)
        for owner, attrs in metadata_owners.items()
        for name, expected in expected_attrs.items()
        if attrs.get(name) != expected
    }
    if mismatched:
        raise ValueError(f"Cannot append native contacts with incompatible metadata: {mismatched}.")
    total_transitions = data_group.attrs["total_transitions"]
    if not isinstance(total_transitions, (int, np.integer)):
        raise ValueError("Expected integer total_transitions metadata.")
    expected_frames = int(total_transitions)
    validate_native_group(native_group, expected_frames)
    assert native_contacts is not None
    _offsets, arrays = native_contacts
    for name in NATIVE_PAIR_FIELDS:
        dataset = cast(h5py.Dataset, native_group[name])
        data = arrays[name]
        if dataset.shape[1:] != data.shape[1:] or dataset.dtype != data.dtype:
            raise ValueError(
                f"Native-contact field {name!r} is incompatible: "
                f"existing={dataset.shape[1:]} {dataset.dtype}, new={data.shape[1:]} {data.dtype}."
            )


def _update_native_metadata(dataset_file: h5py.File, has_native_contacts: bool) -> None:
    """Write schema attributes for native-pair storage."""
    if not has_native_contacts:
        return
    data_group = cast(h5py.Group, dataset_file["data"])
    attrs = {
        "dataset_schema_version": NATIVE_CONTACT_DATASET_SCHEMA_VERSION,
        "contact_storage_schema": NATIVE_CONTACT_STORAGE_SCHEMA,
        "contact_frame": NATIVE_CONTACT_FRAME,
    }
    for name, value in attrs.items():
        data_group.attrs[name] = value
        dataset_file.attrs[name] = value
        cast(h5py.Group, dataset_file["contacts/native_pairs"]).attrs[name] = value


def _append_arrays_atomically(
    data_group: h5py.Group,
    arrays: Mapping[str, Any],
    trajectory_group: h5py.Group | None,
    trajectory_context: Mapping[str, Any],
    *,
    dataset_file: h5py.File,
    native_contacts: tuple[np.ndarray, dict[str, np.ndarray]] | None,
) -> None:
    """Append all fields and roll back their sizes if any write fails."""
    old_count = cast(h5py.Dataset, data_group["states"]).shape[0]
    append_count = arrays["states"].shape[0]
    resized: list[h5py.Dataset] = []
    native_old_sizes: dict[h5py.Dataset, int] = {}
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
        if native_contacts is not None:
            new_offsets, native_arrays = native_contacts
            native_group = cast(h5py.Group, dataset_file["contacts/native_pairs"])
            offsets_dataset = cast(h5py.Dataset, native_group["frame_offsets"])
            old_offset_count = offsets_dataset.shape[0]
            old_pair_count = int(offsets_dataset[-1])
            native_old_sizes[offsets_dataset] = old_offset_count
            offsets_dataset.resize((old_offset_count + new_offsets.shape[0] - 1,))
            offsets_dataset[old_offset_count:] = new_offsets[1:] + old_pair_count
            for name in NATIVE_PAIR_FIELDS:
                dataset = cast(h5py.Dataset, native_group[name])
                data = native_arrays[name]
                native_old_sizes[dataset] = dataset.shape[0]
                dataset.resize((old_pair_count + data.shape[0], *dataset.shape[1:]))
                dataset[old_pair_count:] = data
    except Exception:
        for dataset, old_size in native_old_sizes.items():
            dataset.resize((old_size, *dataset.shape[1:]))
        for dataset in resized:
            dataset.resize((old_count, *dataset.shape[1:]))
        raise


def _decode_body_names(value: Any) -> tuple[str, ...]:
    """Decode an HDF5 string-array attribute into body names."""
    return tuple(item.decode() if isinstance(item, bytes) else str(item) for item in np.asarray(value).reshape(-1))


def _validate_requested_body_net_contact_force_body_names(
    arrays: Mapping[str, Any],
    body_names: Sequence[str] | None,
) -> None:
    """Validate force-label body names before creating or resizing datasets."""
    if body_names is None:
        return
    force_key = "body_net_contact_forces_w"
    if force_key not in arrays:
        raise ValueError("Body net contact-force body names require body_net_contact_forces_w labels.")
    names = tuple(str(name) for name in body_names)
    expected_count = arrays[force_key].shape[-2]
    if len(names) != expected_count:
        raise ValueError(
            "Body net contact-force body-name count must match the force-label body axis: "
            f"names={len(names)}, labels={expected_count}."
        )
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError(f"Body net contact-force body names must be non-empty and unique, got {names}.")


def _validate_body_net_contact_force_body_names(
    data_group: h5py.Group,
    body_names: Sequence[str] | None,
) -> None:
    """Require appended force-label chunks to retain the existing body-slot order."""
    attr_name = "body_net_contact_force_body_names"
    if body_names is None or attr_name not in data_group.attrs:
        return
    existing = _decode_body_names(data_group.attrs[attr_name])
    requested = tuple(str(name) for name in body_names)
    if existing != requested:
        raise ValueError(
            "Cannot append body net contact-force labels with a different body-name order: "
            f"existing={existing}, requested={requested}."
        )


def _write_body_net_contact_force_body_names(
    data_group: h5py.Group,
    body_names: Sequence[str] | None,
) -> None:
    """Write primary body names in the same slot order as force labels."""
    if body_names is None:
        return
    force_key = "body_net_contact_forces_w"
    if force_key not in data_group:
        raise ValueError("Body net contact-force body names require body_net_contact_forces_w labels.")
    names = tuple(str(name) for name in body_names)
    expected_count = cast(h5py.Dataset, data_group[force_key]).shape[-2]
    if len(names) != expected_count:
        raise ValueError(
            "Body net contact-force body-name count must match the force-label body axis: "
            f"names={len(names)}, labels={expected_count}."
        )
    if len(set(names)) != len(names):
        raise ValueError(f"Body net contact-force body names must be unique, got {names}.")
    data_group.attrs["body_net_contact_force_body_names"] = np.asarray(
        names,
        dtype=h5py.string_dtype(encoding="utf-8"),
    )


def _update_metadata(data_group: h5py.Group) -> None:
    """Update aggregate metadata from successfully written datasets."""
    states = cast(h5py.Dataset, data_group["states"])
    data_group.attrs["total_trajectories"] = states.shape[0]
    data_group.attrs["total_transitions"] = states.shape[0] * states.shape[1]
    data_group.attrs["state_dim"] = states.shape[-1]
    data_group.attrs["next_state_dim"] = cast(h5py.Dataset, data_group["next_states"]).shape[-1]
    if "finger_object_normal_forces_w" in data_group:
        from isaaclab_neural.contacts.finger_object_force import METADATA

        for key, value in METADATA.items():
            data_group.attrs[key] = value
    if "body_net_contact_forces_w" in data_group:
        forces = cast(h5py.Dataset, data_group["body_net_contact_forces_w"])
        data_group.attrs["body_net_contact_force_schema"] = "per_primary_body_v1"
        data_group.attrs["body_net_contact_force_frame"] = "world"
        data_group.attrs["body_net_contact_force_units"] = "N"
        data_group.attrs["body_net_contact_force_timing"] = "post_solve_t_to_t_plus_1"
        data_group.attrs["num_primary_bodies"] = forces.shape[-2]
    data_group.attrs["joint_f_dim"] = cast(h5py.Dataset, data_group["joint_f"]).shape[-1]
    if "actions" in data_group:
        data_group.attrs["action_dim"] = cast(h5py.Dataset, data_group["actions"]).shape[-1]
