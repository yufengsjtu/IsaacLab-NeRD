# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone helpers for transition-paired contact-representation datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from isaaclab_neural.contacts.contact_set_schema import (
    CONTACT_REPRESENTATION_TOKENS,
    CONTACT_TOKEN_DIM,
    CONTACT_TOKEN_SOLVER_ACTIVE_FIELD,
)
from isaaclab_neural.generate.adapter import DataGenerationAdapter
from isaaclab_neural.generate.sampler import ActionTrajectorySampler

_CONTACT_DATA_KEYS = {
    "contact_depths",
    "contact_masks",
    "contact_normals",
    "contact_points_0",
    "contact_points_1",
    "contact_thicknesses_0",
    "contact_thicknesses_1",
    "contact_token_body_ids",
    "contact_token_overflow",
    "contact_token_solver_active",
    "contact_token_world_ids",
    "contact_tokens",
}
_PRIMARY_ROLLOUT_CONTACT_KEYS = {
    "contacts",
    "contact_token_body_ids",
    "contact_token_world_ids",
}
_PAIRED_ROLLOUT_KEYS = {
    "paired_contacts",
    "paired_contact_token_body_ids",
    "paired_contact_token_world_ids",
}
_PAIRED_POLICY_SUITE_EXPECTED = {
    "schema_version": 1,
    "requested_transitions": 1_000_000,
    "num_envs": 1024,
    "trajectory_length": 400,
    "seed": 40,
    "randomize_pd_gains": False,
    "task": "Isaac-Velocity-Rough-Anymal-C-v0",
    "contact_packing_policy": "body_round_robin_pair_atomic",
    "max_contact_tokens": 64,
}
_PAIRED_POLICY_CHECKPOINT = {
    "size_bytes": 6_882_293,
    "sha256": "833870337db02e88a660dd0d0b7921ea2a19dcb9c62bffe9660b3775976f3d3a",
}
_PAIRED_POLICY_AGENT_CONFIG = {
    "size_bytes": 1_326,
    "sha256": "9d5feef5b5ad6e9097ea3d740a14d74460cf54108510954d78b173950f7e852e",
}


def sha256_file(path: str | Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for a local file."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def truncation_summary_delta(
    current: Mapping[str, int | float],
    baseline: Mapping[str, int | float],
) -> dict[str, int | float]:
    """Subtract cumulative adapter counters and recompute the interval ratio."""
    if set(current) != set(baseline):
        raise ValueError("Truncation summary fields changed during paired data generation.")
    delta: dict[str, int | float] = {}
    for name, current_value in current.items():
        if name == "truncated_frame_ratio":
            continue
        value = current_value - baseline[name]
        if value < 0:
            raise ValueError(f"Truncation counter {name!r} decreased during paired data generation.")
        delta[name] = int(value) if isinstance(current_value, int) else float(value)
    frames = int(delta["frames"])
    truncated_frames = int(delta["truncated_frames"])
    delta["truncated_frame_ratio"] = truncated_frames / max(frames, 1)
    return delta


def validate_paired_policy_suite_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable policy-suite provenance used for paired evaluation."""
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in _PAIRED_POLICY_SUITE_EXPECTED.items()
        if manifest.get(key) != value
    }
    for field, expected in (
        ("policy_checkpoint", _PAIRED_POLICY_CHECKPOINT),
        ("policy_agent_config", _PAIRED_POLICY_AGENT_CONFIG),
    ):
        actual = manifest.get(field)
        if not isinstance(actual, Mapping):
            mismatches[field] = {"expected": expected, "actual": actual}
            continue
        for key, value in expected.items():
            if actual.get(key) != value:
                mismatches[f"{field}.{key}"] = {"expected": value, "actual": actual.get(key)}
    if mismatches:
        raise ValueError(f"Paired policy-suite manifest does not match the frozen contract: {mismatches}.")

    pairing = manifest.get("pairing")
    if not isinstance(pairing, Mapping):
        raise ValueError("Paired policy-suite manifest is missing pairing identity.")
    actual_transitions = int(manifest.get("actual_transitions", -1))
    trajectory_length = int(manifest["trajectory_length"])
    if actual_transitions < int(manifest["requested_transitions"]):
        raise ValueError("Paired policy suite contains fewer transitions than requested.")
    if actual_transitions % trajectory_length != 0:
        raise ValueError("Paired policy-suite transition count is not a whole number of trajectories.")
    if actual_transitions != int(pairing.get("total_transitions", -1)):
        raise ValueError("Paired policy-suite transition count disagrees with pairing identity.")
    if actual_transitions // trajectory_length != int(pairing.get("num_trajectories", -1)):
        raise ValueError("Paired policy-suite trajectory count disagrees with pairing identity.")
    if trajectory_length != int(pairing.get("steps_per_trajectory", -1)):
        raise ValueError("Paired policy-suite trajectory length disagrees with pairing identity.")
    return {
        **_PAIRED_POLICY_SUITE_EXPECTED,
        "actual_transitions": actual_transitions,
        "policy_checkpoint": dict(_PAIRED_POLICY_CHECKPOINT),
        "policy_agent_config": dict(_PAIRED_POLICY_AGENT_CONFIG),
    }


class PairedDataGenerationAdapter(DataGenerationAdapter):
    """Record two contact representations from each exact Newton frame."""

    def __init__(
        self,
        env: Any,
        solver_cfg: Any,
        *,
        paired_contact_representation: str = CONTACT_REPRESENTATION_TOKENS,
    ):
        # DataGenerationAdapter.__init__ dynamically calls self.sync(), so the
        # secondary adapter must exist as an explicit empty state first.
        self.paired_contact_adapter = None
        self._paired_body_world_torch: torch.Tensor | None = None
        super().__init__(env, solver_cfg)
        if self.contact_mode != "newton_native":
            raise ValueError("Paired contact generation requires contact_mode='newton_native'.")
        if paired_contact_representation == getattr(self.solver, "contact_representation", None):
            raise ValueError("The paired contact representation must differ from the primary representation.")

        from isaaclab_neural.contacts import NewtonContactAdapter

        self.paired_contact_adapter = NewtonContactAdapter(
            self.model,
            num_contacts_per_env=int(solver_cfg.num_contacts_per_env),
            device=str(self.device),
            packing_policy=solver_cfg.contact_packing_policy,
            contact_representation=paired_contact_representation,
            contact_filter="none",
            max_contact_tokens=int(solver_cfg.max_contact_tokens),
        )
        self.sync()

    def sync(self) -> None:
        """Synchronize both encoders from one collision result and one state."""
        contacts = self._prepare_contacts()
        self.backend.sync_solver(self.solver, contacts)
        if self.paired_contact_adapter is not None:
            self.paired_contact_adapter.update(contacts, self.state)

    @property
    def paired_contact_token_body_ids(self) -> torch.Tensor:
        """Packed owner-body ids for the secondary contact representation."""
        if self.paired_contact_adapter is None:
            raise RuntimeError("The paired contact adapter has not been initialized.")
        return self.paired_contact_adapter.contact_token_body_ids

    @property
    def paired_contact_token_world_ids(self) -> torch.Tensor:
        """Newton world ids for secondary packed owner-body ids."""
        body_ids = self.paired_contact_token_body_ids
        world_ids = torch.full_like(body_ids, -1)
        valid = body_ids >= 0
        if not valid.any():
            return world_ids
        body_world = getattr(self.model, "body_world", None)
        if body_world is None:
            raise RuntimeError("Newton body_world metadata is required for paired contact-token identities.")
        if self._paired_body_world_torch is None or self._paired_body_world_torch.device != body_ids.device:
            self._paired_body_world_torch = torch.as_tensor(
                body_world.numpy(),
                device=body_ids.device,
                dtype=torch.long,
            )
        world_ids[valid] = self._paired_body_world_torch.index_select(0, body_ids[valid])
        return world_ids

    def paired_raw_contact_inputs(self) -> dict[str, torch.Tensor]:
        """Return secondary raw contact tensors with a singleton time axis."""
        if self.paired_contact_adapter is None:
            raise RuntimeError("The paired contact adapter has not been initialized.")
        return {name: value.unsqueeze(1) for name, value in self.paired_contact_adapter.to_neural_inputs().items()}

    def paired_contact_truncation_summary(self) -> dict[str, int | float]:
        """Return secondary contact-packing statistics."""
        if self.paired_contact_adapter is None:
            raise RuntimeError("The paired contact adapter has not been initialized.")
        return self.paired_contact_adapter.truncation_summary()


class _PairedContactSamplerMixin:
    """Add secondary-contact buffers to an existing trajectory sampler."""

    adapter: PairedDataGenerationAdapter

    def allocate_batch_buffers(self, record_actions: bool = True) -> dict[str, Any]:
        buffers = super().allocate_batch_buffers(record_actions=record_actions)
        paired_adapter = self.adapter.paired_contact_adapter
        if paired_adapter is None:
            raise RuntimeError("The paired contact adapter has not been initialized.")
        max_tokens = int(paired_adapter.max_contact_tokens)
        num_envs = self.adapter.num_envs
        trajectory_length = self.trajectory_length
        buffers["paired_contacts"] = {
            "contact_tokens": torch.empty(
                (num_envs, trajectory_length, max_tokens, CONTACT_TOKEN_DIM),
                device=self.data_device,
            ),
            "contact_token_overflow": torch.empty(
                (num_envs, trajectory_length),
                dtype=torch.long,
                device=self.data_device,
            ),
            CONTACT_TOKEN_SOLVER_ACTIVE_FIELD: torch.empty(
                (num_envs, trajectory_length, max_tokens),
                dtype=torch.bool,
                device=self.data_device,
            ),
        }
        buffers["paired_contact_token_body_ids"] = torch.empty(
            (num_envs, trajectory_length, max_tokens),
            dtype=torch.long,
            device=self.data_device,
        )
        buffers["paired_contact_token_world_ids"] = torch.empty(
            (num_envs, trajectory_length, max_tokens),
            dtype=torch.long,
            device=self.data_device,
        )
        return buffers

    def _copy_before_step(self, buffers: dict[str, Any], step: int) -> None:
        super()._copy_before_step(buffers, step)
        inputs = self.adapter.paired_raw_contact_inputs()
        body_ids = self.adapter.paired_contact_token_body_ids
        world_ids = self.adapter.paired_contact_token_world_ids
        token_valid = inputs["contact_tokens"][..., 0] > 0.5
        expected_world_ids = self.adapter.contact_world_ids.unsqueeze(-1).expand_as(body_ids)
        mismatched = token_valid.squeeze(1) & (world_ids != expected_world_ids)
        if mismatched.any():
            rows, slots = torch.nonzero(mismatched, as_tuple=True)
            raise RuntimeError(
                "Paired contact-token rows do not match owner Newton worlds: "
                f"rows={rows[:8].tolist()}, slots={slots[:8].tolist()}."
            )
        self._copy_input(buffers["paired_contact_token_body_ids"][:, step], body_ids)
        self._copy_input(buffers["paired_contact_token_world_ids"][:, step], world_ids)
        for name, destination in buffers["paired_contacts"].items():
            self._copy_input(destination[:, step], inputs[name])


class PairedActionTrajectorySampler(_PairedContactSamplerMixin, ActionTrajectorySampler):
    """Generic trajectory sampler that records two contact representations."""


def split_paired_rollouts(rollouts: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a dual-contact rollout while reusing every non-contact tensor."""
    missing = sorted(_PAIRED_ROLLOUT_KEYS - set(rollouts))
    if missing:
        raise ValueError(f"Paired rollouts are missing contact fields: {missing}.")

    primary = {key: value for key, value in rollouts.items() if key not in _PAIRED_ROLLOUT_KEYS}
    paired = {
        key: value for key, value in rollouts.items() if key not in _PAIRED_ROLLOUT_KEYS | _PRIMARY_ROLLOUT_CONTACT_KEYS
    }
    paired.update(
        {
            "contacts": rollouts["paired_contacts"],
            "contact_token_body_ids": rollouts["paired_contact_token_body_ids"],
            "contact_token_world_ids": rollouts["paired_contact_token_world_ids"],
        }
    )
    return primary, paired


def _update_dataset_hash(
    digest: Any,
    name: str,
    dataset: h5py.Dataset,
    *,
    chunk_size: int,
) -> None:
    digest.update(name.encode("utf-8"))
    digest.update(dataset.dtype.str.encode("ascii"))
    digest.update(json.dumps(dataset.shape).encode("ascii"))
    for start in range(0, dataset.shape[0], chunk_size):
        values = np.ascontiguousarray(dataset[start : start + chunk_size])
        digest.update(values.tobytes())


def _compare_and_hash_datasets(
    left: h5py.Group,
    right: h5py.Group,
    names: list[str],
    *,
    label: str,
    chunk_size: int,
) -> str:
    left_digest = hashlib.sha256()
    right_digest = hashlib.sha256()
    for name in names:
        left_dataset = left[name]
        right_dataset = right[name]
        if left_dataset.shape != right_dataset.shape or left_dataset.dtype != right_dataset.dtype:
            raise ValueError(
                f"{label} differ for {name!r}: "
                f"{left_dataset.shape}/{left_dataset.dtype} != "
                f"{right_dataset.shape}/{right_dataset.dtype}."
            )
        for start in range(0, left_dataset.shape[0], chunk_size):
            left_values = np.ascontiguousarray(left_dataset[start : start + chunk_size])
            right_values = np.ascontiguousarray(right_dataset[start : start + chunk_size])
            if left_values.tobytes() != right_values.tobytes():
                raise ValueError(f"{label} differ for {name!r} at trajectory {start}.")
        _update_dataset_hash(left_digest, name, left_dataset, chunk_size=chunk_size)
        _update_dataset_hash(right_digest, name, right_dataset, chunk_size=chunk_size)
    if left_digest.digest() != right_digest.digest():
        raise ValueError(f"{label} differ.")
    return left_digest.hexdigest()


def _normalized_attribute(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _terrain_attributes(handle: h5py.File) -> dict[str, Any]:
    if "context" not in handle or "terrain" not in handle["context"]:
        return {}
    terrain = handle["context"]["terrain"]
    return {str(name): _normalized_attribute(value) for name, value in terrain.attrs.items()}


def validate_paired_hdf5_files(
    primary_path: str | Path,
    paired_path: str | Path,
    *,
    chunk_size: int = 64,
) -> dict[str, Any]:
    """Verify that two HDF5 files differ only in contact representation."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    with (
        h5py.File(Path(primary_path).expanduser(), "r") as primary_file,
        h5py.File(Path(paired_path).expanduser(), "r") as paired_file,
    ):
        primary_data = primary_file["data"]
        paired_data = paired_file["data"]
        primary_shared = set(primary_data) - _CONTACT_DATA_KEYS
        paired_shared = set(paired_data) - _CONTACT_DATA_KEYS
        if primary_shared != paired_shared:
            raise ValueError(
                "Shared rollout data keys differ: "
                f"primary_only={sorted(primary_shared - paired_shared)}, "
                f"paired_only={sorted(paired_shared - primary_shared)}."
            )
        shared_data_keys = sorted(primary_shared)
        data_digest = _compare_and_hash_datasets(
            primary_data,
            paired_data,
            shared_data_keys,
            label="shared rollout data",
            chunk_size=chunk_size,
        )

        primary_context = primary_file["context"]["trajectories"]
        paired_context = paired_file["context"]["trajectories"]
        if set(primary_context) != set(paired_context):
            raise ValueError("Shared trajectory context keys differ.")
        context_keys = sorted(primary_context)
        context_digest = hashlib.sha256()
        context_digest.update(
            _compare_and_hash_datasets(
                primary_context,
                paired_context,
                context_keys,
                label="shared trajectory context",
                chunk_size=chunk_size,
            ).encode("ascii")
        )

        primary_terrain = _terrain_attributes(primary_file)
        paired_terrain = _terrain_attributes(paired_file)
        if primary_terrain != paired_terrain:
            raise ValueError("Shared terrain context differs.")
        context_digest.update(json.dumps(primary_terrain, sort_keys=True, separators=(",", ":")).encode("utf-8"))

        states = primary_data["states"]
        return {
            "num_trajectories": int(states.shape[0]),
            "steps_per_trajectory": int(states.shape[1]),
            "total_transitions": int(states.shape[0] * states.shape[1]),
            "shared_data_keys": shared_data_keys,
            "shared_context_keys": context_keys,
            "shared_data_sha256": data_digest,
            "shared_context_sha256": context_digest.hexdigest(),
        }
