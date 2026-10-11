# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable native-contact contexts for bounded on-policy dynamics queries."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .hdf5 import write_rollouts_to_hdf5
from .native_contacts import NativeContactProjection, read_and_project_frames, validate_native_storage

CONTEXT_FIELDS = ("states", "joint_f", "root_body_q", "root_body_qd", "gravity_dir")
CONTEXT_MODE = "counterfactual_context"
QUERY_MODE = "opd_queries"


def file_sha256(path: str | Path) -> str:
    """Hash an artifact without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_full_history_labels(
    states: torch.Tensor,
    targets: torch.Tensor,
    anchor: torch.Tensor,
    identities: torch.Tensor,
    frame_ids: torch.Tensor,
    episode_ids: torch.Tensor,
) -> None:
    """Validate real consecutive pre/post-physics state labels without reset crossings or broadcasting."""
    if states.ndim != 3 or states.shape != targets.shape or anchor.shape != states[:, -1].shape:
        raise ValueError("Full-history state/target shapes must match exactly.")
    if not torch.isfinite(states).all() or not torch.isfinite(targets).all():
        raise ValueError("Full-history states and labels must be finite.")
    if identities.shape != (len(states), 4) or identities.dtype != torch.int64:
        raise ValueError("Full-history labels require integer query identities.")
    if frame_ids.shape != states.shape[:2] or frame_ids.dtype != torch.int64:
        raise ValueError("Full-history labels require consecutive physical frame IDs.")
    if (
        not torch.equal(frame_ids[:, -1], identities[:, 2])
        or not bool((frame_ids >= 0).all())
        or not bool((frame_ids[:, 1:] - frame_ids[:, :-1] == 1).all())
    ):
        raise ValueError("Full-history labels require consecutive physical frame IDs ending at the anchor.")
    if (
        episode_ids.shape != frame_ids.shape
        or episode_ids.dtype != torch.int64
        or not bool((episode_ids == identities[:, 1, None]).all())
    ):
        raise ValueError("Full-history labels cannot cross an episode reset.")
    if not torch.equal(targets[:, -1], anchor):
        raise ValueError("Full-history final label differs from its recorded anchor.")
    if not torch.allclose(targets[:, :-1], states[:, 1:], rtol=0, atol=1e-5):
        raise ValueError("Recorded pre/post-physics history continuity failed.")


def write_context_shard(
    path: str | Path,
    rollouts: dict,
    *,
    metadata: dict[str, Any],
    labels: torch.Tensor | None = None,
    diagnostics: dict[str, torch.Tensor] | None = None,
    sequence_labels: torch.Tensor | None = None,
) -> None:
    """Publish an immutable context shard with optional MJ anchor and full-history labels.

    Args:
        path: New HDF5 artifact, never overwritten.
        rollouts: Raw SI state/effort histories [batch, history, features] and native CSR pairs.
        metadata: Physics interval [s], split, provenance hashes and episode family IDs.
        labels: Oracle next states [batch, state_dim], or None for unlabelled queries.
        diagnostics: Per-query tensors not used as model input or supervision.
        sequence_labels: Actual post-step states for every real history position, never inferred from placeholders.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    states = rollouts["states"]
    if states.ndim != 3 or states.shape[0] < 1 or states.shape[1] < 1:
        raise ValueError("Contexts require nonempty [batch, history, state_dim] states.")
    dt = metadata.get("physics_dt", 0)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("physics_dt must be finite and positive [s].")
    if metadata.get("split") not in {"train", "validation", "test", "diagnostic"}:
        raise ValueError("An explicit split is required.")
    if len(metadata.get("families", [])) != states.shape[0]:
        raise ValueError("One episode family identifier is required for each context.")
    if not metadata.get("policy_hash") or not metadata.get("dynamics_hash"):
        raise ValueError("Policy and dynamics provenance hashes are required.")
    for key in CONTEXT_FIELDS:
        if not torch.isfinite(rollouts[key]).all():
            raise ValueError(f"Context {key} must be finite.")
    if labels is not None:
        if labels.shape != states[:, -1].shape or not torch.isfinite(labels).all():
            raise ValueError("Oracle labels must be finite with shape [batch, state_dim].")
        if metadata.get("label_origin") != "mjwarp":
            raise ValueError("Supervised OPD contexts require explicit MJWarp label provenance.")
    if sequence_labels is not None:
        if (
            labels is None
            or metadata.get("source_kind") != "mjwarp"
            or metadata.get("sequence_label_origin") != "observed_mj_post_step"
        ):
            raise ValueError("Full-history supervision requires observed real MJ transitions.")
        required = ("query_identity", "history_frame_ids", "history_episode_ids")
        if diagnostics is None or any(key not in diagnostics for key in required):
            raise ValueError("Full-history labels require frame and episode provenance.")
        validate_full_history_labels(states, sequence_labels, labels, *(diagnostics[key] for key in required))
    if "native_contacts" not in rollouts:
        raise ValueError("OPD contexts require native contact CSR, not baked contact tokens.")
    payload = {key: rollouts[key] for key in CONTEXT_FIELDS}
    payload["next_states"] = states.clone() if sequence_labels is None else sequence_labels.clone()
    if labels is not None:
        payload["next_states"][:, -1] = labels
    payload["native_contacts"] = rollouts["native_contacts"]
    payload["trajectory_context"] = rollouts["trajectory_context"]
    for key in ("actions", "action_targets"):
        if key in rollouts:
            payload[key] = rollouts[key]
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    os.close(descriptor)
    try:
        write_rollouts_to_hdf5(temporary, payload, env_name="G1-Native")
        with h5py.File(temporary, "r+") as file:
            file["data"].attrs["mode"] = CONTEXT_MODE if labels is not None else QUERY_MODE
            file.attrs["opd_metadata"] = json.dumps(metadata, allow_nan=False, sort_keys=True)
            group = file.create_group("opd")
            if labels is not None:
                group.create_dataset("target_next_states", data=labels.detach().cpu().numpy())
            if sequence_labels is not None:
                group.create_dataset("sequence_next_states", data=sequence_labels.detach().cpu().numpy())
            for key, value in (diagnostics or {}).items():
                if value.shape[0] != states.shape[0]:
                    raise ValueError(f"Diagnostic {key} must have one row per query.")
                group.create_dataset(key, data=value.detach().cpu().numpy())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


class ContextDataset(Dataset):
    """Read anchor-only oracle contexts or explicitly labelled real full histories."""

    def __init__(
        self,
        paths: list[str | Path],
        *,
        split: str,
        physics_dt: float,
        max_history: int = 10,
        max_contact_tokens: int = 64,
        contact_filter: str = "solver_active",
        state_dim: int | None = None,
        joint_f_dim: int | None = None,
        supervision: str = "anchor",
    ):
        if supervision not in {"anchor", "all_positions"}:
            raise ValueError("Unknown context supervision scope.")
        self.supervision = supervision
        self.paths = []
        for pattern in paths:
            matches = sorted(glob(str(pattern)))
            if not matches:
                raise FileNotFoundError(pattern)
            self.paths.extend(Path(path).resolve() for path in matches)
        if len(set(self.paths)) != len(self.paths):
            raise ValueError("Duplicate context paths would silently reweight samples.")
        self.rows: list[tuple[int, int]] = []
        self.lengths: list[int] = []
        self.sources: list[str] = []
        self.families: set[str] = set()
        self.projection = NativeContactProjection(
            "contact_tokens", max_contact_tokens, "body_round_robin_pair_atomic", contact_filter
        )
        self.data_keys = (*CONTEXT_FIELDS, "next_states", "supervision_mask", *self.projection.output_keys)
        for file_id, path in enumerate(self.paths):
            with h5py.File(path, "r") as file:
                if file["data"].attrs.get("mode") != CONTEXT_MODE:
                    raise ValueError(f"Refusing unlabelled/non-context data: {path}")
                metadata = json.loads(file.attrs["opd_metadata"])
                if metadata["split"] != split:
                    raise ValueError(f"Dataset split mismatch for {path}: expected {split}.")
                if metadata.get("label_origin") != "mjwarp":
                    raise ValueError("Contexts require MJWarp labels.")
                if not np.isclose(metadata["physics_dt"], physics_dt, rtol=0, atol=1e-10):
                    raise ValueError(f"physics_dt mismatch for {path}.")
                count, length, width = file["data/states"].shape
                if state_dim is not None and width != state_dim:
                    raise ValueError(f"State dimension mismatch: {width} != {state_dim}.")
                if joint_f_dim is not None and file["data/joint_f"].shape[-1] != joint_f_dim:
                    raise ValueError("Joint force dimension mismatch.")
                if not 1 <= length <= max_history:
                    raise ValueError(f"Invalid context length {length}; maximum is {max_history}.")
                validate_native_storage(file, count * length)
                if file["opd/target_next_states"].shape != (count, file["data/states"].shape[-1]):
                    raise ValueError("Invalid anchor target shape.")
                if len(metadata["families"]) != count:
                    raise ValueError("Missing query family provenance.")
                if supervision == "all_positions":
                    required = ("sequence_next_states", "query_identity", "history_frame_ids", "history_episode_ids")
                    if (
                        metadata.get("source_kind") != "mjwarp"
                        or metadata.get("sequence_label_origin") != "observed_mj_post_step"
                        or any(f"opd/{key}" not in file for key in required)
                    ):
                        raise ValueError(
                            "Explicit real full-history labels are required; anchor-only contexts cannot be promoted."
                        )
                    validate_full_history_labels(
                        torch.from_numpy(file["data/states"][:]),
                        torch.from_numpy(file["opd/sequence_next_states"][:]),
                        torch.from_numpy(file["opd/target_next_states"][:]),
                        *(torch.from_numpy(file[f"opd/{key}"][:]) for key in required[1:]),
                    )
                self.families.update(metadata["families"])
                self.rows.extend((file_id, row) for row in range(count))
                self.lengths.extend([length] * count)
                source = (
                    "reference"
                    if metadata.get("pool") == "reference"
                    else {"mjwarp": "current_mj", "nerd": "student"}.get(metadata.get("source_kind"), "unknown")
                )
                self.sources.extend([source] * count)
        if not self.rows:
            raise ValueError("No labelled contexts.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        file_id, row = self.rows[index]
        length = self.lengths[index]
        with h5py.File(self.paths[file_id], "r") as file:
            sample = {key: torch.from_numpy(np.asarray(file[f"data/{key}"][row])) for key in CONTEXT_FIELDS}
            if self.supervision == "all_positions":
                sample["next_states"] = torch.from_numpy(np.asarray(file["opd/sequence_next_states"][row]))
                mask = torch.ones(length, dtype=torch.bool)
            else:
                sample["next_states"] = sample["states"].clone()
                sample["next_states"][-1] = torch.from_numpy(np.asarray(file["opd/target_next_states"][row]))
                mask = torch.zeros(length, dtype=torch.bool)
                mask[-1] = True
            sample["supervision_mask"] = mask
            projected = read_and_project_frames(
                file["contacts/native_pairs"],
                np.arange(row * length, (row + 1) * length, dtype=np.int64),
                self.projection,
            )
            sample.update({key: torch.from_numpy(value) for key, value in projected.items()})
        return sample

    def shuffle(self) -> None:
        """Leave row identity stable; the length-bucket sampler controls ordering."""


class ContextBatchSampler(Sampler):
    """Batch homogeneous history lengths without padding or using global RNG."""

    def __init__(self, dataset: ContextDataset, batch_size: int, *, shuffle: bool, seed: int):
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.groups: dict[int, list[int]] = defaultdict(list)
        for index, length in enumerate(dataset.lengths):
            self.groups[length].append(index)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return sum((len(rows) + self.batch_size - 1) // self.batch_size for rows in self.groups.values())

    def __iter__(self):
        batches = []
        for rows in self.groups.values():
            indices = np.asarray(rows, dtype=np.int64)
            if self.shuffle:
                indices = self.rng.permutation(indices)
            batches.extend(
                indices[start : start + self.batch_size].tolist() for start in range(0, len(indices), self.batch_size)
            )
        if self.shuffle:
            self.rng.shuffle(batches)
        yield from batches


class ContextSamplingAudit:
    """Count consumed anchor exposures and nominal loss coefficients, not gradient norms."""

    def __init__(self, dataset: ContextDataset):
        self.dataset = dataset
        self.updates = 0
        self.exposures = Counter()
        self.loss_mass = Counter()
        self.batch_sizes = Counter()
        self.row_exposures = Counter()
        self.trace = hashlib.sha256()

    def record(self, indices: list[int]) -> None:
        self.updates += 1
        self.batch_sizes[len(indices)] += 1
        self.trace.update(np.asarray([len(indices), *indices], dtype="<i8").tobytes())
        for index in indices:
            key = f"{self.dataset.sources[index]}/h{self.dataset.lengths[index]:02d}"
            self.exposures[key] += 1
            self.loss_mass[key] += 1 / len(indices)
            self.row_exposures[index] += 1

    def summary(self) -> dict[str, Any]:
        inventory = Counter(f"{s}/h{h:02d}" for s, h in zip(self.dataset.sources, self.dataset.lengths))
        return {
            "optimizer_updates": self.updates,
            "anchor_exposures": sum(self.exposures.values()),
            "unique_rows_seen": len(self.row_exposures),
            "dataset_rows": len(self.dataset),
            "repeat_exposures": sum(self.row_exposures.values()) - len(self.row_exposures),
            "maximum_row_repetitions": max(self.row_exposures.values(), default=0),
            "update_batch_sizes": dict(self.batch_sizes),
            "inventory": dict(inventory),
            "exposures": dict(self.exposures),
            "nominal_loss_weight_fraction": {k: v / max(self.updates, 1) for k, v in self.loss_mass.items()},
            "index_trace_sha256": self.trace.hexdigest(),
            "interpretation": "Loss coefficients under per-update mean reduction, not measured gradient contributions",
        }


class ContextUpdateSampler(Sampler):
    """Sample fixed source quotas per optimizer update using an explicit length prior.

    Sampling is with replacement within source/length buckets. Missing requested
    buckets fail rather than silently redistributing the configured loss budget.
    """

    def __init__(
        self,
        dataset: ContextDataset,
        batch_size: int,
        *,
        source_counts: dict[str, int],
        length_weights: dict[int, float],
        seed: int,
    ):
        if not source_counts or any(
            not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in source_counts.values()
        ):
            raise ValueError("Source counts must be positive integers.")
        if sum(source_counts.values()) != batch_size:
            raise ValueError("Source counts must sum to the optimizer update batch size.")
        self.lengths = np.asarray(sorted(int(k) for k in length_weights), dtype=np.int64)
        weights = np.asarray([length_weights[int(k)] for k in self.lengths], dtype=np.float64)
        if not len(weights) or not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Length weights must be finite and positive.")
        self.probabilities = weights / weights.sum()
        self.groups = defaultdict(list)
        for index, (source, length) in enumerate(zip(dataset.sources, dataset.lengths)):
            self.groups[source, length].append(index)
        for source in source_counts:
            for length in self.lengths:
                if not self.groups[source, int(length)]:
                    raise ValueError(f"Missing requested context bucket: {source}/h{length}.")
        self.dataset = dataset
        self.batch_size = batch_size
        self.source_counts = dict(source_counts)
        self.rng = np.random.default_rng(seed)
        self.audit = ContextSamplingAudit(dataset)

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for _ in range(len(self)):
            indices = []
            for source, count in self.source_counts.items():
                selected_lengths = self.rng.choice(self.lengths, size=count, p=self.probabilities)
                for length in self.lengths:
                    size = int((selected_lengths == length).sum())
                    indices.extend(self.rng.choice(self.groups[source, int(length)], size=size, replace=True).tolist())
            self.rng.shuffle(indices)
            self.audit.record(indices)
            yield indices


class CurrentEpochSampler(Sampler):
    """Traverse current rows once per epoch, mixing reference/replay without small tail updates.

    Pool indices are separate namespaces. Current rows must already be unique,
    labelled training samples; validation data must never be passed to these pools.
    """

    def __init__(
        self,
        current_count: int,
        *,
        reference_count: int,
        replay_stage_sizes: list[int],
        seed: int,
        current_batch_size: int = 256,
    ):
        if current_batch_size < 1 or current_count < current_batch_size:
            raise ValueError("Insufficient current samples for a complete logical update.")
        if reference_count < 1:
            raise ValueError("A nonempty reference pool is required.")
        if any(size < 1 for size in replay_stage_sizes):
            raise ValueError("Replay stages must be nonempty.")
        self.current_count = current_count
        self.current_batch_size = current_batch_size
        self.reference_count = reference_count
        self.replay_stage_sizes = tuple(replay_stage_sizes)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.current_count // self.current_batch_size

    def __iter__(self):
        for current in np.array_split(self.rng.permutation(self.current_count), len(self)):
            n = len(current)
            replay = []
            reference_size = n
            weights = {"current": 0.5, "reference": 0.5, "replay": 0.0}
            if self.replay_stage_sizes:
                reference_size = n // 2
                order = self.rng.permutation(len(self.replay_stage_sizes))
                stages = np.resize(order, n - reference_size)
                for stage in stages:
                    replay.append((int(stage), int(self.rng.integers(self.replay_stage_sizes[stage]))))
                weights = {"current": 0.5, "reference": 0.25, "replay": 0.25}
            yield {
                "current": current.tolist(),
                "reference": self.rng.integers(self.reference_count, size=reference_size).tolist(),
                "replay": replay,
                "source_weights": weights,
            }


def collate_context_update(samples: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    """Split one logical optimizer batch into exact-length microbatches, without padding."""
    from torch.utils.data import default_collate

    groups = defaultdict(list)
    for sample in samples:
        mask = sample["supervision_mask"]
        if mask.dtype != torch.bool or int(mask.sum()) != 1 or not bool(mask[-1]):
            raise ValueError("Context updates require exactly one anchor label per sample.")
        groups[len(sample["states"])].append(sample)
    return [default_collate(groups[length]) for length in sorted(groups)]


def audit_dataset(path: str | Path, *, expected_state_dim: int | None = None) -> dict[str, Any]:
    """Inspect stored schema and a bounded force sample without trusting filenames."""
    path = Path(path)
    problems = []
    report: dict[str, Any] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
    with h5py.File(path, "r") as file:
        data = file.get("data")
        if data is None:
            return {**report, "eligible_for_rehearsal": False, "problems": ["Missing /data"]}
        report["mode"] = str(data.attrs.get("mode", "missing"))
        report["states_shape"] = list(data["states"].shape) if "states" in data else None
        report["stored_fields"] = sorted(data)
        if expected_state_dim is not None and ("states" not in data or data["states"].shape[-1] != expected_state_dim):
            problems.append(f"State dimension does not match expected {expected_state_dim}")
        if any(key.startswith("contact_") for key in data):
            problems.append("Baked contacts are not current native CSR storage")
        try:
            shape = data["states"].shape
            native = validate_native_storage(file, int(np.prod(shape[:2])))
            if native is None:
                problems.append("Missing native contacts")
        except (KeyError, ValueError) as error:
            problems.append(str(error))
        if "joint_f" in data:
            force = np.asarray(data["joint_f"][: min(8, data["joint_f"].shape[0])])
            report["sampled_joint_f_abs_max"] = float(np.abs(force).max()) if force.size else 0.0
            report["sampled_joint_f_rms"] = float(np.sqrt(np.mean(force**2))) if force.size else 0.0
            if not force.size or not np.isfinite(force).all() or not np.any(force):
                problems.append("Sampled joint_f is empty, non-finite or identically zero")
        else:
            problems.append("Missing joint_f")
        metadata = json.loads(file.attrs.get("opd_metadata", "{}"))
        if (
            metadata.get("label_origin") != "mjwarp"
            or metadata.get("pool") != "reference"
            or metadata.get("split") != "train"
        ):
            problems.append("No verified training-reference provenance; manual audit/conversion required")
        report["metadata"] = metadata
    report["problems"] = problems
    report["eligible_for_rehearsal"] = not problems
    report["audit_scope"] = "Schema/provenance gate and bounded force sample, not exhaustive physical validation"
    return report


def main() -> None:
    """Reconstruct a context sampler's exposure budget without loading model inputs."""
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = yaml.safe_load(args.audit_config.read_text())["algorithm"]
    dataset = ContextDataset(cfg["dataset"]["context_datasets"]["train"], split="train", physics_dt=cfg["data_dt"])
    sampling = cfg["dataset"].get("context_sampling")
    sampler = (
        ContextUpdateSampler(dataset, cfg["batch_size"], seed=args.seed, **sampling)
        if sampling is not None
        else ContextBatchSampler(dataset, cfg["batch_size"], shuffle=True, seed=args.seed)
    )
    audit = ContextSamplingAudit(dataset)
    iterator = iter(sampler)
    for _ in range(args.updates):
        try:
            indices = next(iterator)
        except StopIteration:
            iterator = iter(sampler)
            indices = next(iterator)
        audit.record(indices)
    report = audit.summary()
    report["config_sha256"] = file_sha256(args.audit_config)
    report["shards"] = {str(path): file_sha256(path) for path in dataset.paths}
    report["sampling_seed"] = args.seed
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
