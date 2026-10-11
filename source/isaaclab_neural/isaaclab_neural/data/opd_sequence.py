# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Versioned, immutable action-sequence queries with real histories and terminal prefixes."""

from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from isaaclab_neural.contacts.contact_set_schema import NATIVE_CONTACT_VALUE_FIELDS

from .opd import CONTEXT_FIELDS

SEQUENCE_FORMAT = "opd_action_sequences_v1"
SEQUENCE_ACTION_PROTOCOL = "recorded_target_sequence"


def _compact_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone(memory_format=torch.contiguous_format)
    if isinstance(value, dict):
        return {key: _compact_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_compact_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_compact_copy(item) for item in value)
    return copy.deepcopy(value)


class SequenceWindowBuffer:
    """Assemble non-overlapping control-aligned windows without stepping or resetting an environment.

    Histories are supplied by the recorder in raw SI units. Terminal and rollout-end
    prefixes are ragged, never padded with invented states or future controls.
    """

    def __init__(
        self,
        env_ids: list[int],
        *,
        history_steps: int = 10,
        sequence_controls: int = 4,
        decimation: int = 4,
        record_future_inputs: bool = False,
    ):
        if not env_ids or len(set(env_ids)) != len(env_ids) or min(env_ids) < 0:
            raise ValueError("Selected environment IDs must be distinct and nonnegative.")
        if min(history_steps, sequence_controls, decimation) < 1:
            raise ValueError("History, horizon and decimation must be positive.")
        self.env_ids = tuple(env_ids)
        self.history_steps, self.sequence_controls, self.decimation = history_steps, sequence_controls, decimation
        self.record_future_inputs = record_future_inputs
        self.iteration_id = None
        self.pending = {}
        self.records = []
        self.last_frame = -1
        self.control_finished = True

    def begin_iteration(self, iteration_id: int) -> None:
        """Open an explicit policy/model-version boundary after the previous control step."""
        if not self.control_finished or self.pending or self.records:
            raise RuntimeError("Finish the control step and export the previous iteration before opening another.")
        if iteration_id < 0 or (self.iteration_id is not None and iteration_id <= self.iteration_id):
            raise ValueError("Iteration IDs must increase.")
        self.iteration_id = iteration_id

    def append_frame(self, frame_id: int, histories: dict, next_states: dict, episode_ids: dict) -> None:
        """Append a physical transition, with states in raw solver SI coordinates."""
        if self.iteration_id is None or frame_id != self.last_frame + 1:
            raise ValueError("Missing iteration or non-contiguous physical frame.")
        phase = frame_id % self.decimation
        if phase == 0 and not self.control_finished:
            raise RuntimeError("finish_control_step must be called before the next control step.")
        for env_id in self.env_ids:
            if env_id not in histories:
                if env_id in self.pending:
                    raise ValueError("Missing history for an active sequence.")
                continue
            history = histories[env_id]
            if env_id not in self.pending:
                if phase or len(history) != self.history_steps:
                    continue
                self.pending[env_id] = {
                    "env_id": env_id,
                    "episode_id": int(episode_ids[env_id]),
                    "iteration_id": self.iteration_id,
                    "anchor_frame": frame_id,
                    "history": _compact_copy(list(history)),
                    "actions": [],
                    "action_targets": [],
                    "student_states": [],
                    "frame_ids": [],
                    "episode_ids": [],
                }
            row = self.pending[env_id]
            if int(episode_ids[env_id]) != row["episode_id"]:
                raise ValueError("A sequence cannot cross an episode reset.")
            latest = history[-1]
            if phase == 0:
                row["actions"].append(latest["actions"].detach().cpu().clone())
                row["action_targets"].append(latest["action_targets"].detach().cpu().clone())
            elif not torch.equal(row["action_targets"][-1], latest["action_targets"].cpu()):
                raise ValueError("Processed target changed inside a control hold.")
            row["student_states"].append(next_states[env_id].detach().cpu().clone())
            if self.record_future_inputs:
                recorded = _compact_copy(latest)
                recorded["observed_next_state"] = next_states[env_id].detach().cpu().clone()
                row.setdefault("future_inputs", []).append(recorded)
            row["frame_ids"].append(frame_id)
            row["episode_ids"].append(int(episode_ids[env_id]))
        self.last_frame = frame_id
        self.control_finished = False

    def _finish(self, env_id: int, reason: str) -> None:
        row = self.pending.pop(env_id)
        for key in ("actions", "action_targets", "student_states"):
            row[key] = torch.stack(row[key])
        for key in ("frame_ids", "episode_ids"):
            row[key] = torch.tensor(row[key], dtype=torch.int64)
        row["terminal"] = reason == "reset"
        row["end_reason"] = reason
        self.records.append(row)

    def finish_control_step(self, done_ids: set[int]) -> None:
        """Commit terminal pre-reset transitions; never append a reset state to the same window."""
        if self.control_finished or (self.last_frame + 1) % self.decimation:
            raise RuntimeError("A complete control step is required before finish_control_step.")
        for env_id, row in list(self.pending.items()):
            if env_id in done_ids:
                self._finish(env_id, "reset")
            elif len(row["actions"]) == self.sequence_controls:
                self._finish(env_id, "horizon")
        self.control_finished = True

    def end_iteration(self) -> list[dict[str, Any]]:
        """Transfer completed windows and valid control-aligned prefixes to an independent dataset."""
        if not self.control_finished:
            raise RuntimeError("Cannot export in the middle of a control step.")
        for env_id in list(self.pending):
            self._finish(env_id, "iteration_end")
        result, self.records = self.records, []
        return result


def partition_sequence_records(
    records: list[dict], *, train_env_ids: set[int], validation_env_ids: set[int]
) -> tuple[list[dict], list[dict]]:
    """Partition one rollout by fixed world roles, never by overlapping random windows."""
    if not train_env_ids or not validation_env_ids or train_env_ids & validation_env_ids:
        raise ValueError("Training and validation world roles must be nonempty and disjoint.")
    if any(row["env_id"] not in train_env_ids | validation_env_ids for row in records):
        raise ValueError("Every recorded world needs an explicit train/validation role.")
    return (
        [row for row in records if row["env_id"] in train_env_ids],
        [row for row in records if row["env_id"] in validation_env_ids],
    )


def _validate_sequence_history(row: dict, meta: dict) -> None:
    if any(
        not isinstance(row[key], int) or isinstance(row[key], bool)
        for key in ("env_id", "episode_id", "anchor_frame", "iteration_id")
    ):
        raise ValueError("Sequence identities must be integers.")
    if row["frame_ids"].dtype != torch.int64 or row["episode_ids"].dtype != torch.int64:
        raise ValueError("Frame/episode provenance must use int64 tensors.")
    if len(row["history"]) != meta["history_steps"]:
        raise ValueError("Missing real history; do not pad histories.")
    anchor = row["history"][-1]
    if meta.get("initial_physical_state_protocol") == "native_newton_state_v1":
        native_state = anchor.get("newton_state", {})
        for key in ("joint_q", "joint_qd", "body_q", "body_qd", "body_f"):
            value = native_state.get(key)
            if value is None or not value.numel() or not torch.isfinite(value).all():
                raise ValueError(f"Missing or non-finite native initial {key}.")
    if meta.get("initial_solver_state_protocol") == "recorded_mj_qacc_warmstart":
        warmstart = anchor.get("solver_state", {}).get("qacc_warmstart")
        if warmstart is None or warmstart.ndim != 1 or not torch.isfinite(warmstart).all():
            raise ValueError("Recorded warm-start data is missing or non-finite.")
    for frame in row["history"]:
        for key in (*CONTEXT_FIELDS, "actions", "action_targets"):
            if frame[key].shape != anchor[key].shape or not torch.isfinite(frame[key]).all():
                raise ValueError(f"Invalid or non-finite history {key}.")
        native = frame["native"]
        pairs = len(native["world_ids"])
        if not bool((native["world_ids"] == row["env_id"]).all()):
            raise ValueError("History native contacts belong to a different world.")
        for key in NATIVE_CONTACT_VALUE_FIELDS:
            if len(native[key]) != pairs or not torch.isfinite(native[key]).all():
                raise ValueError("Invalid native contact history.")


def _validate_sequence_labels(row: dict, metadata: dict, require_labels: bool) -> None:
    if require_labels and "oracle_states" not in row:
        raise ValueError("Unlabelled sequences cannot train a dynamics model.")
    if "oracle_states" in row:
        if metadata.get("label_origin") != "mjwarp" or row["oracle_states"].shape != row["student_states"].shape:
            raise ValueError("Missing MJ label provenance or wrong label shape.")
        if not torch.isfinite(row["oracle_states"]).all():
            raise ValueError("Non-finite oracle labels.")


def validate_sequence_shard(shard: dict, *, require_labels: bool = False) -> None:
    """Reject wrong protocols, reset crossings, missing histories and non-finite labels."""
    if shard.get("format") != SEQUENCE_FORMAT:
        raise ValueError("Unsupported sequence format; held-target contexts are not action sequences.")
    meta = shard["metadata"]
    if meta.get("action_protocol") != SEQUENCE_ACTION_PROTOCOL:
        raise ValueError("Wrong sequence action protocol.")
    if meta.get("actuator_state_protocol") != "stateless_implicit_pd":
        raise ValueError("Sequence v1 supports only stateless implicit PD.")
    if not np.isfinite(meta["physics_dt"]) or meta["physics_dt"] <= 0:
        raise ValueError("Invalid physics_dt [s].")
    if meta["split"] not in {"train", "validation", "test", "diagnostic"}:
        raise ValueError("Explicit split required.")
    if meta.get("source_kind") not in {"nerd", "mjwarp"}:
        raise ValueError("Explicit source backend required.")
    if meta.get("initial_physical_state_protocol", "solver_canonical_state") not in {
        "solver_canonical_state",
        "native_newton_state_v1",
    }:
        raise ValueError("Unknown physical-state protocol.")
    for key in ("collection_id", "policy_hash", "dynamics_hash", "protocol"):
        if not meta.get(key):
            raise ValueError(f"Missing sequence provenance: {key}.")
    h, k, decimation = (meta[key] for key in ("history_steps", "sequence_controls", "decimation"))
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in (h, k, decimation)):
        raise ValueError("Invalid sequence lengths or decimation.")
    solver_state_protocol = meta.get("initial_solver_state_protocol", "canonical_cold_reset")
    if solver_state_protocol not in {"canonical_cold_reset", "recorded_mj_qacc_warmstart"}:
        raise ValueError("Unknown initial solver-state protocol.")
    if solver_state_protocol == "recorded_mj_qacc_warmstart" and meta["source_kind"] != "mjwarp":
        raise ValueError("NeRD cannot supply recorded MJ solver memory.")
    records = shard["records"]
    if not records:
        raise ValueError("An empty sequence dataset is not trainable.")
    identities = set()
    for row in records:
        identity = tuple(row[key] for key in ("env_id", "episode_id", "anchor_frame", "iteration_id"))
        if identity in identities:
            raise ValueError("Duplicate current sequence identity.")
        identities.add(identity)
        if min(identity) < 0 or row["iteration_id"] != meta["iteration_id"]:
            raise ValueError("Sequence iteration/version mismatch.")
        length = len(row["actions"])
        steps = length * decimation
        if not 1 <= length <= k or row["anchor_frame"] % decimation:
            raise ValueError("Sequences must start on control boundaries with a valid horizon.")
        if row["actions"].ndim != 2 or row["action_targets"].shape != row["actions"].shape:
            raise ValueError("Actions and targets must have shape [control_steps, joints].")
        if not torch.equal(row["frame_ids"], torch.arange(row["anchor_frame"], row["anchor_frame"] + steps)):
            raise ValueError("Non-contiguous sequence frames.")
        if row["episode_ids"].shape != (steps,) or not bool((row["episode_ids"] == row["episode_id"]).all()):
            raise ValueError("Sequence crosses an episode boundary.")
        _validate_sequence_history(row, meta)
        width = row["history"][-1]["states"].numel()
        if row["student_states"].shape != (steps, width):
            raise ValueError("Recorded successor states must include every physical frame.")
        if row["end_reason"] not in {"horizon", "reset", "iteration_end"} or row["terminal"] != (
            row["end_reason"] == "reset"
        ):
            raise ValueError("Invalid terminal/end-reason contract.")
        if row["end_reason"] == "horizon" and length != k:
            raise ValueError("A horizon-complete record must contain the full target sequence.")
        for key in ("actions", "action_targets", "student_states"):
            if not torch.isfinite(row[key]).all():
                raise ValueError(f"Non-finite sequence {key}.")
        for key in ("actions", "action_targets"):
            if not torch.equal(row["history"][-1][key], row[key][0]):
                raise ValueError("History anchor and first executed action disagree.")
        _validate_sequence_labels(row, meta, require_labels)


def validate_collector_training_role(metadata: dict[str, Any]) -> str:
    """Require independent collector provenance before consuming sequence training labels.

    Args:
        metadata: Sequence metadata, in addition to the structural shard validation.

    Returns:
        The observed-MJ or model-history/MJ-future source classification. This does not
        create past-state labels or assert that a full-history target array exists.
    """
    if (
        metadata.get("split") != "train"
        or metadata.get("evaluation_only") is not False
        or metadata.get("policy_role") != "independent_collector"
        or metadata.get("label_origin") != "mjwarp"
    ):
        raise ValueError("Only independent collector training data with real MJ labels may enter this objective.")
    if metadata.get("source_kind") == "nerd" and metadata.get("history_origin") == (
        "actual_neural_runtime_including_reset_reconstruction"
    ):
        return "model_history_MJ_future"
    if metadata.get("source_kind") == "mjwarp" and metadata.get("history_origin") == "physical_frames":
        return "observed_MJ"
    raise ValueError("Collector history origin is missing or incompatible with its source backend.")


def observed_sequence_rollouts(records: list[dict], metadata: dict) -> dict[str, Any]:
    """Pack complete real-MJ histories/futures in raw SI units for recorded-force evaluation."""
    if metadata.get("source_kind") != "mjwarp" or metadata.get("label_origin") != "mjwarp":
        raise ValueError("Recorded-force trajectories require real MJ observations.")
    steps = metadata["sequence_controls"] * metadata["decimation"]
    complete = [row for row in records if row["end_reason"] == "horizon"]
    if not complete:
        raise ValueError("No complete observed future sequences are available.")
    histories = []
    for row in complete:
        future = row.get("future_inputs", [])
        if len(future) != steps or len(row["history"]) != metadata["history_steps"]:
            raise ValueError("Every physical future input must be recorded; do not invent contacts or forces.")
        frames = row["history"][:-1] + future
        if any("observed_next_state" not in frame for frame in frames):
            raise ValueError("Missing observed next state in history/future frames.")
        states = torch.stack([frame["states"] for frame in frames])
        targets = torch.stack([frame["observed_next_state"] for frame in frames])
        if not torch.allclose(states[1:], targets[:-1], rtol=0, atol=1e-5):
            raise ValueError("Observed trajectory continuity failed.")
        if not torch.equal(targets[metadata["history_steps"] - 1 :], row["oracle_states"]):
            raise ValueError("Observed trajectory targets disagree with real-MJ sequence labels.")
        histories.append(frames)
    result = {
        key: torch.stack([torch.stack([frame[key] for frame in frames]) for frames in histories])
        for key in (*CONTEXT_FIELDS, "actions", "action_targets")
    }
    result["next_states"] = torch.stack(
        [torch.stack([frame["observed_next_state"] for frame in frames]) for frames in histories]
    )
    pairs = [frame["native"] for frames in histories for frame in frames]
    result["native_contacts"] = {key: torch.cat([frame[key] for frame in pairs]) for key in NATIVE_CONTACT_VALUE_FIELDS}
    result["native_contacts"]["frame_counts"] = torch.tensor(
        [len(frame["world_ids"]) for frame in pairs], dtype=torch.int64
    ).reshape(len(histories), -1)
    worlds = torch.tensor([row["env_id"] for row in complete], dtype=torch.int64)
    result["trajectory_context"] = {key: worlds for key in ("state_world_id", "root_world_id", "contact_world_id")}
    return result


def write_sequence_shard(path: str | Path, records: list[dict], *, metadata: dict) -> None:
    """Publish raw SI sequence/history tensors without overwriting existing data.

    States use the solver's raw coordinate contract; targets are positions [m or
    rad, depending on joint type]. Variable-length future prefixes are stored
    without padding. This format is deliberately distinct from old HDF5 contexts.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    meta = json.loads(json.dumps(metadata, allow_nan=False))
    meta.setdefault("action_protocol", SEQUENCE_ACTION_PROTOCOL)
    meta.setdefault("actuator_state_protocol", "stateless_implicit_pd")
    meta.setdefault("initial_solver_state_protocol", "canonical_cold_reset")
    shard = {"format": SEQUENCE_FORMAT, "metadata": meta, "records": records}
    validate_sequence_shard(shard)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(shard, temporary)
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def read_sequence_shard(path: str | Path, *, require_labels: bool = False, memory_map: bool = False) -> dict:
    """Load tensor/primitive-only sequence storage and validate its protocol.

    Args:
        path: Path to the serialized sequence shard.
        require_labels: Whether to require observed oracle labels.
        memory_map: Whether to map an immutable Torch ZIP file instead of eagerly loading tensor storage.

    Returns:
        The validated sequence shard. A mapped source file must remain unchanged while its tensors are used.
    """
    shard = torch.load(path, map_location="cpu", weights_only=True, mmap=memory_map)
    validate_sequence_shard(shard, require_labels=require_labels)
    return shard
