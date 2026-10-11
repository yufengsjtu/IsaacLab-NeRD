# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""True action sequences retain terminal states and cannot masquerade as held-target queries."""

import copy

import pytest
import torch
from isaaclab_neural.data.opd import CONTEXT_FIELDS
from isaaclab_neural.data.opd_sequence import SequenceWindowBuffer, read_sequence_shard, write_sequence_shard
from native_hdf5_samples import make_native_rollouts


def _history(env_id=0):
    data = make_native_rollouts(num_trajectories=1, num_steps=2, trajectory_start=env_id)
    return [
        {
            **{key: data[key][0, t].clone() for key in CONTEXT_FIELDS},
            "actions": torch.tensor([float(t), 0.0]),
            "action_targets": torch.tensor([float(t), 0.0]),
            "native": {k: v[t : t + 1].clone() for k, v in data["native_contacts"].items() if k != "frame_counts"},
        }
        for t in range(2)
    ]


def _metadata():
    return {
        "physics_dt": 0.005,
        "decimation": 2,
        "history_steps": 2,
        "sequence_controls": 2,
        "split": "train",
        "collection_id": "test",
        "policy_hash": "p",
        "dynamics_hash": "d",
        "iteration_id": 0,
        "source_kind": "mjwarp",
        "protocol": {"joint_names": ["a", "b"]},
    }


def _records():
    buffer = SequenceWindowBuffer([0, 3], history_steps=2, sequence_controls=2, decimation=2)
    buffer.begin_iteration(0)
    histories = {i: _history(i) for i in (0, 3)}
    for frame in range(4):
        for history in histories.values():
            history[-1]["actions"] = history[-1]["action_targets"] = torch.tensor([float(frame // 2), 0.0])
        buffer.append_frame(frame, histories, {0: torch.ones(4) * frame, 3: torch.ones(4) * frame}, {0: 0, 3: 0})
        if frame % 2 == 1:
            buffer.finish_control_step({0} if frame == 1 else set())
            if frame == 1:
                histories.pop(0)
    return buffer.end_iteration()


def test_collector_training_role_rejects_task_policy_data_and_synthetic_labels():
    from isaaclab_neural.data.opd_sequence import validate_collector_training_role

    metadata = {
        **_metadata(),
        "source_kind": "nerd",
        "label_origin": "mjwarp",
        "evaluation_only": False,
        "policy_role": "independent_collector",
        "history_origin": "actual_neural_runtime_including_reset_reconstruction",
    }
    assert validate_collector_training_role(metadata) == "model_history_MJ_future"
    assert (
        validate_collector_training_role({**metadata, "source_kind": "mjwarp", "history_origin": "physical_frames"})
        == "observed_MJ"
    )
    for change in (
        {"split": "validation"},
        {"evaluation_only": True},
        {"policy_role": "task_policy_evaluation"},
        {"label_origin": "teacher_prediction"},
        {"history_origin": "fabricated_truth"},
    ):
        with pytest.raises(ValueError):
            validate_collector_training_role({**metadata, **change})


def test_sequence_buffer_preserves_terminal_prefix_and_changing_targets():
    records = _records()
    assert len(records) == 2
    terminal = next(r for r in records if r["env_id"] == 0)
    full = next(r for r in records if r["env_id"] == 3)
    assert terminal["terminal"] and terminal["end_reason"] == "reset"
    assert terminal["student_states"].shape == (2, 4)
    assert terminal["actions"].shape == (1, 2)
    assert full["end_reason"] == "horizon"
    assert full["action_targets"][:, 0].tolist() == [0, 1]
    assert full["frame_ids"].tolist() == [0, 1, 2, 3]
    assert full["history"][-1]["action_targets"][0] == 0


def test_optional_future_inputs_preserve_actual_pre_and_post_states():
    buffer = SequenceWindowBuffer([0], history_steps=2, sequence_controls=2, decimation=2, record_future_inputs=True)
    buffer.begin_iteration(0)
    history = _history()
    for frame in range(4):
        history[-1]["states"] = torch.full((4,), float(frame))
        history[-1]["actions"] = history[-1]["action_targets"] = torch.tensor([float(frame // 2), 0.0])
        buffer.append_frame(frame, {0: history}, {0: torch.full((4,), float(frame + 1))}, {0: 0})
        if frame % 2:
            buffer.finish_control_step(set())
    record = buffer.end_iteration()[0]
    assert len(record["future_inputs"]) == 4
    assert torch.equal(record["future_inputs"][2]["states"], torch.full((4,), 2.0))
    assert torch.equal(record["future_inputs"][2]["observed_next_state"], torch.full((4,), 3.0))
    history[-1]["states"].zero_()
    assert torch.equal(record["future_inputs"][-1]["states"], torch.full((4,), 3.0))


def test_recorded_tensor_views_do_not_retain_other_worlds_storage(tmp_path):
    bulk = torch.arange(64 * 4, dtype=torch.float32).reshape(64, 4)
    history = _history()
    history[-1]["states"] = bulk[3]
    history[-1]["newton_state"] = {"joint_q": bulk[7]}
    history[-1]["solver_state"] = {"qacc_warmstart": bulk[11]}
    buffer = SequenceWindowBuffer([0], history_steps=2, sequence_controls=1, decimation=1, record_future_inputs=True)
    buffer.begin_iteration(0)
    buffer.append_frame(0, {0: history}, {0: torch.ones(4)}, {0: 0})
    buffer.finish_control_step(set())
    row = buffer.end_iteration()[0]
    for frame in (row["history"][-1], row["future_inputs"][0]):
        for tensor in (frame["states"], frame["newton_state"]["joint_q"], frame["solver_state"]["qacc_warmstart"]):
            assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
            assert tensor.storage_offset() == 0
    bulk.zero_()
    torch.testing.assert_close(row["history"][-1]["states"], torch.arange(12, 16, dtype=torch.float32))
    torch.testing.assert_close(
        row["future_inputs"][0]["newton_state"]["joint_q"], torch.arange(28, 32, dtype=torch.float32)
    )
    path = tmp_path / "compact.pt"
    torch.save(row, path)
    restored = torch.load(path, weights_only=True)
    torch.testing.assert_close(restored["history"][-1]["states"], row["history"][-1]["states"])


def test_observed_rollouts_export_real_force_inputs_without_fabricating_history():
    from isaaclab_neural.data.opd_sequence import observed_sequence_rollouts

    data = make_native_rollouts(num_trajectories=1, num_steps=5)
    frames = []
    for index in range(5):
        frames.append(
            {
                **{key: data[key][0, index].clone() for key in CONTEXT_FIELDS},
                "states": torch.full((4,), float(index)),
                "observed_next_state": torch.full((4,), float(index + 1)),
                "actions": torch.zeros(2),
                "action_targets": torch.ones(2),
                "native": {
                    key: value[index : index + 1].clone()
                    for key, value in data["native_contacts"].items()
                    if key != "frame_counts"
                },
            }
        )
    row = {
        "history": frames[:2],
        "future_inputs": frames[1:],
        "oracle_states": torch.stack([frame["observed_next_state"] for frame in frames[1:]]),
        "env_id": 0,
        "end_reason": "horizon",
    }
    meta = {**_metadata(), "label_origin": "mjwarp"}
    result = observed_sequence_rollouts([row], meta)
    torch.testing.assert_close(result["states"][:, 1:], result["next_states"][:, :-1])
    assert result["states"].shape == (1, 5, 4)
    with pytest.raises(ValueError, match="real MJ"):
        observed_sequence_rollouts([row], {**meta, "source_kind": "nerd"})
    bad = copy.deepcopy(row)
    bad["future_inputs"][2]["states"].add_(10)
    with pytest.raises(ValueError, match="continuity"):
        observed_sequence_rollouts([bad], meta)
    with pytest.raises(ValueError, match="future"):
        observed_sequence_rollouts([{key: value for key, value in row.items() if key != "future_inputs"}], meta)


def test_sequence_buffer_rejects_episode_or_iteration_crossing():
    buffer = SequenceWindowBuffer([0], history_steps=2, sequence_controls=2, decimation=2)
    buffer.begin_iteration(0)
    buffer.append_frame(0, {0: _history()}, {0: torch.zeros(4)}, {0: 0})
    with pytest.raises(RuntimeError, match="control"):
        buffer.begin_iteration(1)
    with pytest.raises(ValueError, match="episode"):
        buffer.append_frame(1, {0: _history()}, {0: torch.zeros(4)}, {0: 1})


def test_iteration_end_keeps_prefix_and_opens_a_new_version():
    buffer = SequenceWindowBuffer([0], history_steps=2, sequence_controls=2, decimation=2)
    buffer.begin_iteration(0)
    history = _history()
    for frame in range(2):
        buffer.append_frame(frame, {0: history}, {0: torch.zeros(4)}, {0: 0})
    buffer.finish_control_step(set())
    rows = buffer.end_iteration()
    history[-1]["states"].fill_(99)
    assert rows[0]["end_reason"] == "iteration_end"
    assert rows[0]["student_states"].shape[0] == 2
    assert not bool((rows[0]["history"][-1]["states"] == 99).any())
    buffer.begin_iteration(1)
    for frame in range(2, 4):
        buffer.append_frame(frame, {0: history}, {0: torch.zeros(4)}, {0: 0})
    buffer.finish_control_step(set())
    assert buffer.end_iteration()[0]["iteration_id"] == 1


def test_sequence_buffer_rejects_target_change_inside_control_hold():
    buffer = SequenceWindowBuffer([0], history_steps=2, sequence_controls=2, decimation=2)
    buffer.begin_iteration(0)
    history = _history()
    buffer.append_frame(0, {0: history}, {0: torch.zeros(4)}, {0: 0})
    history[-1]["action_targets"].add_(1)
    with pytest.raises(ValueError, match="control hold"):
        buffer.append_frame(1, {0: history}, {0: torch.zeros(4)}, {0: 0})


def test_sequence_shard_roundtrip_and_wrong_protocol_rejection(tmp_path):
    records = _records()
    path = tmp_path / "sequence.pt"
    write_sequence_shard(path, records, metadata=_metadata())
    restored = read_sequence_shard(path)
    assert restored["metadata"]["action_protocol"] == "recorded_target_sequence"
    torch.testing.assert_close(restored["records"][1]["action_targets"], records[1]["action_targets"])
    mapped = read_sequence_shard(path, memory_map=True)
    torch.testing.assert_close(mapped["records"][1]["action_targets"], restored["records"][1]["action_targets"])
    with pytest.raises(FileExistsError):
        write_sequence_shard(path, records, metadata=_metadata())
    with pytest.raises(ValueError, match="label"):
        read_sequence_shard(path, require_labels=True)
    bad = copy.deepcopy(records)
    bad[1]["episode_ids"][-1] += 1
    with pytest.raises(ValueError, match="episode"):
        write_sequence_shard(tmp_path / "bad.pt", bad, metadata=_metadata())
    bad_meta = {**_metadata(), "action_protocol": "hold_recorded_target"}
    with pytest.raises(ValueError, match="protocol"):
        write_sequence_shard(tmp_path / "wrong.pt", records, metadata=bad_meta)


def test_sequence_initial_state_protocols_do_not_silently_fallback(tmp_path):
    records = _records()
    with pytest.raises(ValueError, match="warm-start"):
        write_sequence_shard(
            tmp_path / "missing_warm.pt",
            records,
            metadata={**_metadata(), "initial_solver_state_protocol": "recorded_mj_qacc_warmstart"},
        )
    with pytest.raises(ValueError, match="native initial"):
        write_sequence_shard(
            tmp_path / "missing_native.pt",
            records,
            metadata={**_metadata(), "initial_physical_state_protocol": "native_newton_state_v1"},
        )
    with pytest.raises(ValueError, match="NeRD"):
        write_sequence_shard(
            tmp_path / "fake_memory.pt",
            records,
            metadata={
                **_metadata(),
                "source_kind": "nerd",
                "initial_solver_state_protocol": "recorded_mj_qacc_warmstart",
            },
        )


def test_sequence_world_roles_are_disjoint_and_exhaustive():
    from isaaclab_neural.data.opd_sequence import partition_sequence_records

    records = _records()
    train, validation = partition_sequence_records(records, train_env_ids={0}, validation_env_ids={3})
    assert [r["env_id"] for r in train] == [0]
    assert [r["env_id"] for r in validation] == [3]
    with pytest.raises(ValueError, match="disjoint"):
        partition_sequence_records(records, train_env_ids={0, 3}, validation_env_ids={3})
    with pytest.raises(ValueError, match="explicit"):
        partition_sequence_records(records, train_env_ids={0}, validation_env_ids={2})
