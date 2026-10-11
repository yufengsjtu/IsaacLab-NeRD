# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read-only callback and physical-unit evaluation contracts."""

from types import SimpleNamespace

import h5py
import pytest
import torch
import warp as wp
from isaaclab_neural.data.opd import write_context_shard
from isaaclab_neural.eval.dynamics_consistency import state_errors, summarize_errors
from isaaclab_neural.generate.opd import ContextRecorder, _raw_rollouts
from native_hdf5_samples import make_native_rollouts


def _guard():
    recorder = ContextRecorder.__new__(ContextRecorder)
    recorder.env = SimpleNamespace(device="cpu")
    recorder.frame = 0
    recorder.verify_frames = 1
    recorder.verified_callbacks = 0
    recorder.manager = SimpleNamespace(
        _state_0=SimpleNamespace(), _contacts=None, _control=SimpleNamespace(joint_f=wp.array([1.0, 2.0], device="cpu"))
    )
    return recorder


def test_recorder_guard_allows_reads_and_rejects_writes():
    recorder = _guard()
    value = recorder._guarded(lambda: recorder.manager._control.joint_f.numpy().copy())
    assert value.tolist() == [1, 2]
    assert recorder.verified_callbacks == 1
    with pytest.raises(RuntimeError, match="mutated physics buffer"):
        recorder._guarded(lambda: recorder.manager._control.joint_f.zero_())


def test_recorder_guard_checks_current_newton_target_names():
    recorder = _guard()
    recorder.manager._control.joint_target_q = wp.array([1.0, 2.0], device="cpu")
    with pytest.raises(RuntimeError, match="joint_target_q"):
        recorder._guarded(lambda: recorder.manager._control.joint_target_q.zero_())


def test_recorder_guard_rejects_policy_rng_consumption():
    recorder = _guard()
    state = torch.get_rng_state()
    try:
        with pytest.raises(RuntimeError, match="RNG"):
            recorder._guarded(lambda: torch.rand(1))
    finally:
        torch.set_rng_state(state)


def test_batched_selected_snapshot_matches_per_world_reference_and_copies_inputs():
    from isaaclab_neural.contacts.contact_set_schema import NATIVE_CONTACT_VALUE_FIELDS
    from isaaclab_neural.data.opd import CONTEXT_FIELDS

    recorder = ContextRecorder.__new__(ContextRecorder)
    recorder.ids = [0, 2]
    recorder.selected = torch.tensor(recorder.ids)
    raw = {key: torch.arange(12, dtype=torch.float32).reshape(3, 1, 4) for key in CONTEXT_FIELDS}
    solver = SimpleNamespace(get_raw_neural_model_inputs=lambda: raw)
    term = SimpleNamespace(raw_actions=torch.arange(6).reshape(3, 2), processed_actions=torch.ones(3, 2))
    recorder.env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda _: term))
    pairs = {key: torch.arange(12).reshape(3, 4) for key in NATIVE_CONTACT_VALUE_FIELDS}
    pairs["world_ids"] = torch.tensor([0, 1, 2])
    expected = {i: recorder._snapshot(solver, pairs, i) for i in recorder.ids}
    actual = recorder._batch_snapshot(solver, pairs)
    assert set(actual) == {0, 2}
    for env_id in recorder.ids:
        for key in (*CONTEXT_FIELDS, "actions", "action_targets"):
            torch.testing.assert_close(actual[env_id][key], expected[env_id][key], rtol=0, atol=0)
        for key in NATIVE_CONTACT_VALUE_FIELDS:
            torch.testing.assert_close(actual[env_id]["native"][key], expected[env_id]["native"][key], rtol=0, atol=0)
    actual[0]["states"].zero_()
    assert raw["states"][0].sum() != 0


def test_state_metrics_use_si_and_quaternion_sign_equivalence():
    target = torch.zeros(2, 87)
    target[:, 6] = 1
    predicted = target.clone()
    predicted[0, 6] = -1
    predicted[1, 0] = 0.01
    predicted[1, 44] = 2
    predicted[1, 47] = 3
    errors = state_errors(predicted, target, 44)
    torch.testing.assert_close(errors["root_orientation_rad"], torch.zeros(2))
    torch.testing.assert_close(errors["root_position_m"], torch.tensor([0.0, 0.01]))
    torch.testing.assert_close(errors["root_linear_velocity_mps"], torch.tensor([0.0, 2.0]))
    torch.testing.assert_close(errors["root_angular_velocity_radps"], torch.tensor([0.0, 3.0]))
    assert summarize_errors({"failed": torch.tensor([float("nan"), 1.0])})["failed"]["non_finite"] == 1


def test_oracle_reader_preserves_native_contacts_and_world_context(tmp_path):
    data = make_native_rollouts(num_steps=1)
    data["actions"] = torch.zeros(2, 1, 2)
    data["action_targets"] = torch.ones(2, 1, 2)
    path = tmp_path / "queries.hdf5"
    metadata = {"physics_dt": 0.005, "split": "train", "families": ["0", "1"], "policy_hash": "p", "dynamics_hash": "d"}
    write_context_shard(path, data, metadata=metadata)
    with h5py.File(path, "r") as file:
        restored = _raw_rollouts(file)
    for key in data["native_contacts"]:
        torch.testing.assert_close(restored["native_contacts"][key], data["native_contacts"][key])
    torch.testing.assert_close(restored["action_targets"], data["action_targets"])
    assert restored["trajectory_context"]["state_world_id"].tolist() == [0, 1]


def test_gate_rejects_different_queries_and_nonfinite_predictions():
    import copy

    from isaaclab_neural.eval.dynamics_consistency import compare_evaluations

    row = {"mean": 1.0, "p95": 2.0, "count": 3, "non_finite": 0}
    baseline = {
        "shards": [{"sha256": "same", "queries": 3, "horizon_physics_steps": 10}],
        "aggregate": {"one_step": {"velocity": dict(row)}, "held_target_rollout": {"velocity": dict(row)}},
    }
    candidate = copy.deepcopy(baseline)
    candidate["aggregate"]["held_target_rollout"]["velocity"]["mean"] = 0.7
    settings = {"improvement": 0.2, "regression": 0.1, "tolerances": {"velocity": 0.001}}
    assert compare_evaluations(baseline, candidate, **settings)["passed"]
    candidate["aggregate"]["one_step"]["velocity"]["non_finite"] = 1
    assert not compare_evaluations(baseline, candidate, **settings)["passed"]
    candidate["shards"][0]["sha256"] = "changed"
    with pytest.raises(ValueError, match="identical"):
        compare_evaluations(baseline, candidate, **settings)


def test_all_failed_worlds_keep_last_valid_inputs_without_fake_labels(tmp_path):
    from isaaclab_neural.data.opd import CONTEXT_FIELDS, QUERY_MODE

    data = make_native_rollouts(num_steps=1)
    recorder = ContextRecorder.__new__(ContextRecorder)
    recorder.neural = object()
    recorder.rows = {}
    recorder.failures = []
    for env_id in range(2):
        frame = {key: data[key][env_id, 0] for key in CONTEXT_FIELDS}
        frame["actions"] = frame["action_targets"] = torch.zeros(2)
        frame["native"] = {
            key: value[env_id : env_id + 1] for key, value in data["native_contacts"].items() if key != "frame_counts"
        }
        recorder.failures.append(
            {
                "history": [frame],
                "next": torch.full((4,), float("nan")),
                "env_id": env_id,
                "episode_id": 0,
                "frame_id": 3,
                "policy_update_id": 0,
            }
        )
    metadata = {
        "collection_id": "failure",
        "physics_dt": 0.005,
        "split": "train",
        "policy_hash": "p",
        "dynamics_hash": "d",
    }
    paths = recorder.export(tmp_path, metadata)
    with h5py.File(paths[0], "r") as file:
        assert file["data"].attrs["mode"] == QUERY_MODE
        assert file["data/states"].shape[0] == 2
        assert "target_next_states" not in file["opd"]
        assert file["opd/query_identity"][:, 0].tolist() == [0, 1]
        assert torch.isfinite(torch.from_numpy(file["data/states"][:])).all()
        assert torch.isnan(torch.from_numpy(file["opd/student_next_states"][:])).all()
