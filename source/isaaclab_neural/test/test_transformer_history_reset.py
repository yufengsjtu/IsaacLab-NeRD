# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for transformer history handling on environment reset."""

from collections import deque

import torch
from isaaclab_neural.solvers.transformer_neural_solver import TransformerNeuralSolver


class _TransformerTestHarness(TransformerNeuralSolver):
    history_snapshot = TransformerNeuralSolver._history_snapshot


def _history_solver(*, num_envs: int, history_length: int = 4) -> _TransformerTestHarness:
    """Build a solver stub with a filled history window."""
    solver = _TransformerTestHarness.__new__(_TransformerTestHarness)
    solver.num_envs = num_envs
    solver.num_states_history = history_length
    solver.torch_device = "cpu"
    solver.predicted_body_net_contact_forces_b = None
    solver.use_kv_cache = False
    solver.use_feature_cache = False
    solver.feature_cache = None
    solver.feature_cache_rebuild_env_ids = None
    solver.kv_cache = None
    solver.kv_cache_rebuild_env_ids = None
    solver.kv_cache_rebuild_inputs = None
    solver.root_body_q = torch.zeros(num_envs, 7)
    solver.states = torch.zeros(num_envs, 4)
    solver.states_embedding = torch.zeros(num_envs, 4)
    solver.joint_f = torch.zeros(num_envs, 2)
    solver.gravity_dir = torch.zeros(num_envs, 3)
    solver.contact_inputs = {
        "contact_masks": torch.zeros(num_envs, 3, dtype=torch.bool),
        "contact_depths": torch.zeros(num_envs, 3),
    }
    solver.states_history = deque(maxlen=history_length)
    for step in range(history_length):
        states = torch.stack(
            [torch.full((4,), float((env_id + 1) * 100 + step)) for env_id in range(num_envs)],
            dim=0,
        )
        solver.states_history.append(
            {
                "root_body_q": torch.full((num_envs, 7), float(step + 1)),
                "states": states,
                "states_embedding": states.clone(),
                "joint_f": torch.full((num_envs, 2), float(step + 1)),
                "gravity_dir": torch.full((num_envs, 3), float(step + 1)),
                "contact_masks": torch.ones(num_envs, 3, dtype=torch.bool),
                "contact_depths": torch.full((num_envs, 3), float(step + 1)),
            }
        )
    return solver


def _set_current_state(solver: TransformerNeuralSolver, value: float) -> None:
    """Write a standing-pose snapshot into the live solver buffers."""
    solver.root_body_q.fill_(value)
    solver.root_body_q[:, 3] = 1.0
    solver.states.fill_(value)
    solver.states_embedding.fill_(value)
    solver.joint_f.fill_(value)
    solver.gravity_dir.fill_(value)
    solver.contact_inputs["contact_masks"].zero_()
    solver.contact_inputs["contact_depths"].zero_()


def test_reset_none_clears_history():
    solver = _history_solver(num_envs=2)

    solver.reset()

    assert len(solver.states_history) == 0
    assert solver.feature_cache is None
    assert solver.kv_cache is None


def test_reset_all_env_ids_clears_history_like_episode_start():
    solver = _history_solver(num_envs=2)
    _set_current_state(solver, 42.0)

    solver.reset(torch.tensor([0, 1], dtype=torch.int32))

    assert len(solver.states_history) == 0


def test_reset_single_env_clears_history_when_num_envs_is_one():
    solver = _history_solver(num_envs=1)
    _set_current_state(solver, 42.0)

    solver.reset([0])

    assert len(solver.states_history) == 0


def test_partial_reset_copies_current_state_instead_of_zeros():
    solver = _history_solver(num_envs=2, history_length=4)
    original_env1 = [entry["states"][1].clone() for entry in solver.states_history]
    _set_current_state(solver, 42.0)

    solver.reset([0])

    assert len(solver.states_history) == 4
    snapshot = solver.history_snapshot()
    for entry in solver.states_history:
        torch.testing.assert_close(entry["states"][0], snapshot["states"][0])
        torch.testing.assert_close(entry["root_body_q"][0], snapshot["root_body_q"][0])
        torch.testing.assert_close(entry["joint_f"][0], snapshot["joint_f"][0])
        torch.testing.assert_close(entry["gravity_dir"][0], snapshot["gravity_dir"][0])
        torch.testing.assert_close(entry["contact_depths"][0], snapshot["contact_depths"][0])
        assert torch.equal(entry["contact_masks"][0], snapshot["contact_masks"][0])
        assert not torch.equal(entry["states"][0], torch.zeros_like(entry["states"][0]))
    for entry, expected in zip(solver.states_history, original_env1, strict=True):
        torch.testing.assert_close(entry["states"][1], expected)
    assert solver.feature_cache_rebuild_env_ids is not None
    assert torch.equal(solver.feature_cache_rebuild_env_ids, torch.tensor([0]))
