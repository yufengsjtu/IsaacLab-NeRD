# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for dataset-backed transformer rollout history."""

from collections import deque

import torch
from isaaclab_neural.solvers.transformer_neural_solver import TransformerNeuralSolver


class _FakeTransformerSolver:
    num_envs = 2
    num_states_history = 10
    torch_device = torch.device("cpu")
    root_body_q = torch.zeros(2, 7)
    states = torch.zeros(2, 4)
    joint_f = torch.zeros(2, 2)
    gravity_dir = torch.zeros(2, 3)
    contacts = {
        "contact_masks": torch.zeros(2, 3, dtype=torch.bool),
        "contact_depths": torch.zeros(2, 3),
    }

    def reset_states_history(self):
        self.states_history = deque(maxlen=self.num_states_history)

    @staticmethod
    def embed_states(states):
        return states.clone()


def _history(history_length: int = 9):
    shape = (2, history_length)
    return {
        "root_body_q": torch.zeros(*shape, 7),
        "states": torch.arange(2 * history_length * 4, dtype=torch.float32).reshape(*shape, 4),
        "joint_f": torch.zeros(*shape, 2),
        "gravity_dir": torch.zeros(*shape, 3),
        "contact_masks": torch.zeros(*shape, 3, dtype=torch.bool),
        "contact_depths": torch.zeros(*shape, 3),
    }


def test_preload_states_history_preserves_order_and_dtypes():
    solver = _FakeTransformerSolver()

    TransformerNeuralSolver.preload_states_history(solver, _history())

    assert len(solver.states_history) == 9
    assert torch.equal(solver.states_history[0]["states"], _history()["states"][:, 0])
    assert torch.equal(solver.states_history[-1]["states"], _history()["states"][:, -1])
    assert solver.states_history[0]["contact_masks"].dtype == torch.bool
    assert torch.equal(solver.states_history[-1]["states_embedding"], solver.states_history[-1]["states"])


def test_preload_states_history_rejects_overlong_window():
    solver = _FakeTransformerSolver()

    try:
        TransformerNeuralSolver.preload_states_history(solver, _history(history_length=11))
    except ValueError as error:
        assert "between 1 and 10" in str(error)
    else:
        raise AssertionError("Expected overlong history to be rejected.")


def test_preload_states_history_rejects_mismatched_contact_shape():
    solver = _FakeTransformerSolver()
    history = _history()
    history["contact_depths"] = torch.zeros(2, 9, 4)

    try:
        TransformerNeuralSolver.preload_states_history(solver, history)
    except ValueError as error:
        assert "History contact_depths must have shape" in str(error)
    else:
        raise AssertionError("Expected mismatched history shape to be rejected.")


def test_preload_states_history_squeezes_scalar_overflow_feature_axis():
    solver = _FakeTransformerSolver()
    solver.contacts = {
        "contact_tokens": torch.zeros(2, 3, 17),
        "contact_token_overflow": torch.zeros(2, dtype=torch.long),
    }
    history = {
        **{key: value for key, value in _history().items() if not key.startswith("contact_")},
        "contact_tokens": torch.zeros(2, 9, 3, 17),
        "contact_token_overflow": torch.zeros(2, 9, 1, dtype=torch.long),
    }

    TransformerNeuralSolver.preload_states_history(solver, history)

    assert solver.states_history[0]["contact_token_overflow"].shape == (2,)
    assert solver.states_history[0]["contact_token_overflow"].dtype == torch.long


def test_online_history_snapshot_does_not_alias_contact_buffers():
    solver = _FakeTransformerSolver()
    solver.root_body_q = torch.zeros(2, 7)
    solver.states = torch.zeros(2, 4)
    solver.states_embedding = torch.zeros(2, 4)
    solver.joint_f = torch.zeros(2, 2)
    solver.gravity_dir = torch.zeros(2, 3)
    solver.contacts = {
        "contact_tokens": torch.zeros(2, 3, 17),
        "contact_token_overflow": torch.zeros(2, dtype=torch.long),
    }

    snapshot = TransformerNeuralSolver._history_snapshot(solver)
    solver.contacts["contact_tokens"].fill_(1.0)
    solver.contacts["contact_token_overflow"].fill_(2)

    assert torch.count_nonzero(snapshot["contact_tokens"]) == 0
    assert torch.count_nonzero(snapshot["contact_token_overflow"]) == 0
