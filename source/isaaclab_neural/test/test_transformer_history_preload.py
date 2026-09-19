# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for dataset-backed transformer rollout history."""

from collections import deque

import torch
from isaaclab_neural.solvers.transformer_neural_solver import TransformerNeuralSolver


class _FakeTransformerSolver(TransformerNeuralSolver):
    history_snapshot = TransformerNeuralSolver._history_snapshot
    fill_history_rows = TransformerNeuralSolver._fill_history_rows_from_current_state

    def __init__(self):
        self.num_envs = 2
        self.num_states_history = 10
        self.use_kv_cache = False
        self.use_feature_cache = False
        self.torch_device = "cpu"
        self.root_body_q = torch.zeros(2, 7)
        self.states = torch.zeros(2, 4)
        self.states_embedding = torch.zeros(2, 4)
        self.states_embedding_type = "identical"
        self.joint_f = torch.zeros(2, 2)
        self.gravity_dir = torch.zeros(2, 3)
        self.contact_inputs = {
            "contact_masks": torch.zeros(2, 3, dtype=torch.bool),
            "contact_depths": torch.zeros(2, 3),
        }
        self.reset_states_history()


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


def _collate_history_like_solver(states_history: deque) -> dict[str, torch.Tensor]:
    """Mirror TransformerNeuralSolver history assembly."""
    model_inputs = torch.utils.data.default_collate(list(states_history))
    for key, value in model_inputs.items():
        assert value.ndim >= 2
        model_inputs[key] = value.transpose(0, 1).contiguous()
    return model_inputs


def test_history_collate_supports_flat_and_token_ranks():
    batch, time, tokens, dim = 4, 3, 128, 17
    history = deque(maxlen=time)
    for step in range(time):
        history.append(
            {
                "states": torch.full((batch, 37), float(step)),
                "contact_tokens": torch.full((batch, tokens, dim), float(step)),
                "contact_token_overflow": torch.full((batch,), step, dtype=torch.long),
                "gravity_dir": torch.zeros(batch, 3),
            }
        )

    model_inputs = _collate_history_like_solver(history)

    assert model_inputs["states"].shape == (batch, time, 37)
    assert model_inputs["contact_tokens"].shape == (batch, time, tokens, dim)
    assert model_inputs["contact_token_overflow"].shape == (batch, time)
    assert model_inputs["gravity_dir"].shape == (batch, time, 3)
    assert torch.equal(model_inputs["states"][0, :, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(model_inputs["contact_token_overflow"][0], torch.tensor([0, 1, 2]))


def test_preload_states_history_preserves_order_and_dtypes():
    solver = _FakeTransformerSolver()

    solver.preload_states_history(_history())

    assert len(solver.states_history) == 9
    assert torch.equal(solver.states_history[0]["states"], _history()["states"][:, 0])
    assert torch.equal(solver.states_history[-1]["states"], _history()["states"][:, -1])
    assert solver.states_history[0]["contact_masks"].dtype == torch.bool
    assert torch.equal(solver.states_history[-1]["states_embedding"], solver.states_history[-1]["states"])


def test_preload_states_history_rejects_overlong_window():
    solver = _FakeTransformerSolver()

    try:
        solver.preload_states_history(_history(history_length=11))
    except ValueError as error:
        assert "between 1 and 10" in str(error)
    else:
        raise AssertionError("Expected overlong history to be rejected.")


def test_preload_states_history_rejects_mismatched_contact_shape():
    solver = _FakeTransformerSolver()
    history = _history()
    history["contact_depths"] = torch.zeros(2, 9, 4)

    try:
        solver.preload_states_history(history)
    except ValueError as error:
        assert "History contact_depths must have shape" in str(error)
    else:
        raise AssertionError("Expected mismatched history shape to be rejected.")


def test_preload_states_history_squeezes_scalar_overflow_feature_axis():
    solver = _FakeTransformerSolver()
    solver.contact_inputs = {
        "contact_tokens": torch.zeros(2, 3, 17),
        "contact_token_overflow": torch.zeros(2, dtype=torch.long),
    }
    history = {
        **{key: value for key, value in _history().items() if not key.startswith("contact_")},
        "contact_tokens": torch.zeros(2, 9, 3, 17),
        "contact_token_overflow": torch.zeros(2, 9, 1, dtype=torch.long),
    }

    solver.preload_states_history(history)

    assert solver.states_history[0]["contact_token_overflow"].shape == (2,)
    assert solver.states_history[0]["contact_token_overflow"].dtype == torch.long


def test_online_history_snapshot_does_not_alias_contact_buffers():
    solver = _FakeTransformerSolver()
    solver.root_body_q = torch.zeros(2, 7)
    solver.states = torch.zeros(2, 4)
    solver.states_embedding = torch.zeros(2, 4)
    solver.joint_f = torch.zeros(2, 2)
    solver.gravity_dir = torch.zeros(2, 3)
    solver.contact_inputs = {
        "contact_tokens": torch.zeros(2, 3, 17),
        "contact_token_overflow": torch.zeros(2, dtype=torch.long),
    }

    snapshot = solver.history_snapshot()
    solver.contact_inputs["contact_tokens"].fill_(1.0)
    solver.contact_inputs["contact_token_overflow"].fill_(2)

    assert torch.count_nonzero(snapshot["contact_tokens"]) == 0
    assert torch.count_nonzero(snapshot["contact_token_overflow"]) == 0


def test_partial_history_fill_copies_current_snapshot_into_reset_rows():
    solver = _FakeTransformerSolver()
    solver.reset_states_history()
    solver.states_embedding = torch.zeros(2, 4)
    solver.states_history.append(solver.history_snapshot())
    solver.states_history.append(solver.history_snapshot())
    solver.states = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    solver.states_embedding = solver.states.clone()
    solver.joint_f = torch.tensor([[9.0, 10.0], [11.0, 12.0]])

    solver.fill_history_rows(torch.tensor([1]))

    for entry in solver.states_history:
        assert torch.equal(entry["states"][0], torch.zeros(4))
        assert torch.equal(entry["joint_f"][0], torch.zeros(2))
        assert torch.equal(entry["states"][1], solver.states[1])
        assert torch.equal(entry["joint_f"][1], solver.joint_f[1])


def test_partial_history_fill_cannot_mutate_inference_mode_clones():
    solver = _FakeTransformerSolver()
    solver.reset_states_history()
    solver.states_embedding = torch.zeros(2, 4)
    with torch.inference_mode():
        solver.states_history.append(solver.history_snapshot())
    solver.states = torch.ones(2, 4)
    solver.states_embedding = torch.ones(2, 4)
    solver.joint_f = torch.ones(2, 2)

    try:
        solver.fill_history_rows(torch.tensor([0]))
    except RuntimeError as error:
        assert "inference" in str(error).lower()
        return
    raise AssertionError("Expected inplace update to inference-mode history tensors to fail.")
