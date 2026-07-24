# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for TransformerNeuralSolver history batching without Newton."""

from collections import deque

import torch


def _collate_history_like_solver(states_history: deque) -> dict[str, torch.Tensor]:
    """Mirror TransformerNeuralSolver.get_neural_model_inputs history assembly."""
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
    # Time axis must preserve history order after [T,B,...] -> [B,T,...].
    assert torch.equal(model_inputs["states"][0, :, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(model_inputs["contact_token_overflow"][0], torch.tensor([0, 1, 2]))
