# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for contact padding at the neural encoder boundary."""

from typing import Any, cast

import pytest
import torch
import torch.nn as nn
from isaaclab_neural.contacts.tensor_utils import (
    MaskedContactMoments,
    mask_inactive_contact_fields,
)
from isaaclab_neural.models.models import ModelMixedInput
from isaaclab_neural.solvers.neural_solver import NeuralSolver
from isaaclab_neural.utils.running_mean_std import RunningMeanStd


class _PassThrough(nn.Module):
    def forward(self, inputs, deterministic=False):
        del deterministic
        return inputs


def test_contact_processing_is_derived_from_contact_mode():
    input_sample = {"states_embedding": torch.zeros(1, 1, 2)}
    input_cfg = {"low_dim": ["states_embedding"]}
    network_cfg = {
        "normalize_input": False,
        "normalize_output": False,
        "encoder": {
            "low_dim": {
                "layer_sizes": [],
                "activation": "relu",
                "layernorm": False,
            }
        },
        "model": {
            "output_tanh": False,
            "mlp": {
                "layer_sizes": [],
                "activation": "relu",
                "layernorm": False,
            },
        },
    }

    native_model = ModelMixedInput(
        input_sample,
        1,
        input_cfg,
        network_cfg,
        device="cpu",
        contact_mode="newton_native",
    )
    fixed_model = ModelMixedInput(
        input_sample,
        1,
        input_cfg,
        network_cfg,
        device="cpu",
        contact_mode="fixed_ground",
    )

    assert native_model.use_native_contact_processing
    assert not fixed_model.use_native_contact_processing


def test_mask_inactive_contact_features_zeros_normalized_padding():
    inputs = {
        "contact_masks": torch.tensor([[[True, False]]]),
        "contact_normals": torch.tensor([[[1.0, 2.0, 3.0, -4.0, -5.0, -6.0]]]),
        "contact_depths": torch.tensor([[[0.25, -3.0]]]),
        "states_embedding": torch.ones(1, 1, 4),
    }

    mask_inactive_contact_fields(inputs)

    torch.testing.assert_close(inputs["contact_normals"], torch.tensor([[[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]]]))
    torch.testing.assert_close(inputs["contact_depths"], torch.tensor([[[0.25, 0.0]]]))
    torch.testing.assert_close(inputs["states_embedding"], torch.ones(1, 1, 4))


def test_mask_inactive_contact_features_rejects_incompatible_shapes():
    inputs = {
        "contact_masks": torch.tensor([[[True, False]]]),
        "contact_normals": torch.ones(1, 1, 5),
    }

    with pytest.raises(ValueError, match="incompatible with contact mask"):
        mask_inactive_contact_fields(inputs)


def test_forward_reapplies_mask_after_normalization():
    model = ModelMixedInput.__new__(ModelMixedInput)
    nn.Module.__init__(model)
    model.normalize_input = True
    model.use_native_contact_processing = True
    model.normalize_output = False
    model.output_tanh = False
    model.is_rnn = False
    model.is_transformer = False
    model.low_dim_input_names = ["contact_normals", "contact_depths"]
    model.encoders = nn.ModuleDict({"low_dim": nn.Identity()})
    model.model = cast(Any, _PassThrough())
    model.input_rms = nn.ModuleDict(
        {
            "contact_normals": RunningMeanStd(shape=(2, 3), device="cpu"),
            "contact_depths": RunningMeanStd(shape=(2, 1), device="cpu"),
        }
    )
    for rms in model.input_rms.values():
        rms_module = cast(Any, rms)
        rms_module.mean.fill_(1.0)
        rms_module.var.fill_(1.0)

    inputs = {
        "contact_masks": torch.tensor([[[True, False]]]),
        "contact_normals": torch.tensor([[[2.0, 3.0, 4.0, 0.0, 0.0, 0.0]]]),
        "contact_depths": torch.tensor([[[0.5, 0.0]]]),
    }
    output = model(inputs)

    torch.testing.assert_close(output[..., 3:6], torch.zeros_like(output[..., 3:6]))
    torch.testing.assert_close(output[..., -1], torch.zeros_like(output[..., -1]))


def test_masked_contact_rms_uses_valid_values_per_slot():
    masks = torch.tensor([[[True, False, True]], [[False, True, False]]])
    values = torch.tensor(
        [
            [[[1.0, 2.0], [100.0, 200.0], [3.0, 4.0]]],
            [[[300.0, 400.0], [5.0, 6.0], [500.0, 600.0]]],
        ]
    ).reshape(2, 1, 6)

    moments = MaskedContactMoments.from_batch(values, masks)
    moments.update(values, masks)
    rms, counts = moments.finalize("cpu", min_samples=1)

    torch.testing.assert_close(rms.mean, torch.tensor([[1.0, 2.0], [5.0, 6.0], [3.0, 4.0]]))
    torch.testing.assert_close(rms.var, torch.zeros(3, 2))
    torch.testing.assert_close(counts, torch.ones(3))


def test_sparse_contact_slots_use_pooled_rms_fallback():
    masks = torch.tensor([[[True, True]], [[True, False]]])
    values = torch.tensor([[[1.0, 10.0]], [[3.0, 100.0]]])
    moments = MaskedContactMoments.from_batch(values, masks)
    moments.update(values, masks)

    rms, counts = moments.finalize("cpu", min_samples=2)
    rms_module = cast(Any, rms)

    # Slot 0 has two samples and keeps its own statistics.
    torch.testing.assert_close(rms_module.mean[0], torch.tensor([2.0]))
    torch.testing.assert_close(rms_module.var[0], torch.tensor([1.0]))
    # Slot 1 has one valid sample and falls back to all valid contacts.
    torch.testing.assert_close(rms_module.mean[1], torch.tensor([14.0 / 3.0]))
    torch.testing.assert_close(rms_module.var[1], torch.tensor([14.888889]), atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(counts, torch.tensor([2.0, 1.0]))


def test_native_contact_masks_cannot_be_reconstructed_from_separation():
    solver = NeuralSolver.__new__(NeuralSolver)
    solver.contact_mode = "newton_native"
    values = torch.zeros(1, 1, 2)

    with pytest.raises(ValueError, match="require explicit contact_masks"):
        solver.get_contact_masks(values, values, values)
