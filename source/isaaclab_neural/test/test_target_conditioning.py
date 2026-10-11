# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Zero-initialized target conditioning preserves the baseline and equal-capacity control."""

import torch
from isaaclab_neural.eval.target_conditioning import TargetErrorModel
from isaaclab_neural.models.models import ModelMixedInput


def _baseline():
    return ModelMixedInput(
        {"states_embedding": torch.zeros(2, 3, 4)},
        4,
        {"low_dim": ["states_embedding"]},
        {
            "normalize_input": False,
            "normalize_output": False,
            "encoder": {"low_dim": {"layer_sizes": [], "activation": "relu", "layernorm": False}},
            "transformer": {"n_layer": 1, "n_head": 2, "n_embd": 8, "block_size": 10, "dropout": 0.0, "bias": False},
            "model": {"mlp": {"layer_sizes": [], "activation": "relu", "layernorm": False}, "output_tanh": False},
        },
        device="cpu",
    )


def test_expansion_preserves_all_old_columns_bias_and_initial_predictions():
    base = _baseline().eval()
    enabled = TargetErrorModel(base, torch.tensor([2.0, -1.0]), torch.tensor([4.0, 9.0]), masked=False).eval()
    masked = TargetErrorModel(base, torch.tensor([2.0, -1.0]), torch.tensor([4.0, 9.0]), masked=True).eval()
    x = {"states_embedding": torch.randn(2, 3, 4), "joint_target_error": torch.randn(2, 3, 2)}
    expected = base({k: v.clone() for k, v in x.items()})
    for model in (enabled, masked):
        torch.testing.assert_close(model(x), expected, rtol=1e-5, atol=1e-6)
        old = base.transformer_model.transformer.wte
        new = model.base.transformer_model.transformer.wte
        torch.testing.assert_close(new.original.weight, old.weight)
        torch.testing.assert_close(new.original.bias, old.bias)
        assert not new.target.weight.any()
    assert sum(p.numel() for p in enabled.parameters()) == sum(p.numel() for p in masked.parameters())


def test_only_enabled_target_columns_receive_gradient():
    base = _baseline()
    for masked in (True, False):
        model = TargetErrorModel(base, torch.zeros(2), torch.ones(2), masked=masked)
        x = {"states_embedding": torch.randn(2, 3, 4), "joint_target_error": torch.ones(2, 3, 2)}
        original = x["joint_target_error"].clone()
        model(x).square().mean().backward()
        gradient = model.base.transformer_model.transformer.wte.target.weight.grad
        assert bool(gradient.abs().sum() > 0) is (not masked)
        torch.testing.assert_close(x["joint_target_error"], original)


def test_runtime_target_history_matches_offline_feature():
    model = TargetErrorModel(_baseline(), torch.zeros(2), torch.ones(2), masked=False).eval()
    torch.nn.init.normal_(model.base.transformer_model.transformer.wte.target.weight)
    states = torch.randn(2, 3, 9)
    error = torch.randn(2, 3, 2)
    inputs = {"states_embedding": torch.randn(2, 3, 4), "states": states, "next_states": None}
    model.runtime_targets = states[..., 7:9] + error
    offline = model({**inputs, "joint_target_error": error})
    torch.testing.assert_close(model(inputs, single_step=True), offline[:, -1:], rtol=1e-5, atol=1e-5)
