# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Model-construction tests for the native15 self-collision representations."""

import pytest
import torch
from isaaclab_neural.contacts.contact_set_schema import ACTIVE15_TOKEN_DIM
from isaaclab_neural.models.body_routed_active15_model import BodyRoutedActive15Encoder
from isaaclab_neural.models.models import ModelMixedInput


def _model(contact_representation: str, encoder_type: str) -> ModelMixedInput:
    sample = {
        "states_embedding": torch.randn(2, 3, 58),
        "contact_tokens": torch.zeros(2, 3, 64, ACTIVE15_TOKEN_DIM),
    }
    input_cfg = {
        "low_dim": ["states_embedding"],
        "contact_set": {
            "dim": ACTIVE15_TOKEN_DIM,
            "encoder_type": encoder_type,
            "num_bodies": 17,
            "body_latent_dim": 64,
            "hidden_dim": 32,
        },
    }
    network_cfg = {
        "normalize_input": False,
        "normalize_output": False,
        "encoder": {"low_dim": {"layer_sizes": [], "activation": "relu", "layernorm": False}},
        "model": {"mlp": {"layer_sizes": [], "activation": "relu", "layernorm": False}},
    }
    return ModelMixedInput(
        input_sample=sample,
        output_dim=2,
        input_cfg=input_cfg,
        network_cfg=network_cfg,
        device="cpu",
        contact_representation=contact_representation,
    )


def test_active15_self_reuses_the_matched_body_routed_encoder() -> None:
    model = _model("active15_self_tokens", "body_routed_active15")
    assert isinstance(model.encoders["contact_set"], BodyRoutedActive15Encoder)


def test_active15_self_rejects_the_raw15_encoder() -> None:
    with pytest.raises(ValueError, match="configured together"):
        _model("active15_self_tokens", "body_routed_raw15")
