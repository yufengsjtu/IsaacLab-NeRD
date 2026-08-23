# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for body-routed Deep Sets contact encoding."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from isaaclab_neural.contacts.contact_set_schema import CONTACT_TOKEN_DIM
from isaaclab_neural.contacts.tensor_utils import ContactTokenMoments, normalize_contact_tokens
from isaaclab_neural.models.body_routed_contact_model import BodyRoutedContactEncoder
from isaaclab_neural.models.models import ModelMixedInput
from isaaclab_neural.utils.checkpoint import load_checkpoint, reconstruct_model_from_checkpoint, save_checkpoint

PACKAGE_ROOT = Path(__file__).parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
CFG_DIR = PACKAGE_ROOT / "isaaclab_neural" / "train" / "cfg" / "Anymal"
PRESET_DIR = REPOSITORY_ROOT / "osmo_scripts" / "presets"


def _token(
    body_slot: int,
    *,
    point_x: float = 0.1,
    other_body_slot: int = -1,
    other_is_dynamic: bool = False,
) -> torch.Tensor:
    token = torch.zeros(CONTACT_TOKEN_DIM)
    token[0] = 1.0
    token[1] = body_slot
    token[2] = other_body_slot
    token[3] = float(other_is_dynamic)
    token[4] = point_x
    token[9] = 1.0
    token[13] = -0.01
    return token


def _mixed_input_network_cfg() -> dict:
    return {
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
            "mlp": {
                "layer_sizes": [],
                "activation": "relu",
                "layernorm": False,
            }
        },
    }


def _model_input_sample() -> dict[str, torch.Tensor]:
    inputs = {
        "states_embedding": torch.randn(2, 3, 4),
        "contact_tokens": torch.zeros(2, 3, 5, CONTACT_TOKEN_DIM),
    }
    inputs["contact_tokens"][0, 0, 0] = _token(body_slot=2)
    inputs["contact_tokens"][1, 1, 0] = _token(body_slot=0, point_x=0.2)
    return inputs


def _input_cfg(num_bodies: int = 3) -> dict:
    return {
        "low_dim": ["states_embedding"],
        "contact_set": {
            "dim": CONTACT_TOKEN_DIM,
            "encoder_type": "body_routed",
            "num_bodies": num_bodies,
            "body_latent_dim": 8,
            "hidden_dim": 16,
            "max_other_bodies": 8,
        },
    }


def _solver_stub(input_sample: dict[str, torch.Tensor], num_bodies: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        get_neural_model_inputs=lambda: {key: value.clone() for key, value in input_sample.items()},
        prediction_dim=2,
        contact_mode="newton_native",
        num_contact_bodies_per_env=num_bodies,
    )


def _encoded_contacts(encoder: BodyRoutedContactEncoder, tokens: torch.Tensor) -> torch.Tensor:
    encoded = encoder.phi(tokens[..., 4:])
    other_ids = tokens[..., 2].long()
    other_dynamic = tokens[..., 3] > 0.5
    known = other_ids >= 0
    known_embedding = encoder.other_body_embed(other_ids.clamp(min=0, max=encoder.other_body_embed.num_embeddings - 1))
    unknown_embedding = torch.where(
        other_dynamic.unsqueeze(-1),
        encoder.other_foreign_embed.expand_as(known_embedding),
        encoder.other_static_embed.expand_as(known_embedding),
    )
    return encoded + torch.where(known.unsqueeze(-1), known_embedding, unknown_embedding)


def test_body_routed_encoder_defaults_match_a_token17_contract() -> None:
    encoder = BodyRoutedContactEncoder(num_bodies=17)

    assert encoder.body_latent_dim == 64
    assert encoder.phi[0].in_features == 13
    assert encoder.phi[0].out_features == 32
    assert encoder.phi[-1].out_features == 64
    assert encoder.rho.in_features == 64
    assert encoder.rho.out_features == 64
    assert encoder.out_features == 17 * 64


def test_body_routed_encoder_matches_explicit_per_body_reference_sum() -> None:
    torch.manual_seed(0)
    encoder = BodyRoutedContactEncoder(num_bodies=3, body_latent_dim=8, hidden_dim=16, max_other_bodies=8)
    tokens = torch.zeros(1, 5, CONTACT_TOKEN_DIM)
    tokens[0, 0] = _token(body_slot=0, point_x=0.1)
    tokens[0, 1] = _token(body_slot=2, point_x=0.2, other_body_slot=1, other_is_dynamic=True)
    tokens[0, 2] = _token(body_slot=0, point_x=0.3)

    encoded = _encoded_contacts(encoder, tokens)
    expected = torch.zeros(1, 3, 8)
    expected[0, 0] = encoder.rho(encoded[0, 0] + encoded[0, 2])
    expected[0, 2] = encoder.rho(encoded[0, 1])

    torch.testing.assert_close(encoder(tokens), expected)


def test_body_routed_encoder_is_permutation_invariant_and_ignores_invalid_tokens() -> None:
    torch.manual_seed(0)
    encoder = BodyRoutedContactEncoder(num_bodies=2, body_latent_dim=8, hidden_dim=16)
    tokens = torch.zeros(1, 5, CONTACT_TOKEN_DIM)
    tokens[0, 0] = _token(body_slot=0, point_x=0.1)
    tokens[0, 1] = _token(body_slot=1, point_x=0.2)
    tokens[0, 2] = _token(body_slot=0, point_x=0.3)
    tokens[0, 3, 1:] = torch.randn(CONTACT_TOKEN_DIM - 1)
    tokens[0, 4] = _token(body_slot=2, point_x=10.0)
    shuffled = tokens[:, torch.tensor([2, 4, 0, 3, 1])]

    torch.testing.assert_close(encoder(tokens), encoder(shuffled))


def test_body_routed_encoder_preserves_other_body_identity() -> None:
    torch.manual_seed(0)
    encoder = BodyRoutedContactEncoder(num_bodies=2, body_latent_dim=8, hidden_dim=16, max_other_bodies=4)
    first = torch.zeros(1, 1, CONTACT_TOKEN_DIM)
    first[0, 0] = _token(body_slot=0, other_body_slot=0, other_is_dynamic=True)
    second = first.clone()
    second[0, 0, 2] = 1.0

    with torch.no_grad():
        encoder.other_body_embed.weight[0].zero_()
        encoder.other_body_embed.weight[1].fill_(1.0)

    assert not torch.allclose(encoder(first), encoder(second))


def test_body_routed_encoder_recovers_rms_scaled_categorical_ids() -> None:
    torch.manual_seed(0)
    encoder = BodyRoutedContactEncoder(num_bodies=17, body_latent_dim=8, hidden_dim=16)
    tokens = torch.zeros(1, 3, CONTACT_TOKEN_DIM)
    tokens[0, 0] = _token(body_slot=1, other_body_slot=-1)
    tokens[0, 1] = _token(body_slot=16, point_x=0.2, other_body_slot=1, other_is_dynamic=True)

    moments = ContactTokenMoments(CONTACT_TOKEN_DIM, device="cpu")
    moments.update(tokens)
    normalized = normalize_contact_tokens(tokens, moments.finalize(device="cpu"))
    categorical_reference = normalized.clone()
    categorical_reference[..., :4] = tokens[..., :4]

    torch.testing.assert_close(encoder(normalized), encoder(categorical_reference))


def test_body_routed_encoder_supports_step_history_and_zero_capacity_inputs() -> None:
    encoder = BodyRoutedContactEncoder(num_bodies=3, body_latent_dim=8, hidden_dim=16)

    step_output = encoder(torch.zeros(2, 4, CONTACT_TOKEN_DIM))
    history_output = encoder(torch.zeros(2, 5, 4, CONTACT_TOKEN_DIM))
    zero_capacity_output = encoder(torch.zeros(2, 5, 0, CONTACT_TOKEN_DIM))

    assert step_output.shape == (2, 3, 8)
    assert history_output.shape == (2, 5, 3, 8)
    assert zero_capacity_output.shape == (2, 5, 3, 8)
    torch.testing.assert_close(step_output, torch.zeros_like(step_output))
    torch.testing.assert_close(history_output, torch.zeros_like(history_output))
    torch.testing.assert_close(zero_capacity_output, torch.zeros_like(zero_capacity_output))


def test_body_routed_encoder_has_finite_forward_and_backward() -> None:
    torch.manual_seed(0)
    encoder = BodyRoutedContactEncoder(num_bodies=2, body_latent_dim=8, hidden_dim=16)
    tokens = torch.zeros(2, 3, CONTACT_TOKEN_DIM)
    tokens[0, 0] = _token(body_slot=0)
    tokens[1, 0] = _token(body_slot=1)
    tokens[1, 1] = _token(body_slot=1, point_x=0.2)
    tokens.requires_grad_()

    output = encoder(tokens)
    output.square().mean().backward()

    assert torch.isfinite(output).all()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in encoder.parameters()
    )


def test_mixed_input_constructs_and_flattens_body_routed_encoder() -> None:
    input_sample = _model_input_sample()
    model = ModelMixedInput(
        input_sample=input_sample,
        output_dim=2,
        input_cfg=_input_cfg(),
        network_cfg=_mixed_input_network_cfg(),
        device="cpu",
    )

    features = model.extract_input_features(input_sample)

    assert isinstance(model.encoders["contact_set"], BodyRoutedContactEncoder)
    assert model.feature_dim == 4 + 3 * 8
    assert features.shape == (2, 3, 4 + 3 * 8)
    assert torch.count_nonzero(features[0, 0, -8:]) > 0


def test_body_routed_checkpoint_round_trip_preserves_forward_output(tmp_path: Path) -> None:
    torch.manual_seed(0)
    input_sample = _model_input_sample()
    input_cfg = _input_cfg()
    network_cfg = _mixed_input_network_cfg()
    cfg = {"inputs": input_cfg, "network": network_cfg}
    model = ModelMixedInput(
        input_sample=input_sample,
        output_dim=2,
        input_cfg=input_cfg,
        network_cfg=network_cfg,
        device="cpu",
    )
    model.eval()
    expected = model({key: value.clone() for key, value in input_sample.items()})
    checkpoint_path = tmp_path / "body_routed.pt"

    save_checkpoint(checkpoint_path, model, "Anymal-C", cfg)
    checkpoint = load_checkpoint(checkpoint_path)
    reconstructed = reconstruct_model_from_checkpoint(checkpoint, _solver_stub(input_sample), device="cpu")
    actual = reconstructed({key: value.clone() for key, value in input_sample.items()})

    assert isinstance(reconstructed.encoders["contact_set"], BodyRoutedContactEncoder)
    torch.testing.assert_close(actual, expected)


def test_body_routed_config_reuses_shared_token_data_and_training_contract() -> None:
    body_cfg = yaml.safe_load((CFG_DIR / "transformer_rough_native_body_routed.yaml").read_text())
    shared_cfg = yaml.safe_load((CFG_DIR / "transformer_rough_native_shared_per_body.yaml").read_text())
    body_preset = yaml.safe_load((PRESET_DIR / "anymal_rough_newton_native_body_routed.yaml").read_text())
    shared_preset = yaml.safe_load((PRESET_DIR / "anymal_rough_newton_native_shared_per_body.yaml").read_text())

    assert body_cfg["env"] == shared_cfg["env"]
    assert body_cfg["algorithm"] == shared_cfg["algorithm"]
    assert body_cfg["network"] == shared_cfg["network"]
    assert body_cfg["inputs"]["low_dim"] == shared_cfg["inputs"]["low_dim"]
    assert body_cfg["inputs"]["contact_set"] == {
        "dim": CONTACT_TOKEN_DIM,
        "encoder_type": "body_routed",
        "body_latent_dim": 64,
        "hidden_dim": 32,
        "max_other_bodies": 32,
    }
    assert body_preset["workflow"]["dataset_subdir"] == shared_preset["workflow"]["dataset_subdir"]
    body_experiment = copy.deepcopy(body_preset["experiment"])
    shared_experiment = copy.deepcopy(shared_preset["experiment"])
    body_experiment.pop("train_cfg")
    shared_experiment.pop("train_cfg")
    shared_experiment.pop("save_interval")
    shared_experiment.pop("persist_periodic_checkpoints")
    assert body_experiment == shared_experiment
    assert Path(body_preset["experiment"]["train_cfg"]).name == "transformer_rough_native_body_routed.yaml"


def test_body_routed_lr_variant_only_changes_learning_rate() -> None:
    author = yaml.safe_load((CFG_DIR / "transformer_rough_native_body_routed.yaml").read_text())
    variant = yaml.safe_load((CFG_DIR / "transformer_rough_native_body_routed_lr_1e-3.yaml").read_text())

    assert variant["algorithm"]["optimizer"]["lr_start"] == "1e-3"
    assert variant["algorithm"]["optimizer"]["lr_end"] == "1e-4"

    normalized = copy.deepcopy(variant)
    normalized["algorithm"]["optimizer"]["lr_start"] = author["algorithm"]["optimizer"]["lr_start"]
    normalized["algorithm"]["optimizer"]["lr_end"] = author["algorithm"]["optimizer"]["lr_end"]
    assert normalized == author


def test_body_routed_lr_preset_only_changes_names_and_config() -> None:
    author = yaml.safe_load((PRESET_DIR / "anymal_rough_newton_native_body_routed.yaml").read_text())
    variant = yaml.safe_load((PRESET_DIR / "anymal_rough_newton_native_body_routed_lr_1e-3.yaml").read_text())

    normalized = copy.deepcopy(variant)
    normalized["workflow"]["base_name"] = author["workflow"]["base_name"]
    normalized["experiment"]["train_cfg"] = author["experiment"]["train_cfg"]
    assert normalized == author
