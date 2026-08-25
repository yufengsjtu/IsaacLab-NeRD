# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shared per-body contact encoding."""

from types import SimpleNamespace

import pytest
import torch
from isaaclab_neural.contacts.contact_set_schema import CONTACT_TOKEN_DIM, CONTACT_TOKEN_GEOMETRY_SLICE
from isaaclab_neural.models.contact_set_model import ContactSetEncoderBlock
from isaaclab_neural.models.models import ModelMixedInput
from isaaclab_neural.models.shared_per_body_contact_encoder import SharedPerBodyContactEncoder
from isaaclab_neural.utils.checkpoint import (
    load_checkpoint,
    reconstruct_model_from_checkpoint,
    save_checkpoint,
)


def _token(body_slot: int, point_x: float = 0.1, other_body_slot: int = -1) -> torch.Tensor:
    token = torch.zeros(CONTACT_TOKEN_DIM)
    token[0] = 1.0
    token[1] = body_slot
    token[2] = other_body_slot
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


def _solver_stub(input_sample: dict[str, torch.Tensor], num_bodies: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        get_neural_model_inputs=lambda: {key: value.clone() for key, value in input_sample.items()},
        prediction_dim=2,
        contact_mode="newton_native",
        num_contact_bodies_per_env=num_bodies,
    )


def test_shared_per_body_encoder_uses_low_dimensional_default_capacity():
    encoder = SharedPerBodyContactEncoder(num_bodies=3)

    assert encoder.body_latent_dim == 16
    assert encoder.contact_encoder[0].out_features == 64


def test_mixed_input_defaults_to_legacy_global_contact_encoder():
    input_sample = {
        "states_embedding": torch.zeros(2, 3, 4),
        "contact_tokens": torch.zeros(2, 3, 5, CONTACT_TOKEN_DIM),
    }
    input_cfg = {
        "low_dim": ["states_embedding"],
        "contact_set": {
            "dim": CONTACT_TOKEN_DIM,
            "hidden_size": 16,
            "encoder_heads": 4,
        },
    }

    model = ModelMixedInput(
        input_sample=input_sample,
        output_dim=2,
        input_cfg=input_cfg,
        network_cfg=_mixed_input_network_cfg(),
        device="cpu",
    )

    assert isinstance(model.encoders["contact_set"], ContactSetEncoderBlock)
    assert model.feature_dim == 20


def test_mixed_input_constructs_and_flattens_shared_per_body_encoder():
    input_sample = {
        "states_embedding": torch.zeros(2, 3, 4),
        "contact_tokens": torch.zeros(2, 3, 5, CONTACT_TOKEN_DIM),
    }
    input_sample["contact_tokens"][0, 0, 0] = _token(body_slot=2)
    input_cfg = {
        "low_dim": ["states_embedding"],
        "contact_set": {
            "dim": CONTACT_TOKEN_DIM,
            "encoder_type": "shared_per_body",
            "num_bodies": 3,
            "body_latent_dim": 8,
            "hidden_dim": 16,
            "use_other_body_embeddings": False,
        },
    }

    model = ModelMixedInput(
        input_sample=input_sample,
        output_dim=2,
        input_cfg=input_cfg,
        network_cfg=_mixed_input_network_cfg(),
        device="cpu",
    )
    features = model.extract_input_features(input_sample)

    assert isinstance(model.encoders["contact_set"], SharedPerBodyContactEncoder)
    assert model.encoders["contact_set"].use_other_body_embeddings is False
    assert model.feature_dim == 28
    assert features.shape == (2, 3, 28)
    assert torch.count_nonzero(features[0, 0, -8:]) > 0


@pytest.mark.parametrize("use_other_body_embeddings", (True, False))
def test_shared_per_body_checkpoint_round_trip_preserves_forward_output(tmp_path, use_other_body_embeddings: bool):
    torch.manual_seed(0)
    input_sample = _model_input_sample()
    input_cfg = {
        "low_dim": ["states_embedding"],
        "contact_set": {
            "dim": CONTACT_TOKEN_DIM,
            "encoder_type": "shared_per_body",
            "num_bodies": 3,
            "body_latent_dim": 8,
            "hidden_dim": 16,
            "use_other_body_embeddings": use_other_body_embeddings,
        },
    }
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
    checkpoint_path = tmp_path / "shared_per_body.pt"

    save_checkpoint(checkpoint_path, model, "Anymal-C", cfg)
    checkpoint = load_checkpoint(checkpoint_path)
    reconstructed = reconstruct_model_from_checkpoint(checkpoint, _solver_stub(input_sample), device="cpu")
    actual = reconstructed({key: value.clone() for key, value in input_sample.items()})

    assert isinstance(reconstructed.encoders["contact_set"], SharedPerBodyContactEncoder)
    assert reconstructed.encoders["contact_set"].use_other_body_embeddings is use_other_body_embeddings
    torch.testing.assert_close(actual, expected)


def test_global_contact_checkpoint_without_encoder_type_still_reconstructs(tmp_path):
    torch.manual_seed(0)
    input_sample = _model_input_sample()
    input_cfg = {
        "low_dim": ["states_embedding"],
        "contact_set": {
            "dim": CONTACT_TOKEN_DIM,
            "hidden_size": 16,
            "encoder_heads": 4,
        },
    }
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
    checkpoint_path = tmp_path / "global_contact.pt"

    save_checkpoint(checkpoint_path, model, "Anymal-C", cfg)
    checkpoint = load_checkpoint(checkpoint_path)
    reconstructed = reconstruct_model_from_checkpoint(checkpoint, _solver_stub(input_sample), device="cpu")
    actual = reconstructed({key: value.clone() for key, value in input_sample.items()})

    assert isinstance(reconstructed.encoders["contact_set"], ContactSetEncoderBlock)
    torch.testing.assert_close(actual, expected)


def test_checkpoint_reconstruction_rejects_primary_body_count_mismatch():
    input_sample = _model_input_sample()
    cfg = {
        "inputs": {
            "low_dim": ["states_embedding"],
            "contact_set": {
                "dim": CONTACT_TOKEN_DIM,
                "encoder_type": "shared_per_body",
                "num_bodies": 2,
            },
        },
        "network": _mixed_input_network_cfg(),
    }
    checkpoint = {
        "version": 2,
        "cfg": cfg,
        "model_state_dict": {},
    }

    with pytest.raises(ValueError, match="does not match runtime primary-body metadata"):
        reconstruct_model_from_checkpoint(checkpoint, _solver_stub(input_sample, num_bodies=3), device="cpu")


def test_shared_per_body_model_forwards_contact_token_trajectory_dataset(tmp_path):
    import h5py
    import numpy as np
    from isaaclab_neural.data.datasets import TrajectoryDataset

    dataset_path = tmp_path / "contact_tokens.hdf5"
    tokens = np.zeros((2, 4, 5, CONTACT_TOKEN_DIM), dtype=np.float32)
    tokens[0, 0, 0] = _token(body_slot=2).numpy()
    token_body_ids = np.full((2, 4, 5), -1, dtype=np.int64)
    token_body_ids[0, 0, 0] = 2
    token_world_ids = np.zeros((2, 4, 5), dtype=np.int64)
    with h5py.File(dataset_path, "w") as handle:
        group = handle.create_group("data")
        group.attrs["mode"] = "trajectory"
        group.attrs["contact_token_frame"] = "world_v1"
        group.attrs["contact_identity_schema"] = "world_owner_v1"
        group.create_dataset("states", data=np.zeros((2, 4, 4), dtype=np.float32))
        group.create_dataset("next_states", data=np.zeros((2, 4, 4), dtype=np.float32))
        group.create_dataset("joint_f", data=np.zeros((2, 4, 2), dtype=np.float32))
        group.create_dataset("gravity_dir", data=np.zeros((2, 4, 3), dtype=np.float32))
        group.create_dataset("contact_tokens", data=tokens)
        group.create_dataset("contact_token_body_ids", data=token_body_ids)
        group.create_dataset("contact_token_world_ids", data=token_world_ids)
        group.create_dataset("contact_token_overflow", data=np.zeros((2, 4), dtype=np.int64))

    dataset = TrajectoryDataset(dataset_path, sample_sequence_length=3)
    sample = {key: value.unsqueeze(0) for key, value in dataset[0].items()}
    sample["states_embedding"] = sample["states"]
    input_cfg = {
        "low_dim": ["states_embedding", "joint_f", "gravity_dir"],
        "contact_set": {
            "dim": CONTACT_TOKEN_DIM,
            "encoder_type": "shared_per_body",
            "num_bodies": 3,
            "body_latent_dim": 8,
            "hidden_dim": 16,
        },
    }
    model = ModelMixedInput(
        input_sample=sample,
        output_dim=4,
        input_cfg=input_cfg,
        network_cfg=_mixed_input_network_cfg(),
        device="cpu",
        contact_mode="newton_native",
    )

    output = model(sample)

    assert sample["contact_tokens"].shape == (1, 3, 5, CONTACT_TOKEN_DIM)
    assert output.shape == (1, 3, 4)
    assert torch.isfinite(output).all()


def test_shared_per_body_encoder_supports_step_and_history_inputs():
    encoder = SharedPerBodyContactEncoder(num_bodies=3, body_latent_dim=8, hidden_dim=16)

    step_output = encoder(torch.zeros(2, 4, CONTACT_TOKEN_DIM))
    history_output = encoder(torch.zeros(2, 5, 4, CONTACT_TOKEN_DIM))

    assert step_output.shape == (2, 3, 8)
    assert history_output.shape == (2, 5, 3, 8)
    torch.testing.assert_close(step_output, torch.zeros_like(step_output))
    torch.testing.assert_close(history_output, torch.zeros_like(history_output))


def test_shared_per_body_encoder_groups_by_primary_body_and_zeros_empty_bodies():
    torch.manual_seed(0)
    encoder = SharedPerBodyContactEncoder(num_bodies=3, body_latent_dim=8, hidden_dim=16)
    tokens = torch.zeros(1, 4, CONTACT_TOKEN_DIM)
    tokens[0, 0] = _token(body_slot=0, point_x=0.1)
    tokens[0, 1] = _token(body_slot=2, point_x=0.2)

    output = encoder(tokens)

    assert torch.count_nonzero(output[0, 0]) > 0
    torch.testing.assert_close(output[0, 1], torch.zeros(8))
    assert torch.count_nonzero(output[0, 2]) > 0


def test_shared_per_body_encoder_is_permutation_invariant_within_each_body():
    torch.manual_seed(0)
    encoder = SharedPerBodyContactEncoder(num_bodies=2, body_latent_dim=8, hidden_dim=16)
    tokens = torch.zeros(1, 4, CONTACT_TOKEN_DIM)
    tokens[0, 0] = _token(body_slot=0, point_x=0.1)
    tokens[0, 1] = _token(body_slot=1, point_x=0.2)
    tokens[0, 2] = _token(body_slot=0, point_x=0.3)
    shuffled = tokens[:, torch.tensor([2, 0, 3, 1])]

    torch.testing.assert_close(encoder(tokens), encoder(shuffled))


def test_shared_per_body_encoder_ignores_padding_and_invalid_body_slots():
    torch.manual_seed(0)
    encoder = SharedPerBodyContactEncoder(num_bodies=2, body_latent_dim=8, hidden_dim=16)
    tokens = torch.zeros(1, 4, CONTACT_TOKEN_DIM)
    tokens[0, 0] = _token(body_slot=0)
    tokens[0, 1, 1:] = torch.randn(CONTACT_TOKEN_DIM - 1)
    tokens[0, 2] = _token(body_slot=-1)
    tokens[0, 3] = _token(body_slot=2)

    reference = torch.zeros_like(tokens)
    reference[0, 0] = tokens[0, 0]

    torch.testing.assert_close(encoder(tokens), encoder(reference))


def test_shared_per_body_encoder_shares_primary_body_parameters():
    torch.manual_seed(0)
    encoder = SharedPerBodyContactEncoder(num_bodies=2, body_latent_dim=8, hidden_dim=16)
    body_zero_tokens = torch.zeros(1, 1, CONTACT_TOKEN_DIM)
    body_zero_tokens[0, 0] = _token(body_slot=0)
    body_one_tokens = body_zero_tokens.clone()
    body_one_tokens[0, 0, 1] = 1.0

    body_zero_output = encoder(body_zero_tokens)[0, 0]
    body_one_output = encoder(body_one_tokens)[0, 1]

    torch.testing.assert_close(body_zero_output, body_one_output)


def test_shared_per_body_encoder_preserves_contact_count_information():
    torch.manual_seed(0)
    encoder = SharedPerBodyContactEncoder(num_bodies=1, body_latent_dim=8, hidden_dim=16)
    one_contact = torch.zeros(1, 2, CONTACT_TOKEN_DIM)
    one_contact[0, 0] = _token(body_slot=0)
    two_contacts = one_contact.clone()
    two_contacts[0, 1] = one_contact[0, 0]

    assert not torch.allclose(encoder(one_contact), encoder(two_contacts))


def test_shared_per_body_encoder_defaults_to_other_body_embeddings():
    torch.manual_seed(0)
    default = SharedPerBodyContactEncoder(num_bodies=2, body_latent_dim=8, hidden_dim=16)
    torch.manual_seed(0)
    explicit = SharedPerBodyContactEncoder(
        num_bodies=2,
        body_latent_dim=8,
        hidden_dim=16,
        use_other_body_embeddings=True,
    )

    assert default.use_other_body_embeddings is True
    assert default.state_dict().keys() == explicit.state_dict().keys()
    for name, value in default.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[name])


def test_shared_per_body_encoder_can_disable_other_body_embeddings():
    torch.manual_seed(0)
    encoder = SharedPerBodyContactEncoder(
        num_bodies=2,
        body_latent_dim=8,
        hidden_dim=16,
        use_other_body_embeddings=False,
    )
    tokens = torch.zeros(1, 3, CONTACT_TOKEN_DIM)
    tokens[0, 0] = _token(body_slot=0, point_x=0.1, other_body_slot=1)
    tokens[0, 1] = _token(body_slot=0, point_x=0.1, other_body_slot=-1)
    tokens[0, 1, 3] = 1.0
    tokens[0, 2] = _token(body_slot=0, point_x=0.2, other_body_slot=-1)

    encoded = encoder._encode_contacts(tokens)
    expected = encoder.contact_encoder(tokens[..., CONTACT_TOKEN_GEOMETRY_SLICE])

    torch.testing.assert_close(encoded, expected)
    torch.testing.assert_close(encoded[:, 0], encoded[:, 1])

    encoder(tokens).sum().backward()
    assert encoder.other_body_embed.weight.grad is not None
    assert encoder.other_static_embed.grad is not None
    assert encoder.other_foreign_embed.grad is not None
    torch.testing.assert_close(
        encoder.other_body_embed.weight.grad,
        torch.zeros_like(encoder.other_body_embed.weight.grad),
    )
    torch.testing.assert_close(encoder.other_static_embed.grad, torch.zeros_like(encoder.other_static_embed.grad))
    torch.testing.assert_close(encoder.other_foreign_embed.grad, torch.zeros_like(encoder.other_foreign_embed.grad))


def test_shared_per_body_encoder_has_finite_forward_and_backward():
    torch.manual_seed(0)
    encoder = SharedPerBodyContactEncoder(num_bodies=2, body_latent_dim=8, hidden_dim=16)
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
