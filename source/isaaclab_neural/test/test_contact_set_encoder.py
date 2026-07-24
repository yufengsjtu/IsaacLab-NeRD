# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for contact token set encoding."""

from types import SimpleNamespace

import pytest
import torch
from isaaclab_neural.contacts.contact_set_encoder import ContactSetEncoder, transform_contact_tokens_to_body_frame
from isaaclab_neural.contacts.contact_set_schema import CONTACT_TOKEN_DIM
from isaaclab_neural.contacts.tensor_utils import ContactTokenMoments
from isaaclab_neural.models.contact_set_model import ContactSetEncoderBlock


def _token_encoder() -> ContactSetEncoder:
    device = torch.device("cpu")
    model = SimpleNamespace(
        body_count=4,
        body_world=torch.tensor([0, 0, 0, 0]),
        body_com=torch.zeros(4, 3),
    )
    primary_body_mask = torch.tensor([True, True, False, False])
    shape_body_torch = torch.tensor([-1, 0, 1, -1])
    return ContactSetEncoder(
        model=model,
        primary_body_mask=primary_body_mask,
        shape_body_torch=shape_body_torch,
        bodies_per_env=2,
        num_envs=1,
        max_contact_tokens=4,
        device=device,
    )


def test_contact_set_encoder_respects_capacity_and_validity():
    encoder = _token_encoder()
    state = SimpleNamespace(
        body_q=torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ),
        body_qd=torch.zeros(2, 6),
    )
    raw = {
        "shape0": torch.tensor([1, 1]),
        "shape1": torch.tensor([0, 0]),
        "point0_world": torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        "point1_world": torch.tensor([[0.05, 0.0, 0.0], [0.15, 0.0, 0.0]]),
        "normal": torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        "surface_separation": torch.tensor([-0.01, -0.02]),
    }

    tokens = encoder.encode(raw, state)

    assert tokens.shape == (1, 4, CONTACT_TOKEN_DIM)
    assert tokens[0, 0, 0] == 1.0
    assert (tokens[0, :, 0] >= 0.0).all()
    assert (tokens[0, :, 0].sum() <= 4)


def test_contact_token_moments_pin_identity_channels():
    tokens = torch.zeros(2, 3, CONTACT_TOKEN_DIM)
    tokens[0, 0, 0] = 1.0
    tokens[0, 0, 4:7] = 1.0
    tokens[0, 1, 0] = 1.0
    tokens[0, 1, 4:7] = -1.0
    moments = ContactTokenMoments(CONTACT_TOKEN_DIM, "cpu")
    moments.update(tokens)
    rms = moments.finalize("cpu")
    assert rms.mean[0].item() == 0.0
    assert rms.var[0].item() == 1.0
    assert rms.mean[4].abs().item() < 1.0e-5


def test_contact_set_encoder_block_is_permutation_invariant():
    block = ContactSetEncoderBlock(hidden_size=32, num_heads=4, device="cpu")
    tokens = torch.zeros(1, 1, 4, CONTACT_TOKEN_DIM)
    tokens[0, 0, 0, 0] = 1.0
    tokens[0, 0, 0, 4:7] = torch.tensor([0.1, 0.2, 0.3])
    tokens[0, 0, 1, 0] = 1.0
    tokens[0, 0, 1, 4:7] = torch.tensor([0.4, 0.5, 0.6])
    shuffled = tokens.clone()
    shuffled[0, 0, 0], shuffled[0, 0, 1] = shuffled[0, 0, 1].clone(), shuffled[0, 0, 0].clone()

    out_a = block(tokens)
    out_b = block(shuffled)
    torch.testing.assert_close(out_a, out_b, atol=1.0e-5, rtol=1.0e-5)


def test_contact_set_encoder_block_uses_eight_latent_queries_by_default():
    block = ContactSetEncoderBlock(hidden_size=32, num_heads=4, device="cpu")

    assert block.num_latent_queries == 8
    assert block.latent_queries.shape == (8, 32)


def test_contact_set_encoder_block_handles_empty_contact_sets():
    block = ContactSetEncoderBlock(hidden_size=32, num_heads=4, device="cpu")
    padded_tokens = torch.zeros(2, 3, 4, CONTACT_TOKEN_DIM)
    padded_tokens[..., 1:] = torch.randn_like(padded_tokens[..., 1:])
    zero_capacity_tokens = torch.zeros(2, 3, 0, CONTACT_TOKEN_DIM)

    padded_output = block(padded_tokens)
    zero_capacity_output = block(zero_capacity_tokens)

    assert padded_output.shape == (2, 3, 32)
    assert torch.isfinite(padded_output).all()
    torch.testing.assert_close(padded_output, zero_capacity_output, atol=1.0e-5, rtol=1.0e-5)


def test_contact_set_encoder_block_mixed_empty_sets_have_finite_gradients():
    block = ContactSetEncoderBlock(hidden_size=32, num_heads=4, device="cpu")
    tokens = torch.zeros(2, 1, 4, CONTACT_TOKEN_DIM)
    tokens[1, 0, 0, 0] = 1.0
    tokens[1, 0, 0, 4:7] = torch.tensor([0.1, 0.2, 0.3])
    tokens.requires_grad_()

    output = block(tokens)
    output.square().mean().backward()

    assert torch.isfinite(output).all()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in block.parameters())


def test_contact_set_encoder_block_rejects_nonpositive_latent_query_count():
    with pytest.raises(ValueError, match="num_latent_queries"):
        ContactSetEncoderBlock(hidden_size=32, num_heads=4, num_latent_queries=0, device="cpu")


def test_transform_contact_tokens_to_body_frame_zeros_padding():
    tokens = torch.zeros(1, 2, CONTACT_TOKEN_DIM)
    tokens[0, 0, 0] = 1.0
    tokens[0, 0, 4:7] = torch.tensor([1.0, 0.0, 0.0])
    root = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    transformed = transform_contact_tokens_to_body_frame(tokens, root)
    assert transformed[0, 1, 0].item() == 0.0
    torch.testing.assert_close(transformed[0, 0, 4:7], torch.tensor([1.0, 0.0, 0.0]))


def test_pack_rows_overflow_does_not_contaminate_next_env():
    """Dropped tokens must write to a dump row, not the next env's first slot."""
    device = torch.device("cpu")
    # Two envs, two primary bodies each: env0 bodies 0/1, env1 bodies 2/3.
    model = SimpleNamespace(
        body_count=4,
        body_world=torch.tensor([0, 0, 1, 1]),
        body_com=torch.zeros(4, 3),
    )
    encoder = ContactSetEncoder(
        model=model,
        primary_body_mask=torch.tensor([True, True, True, True]),
        shape_body_torch=torch.tensor([-1, 0, 1, 2, 3]),
        bodies_per_env=2,
        num_envs=2,
        max_contact_tokens=1,
        device=device,
    )
    state = SimpleNamespace(
        body_q=torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ),
        body_qd=torch.zeros(4, 6),
    )
    # Two terrain contacts in env0 (shapes 1/2 -> bodies 0/1). Capacity is 1,
    # so one directed token is kept and one is dropped. Env1 has no contacts.
    raw = {
        "shape0": torch.tensor([1, 2]),
        "shape1": torch.tensor([0, 0]),
        "point0_world": torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        "point1_world": torch.tensor([[0.0, 0.0, -0.01], [0.1, 0.0, -0.01]]),
        "normal": torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        "surface_separation": torch.tensor([-0.02, -0.01]),
    }

    tokens = encoder.encode(raw, state)

    assert tokens.shape == (2, 1, CONTACT_TOKEN_DIM)
    assert int(tokens[0, 0, 0].item()) == 1
    assert int(tokens[1, 0, 0].item()) == 0
    assert int(encoder.last_overflow[0].item()) == 1
    assert int(encoder.last_overflow[1].item()) == 0
    # Gap of the kept token should be the more penetrating contact (-0.02),
    # and env1 must remain empty after the overflow write.
    torch.testing.assert_close(tokens[0, 0, 13], torch.tensor(-0.02))
    torch.testing.assert_close(tokens[1, 0], torch.zeros(CONTACT_TOKEN_DIM))


def test_body_slot_is_local_per_environment():
    device = torch.device("cpu")
    model = SimpleNamespace(
        body_count=4,
        body_world=torch.tensor([0, 0, 1, 1]),
        body_com=torch.tensor([[0.0, 0.0, 0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.0]]),
    )
    encoder = ContactSetEncoder(
        model=model,
        primary_body_mask=torch.tensor([True, True, True, True]),
        shape_body_torch=torch.tensor([-1, 0, 2]),
        bodies_per_env=2,
        num_envs=2,
        max_contact_tokens=2,
        device=device,
    )
    state = SimpleNamespace(
        body_q=torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ),
        body_qd=torch.zeros(4, 6),
    )
    raw = {
        "shape0": torch.tensor([1, 2]),
        "shape1": torch.tensor([0, 0]),
        "point0_world": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "point1_world": torch.tensor([[0.0, 0.0, -0.01], [1.0, 0.0, -0.01]]),
        "surface0_world": torch.tensor([[0.0, 0.0, -0.1], [1.0, 0.0, -0.1]]),
        "surface1_world": torch.tensor([[0.0, 0.0, -0.01], [1.0, 0.0, -0.01]]),
        "normal": torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        "surface_separation": torch.tensor([-0.02, -0.02]),
    }

    tokens = encoder.encode(raw, state)

    # Both envs' first primary body should map to local slot 0, not global 0/2.
    assert int(tokens[0, 0, 1].item()) == 0
    assert int(tokens[1, 0, 1].item()) == 0
    # Lever arm uses surface - COM = (-0.1) - (0.1) = -0.2 on z for body0/body2.
    torch.testing.assert_close(tokens[0, 0, 12], torch.tensor(-0.2))
    torch.testing.assert_close(tokens[1, 0, 12], torch.tensor(-0.2))


def test_contact_env_ids_accepts_int32_body_world():
    """Newton body_world is int32; env_ids buffers are long and must match on assign."""
    device = torch.device("cpu")
    model = SimpleNamespace(
        body_count=4,
        body_world=torch.tensor([0, 0, 1, 1], dtype=torch.int32),
        body_com=torch.zeros(4, 3),
    )
    encoder = ContactSetEncoder(
        model=model,
        primary_body_mask=torch.tensor([True, True, True, True]),
        shape_body_torch=torch.tensor([-1, 0, 1, 2, 3]),
        bodies_per_env=2,
        num_envs=2,
        max_contact_tokens=4,
        device=device,
    )
    body_ids = torch.tensor([0, 2, -1], dtype=torch.long)
    shapes = torch.tensor([1, 3, 0], dtype=torch.long)

    env_ids = encoder._contact_env_ids(body_ids, shapes)

    torch.testing.assert_close(env_ids, torch.tensor([0, 1, 0], dtype=torch.long))


def test_eager_trajectory_dataset_preserves_contact_token_rank(tmp_path):
    import h5py
    import numpy as np
    from isaaclab_neural.data.datasets import TrajectoryDataset

    path = tmp_path / "tokens.hdf5"
    tokens = np.zeros((2, 4, 3, CONTACT_TOKEN_DIM), dtype=np.float32)
    tokens[..., 0] = 1.0
    with h5py.File(path, "w") as handle:
        group = handle.create_group("data")
        group.attrs["mode"] = "trajectory"
        group.create_dataset("states", data=np.zeros((2, 4, 8), dtype=np.float32))
        group.create_dataset("next_states", data=np.zeros((2, 4, 8), dtype=np.float32))
        group.create_dataset("joint_f", data=np.zeros((2, 4, 3), dtype=np.float32))
        group.create_dataset("contact_tokens", data=tokens)
        group.create_dataset("root_body_q", data=np.zeros((2, 4, 7), dtype=np.float32))
        group.create_dataset("gravity_dir", data=np.zeros((2, 4, 3), dtype=np.float32))

    dataset = TrajectoryDataset(path, sample_sequence_length=2, max_capacity=100)
    sample = dataset[0]
    assert sample["contact_tokens"].shape == (2, 3, CONTACT_TOKEN_DIM)
