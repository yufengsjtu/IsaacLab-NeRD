# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for contact token set encoding."""

from types import SimpleNamespace

import pytest
import torch
from isaaclab_neural.contacts.contact_set_encoder import (
    ContactSetEncoder,
    _DirectedContactRows,
    _segment_exclusive_sum,
    _segment_rank,
    gather_owner_body_q,
    transform_contact_tokens_to_body_frame,
)
from isaaclab_neural.contacts.contact_set_schema import (
    CONTACT_TOKEN_BODY_SLOT_INDEX,
    CONTACT_TOKEN_DIM,
    NATIVE_CONTACT_INTEGER_FIELDS,
    NATIVE_CONTACT_SCALAR_FIELDS,
    NATIVE_CONTACT_VECTOR_FIELDS,
)
from isaaclab_neural.contacts.tensor_utils import ContactTokenMoments
from isaaclab_neural.data.hdf5 import write_rollouts_to_hdf5
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
    assert tokens[0, :, 0].sum() <= 4


def test_encode_into_matches_allocating_interface():
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
        "shape0": torch.tensor([1]),
        "shape1": torch.tensor([0]),
        "point0_world": torch.tensor([[0.0, 0.0, 0.0]]),
        "point1_world": torch.tensor([[0.0, 0.0, -0.01]]),
        "normal": torch.tensor([[0.0, 0.0, 1.0]]),
        "surface_separation": torch.tensor([-0.01]),
    }
    expected = encoder.encode(raw, state)
    packed = torch.full_like(expected, float("nan"))
    overflow = torch.full((1,), -1, dtype=torch.long)
    body_ids = torch.full((1, 4), -2, dtype=torch.long)

    encoder._encode_into(raw, state, packed, overflow, body_ids)

    torch.testing.assert_close(packed, expected)
    torch.testing.assert_close(overflow, torch.zeros_like(overflow))
    assert int(body_ids[0, 0].item()) == 0
    assert (body_ids[0, 1:] == -1).all()


def test_vectorized_segment_helpers_match_reference():
    keys = torch.tensor([3, 1, 3, 2, 1, 3], dtype=torch.long)
    values = torch.tensor([2, 1, 4, 3, 2, 1], dtype=torch.long)
    ramp = torch.zeros_like(keys)

    rank = _segment_rank(keys, ramp)
    exclusive = _segment_exclusive_sum(keys, values, ramp)

    expected_rank = torch.tensor([0, 0, 1, 0, 1, 2], dtype=torch.long)
    expected_exclusive = torch.tensor([0, 0, 2, 0, 1, 6], dtype=torch.long)
    torch.testing.assert_close(rank, expected_rank)
    torch.testing.assert_close(exclusive, expected_exclusive)


def test_pair_atomic_capacity_never_keeps_only_one_side():
    encoder = _token_encoder()
    encoder.max_contact_tokens = 3
    rows = _DirectedContactRows(
        valid=torch.ones(4, dtype=torch.bool),
        world_id=torch.zeros(4, dtype=torch.long),
        body_id=torch.tensor([0, 0, 1, 1]),
        other_body_id=torch.tensor([1, 1, 0, 0]),
        other_dynamic=torch.ones(4, dtype=torch.bool),
        point_world=torch.zeros(4, 3),
        normal_world=torch.tensor([[0.0, 0.0, 1.0]]).expand(4, 3).clone(),
        lever_world=torch.zeros(4, 3),
        gap=torch.tensor([-0.2, -0.1, -0.2, -0.1]),
        velocity_world=torch.zeros(4, 3),
    )

    packed, overflow, _ = encoder._pack_rows(rows)

    assert int(packed[..., 0].sum().item()) == 2
    assert int(overflow.sum().item()) == 2
    torch.testing.assert_close(packed[0, :2, 13], torch.tensor([-0.2, -0.2]))


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


def test_gather_owner_body_q_selects_per_token_poses():
    tokens = torch.zeros(1, 1, 2, CONTACT_TOKEN_DIM)
    tokens[0, 0, 0, 0] = 1.0
    tokens[0, 0, 1, 0] = 1.0
    tokens[0, 0, 0, CONTACT_TOKEN_BODY_SLOT_INDEX] = 0.0
    tokens[0, 0, 1, CONTACT_TOKEN_BODY_SLOT_INDEX] = 1.0
    primary = torch.zeros(1, 1, 2, 7)
    primary[..., 6] = 1.0
    primary[0, 0, 0, :3] = torch.tensor([0.5, 0.0, 0.0])
    primary[0, 0, 1, :3] = torch.tensor([1.5, 0.0, 0.0])

    gathered = gather_owner_body_q(primary, tokens)

    torch.testing.assert_close(gathered[0, 0, 0, :3], torch.tensor([0.5, 0.0, 0.0]))
    torch.testing.assert_close(gathered[0, 0, 1, :3], torch.tensor([1.5, 0.0, 0.0]))


def test_transform_contact_tokens_to_owner_body_frame_uses_per_token_pose():
    tokens = torch.zeros(1, 1, 2, CONTACT_TOKEN_DIM)
    tokens[0, 0, :, 0] = 1.0
    tokens[0, 0, 0, 4:7] = torch.tensor([0.6, 0.0, 0.0])
    tokens[0, 0, 1, 4:7] = torch.tensor([1.7, 0.0, 0.0])
    tokens[0, 0, 0, CONTACT_TOKEN_BODY_SLOT_INDEX] = 0.0
    tokens[0, 0, 1, CONTACT_TOKEN_BODY_SLOT_INDEX] = 1.0
    primary = torch.zeros(1, 1, 2, 7)
    primary[..., 6] = 1.0
    primary[0, 0, 0, :3] = torch.tensor([0.5, 0.0, 0.0])
    primary[0, 0, 1, :3] = torch.tensor([1.5, 0.0, 0.0])

    transformed = transform_contact_tokens_to_body_frame(tokens, gather_owner_body_q(primary, tokens))

    torch.testing.assert_close(transformed[0, 0, 0, 4:7], torch.tensor([0.1, 0.0, 0.0]))
    torch.testing.assert_close(transformed[0, 0, 1, 4:7], torch.tensor([0.2, 0.0, 0.0]))


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
    from isaaclab_neural.data.datasets import TrajectoryDataset

    path = tmp_path / "tokens.hdf5"
    pair_count = 2 * 4 * 3
    native_contacts = {name: torch.zeros(pair_count, dtype=torch.int64) for name in NATIVE_CONTACT_INTEGER_FIELDS}
    native_contacts.update(
        {name: torch.zeros((pair_count, 3), dtype=torch.float32) for name in NATIVE_CONTACT_VECTOR_FIELDS}
    )
    native_contacts.update(
        {name: torch.zeros(pair_count, dtype=torch.float32) for name in NATIVE_CONTACT_SCALAR_FIELDS}
    )
    native_contacts["frame_counts"] = torch.full((2, 4), 3, dtype=torch.int64)
    native_contacts["body_ids_1"].fill_(-1)
    native_contacts["body_slots_1"].fill_(-1)
    native_contacts["world_ids"] = torch.arange(2, dtype=torch.int64).repeat_interleave(4 * 3)
    native_contacts["normal_w"][:, 2] = 1.0
    world_ids = torch.arange(2, dtype=torch.int64)
    write_rollouts_to_hdf5(
        path,
        {
            "states": torch.zeros((2, 4, 8), dtype=torch.float32),
            "next_states": torch.zeros((2, 4, 8), dtype=torch.float32),
            "joint_f": torch.zeros((2, 4, 3), dtype=torch.float32),
            "root_body_q": torch.zeros((2, 4, 7), dtype=torch.float32),
            "root_body_qd": torch.zeros((2, 4, 6), dtype=torch.float32),
            "gravity_dir": torch.zeros((2, 4, 3), dtype=torch.float32),
            "contacts": {},
            "native_contacts": native_contacts,
            "trajectory_context": {
                "state_world_id": world_ids,
                "root_world_id": world_ids,
                "contact_world_id": world_ids,
            },
        },
        env_name="Test-Env",
    )

    dataset = TrajectoryDataset(
        path,
        sample_sequence_length=2,
        max_capacity=100,
        contact_representation="contact_tokens",
        max_contact_tokens=3,
        contact_packing_policy="body_round_robin_pair_atomic",
    )
    sample = dataset[0]
    assert sample["contact_tokens"].shape == (2, 3, CONTACT_TOKEN_DIM)
