# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for canonical Newton-native contact features."""

from types import SimpleNamespace

import pytest
import torch
from isaaclab_neural.contacts.contact_set_schema import (
    CONTACT_FILTER_SOLVER_ACTIVE,
    CONTACT_REPRESENTATION_ACTIVE15,
    CONTACT_REPRESENTATION_ACTIVE15_SELF,
    CONTACT_REPRESENTATION_TOKENS,
    CONTACT_TOKEN_SELF_COLLISION_FIELD,
    CONTACT_TOKEN_SOLVER_ACTIVE_FIELD,
)
from isaaclab_neural.contacts.native15_self_contact_encoder import Raw15SelfContactEncoder
from isaaclab_neural.contacts.newton_contact_adapter import NewtonContactAdapter
from isaaclab_neural.contacts.packing import get_contact_order, resolve_contact_packing_policy
from isaaclab_neural.contacts.raw15_contact_encoder import Raw15ContactEncoder


def _adapter() -> NewtonContactAdapter:
    adapter = NewtonContactAdapter.__new__(NewtonContactAdapter)
    adapter.shape_body = [-1, 3, 7]
    adapter.shape_body_torch = torch.tensor(adapter.shape_body)
    adapter.primary_body_mask = torch.zeros(8, dtype=torch.bool)
    adapter.primary_body_mask[[3, 7]] = True
    return adapter


def _token_model() -> SimpleNamespace:
    return SimpleNamespace(
        world_count=1,
        body_count=2,
        device="cpu",
        shape_body=torch.tensor([-1, 0, 1]),
        shape_margin=torch.tensor([0.01, 0.01, 0.01]),
        body_world=torch.tensor([0, 0]),
        body_com=torch.zeros(2, 3),
        joint_articulation=None,
        joint_parent=None,
        joint_child=None,
        joint_world=None,
    )


def test_active15_solver_filter_packs_raw15_before_filtering() -> None:
    adapter = NewtonContactAdapter(
        _token_model(),
        num_contacts_per_env=4,
        device="cpu",
        contact_representation=CONTACT_REPRESENTATION_ACTIVE15,
        contact_filter=CONTACT_FILTER_SOLVER_ACTIVE,
        max_contact_tokens=4,
    )

    assert isinstance(adapter._token_encoder, Raw15ContactEncoder)
    assert CONTACT_TOKEN_SOLVER_ACTIVE_FIELD not in adapter.to_neural_inputs()


def test_active15_self_solver_filter_packs_raw_self_and_exposes_provenance() -> None:
    adapter = NewtonContactAdapter(
        _token_model(),
        num_contacts_per_env=4,
        device="cpu",
        contact_representation=CONTACT_REPRESENTATION_ACTIVE15_SELF,
        contact_filter=CONTACT_FILTER_SOLVER_ACTIVE,
        max_contact_tokens=4,
    )

    assert isinstance(adapter._token_encoder, Raw15SelfContactEncoder)
    inputs = adapter.to_neural_inputs()
    assert inputs[CONTACT_TOKEN_SELF_COLLISION_FIELD].shape == (1, 4)
    assert inputs[CONTACT_TOKEN_SELF_COLLISION_FIELD].dtype == torch.bool
    assert CONTACT_TOKEN_SOLVER_ACTIVE_FIELD not in inputs


def test_generic_contact_tokens_expose_aligned_solver_active_sidecar() -> None:
    adapter = NewtonContactAdapter(
        _token_model(),
        num_contacts_per_env=4,
        device="cpu",
        contact_representation=CONTACT_REPRESENTATION_TOKENS,
        max_contact_tokens=4,
    )

    inputs = adapter.to_neural_inputs()

    assert inputs[CONTACT_TOKEN_SOLVER_ACTIVE_FIELD].shape == (1, 4)
    assert inputs[CONTACT_TOKEN_SOLVER_ACTIVE_FIELD].dtype == torch.bool


def test_canonicalize_contact_sides_puts_dynamic_body_first():
    adapter = _adapter()
    shape0 = torch.tensor([0, 2, 2])
    shape1 = torch.tensor([1, 0, 1])
    point0 = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    point1 = torch.tensor([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    normal = torch.tensor([[1.0, 0.0, 0.0]]).expand(3, 3).clone()
    thickness0 = torch.tensor([0.1, 0.2, 0.3])
    thickness1 = torch.tensor([0.4, 0.5, 0.6])

    actual = adapter._canonicalize_contact_sides(
        shape0,
        shape1,
        point0,
        point1,
        normal,
        thickness0,
        thickness1,
    )

    (
        actual_shape0,
        actual_shape1,
        actual_point0,
        actual_point1,
        actual_normal,
        actual_t0,
        actual_t1,
        actual_offset0,
        actual_offset1,
    ) = actual
    torch.testing.assert_close(actual_shape0, torch.tensor([1, 2, 1]))
    torch.testing.assert_close(actual_shape1, torch.tensor([0, 0, 2]))
    torch.testing.assert_close(actual_point0[:, 0], torch.tensor([1.0, 2.0, 5.0]))
    torch.testing.assert_close(actual_point1[:, 0], torch.tensor([0.0, 3.0, 4.0]))
    torch.testing.assert_close(actual_normal[:, 0], torch.tensor([-1.0, 1.0, -1.0]))
    torch.testing.assert_close(actual_t0, torch.tensor([0.4, 0.2, 0.6]))
    torch.testing.assert_close(actual_t1, torch.tensor([0.1, 0.5, 0.3]))
    assert actual_offset0 is None
    assert actual_offset1 is None


def test_shape_body_ids_accepts_int32_shape_indices():
    adapter = _adapter()
    shapes = torch.tensor([0, 2, -1, 99], dtype=torch.int32)

    body_ids = adapter._shape_body_ids(shapes)

    assert body_ids.dtype == torch.long
    torch.testing.assert_close(body_ids, torch.tensor([-1, 7, -1, -1]))


def test_surface_offset_moves_effective_contact_point():
    adapter = _adapter()
    adapter._points_to_world = lambda _shapes, points, _state: points
    point0 = torch.tensor([[0.0, 0.0, 0.0]])
    point1 = torch.tensor([[0.0, 0.0, 0.0]])
    offset0 = torch.tensor([[0.0, 0.0, -0.1]])
    offset1 = torch.tensor([[0.0, 0.0, 0.0]])
    normal = torch.tensor([[0.0, 0.0, 1.0]])

    separation, _, _, surface0, surface1 = adapter._read_separation_and_world_points(
        torch.tensor([1]),
        torch.tensor([0]),
        point0,
        point1,
        normal,
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        offset0,
        offset1,
        object(),
    )

    torch.testing.assert_close(surface0, torch.tensor([[0.0, 0.0, -0.1]]))
    torch.testing.assert_close(surface1, point1)
    torch.testing.assert_close(separation, torch.tensor([0.1]))


def test_body_round_robin_pair_atomic_is_rejected_for_flat_packing():
    with pytest.raises(ValueError, match="contact_tokens"):
        get_contact_order(
            {"surface_separation": torch.tensor([-0.01])},
            packing_policy="body_round_robin_pair_atomic",
        )


def test_canonicalize_contact_sides_prefers_primary_robot_over_dynamic_object():
    adapter = _adapter()
    adapter.primary_body_mask.zero_()
    adapter.primary_body_mask[7] = True

    actual = adapter._canonicalize_contact_sides(
        torch.tensor([1]),
        torch.tensor([2]),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[2.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([0.1]),
        torch.tensor([0.2]),
    )

    shape0, shape1, point0, point1, normal, thickness0, thickness1, _, _ = actual
    torch.testing.assert_close(shape0, torch.tensor([2]))
    torch.testing.assert_close(shape1, torch.tensor([1]))
    torch.testing.assert_close(point0, torch.tensor([[2.0, 0.0, 0.0]]))
    torch.testing.assert_close(point1, torch.tensor([[1.0, 0.0, 0.0]]))
    torch.testing.assert_close(normal, torch.tensor([[-1.0, 0.0, 0.0]]))
    torch.testing.assert_close(thickness0, torch.tensor([0.2]))
    torch.testing.assert_close(thickness1, torch.tensor([0.1]))


def test_primary_body_mask_uses_first_articulation_per_world():
    adapter = NewtonContactAdapter.__new__(NewtonContactAdapter)
    adapter.device = torch.device("cpu")
    adapter.num_envs = 2
    adapter.model = SimpleNamespace(
        body_count=6,
        joint_articulation=torch.tensor([0, 0, 1, 2, 2, 3]),
        joint_parent=torch.tensor([-1, 0, -1, -1, 3, -1]),
        joint_child=torch.tensor([0, 1, 2, 3, 4, 5]),
        joint_world=torch.tensor([0, 0, 0, 1, 1, 1]),
    )

    primary_body_mask, rule = adapter._derive_primary_body_mask()

    torch.testing.assert_close(
        primary_body_mask,
        torch.tensor([True, True, False, True, True, False]),
    )
    assert rule == "primary_articulation_first"


def test_penetration_priority_sorts_signed_surface_separation():
    raw_contacts = {
        "surface_separation": torch.tensor([0.02, -0.1, -0.01]),
        "shape0": torch.tensor([2, 1, 3]),
        "shape1": torch.tensor([0, 0, 0]),
        "point0_world": torch.zeros(3, 3),
        "point1_world": torch.zeros(3, 3),
    }

    order = get_contact_order(raw_contacts, "penetration_priority")

    torch.testing.assert_close(order, torch.tensor([1, 2, 0]))


def test_native_default_packing_policy_is_penetration_priority():
    assert resolve_contact_packing_policy("newton_native", None) == "penetration_priority"
    assert resolve_contact_packing_policy("fixed_ground", None) == "stable_index"
    assert resolve_contact_packing_policy("newton_native", "random") == "random"


def test_penetration_priority_uses_deterministic_shape_tie_break():
    raw_contacts = {
        "surface_separation": torch.tensor([-0.1, -0.1, -0.1]),
        "shape0": torch.tensor([3, 1, 2]),
        "shape1": torch.tensor([0, 0, 0]),
        "point0_world": torch.zeros(3, 3),
        "point1_world": torch.zeros(3, 3),
    }

    order = get_contact_order(raw_contacts, "penetration_priority")

    torch.testing.assert_close(order, torch.tensor([1, 2, 0]))


def test_contact_adapter_reports_truncation():
    adapter = _adapter()
    adapter.num_envs = 1
    adapter.num_contacts_per_env = 1
    adapter.max_contact_tokens = 1
    adapter.contact_representation = "flat"
    adapter.device = torch.device("cpu")
    adapter.packing_policy = "penetration_priority"
    adapter.body_world = [0] * 8
    adapter.body_world_torch = torch.tensor(adapter.body_world, dtype=torch.long)
    adapter.bodies_per_env = 8
    adapter.contact_masks = torch.zeros(1, 1, dtype=torch.bool)
    adapter.contact_normals = torch.zeros(1, 1, 3)
    adapter.contact_depths = torch.zeros(1, 1)
    adapter.contact_thicknesses_0 = torch.zeros(1, 1)
    adapter.contact_thicknesses_1 = torch.zeros(1, 1)
    adapter.contact_points_0 = torch.zeros(1, 1, 3)
    adapter.contact_points_1 = torch.zeros(1, 1, 3)
    adapter.contact_tokens = torch.zeros(1, 1, 17)
    adapter.contact_token_overflow = torch.zeros(1, dtype=torch.long)
    adapter._token_encoder = None
    adapter._contact_frames = 0
    adapter._raw_contacts_total = 0
    adapter._packed_contacts_total = 0
    adapter._dropped_contacts_total = 0
    adapter._truncated_frames = 0
    adapter._truncation_warning_emitted = False
    adapter._packed_contacts_gpu = torch.zeros((), dtype=torch.long)
    adapter._dropped_contacts_gpu = torch.zeros((), dtype=torch.long)
    adapter._truncated_frames_gpu = torch.zeros((), dtype=torch.long)
    raw = {
        "surface_separation": torch.tensor([-0.2, -0.1]),
        "shape0": torch.tensor([1, 1]),
        "shape1": torch.tensor([0, 0]),
        "point0_world": torch.zeros(2, 3),
        "point1_world": torch.zeros(2, 3),
        "normal": torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        "thickness0": torch.zeros(2),
        "thickness1": torch.zeros(2),
    }
    adapter._contact_count = lambda _contacts: 2
    adapter._read_raw_contacts = lambda _contacts, _count, _state: raw

    adapter.update(None, None)

    assert adapter.contact_masks[0, 0]
    assert adapter.contact_depths[0, 0] == -0.2
    assert adapter.truncation_summary() == {
        "frames": 1,
        "raw_contacts": 2,
        "packed_contacts": 1,
        "dropped_contacts": 1,
        "truncated_frames": 1,
        "truncated_frame_ratio": 1.0,
    }


def _flat_pack_fixture() -> tuple[NewtonContactAdapter, dict[str, torch.Tensor]]:
    adapter = _adapter()
    adapter.num_envs = 2
    adapter.num_contacts_per_env = 2
    adapter.contact_representation = "flat"
    adapter.device = torch.device("cpu")
    adapter.packing_policy = "penetration_priority"
    adapter.body_world = [0, 0, 0, 0, 1, 1, 1, 1]
    adapter.body_world_torch = torch.tensor(adapter.body_world, dtype=torch.long)
    adapter.bodies_per_env = 4
    adapter.shape_body = [-1, 3, 7, 1, 5]
    adapter.shape_body_torch = torch.tensor(adapter.shape_body, dtype=torch.long)
    for name, shape, dtype in (
        ("contact_masks", (2, 2), torch.bool),
        ("contact_normals", (2, 2, 3), torch.float32),
        ("contact_depths", (2, 2), torch.float32),
        ("contact_thicknesses_0", (2, 2), torch.float32),
        ("contact_thicknesses_1", (2, 2), torch.float32),
        ("contact_points_0", (2, 2, 3), torch.float32),
        ("contact_points_1", (2, 2, 3), torch.float32),
    ):
        setattr(adapter, name, torch.zeros(shape, dtype=dtype))
    adapter.contact_tokens = torch.zeros(2, 2, 17)
    adapter.contact_token_overflow = torch.zeros(2, dtype=torch.long)
    adapter.contact_token_body_ids = torch.full((2, 2), -1, dtype=torch.long)
    adapter._token_encoder = None
    adapter._packed_contacts_gpu = torch.zeros((), dtype=torch.long)
    adapter._dropped_contacts_gpu = torch.zeros((), dtype=torch.long)
    adapter._truncated_frames_gpu = torch.zeros((), dtype=torch.long)
    adapter._contact_frames = 0
    adapter._raw_contacts_total = 0
    adapter._packed_contacts_total = 0
    adapter._dropped_contacts_total = 0
    adapter._truncated_frames = 0
    adapter._truncation_warning_emitted = True
    raw = {
        # env0 gets three contacts (one dropped); env1 gets two.
        "surface_separation": torch.tensor([-0.3, -0.2, -0.1, -0.25, -0.05]),
        "shape0": torch.tensor([1, 1, 1, 2, 2]),  # bodies 3 (env0), 7 (env1)
        "shape1": torch.tensor([0, 0, 0, 0, 0]),
        "point0_world": torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ]
        ),
        "point1_world": torch.zeros(5, 3),
        "normal": torch.tensor([[0.0, 0.0, -1.0]]).expand(5, 3).clone(),
        "thickness0": torch.zeros(5),
        "thickness1": torch.zeros(5),
    }
    return adapter, raw


def test_vectorized_flat_packing_matches_sequential():
    adapter_vec, raw = _flat_pack_fixture()
    adapter_seq, _ = _flat_pack_fixture()

    adapter_vec.reset_buffers()
    adapter_vec._pack_flat_contacts(raw)

    adapter_seq.reset_buffers()
    adapter_seq._pack_flat_contacts_sequential(raw)

    torch.testing.assert_close(adapter_vec.contact_masks, adapter_seq.contact_masks)
    torch.testing.assert_close(adapter_vec.contact_normals, adapter_seq.contact_normals)
    torch.testing.assert_close(adapter_vec.contact_depths, adapter_seq.contact_depths)
    torch.testing.assert_close(adapter_vec.contact_points_0, adapter_seq.contact_points_0)
    torch.testing.assert_close(adapter_vec.contact_points_1, adapter_seq.contact_points_1)
    assert adapter_vec.truncation_summary()["packed_contacts"] == adapter_seq.truncation_summary()["packed_contacts"]
    assert adapter_vec.truncation_summary()["dropped_contacts"] == adapter_seq.truncation_summary()["dropped_contacts"]
