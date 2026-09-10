# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for canonical Newton-native contact features."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_neural.contacts.native_pair_projector import project_native_pairs
from isaaclab_neural.contacts.newton_contact_adapter import NewtonContactAdapter
from isaaclab_neural.contacts.packing import get_contact_order, resolve_contact_packing_policy
from isaaclab_neural.generate.adapter import DataGenerationAdapter
from isaaclab_neural.solvers.neural_solver import _resolve_root_body_ids


def _adapter() -> NewtonContactAdapter:
    adapter = NewtonContactAdapter.__new__(NewtonContactAdapter)
    adapter.shape_body = [-1, 3, 7]
    adapter.shape_body_torch = torch.tensor(adapter.shape_body)
    adapter.primary_body_mask = torch.zeros(8, dtype=torch.bool)
    adapter.primary_body_mask[[3, 7]] = True
    return adapter


def test_resolve_root_body_ids_uses_articulation_joint_children():
    model = SimpleNamespace(
        body_count=6,
        joint_child=torch.tensor([5, 4, 2, 1]),
        body_world_start=torch.tensor([0, 3]),
    )

    root_body_ids = _resolve_root_body_ids(model, np.asarray([0, 2, 4]), num_envs=2)

    np.testing.assert_array_equal(root_body_ids, np.asarray([5, 2]))


def test_contact_token_world_ids_are_derived_from_owner_bodies():
    adapter = DataGenerationAdapter.__new__(DataGenerationAdapter)
    adapter.device = torch.device("cpu")
    adapter.model = SimpleNamespace(body_world=torch.tensor([1, 0, 1, 0]))
    adapter.contact_adapter = SimpleNamespace(contact_token_body_ids=torch.tensor([[1, -1], [0, 2]], dtype=torch.long))

    world_ids = adapter.contact_token_world_ids

    assert world_ids is not None
    torch.testing.assert_close(world_ids, torch.tensor([[0, -1], [1, 1]], dtype=torch.long))


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


def test_surface_offset_does_not_change_solver_clearance():
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
    torch.testing.assert_close(separation, torch.tensor([0.0]))


def test_warp_geometry_matches_torch_reference():
    adapter = NewtonContactAdapter.__new__(NewtonContactAdapter)
    adapter.device = torch.device("cpu")
    adapter.model = SimpleNamespace(device=wp.get_device("cpu"))
    adapter.shape_body_torch = torch.tensor([-1, 0], dtype=torch.long)
    adapter._torch_view_cache = {}
    adapter._geometry_capacity = 0
    adapter._geometry_buffers = {}
    body_q = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]])
    state = SimpleNamespace(body_q=wp.from_torch(body_q, dtype=wp.transform))
    shape0 = torch.tensor([1], dtype=torch.int32)
    shape1 = torch.tensor([0], dtype=torch.int32)
    point0 = torch.tensor([[0.1, 0.2, 0.3]])
    point1 = torch.tensor([[1.1, 2.2, 3.1]])
    normal = torch.tensor([[0.0, 0.0, 1.0]])
    thickness0 = torch.tensor([0.05])
    thickness1 = torch.tensor([0.02])
    offset0 = torch.tensor([[0.0, 0.0, -0.1]])
    offset1 = torch.zeros(1, 3)

    adapter.use_warp_geometry = False
    expected = tuple(
        value.clone()
        for value in adapter._read_separation_and_world_points(
            shape0,
            shape1,
            point0,
            point1,
            normal,
            thickness0,
            thickness1,
            offset0,
            offset1,
            state,
        )
    )
    adapter.use_warp_geometry = True
    actual = adapter._read_separation_and_world_points(
        shape0,
        shape1,
        point0,
        point1,
        normal,
        thickness0,
        thickness1,
        offset0,
        offset1,
        state,
    )
    wp.synchronize_device(adapter.model.device)

    for actual_value, expected_value in zip(actual, expected):
        torch.testing.assert_close(actual_value, expected_value, atol=1.0e-6, rtol=1.0e-6)


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


def test_primary_body_mask_uses_all_articulation_bodies_per_world():
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
        torch.tensor([True, True, True, True, True, True]),
    )
    assert rule == "all_articulation_bodies"


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
    adapter.contact_filter = "none"
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
    adapter._filtered_contacts_gpu = torch.zeros((), dtype=torch.long)
    adapter._dropped_contacts_gpu = torch.zeros((), dtype=torch.long)
    adapter._truncated_frames_gpu = torch.zeros((), dtype=torch.long)
    raw = {
        "surface_separation": torch.tensor([-0.2, -0.1]),
        "shape0": torch.tensor([1, 1]),
        "shape1": torch.tensor([0, 0]),
        "point0_world": torch.zeros(2, 3),
        "point1_world": torch.tensor([[0.0, 0.0, 0.2], [0.0, 0.0, 0.1]]),
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


def test_shared_native_pair_projection_matches_flat_reference():
    native_pairs = {
        # Frame 0 has three pairs (one is dropped); frame 1 has two.
        "world_ids": torch.tensor([0, 0, 0, 1, 1]),
        "separation": torch.tensor([-0.3, -0.2, -0.1, -0.25, -0.05]),
        "shape_ids_0": torch.tensor([1, 1, 1, 2, 2]),
        "shape_ids_1": torch.zeros(5, dtype=torch.long),
        "points_0_w": torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ]
        ),
        "points_1_w": torch.zeros(5, 3),
        "normal_w": torch.tensor([[0.0, 0.0, -1.0]]).expand(5, 3).clone(),
        "thickness_0": torch.tensor([0.01, 0.02, 0.03, 0.04, 0.05]),
        "thickness_1": torch.tensor([0.11, 0.12, 0.13, 0.14, 0.15]),
    }
    native_pairs["points_1_w"][:, 2] = -(
        native_pairs["separation"] + native_pairs["thickness_0"] + native_pairs["thickness_1"]
    )

    actual = project_native_pairs(
        native_pairs,
        num_frames=2,
        representation="flat",
        packing_policy="penetration_priority",
        flat_capacity=2,
        return_pair_indices=True,
    )

    # Penetration priority retains source pairs [0, 1] for frame 0 and [3, 4] for frame 1.
    selected = torch.tensor([[0, 1], [3, 4]])
    torch.testing.assert_close(actual["_pair_indices"], selected)
    torch.testing.assert_close(actual["contact_masks"], torch.ones(2, 2, dtype=torch.bool))
    for output_name, source_name in (
        ("contact_depths", "separation"),
        ("contact_thicknesses_0", "thickness_0"),
        ("contact_thicknesses_1", "thickness_1"),
    ):
        torch.testing.assert_close(actual[output_name], native_pairs[source_name][selected])
    for output_name, source_name in (
        ("contact_normals", "normal_w"),
        ("contact_points_0", "points_0_w"),
        ("contact_points_1", "points_1_w"),
    ):
        expected = native_pairs[source_name][selected].reshape(2, -1)
        torch.testing.assert_close(actual[output_name], expected)
