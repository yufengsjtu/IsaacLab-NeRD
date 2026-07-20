# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for canonical Newton-native contact features."""

from types import SimpleNamespace

import torch
from isaaclab_neural.contacts.newton_contact_adapter import NewtonContactAdapter
from isaaclab_neural.contacts.packing import get_contact_order, resolve_contact_packing_policy


def _adapter() -> NewtonContactAdapter:
    adapter = NewtonContactAdapter.__new__(NewtonContactAdapter)
    adapter.shape_body = [-1, 3, 7]
    adapter.shape_body_torch = torch.tensor(adapter.shape_body)
    adapter.primary_body_mask = torch.zeros(8, dtype=torch.bool)
    adapter.primary_body_mask[[3, 7]] = True
    return adapter


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

    actual_shape0, actual_shape1, actual_point0, actual_point1, actual_normal, actual_t0, actual_t1 = actual
    torch.testing.assert_close(actual_shape0, torch.tensor([1, 2, 1]))
    torch.testing.assert_close(actual_shape1, torch.tensor([0, 0, 2]))
    torch.testing.assert_close(actual_point0[:, 0], torch.tensor([1.0, 2.0, 5.0]))
    torch.testing.assert_close(actual_point1[:, 0], torch.tensor([0.0, 3.0, 4.0]))
    torch.testing.assert_close(actual_normal[:, 0], torch.tensor([-1.0, 1.0, -1.0]))
    torch.testing.assert_close(actual_t0, torch.tensor([0.4, 0.2, 0.6]))
    torch.testing.assert_close(actual_t1, torch.tensor([0.1, 0.5, 0.3]))


def test_shape_body_ids_accepts_int32_shape_indices():
    adapter = _adapter()
    shapes = torch.tensor([0, 2, -1, 99], dtype=torch.int32)

    body_ids = adapter._shape_body_ids(shapes)

    assert body_ids.dtype == torch.long
    torch.testing.assert_close(body_ids, torch.tensor([-1, 7, -1, -1]))


def test_surface_separation_subtracts_both_thicknesses():
    adapter = _adapter()
    adapter._points_to_world = lambda _shapes, points, _state: points
    point0 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    point1 = torch.tensor([[0.15, 0.0, 0.0], [1.5, 0.0, 0.0]])
    normal = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    separation, _, _ = adapter._read_separation_and_world_points(
        torch.tensor([1, 1]),
        torch.tensor([0, 0]),
        point0,
        point1,
        normal,
        torch.tensor([0.1, 0.2]),
        torch.tensor([0.1, 0.1]),
        object(),
    )

    torch.testing.assert_close(separation, torch.tensor([-0.05, 0.2]))


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

    shape0, shape1, point0, point1, normal, thickness0, thickness1 = actual
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
    adapter.device = torch.device("cpu")
    adapter.packing_policy = "penetration_priority"
    adapter.body_world = [0] * 8
    adapter.contact_masks = torch.zeros(1, 1, dtype=torch.bool)
    adapter.contact_normals = torch.zeros(1, 1, 3)
    adapter.contact_depths = torch.zeros(1, 1)
    adapter.contact_thicknesses_0 = torch.zeros(1, 1)
    adapter.contact_thicknesses_1 = torch.zeros(1, 1)
    adapter.contact_points_0 = torch.zeros(1, 1, 3)
    adapter.contact_points_1 = torch.zeros(1, 1, 3)
    adapter._contact_frames = 0
    adapter._raw_contacts_total = 0
    adapter._packed_contacts_total = 0
    adapter._dropped_contacts_total = 0
    adapter._truncated_frames = 0
    adapter._truncation_warning_emitted = False
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
