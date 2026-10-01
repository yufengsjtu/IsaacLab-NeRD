# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# pyright: reportPrivateUsage=false, reportAttributeAccessIssue=false, reportArgumentType=false

"""Tests for canonical Newton-native contact features.

These are intentional white-box tests built from incomplete Newton adapter stubs.
"""

from types import SimpleNamespace

import pytest
import torch
import warp as wp
from isaaclab_neural.contacts.native_pair_projector import gather_or_fill, project_native_pairs
from isaaclab_neural.contacts.newton_contact_adapter import NewtonContactAdapter
from isaaclab_neural.contacts.packing import get_contact_order, resolve_contact_packing_policy
from isaaclab_neural.generate.adapter import DataGenerationAdapter


def _adapter() -> NewtonContactAdapter:
    adapter = NewtonContactAdapter.__new__(NewtonContactAdapter)
    adapter.shape_body = [-1, 3, 7]
    adapter.shape_body_torch = torch.tensor(adapter.shape_body)
    adapter.primary_body_mask = torch.zeros(8, dtype=torch.bool)
    adapter.primary_body_mask[[3, 7]] = True
    return adapter


def test_contact_token_world_ids_are_derived_from_owner_bodies():
    adapter = DataGenerationAdapter.__new__(DataGenerationAdapter)
    adapter.device = torch.device("cpu")
    adapter.model = SimpleNamespace(body_world=torch.tensor([1, 0, 1, 0]))
    contact_adapter = NewtonContactAdapter.__new__(NewtonContactAdapter)
    contact_adapter.contact_token_body_ids = torch.tensor([[1, -1], [0, 2]], dtype=torch.long)
    adapter.contact_adapter = contact_adapter

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

    actual = adapter.canonicalize_contact_sides(
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
    assert actual_shape0 is not None
    assert actual_shape1 is not None
    assert actual_point0 is not None
    assert actual_point1 is not None
    assert actual_normal is not None
    assert actual_t0 is not None
    assert actual_t1 is not None
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

    actual = adapter.canonicalize_contact_sides(
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
        "surface_separation": torch.tensor([-0.2, -0.1, 0.0]),
        "shape0": torch.tensor([1, 1, 1]),
        "shape1": torch.tensor([0, 0, 0]),
        "point0_world": torch.zeros(3, 3),
        "point1_world": torch.tensor([[0.0, 0.0, 0.2], [0.0, 0.0, 0.1], [0.0, 0.0, 0.0]]),
        "normal": torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0]]),
        "thickness0": torch.zeros(3),
        "thickness1": torch.zeros(3),
    }
    adapter._contact_count = lambda _contacts: 3
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


def test_gather_or_fill_matches_masked_lookup_at_bounds():
    table = torch.tensor([4, -1, 7])
    indices = torch.tensor([[-2, 0, 1], [2, 3, 0]])
    expected = torch.full_like(indices, -9)
    valid = (indices >= 0) & (indices < table.numel())
    expected[valid] = table[indices[valid]]
    torch.testing.assert_close(gather_or_fill(table, indices, -9), expected)
    torch.testing.assert_close(
        gather_or_fill(torch.zeros(0, dtype=torch.long), indices, -1), torch.full_like(indices, -1)
    )
    mask = torch.tensor([True, False])
    assert gather_or_fill(mask, torch.tensor([1, 0, 5]), False).tolist() == [False, True, False]


def test_native_pair_prefilter_keeps_exactly_the_pairs_the_input_filter_admits():
    from isaaclab_neural.contacts.native_pair_projector import native_pair_filter_mask

    # Two worlds with bodies (0, 1) and (2, 3); slot 0 is fixed. Shape 0 is the static ground.
    shape_body = torch.tensor([-1, 0, 1, 2, 3])
    encoder = SimpleNamespace(
        _shape_body_ids=lambda shapes: gather_or_fill(shape_body, shapes, -1),
        _contact_env_ids=lambda bodies, shapes: torch.where(bodies >= 0, bodies // 2, torch.full_like(bodies, -1)),
        body_slot=lambda bodies: torch.where(bodies >= 0, bodies % 2, torch.full_like(bodies, -1)),
    )

    def directed_rows(raw, state):
        body0, body1 = encoder._shape_body_ids(raw["shape0"]), encoder._shape_body_ids(raw["shape1"])
        world = torch.where(body0 >= 0, body0 // 2, torch.where(body1 >= 0, body1 // 2, torch.full_like(body0, -1)))
        return SimpleNamespace(
            body_id=torch.cat((body0, body1)),
            world_id=torch.cat((world, world)),
            lever_world=torch.cat((raw["point0_world"], raw["point1_world"])),
            velocity_world=torch.cat((raw["point1_world"], -raw["point0_world"])),
        )

    encoder.directed_rows = directed_rows
    adapter = NewtonContactAdapter.__new__(NewtonContactAdapter)
    adapter.num_envs = 2
    adapter.contact_filter = "solver_active_movable"
    adapter.fixed_body_slots = (0,)
    adapter.device = torch.device("cpu")
    adapter._native_pair_encoder = encoder
    adapter.shape_body_torch = shape_body
    adapter.body_world_torch = torch.tensor([0, 0, 1, 1])
    shape0 = torch.tensor([1, 2, 1, 3, 4, 2, 0, 1])
    shape1 = torch.tensor([0, 0, 2, 0, 3, 4, 0, 0])
    separation = torch.tensor([-1.0, -1.0, -1.0, 0.5, -1.0, -1.0, -1.0, -1.0])
    normal = torch.tensor([[0.0, 0.0, 1.0]]).repeat(8, 1)
    normal[7] = 0.0  # unresolved: degenerate normal
    points = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    raw = {
        "shape0": shape0,
        "shape1": shape1,
        "normal": normal,
        "point0_world": points,
        "point1_world": points + 1,
        "surface_separation": separation,
        "thickness0": torch.zeros(8),
        "thickness1": torch.zeros(8),
        "_body_q": torch.zeros(5, 7),
    }
    state = SimpleNamespace()

    reference = adapter._build_native_contact_pairs(raw, state)
    admitted = native_pair_filter_mask(reference, adapter.contact_filter, adapter.fixed_body_slots)
    expected = {name: value[admitted] for name, value in reference.items()}
    resolved, eligible = adapter._prefilter_masks(raw)
    kept = torch.nonzero(eligible).reshape(-1)
    subset = {name: value[kept] if not name.startswith("_") else value for name, value in raw.items()}
    actual = adapter._build_native_contact_pairs(subset, state)

    assert int(resolved.sum()) == reference["world_ids"].shape[0] == 6
    assert int(eligible.sum()) == int(admitted.sum()) == 4
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name])
