# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for contact-token and root-body Newton world indexing."""

from types import SimpleNamespace

import numpy as np
import torch
from isaaclab_neural.generate.adapter import DataGenerationAdapter
from isaaclab_neural.solvers.neural_solver import _resolve_root_body_ids


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
    adapter.contact_adapter = SimpleNamespace(
        contact_token_body_ids=torch.tensor([[1, -1], [0, 2]], dtype=torch.long)
    )

    world_ids = adapter.contact_token_world_ids

    assert world_ids is not None
    torch.testing.assert_close(world_ids, torch.tensor([[0, -1], [1, 1]], dtype=torch.long))
