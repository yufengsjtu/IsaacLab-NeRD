# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for world-major NeuralSolver layout resolution."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from isaaclab_neural.solvers.world_layout import resolve_neural_solver_layout


class _Buf:
    def __init__(self, data):
        self._data = np.asarray(data)
        self.shape = self._data.shape

    def numpy(self):
        return self._data


def _one_art_per_world_model() -> SimpleNamespace:
    return SimpleNamespace(
        world_count=2,
        articulation_count=2,
        joint_count=2,
        body_count=2,
        joint_q_start=_Buf([0, 1, 2]),
        joint_qd_start=_Buf([0, 1, 2]),
        articulation_start=_Buf([0, 1, 2]),
        joint_world=_Buf([0, 1]),
        joint_child=_Buf([0, 1]),
        joint_f=_Buf(np.zeros(2)),
    )


def _two_arts_per_world_model() -> SimpleNamespace:
    # Robot: two 1-DoF joints. Cube: one FREE joint (7 q, 6 qd).
    return SimpleNamespace(
        world_count=2,
        articulation_count=4,
        joint_count=6,
        body_count=6,
        joint_q_start=_Buf([0, 1, 2, 9, 10, 11, 18]),
        joint_qd_start=_Buf([0, 1, 2, 8, 9, 10, 16]),
        articulation_start=_Buf([0, 2, 3, 5, 6]),
        joint_world=_Buf([0, 0, 0, 1, 1, 1]),
        joint_child=_Buf([0, 1, 2, 3, 4, 5]),
        joint_f=_Buf(np.zeros(16)),
    )


def test_layout_matches_single_articulation_worlds() -> None:
    layout = resolve_neural_solver_layout(_one_art_per_world_model())

    assert layout.num_envs == 2
    assert layout.articulations_per_env == 1
    assert layout.dof_q_per_env == 1
    assert layout.dof_qd_per_env == 1
    assert layout.joint_f_dim == 1
    np.testing.assert_array_equal(layout.root_body_ids, np.asarray([0, 1]))
    np.testing.assert_array_equal(layout.state_world_ids, np.asarray([0, 1]))


def test_layout_concatenates_multiple_articulations_per_world() -> None:
    layout = resolve_neural_solver_layout(_two_arts_per_world_model())

    assert layout.num_envs == 2
    assert layout.articulations_per_env == 2
    assert layout.dof_q_per_env == 9
    assert layout.dof_qd_per_env == 8
    assert layout.num_joints_per_env == 3
    assert layout.joint_f_dim == 8
    np.testing.assert_array_equal(layout.root_body_ids, np.asarray([0, 3]))


def test_layout_rejects_noncontiguous_worlds() -> None:
    model = _two_arts_per_world_model()
    model.joint_world = _Buf([0, 1, 0, 1, 0, 1])

    with pytest.raises(ValueError, match="world-contiguous"):
        resolve_neural_solver_layout(model)


def test_layout_rejects_uneven_articulation_counts() -> None:
    model = _one_art_per_world_model()
    model.articulation_count = 3

    with pytest.raises(ValueError, match="same number of articulations"):
        resolve_neural_solver_layout(model)
