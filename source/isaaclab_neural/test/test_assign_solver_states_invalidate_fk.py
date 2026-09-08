# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for pool-reset kinematics invalidation."""

from __future__ import annotations

import torch

from isaaclab_neural.generate.adapter import NewtonDataGenerationBackend


class _FakeManager:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def invalidate_fk(self, *args, **kwargs) -> None:
        del args, kwargs
        self.calls.append("invalidate_fk")

    def forward(self) -> None:
        self.calls.append("forward")


class _FakeSolver:
    def __init__(self) -> None:
        self.assigned = None

    def _assign_states_from_torch(self, state, states: torch.Tensor) -> None:
        self.assigned = (state, tuple(states.shape))


def test_assign_solver_states_invalidates_fk_before_forward() -> None:
    backend = NewtonDataGenerationBackend.__new__(NewtonDataGenerationBackend)
    manager = _FakeManager()
    backend._manager = manager
    backend.state = object()
    solver = _FakeSolver()

    backend.assign_solver_states(solver, torch.zeros(2, 4))

    assert solver.assigned[1] == (2, 4)
    assert manager.calls == ["invalidate_fk", "forward"]
