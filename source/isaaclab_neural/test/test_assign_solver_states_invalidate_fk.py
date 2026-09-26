# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for pool-reset kinematics invalidation."""

from __future__ import annotations

import torch
from isaaclab_neural.envs.neural_env_wrapper import NeuralEnvAdapter
from isaaclab_neural.generate.adapter import NewtonDataGenerationBackend
from isaaclab_neural.physics.nerd_newton_manager import NewtonNerdManager


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
        self.num_envs = 2
        self.torch_device = torch.device("cpu")

    def _assign_states_from_torch(self, state, states: torch.Tensor) -> None:
        self.assigned = (state, tuple(states.shape))

    def reset(self, env_ids=None) -> None:
        self.reset_ids = env_ids


def test_assign_solver_states_invalidates_fk_before_forward() -> None:
    backend = NewtonDataGenerationBackend.__new__(NewtonDataGenerationBackend)
    manager = _FakeManager()
    backend._manager = manager
    backend.state = object()
    solver = _FakeSolver()

    backend.assign_solver_states(solver, torch.zeros(2, 4))

    assert solver.assigned[1] == (2, 4)
    assert manager.calls == ["invalidate_fk", "forward"]


def test_neural_env_adapter_explicit_reset_invalidates_fk_before_forward() -> None:
    manager = _FakeManager()
    solver = _FakeSolver()
    manager._solver = solver
    from types import SimpleNamespace

    manager.backend = SimpleNamespace(state_0=object())
    manager.sync_neural_solver = lambda *, update_history=True: manager.calls.append("sync")

    adapter = NeuralEnvAdapter.__new__(NeuralEnvAdapter)
    adapter.manager = manager
    adapter.env = None

    adapter.reset(torch.zeros(2, 4), reset_history=True)

    assert solver.assigned[1] == (2, 4)
    assert manager.calls == ["invalidate_fk", "forward", "sync"]
    assert solver.reset_ids is None


def test_reset_neural_solver_invalidates_fk_before_forward(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(NewtonNerdManager, "_nerd_active", True, raising=False)
    monkeypatch.setattr(NewtonNerdManager, "_solver", object(), raising=False)
    monkeypatch.setattr(NewtonNerdManager, "invalidate_fk", classmethod(lambda cls: calls.append("invalidate_fk")))
    monkeypatch.setattr(NewtonNerdManager, "forward", classmethod(lambda cls: calls.append("forward")))
    monkeypatch.setattr(
        NewtonNerdManager,
        "sync_neural_solver",
        classmethod(lambda cls, *, update_history=True: calls.append("sync")),
    )

    NewtonNerdManager.reset_neural_solver(env_ids=None, reset_history=False)

    assert calls == ["invalidate_fk", "forward", "sync"]
