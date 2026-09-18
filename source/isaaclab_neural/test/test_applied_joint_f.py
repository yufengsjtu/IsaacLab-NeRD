# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for copying implicit-actuator shadow PD into NeRD ``joint_f``."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from isaaclab_neural.envs.neural_env_wrapper import NeuralEnvAdapter
from isaaclab_neural.generate.adapter import DataGenerationAdapter
from isaaclab_neural.physics.applied_joint_f import (
    backend_applied_effort,
    copy_applied_effort_into_sim_joint_f,
    copy_scene_applied_effort_into_sim_joint_f,
    reorder_user_effort_to_backend,
)
from isaaclab_neural.physics.nerd_newton_manager import NewtonNerdManager


def test_reorder_user_effort_to_backend_keeps_identity() -> None:
    applied = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    torch.testing.assert_close(reorder_user_effort_to_backend(applied, has_ordering=False), applied)


def test_reorder_user_effort_to_backend_scatters_permutation() -> None:
    applied = torch.tensor([[1.0, 2.0, 3.0]])
    user_to_backend = torch.tensor([2, 0, 1])

    backend = reorder_user_effort_to_backend(applied, user_to_backend=user_to_backend, has_ordering=True)

    torch.testing.assert_close(backend, torch.tensor([[2.0, 3.0, 1.0]]))


def test_copy_applied_effort_writes_backend_order_without_touching_control_source() -> None:
    applied = torch.tensor([[8.0, -3.0, 0.5]])
    sim_joint_f = torch.zeros(1, 3)
    articulation = SimpleNamespace(
        actuators=SimpleNamespace(applied_effort=applied),
        data=SimpleNamespace(has_joint_ordering=True, _sim_bind_joint_effort=sim_joint_f),
        _joint_user_to_backend_map=lambda: torch.tensor([1, 2, 0]),
    )

    assert copy_applied_effort_into_sim_joint_f(articulation)
    torch.testing.assert_close(sim_joint_f, torch.tensor([[0.5, 8.0, -3.0]]))
    torch.testing.assert_close(applied, torch.tensor([[8.0, -3.0, 0.5]]))


def test_copy_floating_base_effort_preserves_base_dofs_and_warp_ordering() -> None:
    import warp as wp

    applied = torch.tensor([[8.0, -3.0, 0.5], [2.0, 4.0, -1.0]])
    control = torch.zeros(2, 9)
    control[:, :6] = torch.arange(6)
    original_base = control[:, :6].clone()
    articulation = SimpleNamespace(
        actuators=SimpleNamespace(applied_effort=applied, has_implicit_actuators=True),
        data=SimpleNamespace(has_joint_ordering=True, _sim_bind_joint_effort=wp.from_torch(control[:, 6:])),
        _joint_user_to_backend_map=lambda: wp.array([1, 2, 0], dtype=wp.int32, device="cpu"),
    )

    assert copy_applied_effort_into_sim_joint_f(articulation)

    torch.testing.assert_close(control[:, :6], original_base)
    torch.testing.assert_close(control[:, 6:], applied[:, [2, 0, 1]])


def test_copy_keeps_explicit_actuator_control_unchanged() -> None:
    control = torch.tensor([[2.0, 3.0]])
    original = control.clone()
    articulation = SimpleNamespace(
        actuators=SimpleNamespace(applied_effort=torch.zeros_like(control), has_implicit_actuators=False),
        data=SimpleNamespace(has_joint_ordering=False, _sim_bind_joint_effort=control),
    )

    assert not copy_applied_effort_into_sim_joint_f(articulation)
    torch.testing.assert_close(control, original)


def test_copy_scene_skips_articulations_without_actuators() -> None:
    sim_joint_f = torch.zeros(2, 2)
    robot = SimpleNamespace(
        actuators=SimpleNamespace(applied_effort=torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
        data=SimpleNamespace(has_joint_ordering=False, _sim_bind_joint_effort=sim_joint_f),
    )
    cube = SimpleNamespace(actuators=None, data=SimpleNamespace(_sim_bind_joint_effort=torch.zeros(2, 0)))
    scene = SimpleNamespace(articulations={"robot": robot, "cube": cube})

    assert copy_scene_applied_effort_into_sim_joint_f(scene) == 1
    torch.testing.assert_close(sim_joint_f, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert backend_applied_effort(cube) is None


def test_nerd_post_actuator_hooks_run_telemetry_before_overlay(monkeypatch) -> None:
    order: list[str] = []
    NewtonNerdManager._post_actuator_callbacks = [lambda: order.append("telemetry")]
    monkeypatch.setattr(
        NewtonNerdManager,
        "_overlay_applied_effort_into_joint_f",
        classmethod(lambda cls: order.append("overlay")),
    )
    try:
        NewtonNerdManager._run_post_actuator_hooks()
    finally:
        NewtonNerdManager._post_actuator_callbacks = []

    assert order == ["telemetry", "overlay"]


def test_nerd_overlay_is_skipped_when_inactive(monkeypatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        "isaaclab_neural.physics.applied_joint_f.copy_scene_applied_effort_into_sim_joint_f",
        lambda scene: calls.append(scene),
    )
    NewtonNerdManager._nerd_active = False
    NewtonNerdManager._applied_effort_scenes = [object()]
    try:
        NewtonNerdManager._overlay_applied_effort_into_joint_f()
    finally:
        NewtonNerdManager._applied_effort_scenes = []
        NewtonNerdManager._nerd_active = False

    assert calls == []


def test_nerd_overlay_copies_registered_scenes_when_active(monkeypatch) -> None:
    scene = object()
    calls: list[object] = []
    monkeypatch.setattr(
        "isaaclab_neural.physics.applied_joint_f.copy_scene_applied_effort_into_sim_joint_f",
        lambda value: calls.append(value),
    )
    NewtonNerdManager._nerd_active = True
    NewtonNerdManager._applied_effort_scenes = [scene]
    try:
        NewtonNerdManager._overlay_applied_effort_into_joint_f()
    finally:
        NewtonNerdManager._applied_effort_scenes = []
        NewtonNerdManager._nerd_active = False

    assert calls == [scene]


def test_dataset_adapter_sync_overlays_only_when_requested() -> None:
    overlay_calls: list[bool] = []
    sync_calls: list[object] = []
    adapter = DataGenerationAdapter.__new__(DataGenerationAdapter)
    adapter.env = SimpleNamespace(scene=object())
    adapter.overlay_applied_effort_into_control_joint_f = lambda: overlay_calls.append(True)
    adapter._prepare_contacts = lambda: "contacts"
    adapter.backend = SimpleNamespace(sync_solver=lambda solver, contacts: sync_calls.append((solver, contacts)))
    adapter.solver = object()

    adapter.sync()
    adapter.sync(overlay_applied_effort=True)

    assert overlay_calls == [True]
    assert sync_calls == [(adapter.solver, "contacts"), (adapter.solver, "contacts")]


def test_neural_env_adapter_registers_applied_effort_scene() -> None:
    scene = object()
    env = SimpleNamespace(_reset_idx=lambda _ids: None, scene=scene)
    NewtonNerdManager._applied_effort_scenes = []
    try:
        NeuralEnvAdapter(env, manager=NewtonNerdManager, patch_reset=False)
        NeuralEnvAdapter(env, manager=NewtonNerdManager, patch_reset=False)
        assert NewtonNerdManager._applied_effort_scenes == [scene]
    finally:
        NewtonNerdManager._applied_effort_scenes = []


def test_simulate_physics_only_overlays_unless_skipped(monkeypatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(NewtonNerdManager, "_prepare_nerd_contacts", classmethod(lambda cls: "contacts"))
    monkeypatch.setattr(
        NewtonNerdManager,
        "_overlay_applied_effort_into_joint_f",
        classmethod(lambda cls: calls.append("overlay")),
    )
    monkeypatch.setattr(
        NewtonNerdManager,
        "_run_solver_substeps",
        classmethod(lambda cls, contacts: calls.append(("solve", contacts))),
    )
    monkeypatch.setattr(NewtonNerdManager, "_update_sensors", classmethod(lambda cls, contacts: None))
    monkeypatch.setattr(NewtonNerdManager, "_sync_upstream_newton_state", classmethod(lambda cls: None))
    NewtonNerdManager._nerd_active = True
    NewtonNerdManager._skip_applied_effort_overlay = False
    try:
        NewtonNerdManager._simulate_physics_only()
        with NewtonNerdManager.without_applied_effort_overlay():
            NewtonNerdManager._simulate_physics_only()
        NewtonNerdManager._simulate_physics_only()
    finally:
        NewtonNerdManager._nerd_active = False
        NewtonNerdManager._skip_applied_effort_overlay = False

    assert calls == [
        "overlay",
        ("solve", "contacts"),
        ("solve", "contacts"),
        "overlay",
        ("solve", "contacts"),
    ]


def test_nerd_post_step_callbacks_run_after_solver_before_sensors(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(NewtonNerdManager, "_nerd_active", True)
    monkeypatch.setattr(NewtonNerdManager, "_skip_applied_effort_overlay", True)
    monkeypatch.setattr(NewtonNerdManager, "_solver_dt", 0.005)
    monkeypatch.setattr(NewtonNerdManager, "_num_substeps", 1)
    monkeypatch.setattr(NewtonNerdManager, "_decimation", 2)
    monkeypatch.setattr(NewtonNerdManager, "_adapter", None)
    monkeypatch.setattr(NewtonNerdManager, "_prepare_nerd_contacts", classmethod(lambda cls: None))
    monkeypatch.setattr(NewtonNerdManager, "_run_post_actuator_hooks", classmethod(lambda cls: None))
    monkeypatch.setattr(
        NewtonNerdManager, "_run_solver_substeps", classmethod(lambda cls, contacts: calls.append("solve"))
    )
    monkeypatch.setattr(NewtonNerdManager, "_post_step_callbacks", [lambda: calls.append("post")])
    monkeypatch.setattr(
        NewtonNerdManager, "_update_sensors", classmethod(lambda cls, contacts: calls.append("sensors"))
    )
    monkeypatch.setattr(NewtonNerdManager, "_sync_upstream_newton_state", classmethod(lambda cls: None))
    NewtonNerdManager._simulate_full()
    assert calls == ["solve", "solve", "post", "sensors"]
    calls.clear()
    NewtonNerdManager._simulate_physics_only()
    assert calls == ["solve", "post", "sensors"]


def test_training_eval_joint_f_step_skips_applied_effort_overlay(monkeypatch) -> None:
    from isaaclab_neural.eval.training_evaluator import TrainingRolloutEvaluator

    seen: list[bool] = []
    monkeypatch.setattr("isaaclab_neural.eval.training_evaluator.wp.from_torch", lambda value: value)
    NewtonNerdManager._control = SimpleNamespace(joint_f=SimpleNamespace(assign=lambda _value: None))
    monkeypatch.setattr(
        NewtonNerdManager,
        "_simulate_physics_only",
        classmethod(lambda cls: seen.append(cls._skip_applied_effort_overlay)),
    )
    evaluator = TrainingRolloutEvaluator.__new__(TrainingRolloutEvaluator)
    evaluator.device = "cpu"
    evaluator.neural_env = SimpleNamespace(
        neural_adapter=SimpleNamespace(
            manager=NewtonNerdManager,
            isaaclab_env=SimpleNamespace(),
            sync=lambda **_kwargs: None,
            states_torch=torch.zeros(2, 3),
        )
    )
    try:
        evaluator._step_joint_f(torch.zeros(2, 3))
    finally:
        NewtonNerdManager._control = None
        NewtonNerdManager._skip_applied_effort_overlay = False

    assert seen == [True]
    assert NewtonNerdManager._skip_applied_effort_overlay is False
