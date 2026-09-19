# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for world-major NeuralSolver layout resolution."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import create_autospec

import newton
import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_neural.generate.adapter import DataGenerationAdapter
from isaaclab_neural.generate.sampler import ActionTrajectorySampler
from isaaclab_neural.solvers.neural_solver import NeuralSolver
from isaaclab_neural.solvers.world_layout import resolve_neural_solver_layout


class _SamplerTestHarness(ActionTrajectorySampler):
    trajectory_context = ActionTrajectorySampler._trajectory_context


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


def _make_world_solver(*, with_arm: bool, num_free_bodies: int, frame: str, anchor: str) -> NeuralSolver:
    builder = newton.ModelBuilder()
    for _ in range(2):
        builder.begin_world()
        if with_arm:
            base = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), label="base")
            link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), label="arm")
            fixed = builder.add_joint_fixed(parent=-1, child=base)
            revolute = builder.add_joint_revolute(parent=base, child=link, limit_lower=-2.0, limit_upper=2.0)
            builder.add_articulation([fixed, revolute])
        for index in range(num_free_bodies):
            body = builder.add_link(mass=0.2, inertia=wp.mat33(np.eye(3)), label=f"object_{index}")
            joint = builder.add_joint_free(child=body)
            builder.add_articulation([joint])
        builder.add_shape_box(body=-1, hx=0.5, hy=0.5, hz=0.025)
        builder.end_world()
    return NeuralSolver(
        model=builder.finalize(device="cpu"),
        states_frame=frame,
        anchor_frame_step=anchor,
        states_embedding_type="identical",
    )


@pytest.mark.parametrize("with_arm,num_free_bodies", [(False, 1), (True, 1), (True, 2)])
@pytest.mark.parametrize("frame", ["body", "body_translation_only"])
@pytest.mark.parametrize("anchor", ["first", "last", "every"])
def test_world_frame_conversion_includes_free_objects_after_robot(with_arm, num_free_bodies, frame, anchor) -> None:
    solver = _make_world_solver(with_arm=with_arm, num_free_bodies=num_free_bodies, frame=frame, anchor=anchor)
    states = torch.full((2, 3, solver.state_dim), 0.25)
    root = torch.zeros(2, 3, 7)
    root[..., :3] = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 4.0, 0.0]]])
    root[1, :, :3] += torch.tensor([10.0, -5.0, 1.0])
    root[..., 5:7] = 2.0**-0.5
    free_joints = np.flatnonzero(solver.joint_types == newton.JointType.FREE)
    joint_qd_start = solver.model.joint_qd_start
    assert joint_qd_start is not None
    qd_starts = joint_qd_start.numpy()
    for joint in free_joints:
        q = int(solver.joint_q_start[joint])
        v = solver.dof_q_per_env + int(qd_starts[joint])
        states[..., q : q + 3] = root[..., :3] + torch.tensor([0.4, 0.2, 0.3])
        states[..., q + 3 : q + 7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        states[..., v : v + 6] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    original = states.clone()
    states.requires_grad_()
    converted, converted_next, *_ = solver.convert_coordinate_frame(root, states, states, None, None, None, None)
    expected = original.clone()
    selected_root = root if anchor == "every" else root[:, :1] if anchor == "first" else root[:, -1:]
    for joint in free_joints:
        q = int(solver.joint_q_start[joint])
        v = solver.dof_q_per_env + int(qd_starts[joint])
        position = original[..., q : q + 3] - selected_root[..., :3]
        if frame == "body":
            position = torch.stack((position[..., 1], -position[..., 0], position[..., 2]), dim=-1)
            expected[..., q + 3 : q + 7] = torch.tensor([0.0, 0.0, -(2.0**-0.5), 2.0**-0.5])
            expected[..., v : v + 6] = torch.tensor([2.0, -1.0, 3.0, 5.0, -4.0, 6.0])
        expected[..., q : q + 3] = position
    torch.testing.assert_close(converted, expected)
    torch.testing.assert_close(converted_next, expected)
    torch.testing.assert_close(states, original)
    restored = solver.convert_states_back_to_world(root, converted[:, -1])
    torch.testing.assert_close(restored, original[:, -1])
    converted.square().sum().backward()
    assert states.grad is not None and torch.isfinite(states.grad).all()


@pytest.mark.parametrize("with_terrain", [False, True])
def test_trajectory_context_records_replica_origins(with_terrain) -> None:
    scene = SimpleNamespace(env_origins=torch.tensor([[1.25, -1.25, 0.0], [-1.25, 1.25, 0.0]]))
    expected = scene.env_origins
    if with_terrain:
        scene.terrain = SimpleNamespace(env_origins=scene.env_origins + 5.0)
        expected = scene.terrain.env_origins
    sampler = _SamplerTestHarness.__new__(_SamplerTestHarness)
    sampler.data_device = torch.device("cpu")
    sampler.adapter = create_autospec(
        DataGenerationAdapter,
        instance=True,
        num_envs=2,
        env=SimpleNamespace(scene=scene),
        state_world_ids=torch.arange(2),
        root_world_ids=torch.arange(2),
        contact_world_ids=torch.arange(2),
    )
    context = sampler.trajectory_context()
    torch.testing.assert_close(context["env_origin"], expected)
    expected.add_(1.0)
    assert not torch.equal(context["env_origin"], expected)


def test_world_input_translation_preserves_state_and_contact_identity() -> None:
    solver = _make_world_solver(with_arm=True, num_free_bodies=2, frame="body", anchor="every")
    states = torch.arange(2 * 3 * solver.state_dim, dtype=torch.float32).reshape(2, 3, solver.state_dim)
    tokens = torch.ones(2, 3, 4, 17)
    tokens[..., -1, :] = 0.0
    masks = torch.tensor([True, True, True, False]).expand(2, 3, 4)
    points = torch.ones(2, 3, 4, 3)
    points[..., -1, :] = 0.0
    inputs = {
        "states": states,
        "next_states": states + 1.0,
        "root_body_q": torch.ones(2, 3, 7),
        "primary_body_q": torch.ones(2, 3, solver.num_bodies_per_env, 7),
        "root_body_qd": torch.ones(2, 3, 6),
        "joint_f": torch.ones(2, 3, solver.joint_f_dim),
        "env_origin": torch.zeros(2, 3),
        "contact_tokens": tokens,
        "contact_masks": masks,
        "contact_points_0": points.flatten(-2),
        "contact_points_1": points.flatten(-2),
        "contact_normals": points.flatten(-2),
    }
    original = {key: value.clone() for key, value in inputs.items()}
    shift = torch.tensor([[10.0, -3.0, 2.0], [-20.0, 6.0, 1.0]])
    translated = solver.translate_world_inputs(inputs, shift)
    for key in ("states", "next_states"):
        expected = original[key].clone()
        for joint in solver.free_joint_indices:
            q = int(solver.joint_q_start[joint])
            expected[..., q : q + 3] += shift[:, None, :]
        torch.testing.assert_close(translated[key], expected)
    for key in ("root_body_q", "primary_body_q"):
        expected = original[key].clone()
        expected[..., :3] += shift.reshape(2, *([1] * (expected.ndim - 2)), 3)
        torch.testing.assert_close(translated[key], expected)
    expected_tokens = tokens.clone()
    expected_tokens[..., :-1, 4:7] += shift[:, None, None, :]
    torch.testing.assert_close(translated["contact_tokens"], expected_tokens)
    for key in ("contact_points_0", "contact_points_1"):
        expected_points = points.clone()
        expected_points[..., :-1, :] += shift[:, None, None, :]
        torch.testing.assert_close(translated[key], expected_points.flatten(-2))
    torch.testing.assert_close(translated["env_origin"], shift)
    for key in ("root_body_qd", "joint_f", "contact_masks", "contact_normals"):
        torch.testing.assert_close(translated[key], original[key])
    for key in inputs:
        torch.testing.assert_close(inputs[key], original[key])


def test_neural_contact_inputs_are_separate_from_newton_contacts() -> None:
    solver = _make_world_solver(with_arm=True, num_free_bodies=1, frame="world", anchor="every")
    assert solver.contact_inputs == {}
    assert super(NeuralSolver, solver).contacts is None
    inputs = {"contact_masks": torch.ones(2, 3, dtype=torch.bool)}
    with pytest.warns(DeprecationWarning, match="contact_inputs"):
        solver.contacts = inputs
    assert solver.contact_inputs is inputs
    with pytest.warns(DeprecationWarning, match="contact_inputs"):
        assert solver.contacts is inputs
    assert super(NeuralSolver, solver).contacts is None


def test_neural_step_writes_robot_and_free_body_but_has_no_table_state() -> None:
    solver = _make_world_solver(with_arm=True, num_free_bodies=1, frame="world", anchor="every")
    state_in, state_out = solver.model.state(), solver.model.state()
    joint_q, joint_qd = state_in.joint_q, state_in.joint_qd
    assert joint_q is not None and joint_qd is not None
    newton.eval_fk(solver.model, joint_q, joint_qd, state_in)
    solver.sync_from_newton(state_in, None, solver.model.control().joint_f)
    current = solver.states.clone()
    target = current.clone()
    target[:, 0] = 0.2
    target[:, solver.dof_q_per_env] = 0.3
    free_joint = int(solver.free_joint_indices[0])
    q = int(solver.joint_q_start[free_joint])
    v = solver.dof_q_per_env + int(solver.joint_qd_start[free_joint])
    target[:, q : q + 3] = torch.tensor([0.4, 0.2, 0.5])
    target[:, v : v + 6] = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    prediction = solver.convert_next_states_to_prediction(current, target)

    class PrescribedPrediction(torch.nn.Module):
        def forward(self, inputs, *, single_step):
            return prediction[:, None, :]

    solver.set_neural_solver_model(PrescribedPrediction())
    control = solver.model.control()
    solver.step(state_in, state_out, control)
    solver.sync_from_newton(state_out, None, control.joint_f)
    torch.testing.assert_close(solver.states, target)
    joint_child = solver.model.joint_child
    body_q = state_out.body_q
    shape_body = solver.model.shape_body
    assert joint_child is not None and body_q is not None and shape_body is not None
    body_ids = torch.as_tensor(joint_child.numpy()[free_joint :: solver.num_joints_per_env].copy(), dtype=torch.long)
    cube_poses = wp.to_torch(body_q).index_select(0, body_ids)
    torch.testing.assert_close(cube_poses, target[:, q : q + 7])
    assert solver.state_dim == 1 + 7 + 1 + 6
    assert solver.articulations_per_env == 2
    assert (shape_body.numpy() == -1).all()
