# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for contact-free NeRD runtime setup."""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import create_autospec

import isaaclab_neural.solvers
import pytest
import torch
from isaaclab_neural.data.hdf5 import write_rollouts_to_hdf5
from isaaclab_neural.envs.anymal_nerd_env import (
    NerdAnymalCFlatEnvCfg,
    NerdAnymalCRoughEnvCfg,
    apply_contact_mode_mdp,
)
from isaaclab_neural.envs.cartpole_nerd_env import NerdCartpoleEnvCfg
from isaaclab_neural.generate.adapter import DataGenerationAdapter, NewtonDataGenerationBackend
from isaaclab_neural.generate.sampler import ActionTrajectorySampler
from isaaclab_neural.physics import NerdNewtonCfg, NerdSolverCfg
from isaaclab_neural.physics.nerd_newton_manager import NewtonNerdManager
from isaaclab_neural.rl.rsl_rl.cli_args import apply_neural_model_to_env_cfg
from isaaclab_neural.solvers.neural_solver import NeuralSolver
from isaaclab_neural.solvers.rnn_neural_solver import RNNNeuralSolver
from isaaclab_neural.train.trainers import VanillaTrainer
from isaaclab_newton.physics.newton_manager import NewtonManager
from newton import Control, Model, State


class _ManagerTestHarness(NewtonNerdManager):
    """Expose hooks bound to the production manager, preserving singleton state."""

    build_solver = NewtonNerdManager._build_solver
    load_neural_model = NewtonNerdManager._load_nerd_neural_model
    defer_neural_model_loading = NewtonNerdManager._defer_neural_model_loading
    prepare_contacts = NewtonNerdManager._prepare_nerd_contacts
    sync_upstream_state = NewtonNerdManager._sync_upstream_newton_state
    reset_solver_internals = NewtonNerdManager._reset_solver_internals
    sync_attributes = NewtonNerdManager._UPSTREAM_SYNC_ATTRS

    @staticmethod
    def runtime_state() -> dict[str, object]:
        return {
            "solver": NewtonManager._solver,
            "needs_collision_pipeline": NewtonManager._needs_collision_pipeline,
            "collision_pipeline": NewtonManager._collision_pipeline,
            "contact_adapter": NewtonNerdManager._nerd_contact_adapter,
            "loading_deferred": NewtonNerdManager._neural_model_loading_deferred,
        }


class _SamplerTestHarness(ActionTrajectorySampler):
    copy_before_step = ActionTrajectorySampler._copy_before_step


class _RecurrentTestHarness(RNNNeuralSolver):
    before_model_forward = RNNNeuralSolver._before_model_forward


def test_contact_free_manager_skips_contact_pipeline(monkeypatch) -> None:
    captured_kwargs = {}
    installed_models = []
    fake_solver = SimpleNamespace(
        set_neural_solver_model=lambda model: installed_models.append(model),
        eval=lambda: None,
    )
    fake_neural_model = object()

    def _create_neural_solver(_solver_cfg, **kwargs):
        captured_kwargs.update(kwargs)
        return fake_solver

    monkeypatch.setattr(isaaclab_neural.solvers, "create_neural_solver", _create_neural_solver)
    monkeypatch.setattr(NewtonManager, "_contacts", None)
    monkeypatch.setattr(
        NewtonNerdManager,
        "_load_nerd_neural_model",
        classmethod(lambda cls, cfg_dict, solver: fake_neural_model),
    )
    monkeypatch.setattr(NewtonNerdManager, "_disable_cuda_graph_for_nerd", classmethod(lambda cls: None))
    monkeypatch.setattr(NewtonNerdManager, "_sync_upstream_newton_state", classmethod(lambda cls: None))

    _ManagerTestHarness.build_solver(
        model=Model("cpu"),
        solver_cfg=NerdSolverCfg(contact_mode="newton_native", num_contacts_per_env=0),
    )

    state = _ManagerTestHarness.runtime_state()
    assert state["solver"] is fake_solver
    assert state["needs_collision_pipeline"] is False
    assert state["collision_pipeline"] is None
    assert state["contact_adapter"] is None
    assert captured_kwargs["contacts"] is None
    assert captured_kwargs["contact_adapter"] is None
    assert _ManagerTestHarness.prepare_contacts() is None
    assert installed_models == [fake_neural_model]


def test_runtime_requires_explicit_dynamics_checkpoint() -> None:
    with pytest.raises(ValueError, match="explicit neural_model_path"):
        _ManagerTestHarness.load_neural_model({"neural_model_path": None})


def test_training_can_defer_neural_model_loading(monkeypatch) -> None:
    installed_models = []
    fake_solver = SimpleNamespace(
        set_neural_solver_model=lambda model: installed_models.append(model),
        eval=lambda: pytest.fail("Deferred model must not enter evaluation mode."),
    )

    monkeypatch.setattr(isaaclab_neural.solvers, "create_neural_solver", lambda *_args, **_kwargs: fake_solver)
    monkeypatch.setattr(NewtonManager, "_contacts", None)
    monkeypatch.setattr(
        NewtonNerdManager,
        "_load_nerd_neural_model",
        classmethod(lambda cls, cfg_dict, solver: pytest.fail("Deferred model must not be loaded.")),
    )
    monkeypatch.setattr(NewtonNerdManager, "_disable_cuda_graph_for_nerd", classmethod(lambda cls: None))
    monkeypatch.setattr(NewtonNerdManager, "_sync_upstream_newton_state", classmethod(lambda cls: None))

    with _ManagerTestHarness.defer_neural_model_loading():
        assert _ManagerTestHarness.runtime_state()["loading_deferred"] is True
        _ManagerTestHarness.build_solver(
            model=Model("cpu"),
            solver_cfg=NerdSolverCfg(
                contact_mode="newton_native",
                num_contacts_per_env=0,
            ),
        )

    state = _ManagerTestHarness.runtime_state()
    assert state["solver"] is fake_solver
    assert installed_models == []
    assert state["loading_deferred"] is False


def test_cartpole_uses_contact_free_solver_config() -> None:
    cfg = NerdCartpoleEnvCfg()
    physics_cfg = cfg.sim.physics
    assert isinstance(physics_cfg, NerdNewtonCfg)
    assert physics_cfg.solver_cfg.contact_mode == "newton_native"
    assert physics_cfg.solver_cfg.num_contacts_per_env == 0
    apply_contact_mode_mdp(cfg)


def test_nerd_rl_requires_explicit_dynamics_checkpoint() -> None:
    cfg = NerdCartpoleEnvCfg()
    args = Namespace(
        neural_model_path=None,
        contact_mode=None,
        num_contacts_per_env=None,
        use_kv_cache=False,
    )

    with pytest.raises(ValueError, match="requires --neural-model-path"):
        apply_neural_model_to_env_cfg(cfg, args)


def test_contact_tokens_require_positive_contact_capacities() -> None:
    with pytest.raises(ValueError, match="contact_tokens.*positive contact capacity"):
        NerdSolverCfg(
            contact_mode="newton_native",
            contact_representation="contact_tokens",
            num_contacts_per_env=0,
        )

    with pytest.raises(ValueError, match="max_contact_tokens must be positive"):
        NerdSolverCfg(
            contact_mode="newton_native",
            contact_representation="contact_tokens",
            num_contacts_per_env=1,
            max_contact_tokens=0,
        )


def test_anymal_configs_keep_native_contact_capacity() -> None:
    flat_cfg = NerdAnymalCFlatEnvCfg()
    rough_cfg = NerdAnymalCRoughEnvCfg()
    for cfg in (flat_cfg, rough_cfg):
        physics_cfg = cfg.sim.physics
        assert isinstance(physics_cfg, NerdNewtonCfg)
        assert physics_cfg.solver_cfg.contact_mode == "newton_native"
        assert physics_cfg.solver_cfg.num_contacts_per_env > 0
    assert rough_cfg.terminations.base_contact is None
    assert rough_cfg.rewards.track_lin_vel_xy_exp.weight == 1.0
    assert rough_cfg.rewards.track_ang_vel_z_exp.weight == 0.5


def _contact_free_adapter() -> DataGenerationAdapter:
    adapter = create_autospec(DataGenerationAdapter, instance=True)
    adapter.num_envs = 2
    adapter.num_contacts_per_env = 0
    adapter.state_dim = 4
    adapter.joint_f_dim = 1
    adapter.state_world_ids = torch.arange(2)
    adapter.root_world_ids = torch.arange(2)
    adapter.contact_world_ids = torch.arange(2)
    adapter.solver = SimpleNamespace(contact_representation="flat")
    adapter.model = SimpleNamespace(up_axis=2)
    adapter.env = SimpleNamespace(scene=None)
    adapter.contact_adapter = None
    adapter.body_net_contact_force_recorder = None
    adapter.raw_neural_inputs.return_value = {
        "states": torch.zeros(2, 1, 4),
        "root_body_q": torch.zeros(2, 1, 7),
        "root_body_qd": torch.zeros(2, 1, 6),
    }
    return adapter


def test_contact_free_sampler_omits_contact_tensors() -> None:
    sampler = _SamplerTestHarness.__new__(_SamplerTestHarness)
    sampler.adapter = _contact_free_adapter()
    sampler.trajectory_length = 3
    sampler.data_device = torch.device("cpu")

    buffers = sampler.allocate_batch_buffers(record_actions=False)
    sampler.copy_before_step(buffers, step=0)

    assert buffers["contacts"] == {}


def test_contact_token_generation_rejects_zero_contact_capacity(monkeypatch) -> None:
    backend = SimpleNamespace(
        model=object(),
        state=object(),
        control=object(),
    )
    monkeypatch.setattr(
        "isaaclab_neural.generate.adapter.NewtonDataGenerationBackend",
        lambda device: backend,
    )
    solver_cfg = SimpleNamespace(
        contact_mode="newton_native",
        contact_representation="contact_tokens",
        num_contacts_per_env=0,
    )

    with pytest.raises(ValueError, match="contact_tokens.*positive contact capacity"):
        DataGenerationAdapter(SimpleNamespace(device="cpu"), solver_cfg)


def test_backend_allows_missing_contacts_only_for_contact_free_solver() -> None:
    backend = NewtonDataGenerationBackend.__new__(NewtonDataGenerationBackend)
    backend.state = State()
    backend.control = Control()
    calls = []
    solver = SimpleNamespace(
        num_contacts_per_env=0,
        sync_from_newton=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    backend.sync_solver(solver, None)
    assert len(calls) == 1

    solver.num_contacts_per_env = 1
    with pytest.raises(RuntimeError, match="contacts are not initialized"):
        backend.sync_solver(solver, None)


def test_contact_free_hdf5_schema_accepts_no_contact_fields(tmp_path) -> None:
    arrays = {
        "states": torch.zeros((2, 3, 4), dtype=torch.float32),
        "next_states": torch.zeros((2, 3, 4), dtype=torch.float32),
        "joint_f": torch.zeros((2, 3, 1), dtype=torch.float32),
    }

    dataset_path = tmp_path / "contact_free.hdf5"
    write_rollouts_to_hdf5(dataset_path, arrays, env_name="Cartpole")
    assert dataset_path.is_file()


@pytest.mark.parametrize(
    "field_name",
    [
        "contact_masks",
        "contact_normals",
        "contact_depths",
        "contact_thicknesses_0",
        "contact_thicknesses_1",
        "contact_points_0",
        "contact_points_1",
        "contact_tokens",
        "contact_token_overflow",
        "contact_token_body_ids",
        "contact_token_world_ids",
    ],
)
def test_hdf5_schema_rejects_baked_contact_fields(field_name: str, tmp_path) -> None:
    arrays = {
        "states": torch.zeros((2, 3, 4), dtype=torch.float32),
        "next_states": torch.zeros((2, 3, 4), dtype=torch.float32),
        "joint_f": torch.zeros((2, 3, 1), dtype=torch.float32),
        field_name: torch.zeros((2, 3, 1), dtype=torch.float32),
    }

    with pytest.raises(ValueError, match=rf"baked contact fields are incompatible: \['{field_name}'\]"):
        write_rollouts_to_hdf5(tmp_path / "baked_contacts.hdf5", arrays, env_name="Cartpole")


def test_contact_free_training_batch_needs_no_contact_masks() -> None:
    solver = SimpleNamespace(
        process_neural_model_inputs=lambda data: None,
        convert_next_states_to_prediction=lambda **kwargs: kwargs["next_states"],
    )
    trainer = VanillaTrainer.__new__(VanillaTrainer)
    trainer.force_head_enabled = False
    trainer.device = "cpu"
    trainer.non_blocking_data_transfer = False
    trainer.neural_solver = solver
    trainer.neural_env = SimpleNamespace(frame_dt=0.02)
    data = {
        "states": torch.zeros(2, 3, 4),
        "next_states": torch.zeros(2, 3, 4),
        "joint_f": torch.zeros(2, 3, 1),
        "physics_step": torch.arange(3).reshape(1, 3, 1).expand(2, -1, -1),
    }

    result = trainer.preprocess_data_batch(data)

    assert "contact_masks" not in result
    assert "physics_step" not in result


def test_upstream_sync_attrs_exist_on_nerd_manager() -> None:
    """Fail when the NewtonManager mirror list names a removed class attribute."""
    missing = [name for name in _ManagerTestHarness.sync_attributes if not hasattr(NewtonNerdManager, name)]
    assert missing == [], f"Stale NewtonManager sync attrs: {missing}"
    _ManagerTestHarness.sync_upstream_state()


def test_nerd_manager_skips_newton_solver_internal_reset() -> None:
    """Isaac Lab 3 forward() must not call NeuralSolver.reset(state, world_mask=...)."""
    _ManagerTestHarness.reset_solver_internals(None)


def test_recurrent_solver_preserves_reset_schedule_and_validates_methods(monkeypatch) -> None:
    class RecurrentModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def init_rnn(self, batch_size):
            self.calls.append(("init", batch_size))

        def reset_rnn_hidden_states(self, batch_indices=None):
            self.calls.append(("reset", None if batch_indices is None else batch_indices.tolist()))

    solver = _RecurrentTestHarness.__new__(_RecurrentTestHarness)
    model = RecurrentModel()
    solver.neural_model = model
    solver.num_envs = 2
    solver.torch_device = "cpu"
    solver.reset_seq_length = 2
    solver.predicted_finger_object_normal_forces_b = torch.ones(2, 2, 3)
    solver.reset()
    assert torch.count_nonzero(solver.predicted_finger_object_normal_forces_b) == 0
    solver.predicted_finger_object_normal_forces_b.fill_(1.0)
    solver.reset(torch.tensor([1]))
    torch.testing.assert_close(solver.predicted_finger_object_normal_forces_b[0], torch.ones(2, 3))
    assert torch.count_nonzero(solver.predicted_finger_object_normal_forces_b[1]) == 0
    solver.reset(torch.tensor([], dtype=torch.long))
    for _ in range(3):
        solver.before_model_forward()
    assert model.calls == [("init", 2), ("reset", [1]), ("reset", None), ("reset", None)]
    monkeypatch.setattr(model, "reset_rnn_hidden_states", torch.tensor(1.0))
    with pytest.raises(RuntimeError, match="reset_rnn_hidden_states"):
        solver.reset(torch.tensor([0]))


def test_neural_solver_reset_accepts_newton_world_mask_kwargs() -> None:
    solver = NeuralSolver.__new__(NeuralSolver)
    solver.predicted_body_net_contact_forces_b = None
    solver.reset(object(), world_mask=object(), flags=0)


@pytest.mark.parametrize("prediction_type", ["relative", "absolute"])
@pytest.mark.parametrize("orientation", ["quaternion", "exponential"])
def test_prediction_to_next_states_matches_per_joint_conversion(prediction_type, orientation) -> None:
    from newton import JointType

    # Scalar joints around a FREE and a BALL joint, with a coordinate gap that must split the scalar runs.
    joints = [
        (JointType.REVOLUTE, 0, 1),
        (JointType.REVOLUTE, 1, 2),
        (JointType.FREE, 2, 9),
        (JointType.PRISMATIC, 9, 10),
        (JointType.BALL, 10, 14),
        (JointType.REVOLUTE, 14, 15),
        (JointType.REVOLUTE, 16, 17),
        (JointType.REVOLUTE, 17, 18),
    ]
    solver = NeuralSolver.__new__(NeuralSolver)
    solver.prediction_type = prediction_type
    solver.orientation_prediction_parameterization = orientation
    solver.num_joints_per_env = len(joints)
    solver.joint_types = [joint[0] for joint in joints]
    solver.joint_q_start = [joint[1] for joint in joints]
    solver.joint_q_end = [joint[2] for joint in joints]
    solver.dof_q_per_env = 18
    states = torch.randn(5, 18 + 16)
    rotation_dofs = 4 if orientation == "quaternion" else 3
    prediction = torch.randn(5, 18 - 8 + 2 * rotation_dofs - 1 + 16)

    expected = torch.empty_like(states)
    offset = 0
    for joint_type, start, end in joints:
        if joint_type == JointType.FREE:
            offset += solver._convert_prediction_to_next_states_regular_dofs(
                states[:, start : start + 3], prediction[:, offset:], expected[:, start : start + 3]
            )
            start += 3
        if joint_type in (JointType.FREE, JointType.BALL):
            offset += solver._convert_prediction_to_next_states_orientation_dofs(
                states[:, start : start + 4], prediction[:, offset:], expected[:, start : start + 4]
            )
        else:
            offset += solver._convert_prediction_to_next_states_regular_dofs(
                states[:, start:end], prediction[:, offset:], expected[:, start:end]
            )
    velocity = prediction[:, offset:]
    expected[:, 18:] = velocity if prediction_type == "absolute" else states[:, 18:] + velocity
    actual = solver.convert_prediction_to_next_states(states, prediction)
    # Coordinate 15 belongs to no joint and stays uninitialized in both conversions.
    covered = [index for index in range(states.shape[1]) if index != 15]
    assert torch.equal(actual[:, covered], expected[:, covered])


def test_reset_contacts_are_reused_only_for_an_unchanged_body_state(monkeypatch) -> None:
    import warp as wp

    from isaaclab.physics import PhysicsManager

    state = SimpleNamespace(
        body_q=wp.array(torch.randn(4, 7).numpy(), dtype=wp.float32, device="cpu"),
        body_qd=wp.array(torch.randn(4, 6).numpy(), dtype=wp.float32, device="cpu"),
    )
    collisions = []
    monkeypatch.setattr(PhysicsManager, "_device", "cpu")
    monkeypatch.setattr(NewtonNerdManager, "backend", SimpleNamespace(state_0=state), raising=False)
    monkeypatch.setattr(NewtonNerdManager, "_nerd_contact_adapter", object())
    monkeypatch.setattr(NewtonNerdManager, "_contacts", object())
    monkeypatch.setattr(
        NewtonNerdManager, "_collision_pipeline", SimpleNamespace(collide=lambda *args: collisions.append(args))
    )
    monkeypatch.setattr(NewtonNerdManager, "_reset_contact_snapshot", None)
    reuses = []
    monkeypatch.setattr(
        NewtonNerdManager, "_solver", SimpleNamespace(reuse_contact_inputs_once=lambda: reuses.append(1))
    )

    # Unchanged state after the reset sync: the next collision pass is skipped once, then runs again.
    NewtonNerdManager._record_reset_contact_snapshot()
    assert _ManagerTestHarness.prepare_contacts() is NewtonNerdManager._contacts
    assert (len(collisions), len(reuses)) == (0, 1)
    _ManagerTestHarness.prepare_contacts()
    assert (len(collisions), len(reuses)) == (1, 1)

    # Any intermediate body-state write forces recomputation.
    NewtonNerdManager._record_reset_contact_snapshot()
    wp.to_torch(state.body_qd)[2, 1] += 1e-6
    _ManagerTestHarness.prepare_contacts()
    assert (len(collisions), len(reuses)) == (2, 1)
    assert NewtonNerdManager._reset_contact_snapshot is None

    # The solver skips exactly one adapter update after reuse_contact_inputs_once().
    updates = []
    solver = NeuralSolver.__new__(NeuralSolver)
    solver.contact_adapter = SimpleNamespace(
        update=lambda contacts, newton_state: updates.append(contacts), to_neural_inputs=lambda: {"tokens": 1}
    )
    solver.root_body_ids = torch.tensor([0])
    solver.joint_f_dim = 0
    solver.states = torch.zeros(1, 2)
    solver._acquire_states_to_torch = lambda newton_state, states: None
    solver.wrap2PI = lambda states: None
    solver.reuse_contact_inputs_once()
    solver._update_states(state, "reset-contacts", None)
    solver._update_states(state, "step-contacts", None)
    assert updates == ["step-contacts"]


def test_control_step_mode_folds_decimation_into_one_prediction(monkeypatch) -> None:
    from isaaclab.physics import PhysicsManager

    monkeypatch.setattr(NewtonManager, "handles_decimation", classmethod(lambda cls: False))
    for active, step_mode, expected in (
        (True, "control_step", True),
        (True, "physics_frame", False),
        (False, "control_step", False),
    ):
        monkeypatch.setattr(NewtonNerdManager, "_nerd_active", active)
        monkeypatch.setattr(NewtonNerdManager, "_solver", SimpleNamespace(step_mode=step_mode))
        assert NewtonNerdManager.handles_decimation() is expected

    # The eager path advances one physics frame; the control-step prediction accounts for the rest.
    calls = []
    monkeypatch.setattr(NewtonNerdManager, "_nerd_active", True)
    monkeypatch.setattr(NewtonNerdManager, "_solver", SimpleNamespace(step_mode="control_step"))
    monkeypatch.setattr(NewtonNerdManager, "_decimation", 4, raising=False)
    monkeypatch.setattr(NewtonNerdManager, "_solver_dt", 0.01, raising=False)
    monkeypatch.setattr(NewtonNerdManager, "_num_substeps", 1, raising=False)
    monkeypatch.setattr(PhysicsManager, "_sim_time", 0.0)
    for name in ("_prepare_nerd_contacts", "_overlay_applied_effort_into_joint_f", "_update_sensors"):
        monkeypatch.setattr(NewtonNerdManager, name, classmethod(lambda cls, *args, _name=name: calls.append(_name)))
    monkeypatch.setattr(
        NewtonNerdManager, "_run_solver_substeps", classmethod(lambda cls, contacts: calls.append("solve"))
    )
    monkeypatch.setattr(NewtonNerdManager, "_sync_upstream_newton_state", classmethod(lambda cls: None))
    monkeypatch.setattr(NewtonNerdManager, "_post_step_callbacks", [], raising=False)
    NewtonNerdManager._simulate_physics_only()
    assert calls.count("solve") == 1
    assert PhysicsManager._sim_time == pytest.approx(0.03)


def test_joint_target_offset_is_relative_to_current_joint_coordinates() -> None:
    import numpy as np

    solver = NeuralSolver.__new__(NeuralSolver)
    solver.dof_q_per_env = 2
    solver.states_frame = "world"
    solver.states_embedding_type = None
    solver.is_continuous_dof = np.zeros(4, dtype=bool)
    states = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    inputs = {
        "root_body_q": torch.zeros(1, 1, 7),
        "states": states.clone(),
        "states_embedding": torch.zeros(1, 1, 4),
        "joint_target_q": torch.tensor([[[1.5, 1.0]]]),
        "gravity_dir": torch.zeros(1, 1, 3),
    }
    processed = solver.process_neural_model_inputs(inputs)
    assert "joint_target_q" not in processed
    torch.testing.assert_close(processed["joint_target_offset"], torch.tensor([[[0.5, -1.0]]]))
    torch.testing.assert_close(processed["states"], states)
