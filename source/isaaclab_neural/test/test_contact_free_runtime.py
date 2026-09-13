# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for contact-free NeRD runtime setup."""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace
from typing import cast

import isaaclab_neural.solvers
import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_neural.data.hdf5 import _validate_rollout_arrays
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
from isaaclab_neural.train.trainers import VanillaTrainer
from isaaclab_newton.physics.newton_manager import NewtonManager
from newton import Control, Model, State


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

    NewtonNerdManager._build_solver(
        model=Model(device="cpu"),
        solver_cfg=NerdSolverCfg(contact_mode="newton_native", num_contacts_per_env=0),
    )

    assert NewtonManager._solver is fake_solver
    assert NewtonManager._needs_collision_pipeline is False
    assert NewtonManager._collision_pipeline is None
    assert NewtonNerdManager._nerd_contact_adapter is None
    assert captured_kwargs["contacts"] is None
    assert captured_kwargs["contact_adapter"] is None
    assert NewtonNerdManager._prepare_nerd_contacts() is None
    assert installed_models == [fake_neural_model]


def test_runtime_requires_explicit_dynamics_checkpoint() -> None:
    with pytest.raises(ValueError, match="explicit neural_model_path"):
        NewtonNerdManager._load_nerd_neural_model({"neural_model_path": None})


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

    with NewtonNerdManager._defer_neural_model_loading():
        NewtonNerdManager._build_solver(
            model=Model(device="cpu"),
            solver_cfg=NerdSolverCfg(
                contact_mode="newton_native",
                num_contacts_per_env=0,
            ),
        )

    assert NewtonManager._solver is fake_solver
    assert installed_models == []
    assert NewtonNerdManager._neural_model_loading_deferred is False


def test_cartpole_uses_contact_free_solver_config() -> None:
    cfg = NerdCartpoleEnvCfg()

    assert isinstance(cfg.sim.physics, NerdNewtonCfg)
    assert cfg.sim.physics.solver_cfg.contact_mode == "newton_native"
    assert cfg.sim.physics.solver_cfg.num_contacts_per_env == 0
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
        assert isinstance(cfg.sim.physics, NerdNewtonCfg)
        assert cfg.sim.physics.solver_cfg.contact_mode == "newton_native"
        assert cfg.sim.physics.solver_cfg.num_contacts_per_env > 0
        assert cfg.sim.physics.default_shape_cfg.margin == 0.0
    assert rough_cfg.terminations.base_contact is None
    assert rough_cfg.rewards.track_lin_vel_xy_exp.weight == 1.0
    assert rough_cfg.rewards.track_ang_vel_z_exp.weight == 0.5


class _ContactFreeAdapter:
    num_envs = 2
    num_contacts_per_env = 0
    state_dim = 4
    joint_f_dim = 1
    state_world_ids = torch.arange(2)
    root_world_ids = torch.arange(2)
    contact_world_ids = torch.arange(2)
    solver = SimpleNamespace(contact_representation="flat")
    model = SimpleNamespace(up_axis=2)
    env = SimpleNamespace(scene=None)

    def raw_neural_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "states": torch.zeros(2, 1, 4),
            "root_body_q": torch.zeros(2, 1, 7),
            "root_body_qd": torch.zeros(2, 1, 6),
        }


def test_contact_free_sampler_omits_contact_tensors() -> None:
    sampler = ActionTrajectorySampler.__new__(ActionTrajectorySampler)
    sampler.adapter = cast(DataGenerationAdapter, _ContactFreeAdapter())
    sampler.trajectory_length = 3
    sampler.data_device = torch.device("cpu")

    buffers = sampler.allocate_batch_buffers(record_actions=False)
    sampler._copy_before_step(buffers, step=0)

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
    backend.control.joint_f = wp.zeros(1, dtype=wp.float32, device="cpu")
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


def test_contact_free_hdf5_schema_accepts_no_contact_fields() -> None:
    arrays = {
        "states": np.zeros((2, 3, 4), dtype=np.float32),
        "next_states": np.zeros((2, 3, 4), dtype=np.float32),
        "joint_f": np.zeros((2, 3, 1), dtype=np.float32),
    }

    _validate_rollout_arrays(arrays)


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
def test_hdf5_schema_rejects_baked_contact_fields(field_name: str) -> None:
    arrays = {
        "states": np.zeros((2, 3, 4), dtype=np.float32),
        "next_states": np.zeros((2, 3, 4), dtype=np.float32),
        "joint_f": np.zeros((2, 3, 1), dtype=np.float32),
        field_name: np.zeros((2, 3, 1), dtype=np.float32),
    }

    with pytest.raises(ValueError, match=rf"baked contact fields are incompatible: \['{field_name}'\]"):
        _validate_rollout_arrays(arrays)


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
    }

    result = trainer.preprocess_data_batch(data)

    assert "contact_masks" not in result


def test_upstream_sync_attrs_exist_on_nerd_manager() -> None:
    """Fail when the NewtonManager mirror list names a removed class attribute."""
    missing = [name for name in NewtonNerdManager._UPSTREAM_SYNC_ATTRS if not hasattr(NewtonNerdManager, name)]
    assert missing == [], f"Stale NewtonManager sync attrs: {missing}"
    NewtonNerdManager._sync_upstream_newton_state()


def test_nerd_manager_skips_newton_solver_internal_reset() -> None:
    """Isaac Lab 3 forward() must not call NeuralSolver.reset(state, world_mask=...)."""
    NewtonNerdManager._reset_solver_internals(wp.array([True], dtype=wp.bool, device="cpu"))


def test_neural_solver_reset_accepts_newton_world_mask_kwargs() -> None:
    solver = NeuralSolver.__new__(NeuralSolver)
    solver.predicted_body_net_contact_forces_b = None
    solver.reset(object(), world_mask=object(), flags=0)
