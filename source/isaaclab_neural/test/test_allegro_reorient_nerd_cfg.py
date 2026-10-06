# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config-level tests for the Allegro cube-reorientation NeRD tasks."""

from __future__ import annotations

import gymnasium as gym
import isaaclab_neural.envs  # noqa: F401
import pytest
from isaaclab_neural.envs.allegro_reorient_nerd_env import (
    ALLEGRO_REORIENT_NCONMAX,
    ALLEGRO_REORIENT_NJMAX,
    ALLEGRO_REORIENT_NUM_CONTACTS,
    AllegroReorientDatasetGenEnvCfg,
    AllegroReorientPolicyEnvCfg,
    NerdAllegroReorientEnvCfg,
)
from isaaclab_neural.eval.allegro_reorient_policy import summarize_reorient_episodes
from isaaclab_neural.physics import NerdNewtonCfg
from isaaclab_newton.physics import NewtonCollisionPipelineCfg

from isaaclab_tasks.core.reorient.config.allegro_hand.allegro_hand_common import PhysicsCfg as AllegroPhysicsCfg
from isaaclab_tasks.core.reorient.config.allegro_hand.allegro_hand_manager_env_cfg import AllegroHandManagerEnvCfg
from isaaclab_tasks.utils.hydra import collect_presets


@pytest.mark.parametrize("cfg_type", [AllegroReorientPolicyEnvCfg, AllegroReorientDatasetGenEnvCfg])
def test_allegro_policy_and_dataset_share_newton_contact_dynamics(cfg_type) -> None:
    cfg = cfg_type()
    core = AllegroHandManagerEnvCfg()
    core_solver = core.sim.physics.newton_mjwarp.solver_cfg

    assert isinstance(cfg.sim.physics, AllegroPhysicsCfg)
    newton_cfg = cfg.sim.physics.newton_mjwarp
    assert cfg.sim.physics.default is newton_cfg
    assert newton_cfg.solver_cfg.use_mujoco_contacts is False
    assert isinstance(newton_cfg.collision_cfg, NewtonCollisionPipelineCfg)
    assert newton_cfg.solver_cfg.njmax == ALLEGRO_REORIENT_NJMAX > core_solver.njmax
    assert newton_cfg.solver_cfg.nconmax == ALLEGRO_REORIENT_NCONMAX > core_solver.nconmax
    for field in ("integrator", "cone", "impratio"):
        assert getattr(newton_cfg.solver_cfg, field) == getattr(core_solver, field), field
    assert newton_cfg.num_substeps == core.sim.physics.newton_mjwarp.num_substeps
    for field in ("observations", "actions", "commands", "rewards", "terminations", "events"):
        assert getattr(cfg, field).to_dict() == getattr(core, field).to_dict(), field
    assert cfg.sim.dt == core.sim.dt
    assert cfg.decimation == core.decimation
    assert newton_cfg.use_cuda_graph is (cfg_type is AllegroReorientPolicyEnvCfg)


def test_allegro_nerd_predicts_hand_and_cube_with_teacher_collision() -> None:
    cfg = NerdAllegroReorientEnvCfg()
    teacher = AllegroReorientPolicyEnvCfg().sim.physics.newton_mjwarp

    assert isinstance(cfg.sim.physics, NerdNewtonCfg)
    solver_cfg = cfg.sim.physics.solver_cfg
    assert solver_cfg.contact_mode == "newton_native"
    assert solver_cfg.contact_representation == "contact_tokens"
    assert solver_cfg.num_contacts_per_env == solver_cfg.max_contact_tokens == ALLEGRO_REORIENT_NUM_CONTACTS
    assert solver_cfg.states_frame == "body"
    assert solver_cfg.orientation_prediction_parameterization == "quaternion"
    assert cfg.sim.physics.collision_cfg.to_dict() == teacher.collision_cfg.to_dict()
    assert cfg.sim.physics.num_substeps == 1
    preset_names = {name for fields in collect_presets(cfg).values() for name in fields}
    assert "newton_mjwarp" in preset_names


def test_allegro_gym_ids_are_registered() -> None:
    for env_id in (
        "Isaac-Reorient-Cube-Allegro-Policy-v0",
        "Isaac-Reorient-Cube-Allegro-Dataset-Gen-v0",
        "Isaac-Reorient-Cube-Allegro-NeRD-v0",
    ):
        assert gym.spec(env_id).kwargs["default_agent"] == "rsl_rl"


def test_summarize_reorient_episodes_counts_goals_and_drops() -> None:
    episodes = [
        {"goals": 3, "dropped": False},
        {"goals": 0, "dropped": True},
        {"goals": 6, "dropped": True},
    ]

    summary = summarize_reorient_episodes(episodes)

    assert summary == {"episodes": 3, "mean_goals_per_episode": 3.0, "drops": 2, "drop_rate": 2 / 3}
    assert summarize_reorient_episodes([]) == {
        "episodes": 0,
        "mean_goals_per_episode": None,
        "drops": 0,
        "drop_rate": None,
    }


def test_allegro_training_eval_uses_closed_loop_policy_mode(tmp_path) -> None:
    from isaaclab_neural.train.trainers import VanillaTrainer

    trainer = object.__new__(VanillaTrainer)
    with pytest.raises(FileNotFoundError):
        trainer._init_evaluator(
            {"eval": {"interval": 5, "mode": "allegro_policy", "policy_checkpoint": str(tmp_path / "missing.pt")}}, {}
        )
    policy = tmp_path / "policy.pt"
    policy.touch()
    trainer._init_evaluator(
        {"eval": {"mode": "allegro_policy", "policy_checkpoint": str(policy)}}, {"eval_interval": 5}
    )
    assert trainer.eval_interval == 5 and trainer.evaluator is None and trainer.keyboard_policy_eval is None
    assert trainer.allegro_policy_eval == {"checkpoint": policy, "control_steps": 600, "seed": 43}
    trainer._init_evaluator({"eval": {"interval": 5, "mode": "allegro_policy"}}, {"eval_interval": 0})
    assert trainer.allegro_policy_eval is None


def _recording(path, contract=None):
    import json

    import h5py

    with h5py.File(path, "w") as file:
        group = file.create_group("data")
        if contract is not None:
            group.attrs["recording_contract"] = json.dumps(contract)
    return str(path)


def test_recording_contract_binds_mobility_and_step_mode(tmp_path):
    from types import SimpleNamespace

    import h5py
    from isaaclab_neural.data.native_contacts import read_contact_mobility
    from isaaclab_neural.generate.allegro_dataset import RECORDING_CONTRACT_ATTR
    from isaaclab_neural.train.trainers import VanillaTrainer

    contract = {"step_mode": "control_step", "contact_filter": "solver_active_movable", "contact_fixed_body_slots": [0]}
    recorded = _recording(tmp_path / "recorded.hdf5", contract)
    assert RECORDING_CONTRACT_ATTR == "recording_contract"
    with h5py.File(recorded, "r") as file:
        assert read_contact_mobility(file["data"], "solver_active_movable") == (0,)

    def trainer(**solver):
        solver = {"step_mode": "control_step", "joint_target_input": True, "contact_filter": "none", **solver}
        return SimpleNamespace(neural_solver=SimpleNamespace(contact_adapter=SimpleNamespace(), **solver))

    validate = VanillaTrainer._validate_recording_contracts
    generic = _recording(tmp_path / "generic.hdf5")
    cfg = {"train_datasets": {"a": {"path": recorded}}, "valid_datasets": {"v": generic}}
    validate(trainer(), cfg)
    with pytest.raises(ValueError, match="predicts 'physics_frame'"):
        validate(trainer(step_mode="physics_frame"), cfg)
    with pytest.raises(ValueError, match="joint_target_input"):
        validate(trainer(joint_target_input=False), cfg)
    movable = {"contact_filter": "solver_active_movable"}
    with pytest.raises(ValueError, match="body mobility"):
        validate(SimpleNamespace(neural_solver=SimpleNamespace(**{**vars(trainer().neural_solver), **movable})), cfg)
    solver = SimpleNamespace(
        **{**vars(trainer().neural_solver), **movable, "contact_adapter": SimpleNamespace(fixed_body_slots=(0,))}
    )
    validate(SimpleNamespace(neural_solver=solver), cfg)
    # Recordings without a contract (generic generator, Keyboard) are left to their own validators.
    validate(trainer(step_mode="physics_frame"), {"train_dataset_path": generic})
