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
