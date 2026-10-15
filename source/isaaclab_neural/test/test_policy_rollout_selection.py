# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Closed-loop policy-rollout evaluation used by the control-step ANYmal and Franka Lift recipes."""

from __future__ import annotations

import math

import gymnasium as gym
import isaaclab_neural.envs  # noqa: F401
import pytest
from isaaclab_neural.envs.anymal_nerd_env import (
    AnymalCFlatControlStepDatasetGenEnvCfg,
    AnymalCFlatPolicyEnvCfg,
    NerdAnymalCFlatControlStepEnvCfg,
    NerdAnymalCFlatEnvCfg,
)
from isaaclab_neural.eval.policy_rollout import selection_error

from isaaclab_assets.robots.anymal import ANYDRIVE_3_SIMPLE_ACTUATOR_CFG


def test_selection_prefers_the_checkpoint_closest_to_the_native_reference() -> None:
    summary = {"finite": True, "velocity_tracking_error": 0.30}
    assert selection_error(summary, {"metric": "velocity_tracking_error", "reference": 0.25}) == pytest.approx(0.05)
    # An optimistic model below the native error is not preferred over an equally distant pessimistic one.
    optimistic = {"finite": True, "velocity_tracking_error": 0.20}
    assert selection_error(optimistic, {"metric": "velocity_tracking_error", "reference": 0.25}) == pytest.approx(0.05)
    assert selection_error(summary, {"metric": "velocity_tracking_error"}) == pytest.approx(0.30)
    rate = {"finite": True, "pose_success_rate": 0.4}
    assert selection_error(rate, {"metric": "pose_success_rate", "higher_is_better": True}) == pytest.approx(-0.4)
    assert math.isinf(selection_error({**summary, "finite": False}, {"metric": "velocity_tracking_error"}))
    assert math.isinf(selection_error({"finite": True}, {"metric": "velocity_tracking_error"}))


def test_training_eval_parses_policy_rollout_mode(tmp_path) -> None:
    from isaaclab_neural.train.trainers import VanillaTrainer

    policy = tmp_path / "policy.pt"
    policy.touch()
    eval_cfg = {
        "interval": 5,
        "mode": "policy_rollout",
        "policy_checkpoint": str(policy),
        "policy_task": "Isaac-Velocity-Flat-Anymal-C-Policy-v0",
        "metrics": [{"name": "velocity_tracking_error"}],
        "selection": {"metric": "velocity_tracking_error", "reference": 0.2},
    }
    trainer = object.__new__(VanillaTrainer)
    trainer._init_evaluator({"eval": eval_cfg}, {})
    assert trainer.evaluator is None and trainer.allegro_policy_eval is None
    assert trainer.task_policy_eval["selection"] == {"metric": "velocity_tracking_error", "reference": 0.2}
    assert trainer.task_policy_eval["control_steps"] == 500
    with pytest.raises(ValueError, match="selection.metric"):
        trainer._init_evaluator({"eval": {**eval_cfg, "selection": {"metric": "goals"}}}, {})
    with pytest.raises(FileNotFoundError):
        trainer._init_evaluator({"eval": {**eval_cfg, "policy_checkpoint": str(tmp_path / "missing.pt")}}, {})


def test_anymal_control_step_envs_share_the_stateless_actuator_and_contacts() -> None:
    policy = AnymalCFlatPolicyEnvCfg()
    nerd = NerdAnymalCFlatControlStepEnvCfg()
    for cfg in (policy, nerd):
        actuator = cfg.scene.robot.actuators["legs"]
        assert type(actuator) is type(ANYDRIVE_3_SIMPLE_ACTUATOR_CFG)
        assert actuator.armature == NerdAnymalCFlatEnvCfg().scene.robot.actuators["legs"].armature
        assert cfg.events.push_robot is None
    assert policy.observations.policy.to_dict() == nerd.observations.policy.to_dict()
    newton_cfg = policy.sim.physics.newton_mjwarp
    assert policy.sim.physics.default is newton_cfg
    assert newton_cfg.solver_cfg.use_mujoco_contacts is False
    assert newton_cfg.default_shape_cfg.margin == nerd.sim.physics.default_shape_cfg.margin == 0.0
    assert newton_cfg.collision_cfg.max_triangle_pairs == nerd.sim.physics.collision_cfg.max_triangle_pairs
    assert (policy.sim.dt, policy.decimation) == (nerd.sim.dt, nerd.decimation)
    # The recording twin must expose every physics frame to the recorder.
    recording = AnymalCFlatControlStepDatasetGenEnvCfg()
    assert recording.sim.physics.newton_mjwarp.use_cuda_graph is False
    assert recording.sim.use_newton_actuators is False and policy.sim.use_newton_actuators is True
    assert recording.scene.robot.actuators["legs"].to_dict() == policy.scene.robot.actuators["legs"].to_dict()
    for env_id in (
        "Isaac-Velocity-Flat-Anymal-C-Policy-v0",
        "Isaac-Velocity-Flat-Anymal-C-ControlStep-Dataset-Gen-v0",
        "Isaac-Velocity-Flat-Anymal-C-ControlStep-NeRD-v0",
        "Isaac-Lift-Cube-Franka-Policy-v0",
    ):
        assert "rsl_rl_cfg_entry_point" in gym.spec(env_id).kwargs
