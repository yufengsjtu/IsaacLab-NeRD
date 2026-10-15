# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The simulator-free task registry and the environments it names."""

from __future__ import annotations

import gymnasium as gym
import isaaclab_neural.envs  # noqa: F401
import pytest
from isaaclab_neural.tasks import TASK_SPECS, NerdTaskSpec, expand_experiment, get_task_spec, register_task_spec
from isaaclab_neural.tasks.specs import AUXILIARY_LABELS

REPOSITORY_ROOT = __import__("pathlib").Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("name", sorted(TASK_SPECS))
def test_registered_specs_name_existing_tasks_and_teachers(name) -> None:
    spec = get_task_spec(name)
    for task in (spec.policy_task, spec.record_task, spec.nerd_task):
        assert "rsl_rl_cfg_entry_point" in gym.spec(task).kwargs, task
    if spec.teacher_preset is None or (REPOSITORY_ROOT / spec.teacher_checkpoint).is_file():
        assert (REPOSITORY_ROOT / spec.teacher_checkpoint).is_file(), spec.teacher_checkpoint
    if spec.teacher_preset is not None:
        assert (REPOSITORY_ROOT / "osmo_scripts/presets" / f"{spec.teacher_preset}.yaml").is_file()
    assert set(spec.auxiliary_labels) <= set(AUXILIARY_LABELS)


def test_record_twins_step_every_physics_frame() -> None:
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    for spec in TASK_SPECS.values():
        if spec.record_task == spec.policy_task or spec.pipeline != "env_driven":
            continue
        cfg = load_cfg_from_registry(spec.record_task, "env_cfg_entry_point")
        physics = cfg.sim.physics
        assert getattr(physics, "newton_mjwarp", physics).use_cuda_graph is False, spec.name
        if spec.name != "franka_lift":
            assert cfg.sim.use_newton_actuators is False, spec.name


def test_expand_experiment_fills_defaults_without_overriding_preset_keys() -> None:
    expanded = expand_experiment(
        {
            "task_spec": "anymal_c_rough",
            "env_name": "Custom",
            "policy_evaluation": {"num_envs": 8},
            "throughput_benchmark": {"num_envs": [16]},
        }
    )
    spec = get_task_spec("anymal_c_rough")
    assert expanded["env_name"] == "Custom"
    assert expanded["data_gen_task"] == spec.record_task and expanded["train_task"] == spec.nerd_task
    assert expanded["record_body_net_contact_forces"] is True
    assert expanded["policy_evaluation"]["args"] == spec.evaluation_args()
    assert expanded["policy_evaluation"]["reference"]["metric"] == spec.selection_metric
    assert expanded["throughput_benchmark"]["env_args"] == spec.benchmark_env_args()
    assert expand_experiment({"env_name": "x"}) == {"env_name": "x"}
    assert get_task_spec("franka_reach").collector_args()[-2:] == ["--max-contact-tokens", "0"]


def test_register_task_spec_rejects_duplicates_and_unknown_labels() -> None:
    spec = get_task_spec("cartpole")
    with pytest.raises(ValueError, match="already registered"):
        register_task_spec(spec)
    with pytest.raises(ValueError, match="unknown auxiliary labels"):
        NerdTaskSpec(
            name="x",
            env_name="x",
            policy_task="x",
            record_task="x",
            nerd_task="x",
            teacher_checkpoint="x",
            auxiliary_labels=("torque",),
        )
    with pytest.raises(KeyError, match="Unknown task spec"):
        get_task_spec("missing")
