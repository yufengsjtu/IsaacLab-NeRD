# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task specifications shared by the env-driven collector, closed-loop evaluation, benchmarks and OSMO presets.

A :class:`NerdTaskSpec` names the four registered environments of one task and the defaults every tool needs:

* ``policy_task``: native MJWarp task with CUDA graphs, used for the teacher's runner config, native closed-loop
  evaluation and throughput baselines.
* ``record_task``: native twin that advances every physics frame through Isaac Lab so the env-driven collector can
  observe it (no CUDA graph, no fused Newton actuator loop).
* ``nerd_task``: the same task with NeRD dynamics.
* ``env_name``: dataset environment name, i.e. the directory under the dataset root.

Tools accept ``--task-spec NAME`` and presets ``experiment.task_spec: NAME``; explicit arguments and preset keys
override the spec. The module imports only the standard library so OSMO launchers and tests can read it without a
simulator. Register new tasks with :func:`register_task_spec`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

AUXILIARY_LABELS = ("finger_object_normal_forces", "body_net_contact_forces", "robot_normal_forces")
"""Auxiliary force labels a recipe can record and predict with an auxiliary head."""


@dataclass(frozen=True)
class NerdTaskSpec:
    """Environments, teacher and evaluation defaults of one NeRD task."""

    name: str
    """Spec name, e.g. ``"anymal_c_rough"``."""
    env_name: str
    """Dataset environment name."""
    policy_task: str
    """Graph-enabled native task (teacher runner config, native evaluation and baselines)."""
    record_task: str
    """Native twin stepped frame by frame for env-driven collection."""
    nerd_task: str
    """NeRD task."""
    teacher_checkpoint: str
    """Packaged RSL-RL teacher checkpoint, relative to the repository root."""
    teacher_preset: str | None = None
    """OSMO preset that trains the teacher, when it is not an upstream checkpoint."""
    max_contact_tokens: int = 64
    """Contact-token capacity; ``0`` marks a contact-free task."""
    frames_per_control_step: int = 4
    """Task decimation."""
    num_bodies: int = 1
    """Approximate primary contact bodies, sizing synthetic inputs of model-only timing."""
    metrics: tuple[dict[str, Any], ...] = ()
    """Closed-loop task metrics (see :mod:`isaaclab_neural.eval.policy_rollout`)."""
    selection_metric: str = "reward_per_step"
    """Closed-loop metric whose distance to the native value selects ``best_eval_model``."""
    higher_is_better: bool = True
    """Orientation of ``selection_metric`` when no native reference is set."""
    auxiliary_labels: tuple[str, ...] = ()
    """Auxiliary force labels recorded for this task (subset of :data:`AUXILIARY_LABELS`)."""
    pipeline: str = "env_driven"
    """``"env_driven"`` for the generic collector, ``"so101_keyboard"`` for the Keyboard pipeline."""
    notes: str = ""
    """Task-specific caveats."""
    extra: dict[str, Any] = field(default_factory=dict)
    """Free-form task data for new tools."""

    def __post_init__(self) -> None:
        unknown = sorted(set(self.auxiliary_labels) - set(AUXILIARY_LABELS))
        if unknown:
            raise ValueError(f"{self.name}: unknown auxiliary labels {unknown}; expected {AUXILIARY_LABELS}.")
        if self.max_contact_tokens < 0 or self.frames_per_control_step <= 0:
            raise ValueError(f"{self.name}: contact capacity and decimation must be non-negative and positive.")

    @property
    def contact_free(self) -> bool:
        """Whether the task records no contacts."""
        return self.max_contact_tokens == 0

    def collector_args(self) -> list[str]:
        """Default env-driven collector arguments implied by the spec."""
        args = ["--task", self.record_task, "--env-name", self.env_name, "--policy-task", self.policy_task]
        args += ["--max-contact-tokens", str(self.max_contact_tokens)]
        if "body_net_contact_forces" in self.auxiliary_labels:
            args.append("--record-body-net-contact-forces")
        return args

    def evaluation_args(self) -> list[str]:
        """Default closed-loop evaluation arguments (native task and metric names)."""
        args = ["--task", self.policy_task]
        for metric in self.metrics:
            args += ["--metric", str(metric["name"])]
        return args

    def benchmark_env_args(self) -> list[str]:
        """Default end-to-end benchmark arguments."""
        return ["--task", self.policy_task, "--nerd-task", self.nerd_task]

    def benchmark_model_args(self) -> list[str]:
        """Default model-only benchmark arguments."""
        return ["--num-bodies", str(self.num_bodies), "--frames-per-control-step", str(self.frames_per_control_step)]


TASK_SPECS: dict[str, NerdTaskSpec] = {}
"""Registered task specifications by name."""


def register_task_spec(spec: NerdTaskSpec, *, replace: bool = False) -> NerdTaskSpec:
    """Register ``spec`` under its name; set ``replace`` to redefine an existing task."""
    if spec.name in TASK_SPECS and not replace:
        raise ValueError(f"Task spec {spec.name!r} is already registered.")
    TASK_SPECS[spec.name] = spec
    return spec


def get_task_spec(name: str) -> NerdTaskSpec:
    """Return the registered spec ``name``."""
    try:
        return TASK_SPECS[name]
    except KeyError as error:
        raise KeyError(f"Unknown task spec {name!r}; registered: {sorted(TASK_SPECS)}.") from error


def expand_experiment(experiment: dict[str, Any]) -> dict[str, Any]:
    """Fill a preset ``experiment`` from its ``task_spec`` without overriding keys the preset sets.

    Fills the task ids, dataset environment name, teacher checkpoint, contact capacity, auxiliary recording flags and,
    for ``policy_evaluation`` and ``throughput_benchmark`` blocks, their task arguments and the native reference.
    """
    name = experiment.get("task_spec")
    if not name:
        return experiment
    spec = get_task_spec(str(name))
    result = dict(experiment)
    defaults = {
        "env_name": spec.env_name,
        "robot_name": spec.env_name,
        "data_gen_task": spec.record_task,
        "deployment_task": spec.record_task,
        "policy_task": spec.policy_task,
        "train_task": spec.nerd_task,
        "policy_checkpoint": f"./{spec.teacher_checkpoint}",
        "max_contact_tokens": spec.max_contact_tokens,
        "num_contacts_per_env": spec.max_contact_tokens,
        "dataset_collector": "env_driven",
    }
    if "body_net_contact_forces" in spec.auxiliary_labels:
        defaults["record_body_net_contact_forces"] = True
    for key, value in defaults.items():
        result.setdefault(key, value)
    evaluation = result.get("policy_evaluation")
    if isinstance(evaluation, dict):
        evaluation = dict(evaluation)
        evaluation.setdefault("module", "isaaclab_neural.eval.policy_rollout")
        evaluation.setdefault("args", spec.evaluation_args())
        evaluation.setdefault(
            "reference", {"metric": spec.selection_metric, "cfg_key": "algorithm.eval.selection.reference"}
        )
        result["policy_evaluation"] = evaluation
    bench = result.get("throughput_benchmark")
    if isinstance(bench, dict):
        bench = dict(bench)
        bench.setdefault("module", "isaaclab_neural.eval.task_benchmark")
        bench.setdefault("env_args", spec.benchmark_env_args())
        bench.setdefault("model_args", spec.benchmark_model_args())
        result["throughput_benchmark"] = bench
    return result


_LOCOMOTION_METRICS = ({"name": "velocity_tracking_error", "command": "base_velocity"},)

register_task_spec(
    NerdTaskSpec(
        name="cartpole",
        env_name="Cartpole-ControlStep",
        policy_task="Isaac-Cartpole-Policy-v0",
        record_task="Isaac-Cartpole-ControlStep-Dataset-Gen-v0",
        nerd_task="Isaac-Cartpole-NeRD-v0",
        teacher_checkpoint="pretrained/control_policy/cartpole_control_step/model_149.pt",
        teacher_preset="cartpole_teacher",
        max_contact_tokens=0,
        frames_per_control_step=2,
        num_bodies=2,
        selection_metric="reward_per_step",
        notes="Contact-free; the pole falls are terminations, so the failure rate is reported as well.",
    )
)
register_task_spec(
    NerdTaskSpec(
        name="anymal_c_flat",
        env_name="Anymal-C-Flat-ControlStep",
        policy_task="Isaac-Velocity-Flat-Anymal-C-Policy-v0",
        record_task="Isaac-Velocity-Flat-Anymal-C-ControlStep-Dataset-Gen-v0",
        nerd_task="Isaac-Velocity-Flat-Anymal-C-ControlStep-NeRD-v0",
        teacher_checkpoint="pretrained/control_policy/anymal_c_flat_control_step/model_999.pt",
        teacher_preset="anymal_flat_teacher",
        num_bodies=17,
        metrics=_LOCOMOTION_METRICS,
        selection_metric="velocity_tracking_error",
        higher_is_better=False,
        notes="Stateless ANYdrive actuator: a control-step model evaluates the actuator once per control step.",
    )
)
register_task_spec(
    NerdTaskSpec(
        name="anymal_c_rough",
        env_name="Anymal-C-Rough-ControlStep",
        policy_task="Isaac-Velocity-Rough-Anymal-C-Policy-v0",
        record_task="Isaac-Velocity-Rough-Anymal-C-ControlStep-Dataset-Gen-v0",
        nerd_task="Isaac-Velocity-Rough-Anymal-C-ControlStep-NeRD-v0",
        teacher_checkpoint="pretrained/control_policy/anymal_c_rough_control_step/model_1499.pt",
        teacher_preset="anymal_rough_teacher",
        num_bodies=17,
        metrics=_LOCOMOTION_METRICS,
        selection_metric="velocity_tracking_error",
        higher_is_better=False,
        auxiliary_labels=("body_net_contact_forces",),
        notes="Stateless ANYdrive actuator; per-body net contact forces are an auxiliary prediction target.",
    )
)
register_task_spec(
    NerdTaskSpec(
        name="franka_reach",
        env_name="Franka-Reach-ControlStep",
        policy_task="Isaac-Reach-Franka-Policy-v0",
        record_task="Isaac-Reach-Franka-ControlStep-Dataset-Gen-v0",
        nerd_task="Isaac-Reach-Franka-NeRD-v0",
        teacher_checkpoint="pretrained/control_policy/franka_reach_menagerie/model_999.pt",
        max_contact_tokens=0,
        frames_per_control_step=2,
        num_bodies=11,
        metrics=({"name": "command_error", "command": "ee_pose", "key": "position_error"},),
        selection_metric="command_error",
        higher_is_better=False,
        notes="Contact-free Reach scene with the Menagerie Panda.",
    )
)
register_task_spec(
    NerdTaskSpec(
        name="franka_lift",
        env_name="Franka-Lift-ControlStep",
        policy_task="Isaac-Lift-Cube-Franka-Policy-v0",
        record_task="Isaac-Lift-Cube-Franka-Dataset-Gen-v0",
        nerd_task="Isaac-Lift-Cube-Franka-NeRD-v0",
        teacher_checkpoint="pretrained/official/Isaac-Lift-Franka_newtonmjwarp_none_rsl_rl.pt",
        num_bodies=12,
        metrics=(
            {"name": "pose_success_rate", "command": "object_pose", "position_threshold": 0.05, "orientation_threshold": 0.5},
        ),
        selection_metric="pose_success_rate",
        auxiliary_labels=("finger_object_normal_forces",),
        notes="The finger-object normal-force head feeds the Lift policy's contact observations.",
    )
)
register_task_spec(
    NerdTaskSpec(
        name="allegro_reorient",
        env_name="Allegro-Reorient",
        policy_task="Isaac-Reorient-Cube-Allegro-Policy-v0",
        record_task="Isaac-Reorient-Cube-Allegro-Policy-v0",
        nerd_task="Isaac-Reorient-Cube-Allegro-NeRD-v0",
        teacher_checkpoint="pretrained/control_policy/allegro_reorient/model_4999.pt",
        teacher_preset="allegro_reorient_teacher",
        max_contact_tokens=128,
        num_bodies=22,
        selection_metric="mean_goals_per_episode",
        notes="Closed-loop evaluation uses isaaclab_neural.eval.allegro_reorient_policy (goals and drops).",
    )
)
register_task_spec(
    NerdTaskSpec(
        name="so101_keyboard",
        env_name="SO101-Keyboard-Fixed",
        policy_task="IsaacContrib-Keyboard-SO101",
        record_task="IsaacContrib-Keyboard-SO101",
        nerd_task="IsaacContrib-Keyboard-SO101-NeRD",
        teacher_checkpoint="pretrained/control_policy/so101_keyboard/model_14999.pt",
        max_contact_tokens=128,
        num_bodies=115,
        selection_metric="clean_success_rate",
        auxiliary_labels=("robot_normal_forces",),
        pipeline="so101_keyboard",
        notes="Uses the Keyboard collector, replay validation and evaluator (dataset_profile: so101_keyboard).",
    )
)
