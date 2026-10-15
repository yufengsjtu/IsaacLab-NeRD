# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect env-driven NeRD records with the SO101 Keyboard recording contract (Allegro, ANYmal, Franka Lift).

Run with ``python -m isaaclab_neural.generate.allegro_dataset``. Unlike the generic generator, the
native task steps with its own actions, rewards, terminations and auto-resets: records are observed
after each physics-frame scene update and cut into fixed-length clips that never cross a reset, so
dropped cubes or fallen robots end their episodes instead of filling trajectories.
``--record-control-steps`` stores one record per control step for ``step_mode="control_step"`` models,
with the boundary state, the first frame's applied effort and joint targets as inputs and the last
frame's state as the label. Tasks with ``finger_object_normal_force_sensors`` also record the
post-record filtered finger normal forces. Allegro remains the default task; other tasks pass
``--task``, ``--env-name`` and, when the task registers no RSL-RL config, ``--policy-task``.

Chunks are appended to one HDF5 file whose ``data`` group carries a ``recording_contract`` with the
recording identity, step mode, timing and contact mobility, as unique-transition, sliding-window and
GPU-resident training expect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from importlib.metadata import version
from pathlib import Path
from typing import Any

import h5py
import torch

ALLEGRO_RECORDING_SCHEMA = "allegro_reorient_recording_v1"
"""Recording-contract schema written for the Allegro dataset environment."""

ENV_DRIVEN_RECORDING_SCHEMA = "env_driven_recording_v1"
"""Recording-contract schema written for every other dataset environment."""

FINGER_FORCE_KEY = "finger_object_normal_forces_w"
"""Filtered finger-object normal-force label key [N], shape ``[N, 2, 3]`` per record."""

RECORDING_CONTRACT_ATTR = "recording_contract"
"""HDF5 ``data`` attribute holding the JSON recording contract."""

ALLEGRO_ENV_NAME = "Allegro-Reorient"
"""Dataset environment name shared with the Allegro training recipes."""

COLLECTION_MODES = ("policy", "zero", "random")
"""Action sources: the teacher (optionally with Gaussian noise), zero actions or held uniform actions."""


def _cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to("cpu").clone()


class AllegroFrameRecorder:
    """Observe native scene updates of the Allegro task, preserving its MDP stepping and auto-resets.

    Each physics frame records the pre-frame state, body poses and native contacts, the post-actuator applied
    effort and the post-frame state. The scoped ``scene.update`` hook is restored even if stepping fails.
    """

    def __init__(self, env: Any, adapter: Any):
        if env._physics_handles_decimation:
            raise ValueError("Allegro recording requires one observable scene update per physics frame.")
        self.env = env
        self.adapter = adapter
        self.records_per_step = int(env.cfg.decimation)
        self.frames: list[dict[str, Any]] = []
        self.before: dict[str, Any] = {}

    def _snapshot(self) -> dict[str, Any]:
        raw = self.adapter.raw_neural_inputs()
        result: dict[str, Any] = {
            name: _cpu(raw[name][:, 0])
            for name in ("states", "root_body_q", "root_body_qd", "gravity_dir", "primary_body_q")
        }
        native = self.adapter.native_contact_pairs()
        if native is None:
            raise RuntimeError("Allegro datasets require native contacts.")
        result["native"] = {name: _cpu(value) for name, value in native.items()}
        return result

    def _record(self, action: torch.Tensor, after: dict[str, Any], inputs: dict[str, torch.Tensor]) -> None:
        num_envs = self.env.num_envs
        frame = dict(self.before)
        frame.update(
            next_states=after["states"],
            **inputs,
            actions=_cpu(action),
            terminated=torch.zeros(num_envs, dtype=torch.bool),
            truncated=torch.zeros(num_envs, dtype=torch.bool),
        )
        if getattr(self.adapter, "finger_object_normal_force_sensors", ()):
            frame[FINGER_FORCE_KEY] = _cpu(self.adapter.finger_object_normal_forces_w())
        self.frames.append(frame)
        self.before = after

    def _after_frame(self, action: torch.Tensor) -> None:
        import warp as wp

        control = wp.to_torch(self.adapter.control.joint_f)
        saved = control.clone()
        try:
            self.adapter.sync(overlay_applied_effort=True)
            after = self._snapshot()
            effort = _cpu(self.adapter.raw_neural_inputs()["joint_f"][:, 0])
        finally:
            control.copy_(saved)
        self._record(action, after, {"joint_f": effort})

    def step(self, action: torch.Tensor) -> tuple:
        """Step one native control interval and record its physics frames (or its one control-step record)."""
        self.adapter.sync()
        self.before = self._snapshot()
        scene = self.env.scene
        original = scene.update
        had_override = "update" in vars(scene)
        prior_override = vars(scene).get("update")
        last_counter = self.env._sim_step_counter
        start = len(self.frames)

        def update(*args, **kwargs):
            nonlocal last_counter
            original(*args, **kwargs)
            counter = self.env._sim_step_counter
            if counter != last_counter:
                if counter - last_counter != 1:
                    raise RuntimeError("A physics frame was skipped by the recorder.")
                self._after_frame(action)
                last_counter = counter

        scene.update = update
        try:
            result = self.env.step(action)
        finally:
            if had_override:
                scene.update = prior_override
            else:
                del scene.update
        if len(self.frames) - start != self.records_per_step:
            raise RuntimeError("Recorded record count does not match the control step.")
        self.frames[-1].update(terminated=_cpu(result[2]), truncated=_cpu(result[3]))
        return result

    def clips(self, horizon: int, frame_offset: int) -> dict[str, Any] | None:
        """Cut recorded records into reset-safe clips of ``horizon`` records, then clear the buffer.

        Args:
            horizon: Records per clip.
            frame_offset: Records already written for this recording, keeping source frame identities unique.

        Returns:
            Rollout tensors with native contacts and trajectory context, or ``None`` without a full clip.
        """
        from isaaclab_neural.data.keyboard import keyboard_clip_indices
        from isaaclab_neural.generate.keyboard_dataset import pack_keyboard_contact_clips

        tensors = {
            name: torch.stack([frame[name] for frame in self.frames], dim=1)
            for name in self.frames[0]
            if name != "native"
        }
        selected = keyboard_clip_indices(
            tensors["terminated"] | tensors["truncated"], horizon=horizon, decimation=self.records_per_step
        )
        frames, self.frames = self.frames, []
        if not selected:
            return None
        output: dict[str, Any] = {
            name: torch.stack([value[world, start : start + horizon] for world, start in selected])
            for name, value in tensors.items()
        }
        worlds = torch.tensor([world for world, _ in selected])
        output["trajectory_context"] = {
            "source_env_id": worlds,
            "source_start_frame": torch.tensor([frame_offset + start for _, start in selected]),
            "env_origin": _cpu(self.env.scene.env_origins)[worlds],
            **{name: worlds.clone() for name in ("state_world_id", "root_world_id", "contact_world_id")},
        }
        output["native_contacts"] = pack_keyboard_contact_clips(
            [frame["native"] for frame in frames], selected, horizon=horizon, num_worlds=self.env.num_envs
        )
        return output


class AllegroControlStepRecorder(AllegroFrameRecorder):
    """Record one transition per control step for ``step_mode="control_step"`` NeRD models.

    Inputs follow the control-step runtime: state, body poses and native contacts at the control boundary, and the
    applied effort and joint position targets after the actuators of the step's first physics frame. The label is
    the state after the last physics frame.
    """

    def __init__(self, env: Any, adapter: Any):
        super().__init__(env, adapter)
        self.records_per_step = 1
        self._interval_frames = 0
        self._step_inputs: dict[str, torch.Tensor] = {}

    def step(self, action: torch.Tensor) -> tuple:
        self._interval_frames = 0
        return super().step(action)

    def _after_frame(self, action: torch.Tensor) -> None:
        import warp as wp

        self._interval_frames += 1
        if self._interval_frames == 1:
            control = self.adapter.control
            joint_f = wp.to_torch(control.joint_f)
            saved = joint_f.clone()
            try:
                self.adapter.overlay_applied_effort_into_control_joint_f()
                effort = _cpu(joint_f.view(self.env.num_envs, -1))
            finally:
                joint_f.copy_(saved)
            self._step_inputs = {
                "joint_f": effort,
                "joint_target_q": _cpu(wp.to_torch(control.joint_target_q).view(self.env.num_envs, -1)),
            }
        if self._interval_frames < self.env.cfg.decimation:
            return
        self.adapter.sync()
        self._record(action, self._snapshot(), self._step_inputs)


def recording_contract(env: Any, adapter: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Describe the recording identity, timing, layout and contact mobility for training-time validation."""
    contact_adapter = adapter.contact_adapter
    if contact_adapter is None:
        raise ValueError("Allegro recordings require a contact adapter.")
    result: dict[str, Any] = {
        "schema": ALLEGRO_RECORDING_SCHEMA if args.env_name == ALLEGRO_ENV_NAME else ENV_DRIVEN_RECORDING_SCHEMA,
        "env_name": args.env_name,
        "recording_id": uuid.uuid4().hex,
        "task": args.task,
        "step_mode": "control_step" if args.record_control_steps else "physics_frame",
        "physics_dt": env.physics_dt,
        "decimation": env.cfg.decimation,
        "num_q": adapter.solver.dof_q_per_env,
        "num_qd": adapter.solver.dof_qd_per_env,
        "source_num_envs": env.num_envs,
        "source_seed": args.seed,
        "collection_mode": args.collection_mode,
        "policy_action_noise_std": args.policy_action_noise_std,
        "contact_filter": contact_adapter.contact_filter,
        "joint_f_contract": (
            "post_actuator_shadow_PD_of_the_first_physics_frame"
            if args.record_control_steps
            else "post_actuator_shadow_PD_in_backend_order_for_the_recorded_physics_frame"
        ),
        "versions": {name: version(name) for name in ("newton", "mujoco-warp", "torch", "rsl-rl-lib")},
    }
    if getattr(args, "policy_sha256", None):
        result["policy_sha256"] = args.policy_sha256
    if args.record_control_steps:
        result["joint_target_contract"] = "newton_control_joint_target_q_after_first_frame_actuators"
    if contact_adapter.fixed_body_slots is not None:
        result["contact_fixed_body_slots"] = list(contact_adapter.fixed_body_slots)
    return result


def write_recording_contract(path: Path, contract: dict[str, Any]) -> None:
    """Attach the JSON recording contract to an appended HDF5 recording."""
    with h5py.File(path, "a") as file:
        file["data"].attrs[RECORDING_CONTRACT_ATTR] = json.dumps(contract, sort_keys=True, allow_nan=False)


def _policy_action(policy: Any, wrapped: Any, noise_std: float) -> torch.Tensor:
    action = policy(wrapped.get_observations())
    if noise_std > 0.0:
        action = (action + noise_std * torch.randn_like(action)).clamp(-1.0, 1.0)
    return action


def collect(env: Any, adapter: Any, args: argparse.Namespace, policy: Any, wrapped: Any) -> dict[str, Any]:
    """Collect chunks until ``args.num_transitions`` clip records are written, appending each chunk."""
    from isaaclab_neural.data import append_rollouts_to_hdf5

    recorder = (
        AllegroControlStepRecorder(env, adapter) if args.record_control_steps else AllegroFrameRecorder(env, adapter)
    )
    contract = recording_contract(env, adapter, args)
    held = adapter.zero_action()
    written = recorded = episodes = control_steps = 0
    started = time.monotonic()
    env.reset()
    while written < args.num_transitions:
        for _ in range(args.chunk_control_steps):
            if args.collection_mode == "policy":
                action = _policy_action(policy, wrapped, args.policy_action_noise_std)
            elif args.collection_mode == "random":
                if control_steps % args.random_hold_controls == 0:
                    held.uniform_(-args.random_action_limit, args.random_action_limit)
                action = held
            else:
                action = held
            if not bool(torch.isfinite(action).all()):
                raise RuntimeError("Collection produced non-finite actions.")
            result = recorder.step(action)
            episodes += int((result[2] | result[3]).sum())
            control_steps += 1
        rollouts = recorder.clips(args.horizon, recorded)
        recorded += args.chunk_control_steps * recorder.records_per_step
        if rollouts is None:
            continue
        append_rollouts_to_hdf5(args.dataset, rollouts, args.env_name)
        written += rollouts["states"].shape[0] * rollouts["states"].shape[1]
        print(f"[collect] {written}/{args.num_transitions} records, {episodes} episodes", flush=True)
    write_recording_contract(args.dataset, contract)
    return {
        "dataset": str(args.dataset),
        "contract": contract,
        "written_records": written,
        "control_steps": control_steps,
        "completed_episodes": episodes,
        "seconds": time.monotonic() - started,
        "contact_truncation": adapter.contact_truncation_summary(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-Reorient-Cube-Allegro-Policy-v0")
    parser.add_argument("--env-name", default=ALLEGRO_ENV_NAME, help="Dataset environment name.")
    parser.add_argument("--policy-task", help="Task supplying the fallback RSL-RL runner config; defaults to --task.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--collection-mode", choices=COLLECTION_MODES, default="policy")
    parser.add_argument("--checkpoint", type=Path, help="RSL-RL teacher checkpoint for policy collection.")
    parser.add_argument("--policy-action-noise-std", type=float, default=0.0)
    parser.add_argument("--random-action-limit", type=float, default=1.0)
    parser.add_argument("--random-hold-controls", type=int, default=4)
    parser.add_argument("--record-control-steps", action="store_true")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--num-transitions", type=int, required=True, help="Clip records to write.")
    parser.add_argument("--horizon", type=int, required=True, help="Records per reset-safe clip.")
    parser.add_argument("--chunk-control-steps", type=int, default=64, help="Control steps per appended chunk.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-contact-tokens", type=int, default=128)
    parser.add_argument("--contact-filter", choices=("none", "solver_active", "solver_active_movable"), default="none")
    return parser


def main() -> None:
    import gymnasium as gym

    from isaaclab.app import add_launcher_args, launch_simulation

    from isaaclab_tasks.utils.hydra import resolve_presets
    from isaaclab_tasks.utils.parse_cfg import resolve_task_config

    import isaaclab_neural.envs  # noqa: F401 - registers the NeRD policy and dataset tasks
    from isaaclab_neural.eval.policy_rollout import load_task_policy
    from isaaclab_neural.generate.adapter import DataGenerationAdapter
    from isaaclab_neural.physics import NerdSolverCfg

    parser = build_parser()
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.dataset.exists():
        raise FileExistsError(args.dataset)
    if args.collection_mode == "policy" and (args.checkpoint is None or not args.checkpoint.is_file()):
        parser.error("policy collection requires an existing --checkpoint")
    if args.policy_action_noise_std < 0.0 or (args.policy_action_noise_std > 0.0 and args.collection_mode != "policy"):
        parser.error("--policy-action-noise-std must be non-negative and applies to policy collection only")
    if min(args.num_envs, args.num_transitions, args.horizon, args.chunk_control_steps, args.random_hold_controls) <= 0:
        parser.error("counts must be positive")
    sys.argv = [sys.argv[0]]
    cfg, _ = resolve_task_config(args.task, "rsl_rl_cfg_entry_point", overrides=[])
    cfg = resolve_presets(cfg, {"newton_mjwarp"})
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device or "cuda:0"
    cfg.seed = args.seed
    records_per_control = 1 if args.record_control_steps else cfg.decimation
    if args.horizon % records_per_control or args.chunk_control_steps * records_per_control < args.horizon:
        parser.error("horizon must be a multiple of the records per control step and fit in one chunk")
    torch.manual_seed(args.seed)
    args.dataset.parent.mkdir(parents=True, exist_ok=True)
    with launch_simulation(cfg, args):
        env = gym.make(args.task, cfg=cfg).unwrapped
        try:
            with torch.inference_mode():
                env.reset()
                adapter = DataGenerationAdapter(
                    env,
                    NerdSolverCfg(
                        states_frame="world",
                        num_contacts_per_env=args.max_contact_tokens,
                        contact_representation="contact_tokens",
                        max_contact_tokens=args.max_contact_tokens,
                        contact_filter=args.contact_filter,
                        contact_packing_policy="body_round_robin_pair_atomic",
                        joint_target_input=bool(args.record_control_steps),
                    ),
                )
                policy = wrapped = None
                if args.collection_mode == "policy":
                    wrapped, policy, _ = load_task_policy(
                        env,
                        args.checkpoint,
                        task=args.policy_task or args.task,
                        device=cfg.sim.device,
                        seed=args.seed,
                    )
                    with args.checkpoint.open("rb") as stream:
                        args.policy_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
                report = collect(env, adapter, args, policy, wrapped)
        finally:
            env.close()
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "contract"}), flush=True)


if __name__ == "__main__":
    main()
