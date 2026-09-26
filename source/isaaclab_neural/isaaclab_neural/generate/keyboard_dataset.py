# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect fixed-keyboard physics frames and dynamically replay their recorded actions.

Run with ``python -m isaaclab_neural.generate.keyboard_dataset collect|replay``.
The original manager-based task remains the ground-truth simulator. This module
neither registers a NeRD task nor trains a dynamics model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import h5py
import numpy as np
import torch

from isaaclab_neural.contacts.contact_set_schema import NATIVE_CONTACT_FRAME_COUNTS, NATIVE_CONTACT_VALUE_FIELDS
from isaaclab_neural.data.keyboard import (
    KEYBOARD_MJW_MODEL_FIELDS,
    KEYBOARD_REPLAY_LAYOUT,
    KEYBOARD_SCHEMA,
    KEYBOARD_TASK,
    keyboard_clip_indices,
    keyboard_replay_batches,
    read_keyboard_metadata,
    validate_keyboard_replay,
    write_keyboard_dataset,
)
from isaaclab_neural.generate.adapter import DataGenerationAdapter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

COMMAND_FIELDS = (
    "target",
    "typed",
    "target_len",
    "typed_len",
    "_prev_pressed",
    "_just_reset",
    "distance",
    "prefix_len",
    "max_prefix",
    "min_prefix",
    "new_high",
    "new_low",
    "_start_distance",
    "time_left",
    "command_counter",
    "_env_source",
)
MJW_FIELDS = ("qacc_warmstart", "qfrc_applied", "xfrc_applied", "ctrl", "act", "time")
MJW_MODEL_FIELDS = KEYBOARD_MJW_MODEL_FIELDS


def _mjwarp_model_arrays(adapter: DataGenerationAdapter) -> dict[str, Any]:
    model = adapter.backend._manager._solver.mjw_model
    result = {}
    for name in MJW_MODEL_FIELDS:
        value = model
        for component in name.split("."):
            value = getattr(value, component)
        result["mjwarp_model_" + name.replace(".", "_")] = value
    return result


def _restore_mjwarp_model_context(
    env: Any, adapter: DataGenerationAdapter, context: dict[str, torch.Tensor], env_ids: torch.Tensor | None
) -> None:
    import warp as wp

    keys = {"mjwarp_model_" + name.replace(".", "_") for name in MJW_MODEL_FIELDS}
    if not keys.intersection(context):
        return
    if not keys.issubset(context):
        raise ValueError("Incomplete MJWarp derived-model replay context.")
    manager = adapter.backend._manager
    with wp.ScopedDevice(env.device):
        for change in tuple(manager._model_changes):
            manager._solver.notify_model_changed(change)
        manager._model_changes.clear()
    ids = slice(None) if env_ids is None else env_ids
    for key, array in _mjwarp_model_arrays(adapter).items():
        target = wp.to_torch(array)
        if target.shape != context[key].shape or target.shape[0] != env.num_envs:
            raise ValueError(f"Unexpected MJWarp derived-model replay layout: {key}.")
        target[ids] = context[key].to(target.device)[ids]
    manager._solver._invalidate_contact_fast_path()


def configure_fixed_keyboard(env_cfg: Any, *, variant: int = 0) -> Any:
    """Copy a resolved native config, freezing only keyboard variation and root pose.

    Both reset event paths are frozen at zero offsets from the authored root
    pose. Robot resets, curriculum, rewards, terminations and timing are retained.
    """
    cfg = env_cfg.copy()
    apply_fixed_keyboard(cfg, variant=variant)
    return cfg


def apply_fixed_keyboard(cfg: Any, *, variant: int = 0) -> None:
    """Freeze a resolved keyboard config in place, including both root-reset paths."""
    pool = cfg.scene.keyboard.spawn.assets_cfg
    if variant < 0 or variant >= len(pool):
        raise ValueError(f"keyboard variant must be in [0, {len(pool)}).")
    cfg.scene.keyboard.spawn.assets_cfg = [pool[variant].copy()]
    cfg.scene.keyboard.spawn.random_choice = False
    for term in (cfg.events.reset_keyboard, cfg.commands.typing.reset.pre_solve_reset):
        if term is not None:
            term.params["pose_range"] = {axis: (0.0, 0.0) for axis in term.params["pose_range"]}
            term.params["velocity_range"] = {axis: (0.0, 0.0) for axis in term.params["velocity_range"]}


def pack_keyboard_contact_clips(
    frames: list[dict[str, torch.Tensor]], selected: list[tuple[int, int]], *, horizon: int, num_worlds: int
) -> dict[str, torch.Tensor]:
    """Pack CPU contact frames without rescanning every world's pairs for each clip.

    Args:
        frames: Recorded native pairs for each physics frame, stored on CPU.
        selected: Source world and starting physics-frame index for each clip.
        horizon: Number of physics frames in each clip.
        num_worlds: Number of source worlds.

    Returns:
        Trajectory-major, frame-major counts and contact values, retaining source pair order.
    """
    if not frames or not selected or horizon <= 0 or num_worlds <= 0:
        raise ValueError("Contact packing requires frames, clips, and positive dimensions.")
    indexed = []
    boundaries = np.arange(num_worlds + 1)
    for frame in frames:
        ids = frame["world_ids"].numpy()
        order = np.argsort(ids, kind="stable")
        offsets = np.searchsorted(ids[order], boundaries)
        already_sorted = np.array_equal(order, np.arange(ids.size))
        values = {
            name: frame[name].numpy() if already_sorted else frame[name].numpy()[order]
            for name in NATIVE_CONTACT_VALUE_FIELDS
        }
        indexed.append((offsets, values))
    counts = np.zeros((len(selected), horizon), dtype=np.int64)
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in NATIVE_CONTACT_VALUE_FIELDS}
    for clip, (world, start) in enumerate(selected):
        if not 0 <= world < num_worlds or not 0 <= start <= len(frames) - horizon:
            raise ValueError("Contact clip lies outside the source world timeline.")
        for step in range(horizon):
            offsets, values = indexed[start + step]
            begin, end = offsets[world : world + 2]
            counts[clip, step] = end - begin
            for name in chunks:
                chunks[name].append(values[name][begin:end])
    return {
        NATIVE_CONTACT_FRAME_COUNTS: torch.from_numpy(counts),
        **{name: torch.from_numpy(np.concatenate(values, axis=0)) for name, values in chunks.items()},
    }


def _cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to("cpu").clone()


def _context(env: Any, adapter: DataGenerationAdapter) -> dict[str, torch.Tensor]:
    import warp as wp

    from isaaclab_tasks.core.lift.mdp.utils import get_reset_state

    ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int32)
    command = env.command_manager.get_term("typing")
    result = {
        "reset_state": _cpu(get_reset_state(env, ids, ("robot", "keyboard"), is_relative=True)),
        "episode_length": _cpu(env.episode_length_buf),
        "last_action": _cpu(env.action_manager.action),
        "previous_action": _cpu(env.action_manager.prev_action),
        "source_env_id": torch.arange(env.num_envs),
        "env_origin": _cpu(env.scene.env_origins),
    }
    for name in ("state_world_id", "root_world_id", "contact_world_id"):
        result[name] = torch.arange(env.num_envs)
    for name in COMMAND_FIELDS:
        value = getattr(command, name, None)
        if isinstance(value, torch.Tensor):
            result["command_" + name] = _cpu(value)
    data = adapter.backend.mjwarp_data
    for name in MJW_FIELDS:
        value = getattr(data, name, None)
        if value is not None:
            tensor = wp.to_torch(value)
            if tensor.numel():
                if tensor.shape[0] != env.num_envs:
                    raise ValueError(f"Unexpected MJWarp replay buffer layout: {name}.")
                result["mjwarp_" + name] = _cpu(tensor)
    return result


def runtime_metadata(env: Any, adapter: DataGenerationAdapter, variant: int) -> dict[str, Any]:
    """Describe actual imported physics, mappings, and SI label conventions."""
    from isaaclab.utils.dict import replace_slices_with_strings

    model = adapter.model
    command = env.command_manager.get_term("typing")
    per_world = adapter.solver.num_joints_per_env
    labels = [str(name).rstrip("/").rsplit("/", 1)[-1] for name in model.body_label]
    assert model.joint_child is not None and model.joint_q_start is not None
    child = model.joint_child.numpy()[:per_world]
    starts = model.joint_q_start.numpy()
    keys = [env.scene["keyboard"].body_names[int(i)] for i in command._body_col_idx[0]]
    indices = []
    for key in keys:
        body = labels[: adapter.solver.num_bodies_per_env].index(key)
        joint = int(np.flatnonzero(child == body).item())
        if int(starts[joint + 1] - starts[joint]) != 1:
            raise ValueError("Keyboard keys must have one position coordinate each.")
        indices.append(int(starts[joint]))
    parameters = {}
    for name in (
        "body_mass",
        "body_inertia",
        "joint_type",
        "joint_axis",
        "joint_target_ke",
        "joint_target_kd",
        "joint_limit_lower",
        "joint_limit_upper",
        "joint_armature",
        "joint_friction",
        "joint_damping",
    ):
        array = getattr(model, name, None)
        if array is not None:
            values = array.numpy()
            if values.shape[0] % env.num_envs:
                raise ValueError(f"Non-uniform physics parameter layout: {name}.")
            parameters[name] = values[: values.shape[0] // env.num_envs].tolist()
    sensor = env.scene.sensors["robot_contact"]
    result = {
        "schema": KEYBOARD_SCHEMA,
        "task": KEYBOARD_TASK,
        "keyboard_variant": variant,
        "physics_dt": env.physics_dt,
        "decimation": env.cfg.decimation,
        "num_substeps": env.cfg.sim.physics.num_substeps,
        "num_q": adapter.solver.dof_q_per_env,
        "num_qd": adapter.solver.dof_qd_per_env,
        "key_q_indices": indices,
        "force_frame": "world",
        "force_units": "N",
        "force_timing": "post_solve_before_reset",
        "force_body_names": [str(name).rstrip("/").rsplit("/", 1)[-1] for name in sensor.sensor_names],
        "joint_names": {name: asset.joint_names for name, asset in env.scene.articulations.items()},
        "body_names": {name: asset.body_names for name, asset in env.scene.articulations.items()},
        "asset_order": list(env.scene.articulations),
        "physics_parameters": parameters,
        "solver_cfg": env.cfg.sim.physics.solver_cfg.to_dict(),
        "deterministic": env.cfg.sim.physics.deterministic,
        "use_cuda_graph": env.cfg.sim.physics.use_cuda_graph,
        "deterministic_mode": adapter.backend._manager._deterministic_mode.name.lower(),
        "action_contract": "relative_joint_position_scale_0.02_processed_once_per_control_step",
        "joint_f_contract": "post_actuator_shadow_PD_in_backend_order_for_the_recorded_physics_frame",
        "key_pressed_contract": "physics_frame_position_below_half_travel_not_a_typing_event",
        "typing_after_contract": "post_env_step_at_control_boundaries; ignore_after_terminal_reset",
        "source_num_envs": env.num_envs,
        "source_seed": env.cfg.seed,
        "scene": "fixed",
        "versions": {name: version(name) for name in ("newton", "mujoco-warp", "torch", "rsl-rl-lib")},
        "robot_asset": env.cfg.scene.robot.spawn.to_dict(),
        "robot_actuators": {name: cfg.to_dict() for name, cfg in env.cfg.scene.robot.actuators.items()},
        "keyboard_reset": env.cfg.events.reset_keyboard.to_dict(),
        "typing_reset": env.cfg.commands.typing.reset.to_dict(),
    }
    if adapter.contact_adapter is None:
        raise ValueError("Keyboard recordings require a contact adapter.")
    result["contact_filter"] = adapter.contact_adapter.contact_filter
    if adapter.contact_adapter.fixed_body_slots is not None:
        result["contact_fixed_body_slots"] = list(adapter.contact_adapter.fixed_body_slots)
    return json.loads(json.dumps(replace_slices_with_strings(result), allow_nan=False))


class KeyboardFrameRecorder:
    """Observe native scene updates, preserving normal MDP stepping and auto-reset.

    Capture occurs after the scene update for each physics frame, before control
    rewards/terminations/reset. The scoped hook is restored even if stepping fails.
    """

    def __init__(self, env: Any, adapter: DataGenerationAdapter, *, record_model_context: bool = False):
        self.record_model_context = record_model_context
        if env._physics_handles_decimation:
            raise ValueError("Keyboard recording requires one observable scene update per physics frame.")
        self.env = env
        self.adapter = adapter
        self.frames: list[dict[str, Any]] = []
        self.contexts: list[dict[str, torch.Tensor]] = []
        self.before: dict[str, Any] = {}

    def _snapshot(self) -> dict[str, Any]:
        raw = self.adapter.raw_neural_inputs()
        result: dict[str, Any] = {
            name: _cpu(raw[name][:, 0])
            for name in ("states", "root_body_q", "root_body_qd", "gravity_dir", "primary_body_q")
        }
        result["articulation_root_pose"] = torch.stack(
            [_cpu(asset.data.root_link_pose_w.torch) for asset in self.env.scene.articulations.values()], dim=1
        )
        result["articulation_root_pose"][..., :3] -= _cpu(self.env.scene.env_origins)[:, None]
        native = self.adapter.native_contact_pairs()
        if native is None:
            raise RuntimeError("Keyboard datasets require native contacts.")
        result["native"] = {name: _cpu(value) for name, value in native.items()}
        return result

    def _after_frame(self, action: torch.Tensor) -> None:
        import warp as wp

        env = self.env
        if self.record_model_context and len(self.frames) % env.cfg.decimation == 0:
            model_context = {key: _cpu(wp.to_torch(value)) for key, value in _mjwarp_model_arrays(self.adapter).items()}
            if any(value.shape[0] != env.num_envs for value in model_context.values()):
                raise ValueError("MJWarp derived-model context must have independent per-world storage.")
            self.contexts[-1].update(model_context)
        force = _cpu(env.scene.sensors["robot_contact"].data.net_normal_forces_w_history.torch[:, 0])
        command = env.command_manager.get_term("typing")
        keyboard = env.scene["keyboard"]
        q = keyboard.data.joint_pos.torch[command._inst_idx, command._col_idx]
        limits = keyboard.data.joint_pos_limits.torch[command._inst_idx, command._col_idx]
        pressed = _cpu(q < limits[..., 1] - command.cfg.actuation_fraction * (limits[..., 1] - limits[..., 0]))
        assert self.adapter.control.joint_f is not None
        control = wp.to_torch(self.adapter.control.joint_f)
        saved_control = control.clone()
        try:
            self.adapter.sync(overlay_applied_effort=True)
            after = self._snapshot()
            effort = _cpu(self.adapter.raw_neural_inputs()["joint_f"][:, 0])
        finally:
            control.copy_(saved_control)
        frame = {name: value for name, value in self.before.items()}
        frame.update(
            next_states=after["states"],
            joint_f=effort,
            actions=_cpu(action),
            robot_net_normal_forces_w=force,
            key_pressed=pressed,
            terminated=torch.zeros(env.num_envs, dtype=torch.bool),
            truncated=torch.zeros(env.num_envs, dtype=torch.bool),
            control_boundary=torch.zeros(env.num_envs, dtype=torch.bool),
            typed_after=torch.full_like(_cpu(command.typed), -1),
            reward=torch.zeros(env.num_envs),
        )
        self.frames.append(frame)
        self.before = after

    def step(self, action: torch.Tensor) -> tuple:
        """Step one native control interval and record all of its physics frames."""
        self.contexts.append(_context(self.env, self.adapter))
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
        if len(self.frames) - start != self.env.cfg.decimation:
            raise RuntimeError("Recorded physics-frame count does not match decimation.")
        self.frames[-1].update(
            terminated=_cpu(result[2]),
            truncated=_cpu(result[3]),
            control_boundary=torch.ones(self.env.num_envs, dtype=torch.bool),
            typed_after=_cpu(self.env.command_manager.get_term("typing").typed),
            reward=_cpu(result[1]),
        )
        return result

    def tensors(self) -> dict[str, torch.Tensor]:
        """Stack recorded physics tensors as ``[worlds, frames, ...]``."""
        return {
            name: torch.stack([frame[name] for frame in self.frames], dim=1)
            for name in self.frames[0]
            if name != "native"
        }

    def timeline(self) -> dict[str, dict[str, torch.Tensor]]:
        """Return all synchronized worlds and control-boundary reset context for action replay."""
        return {
            "frames": self.tensors(),
            "contexts": {
                name: torch.stack([context[name] for context in self.contexts], dim=1) for name in self.contexts[0]
            },
        }

    def clips(self, horizon: int) -> dict[str, Any]:
        """Reuse native CSR packing for fixed-length clips that do not cross reset."""
        tensors = self.tensors()
        selected = keyboard_clip_indices(
            tensors["terminated"] | tensors["truncated"], horizon=horizon, decimation=self.env.cfg.decimation
        )
        if not selected:
            raise ValueError("No full reset-safe clips; shorten the horizon or collect more control steps.")
        output: dict[str, Any] = {
            name: torch.stack([value[world, start : start + horizon] for world, start in selected])
            for name, value in tensors.items()
        }
        output["trajectory_context"] = {
            name: torch.stack(
                [self.contexts[start // self.env.cfg.decimation][name][world] for world, start in selected]
            )
            for name in self.contexts[0]
        }
        output["trajectory_context"]["source_start_frame"] = torch.tensor([start for _, start in selected])
        output["native_contacts"] = pack_keyboard_contact_clips(
            [frame["native"] for frame in self.frames], selected, horizon=horizon, num_worlds=self.env.num_envs
        )
        return output


class KeyboardPressPolicy:
    """Small DLS probe using real task actions, without changing targets or rewards."""

    def __init__(self, env: Any, *, action_limit: float = 1.0):
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        if not np.isfinite(action_limit) or action_limit <= 0:
            raise ValueError("Script action limit must be positive and finite.")
        self.action_limit = action_limit
        self.env = env
        self.command = env.command_manager.get_term("typing")
        self.robot = env.scene["robot"]
        self.controller = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls"),
            num_envs=env.num_envs,
            device=env.device,
        )
        self.age = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.targets = torch.zeros(env.num_envs, 3, device=env.device)
        self.normals = torch.zeros_like(self.targets)
        self.reset(torch.ones(env.num_envs, dtype=torch.bool, device=env.device))

    def reset(self, mask: torch.Tensor) -> None:
        from isaaclab.utils.math import quat_apply

        command = self.command
        quat = self.robot.data.body_quat_w.torch[:, command._ik_body_idx]
        tip = self.robot.data.body_pos_w.torch[:, command._ik_body_idx] + quat_apply(quat, command._ik_offset)
        typeable = torch.as_tensor(command.cfg.typeable_slots, device=self.env.device)
        positions = command.key_pos_w()
        slots = typeable[(positions[:, typeable] - tip[:, None]).norm(dim=-1).argmin(dim=1)]
        rows = torch.arange(self.env.num_envs, device=self.env.device)
        self.targets[mask] = positions[rows, slots][mask]
        quat = self.env.scene["keyboard"].data.body_quat_w.torch[
            command._inst_idx[rows, slots], command._body_col_idx[rows, slots]
        ]
        axis = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        self.normals[mask] = quat_apply(quat, axis)[mask]
        self.age[mask] = 0

    def __call__(self) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply, skew_symmetric_matrix

        command = self.command
        quat = self.robot.data.body_quat_w.torch[:, command._ik_body_idx]
        lever = quat_apply(quat, command._ik_offset)
        tip = self.robot.data.body_pos_w.torch[:, command._ik_body_idx] + lever
        jac = self.robot.data.body_link_jacobian_w.torch[
            :, command._ik_jacobi_body_idx, :, command._ik_jacobi_joint_ids
        ].clone()
        jac[:, :3] += torch.bmm(-skew_symmetric_matrix(lever), jac[:, 3:])
        phase = self.age.remainder(140)
        height = torch.where((phase < 40) | (phase >= 100), 0.025, -0.002)
        self.controller.set_command(self.targets + height[:, None] * self.normals, ee_quat=quat)
        q = self.robot.data.joint_pos.torch[:, command._ik_joint_ids]
        desired = self.controller.compute(tip, quat, jac, q)
        limits = self.robot.data.soft_joint_pos_limits.torch[:, command._ik_joint_ids]
        desired = desired.clamp(limits[..., 0], limits[..., 1])
        action = torch.zeros((self.env.num_envs, 6), device=self.env.device)
        action[:, command._ik_joint_ids] = ((desired - q) / 0.02).clamp(-self.action_limit, self.action_limit)
        low = torch.as_tensor(self.env.single_action_space.low, device=self.env.device)
        high = torch.as_tensor(self.env.single_action_space.high, device=self.env.device)
        self.age += 1
        return action.clamp(low, high)


def restore_keyboard_initial_state(
    env: Any, context: dict[str, torch.Tensor], *, env_ids: torch.Tensor | None = None
) -> None:
    """Restore physical and MDP state in the original world slots, without advancing physics.

    With explicit indices, restore only worlds that have already terminated;
    other worlds and their ongoing episodes remain untouched.
    """
    from isaaclab_tasks.core.lift.mdp.utils import set_reset_state

    if not torch.allclose(env.scene.env_origins, context["env_origin"].to(env.device), rtol=0.0, atol=1.0e-6):
        raise ValueError("Replay must preserve source world origins; fixed-base worlds cannot be relocated.")
    if env_ids is None:
        env.reset()
    ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int32) if env_ids is None else env_ids
    set_reset_state(env, context["reset_state"].to(env.device)[ids], ids, ("robot", "keyboard"), is_relative=True)
    env.sim.forward()
    env.scene.update(dt=env.physics_dt if env_ids is None else 0.0)
    env.episode_length_buf[ids] = context["episode_length"].to(env.device)[ids]
    env.action_manager.action[ids] = context["last_action"].to(env.device)[ids]
    env.action_manager.prev_action[ids] = context["previous_action"].to(env.device)[ids]
    command = env.command_manager.get_term("typing")
    for name in COMMAND_FIELDS:
        if "command_" + name in context:
            getattr(command, name)[ids] = context["command_" + name].to(env.device)[ids]


def restore_context(
    env: Any,
    adapter: DataGenerationAdapter,
    context: dict[str, torch.Tensor],
    *,
    env_ids: torch.Tensor | None = None,
) -> None:
    """Restore a clip's initial physical/MDP state and MJWarp warm-start buffers."""
    import warp as wp

    restore_keyboard_initial_state(env, context, env_ids=env_ids)
    _restore_mjwarp_model_context(env, adapter, context, env_ids)
    ids = slice(None) if env_ids is None else env_ids
    data = adapter.backend.mjwarp_data
    for name in MJW_FIELDS:
        if "mjwarp_" + name in context:
            wp.to_torch(getattr(data, name))[ids] = context["mjwarp_" + name].to(env.device)[ids]
    adapter.sync()


def replay_batch(
    env: Any,
    adapter: DataGenerationAdapter,
    expected: dict[str, torch.Tensor],
    context: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    active_worlds: list[int] | None = None,
) -> dict[str, Any]:
    """Re-simulate recorded actions without overwriting states between frames."""
    restore_context(env, adapter, context)
    recorder = KeyboardFrameRecorder(env, adapter)
    horizon = expected["states"].shape[1]
    decimation = env.cfg.decimation
    for step in range(0, horizon, decimation):
        actions = expected["actions"][:, step : step + decimation]
        if not torch.equal(actions, actions[:, :1].expand_as(actions)):
            raise ValueError("Recorded actions changed inside a control interval.")
        recorder.step(actions[:, 0].to(env.device))
    return compare_keyboard_frames(recorder.tensors(), expected, metadata, active_worlds)


def compare_keyboard_frames(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    active_worlds: list[int] | None = None,
) -> dict[str, Any]:
    """Compare native replay frames in physical units without masking mismatched terminations."""
    if active_worlds is not None:
        actual = {name: value[active_worlds] for name, value in actual.items()}
        expected = {name: value[active_worlds] for name, value in expected.items()}
    delta = (actual["next_states"] - expected["next_states"]).abs()
    nq = int(metadata["num_q"])
    keys = metadata["key_q_indices"]
    robot = sorted(set(range(nq)) - set(keys))
    valid_command = expected["control_boundary"] & ~(expected["terminated"] | expected["truncated"])
    return {
        "articulation_root_pose_error": (
            float((actual["articulation_root_pose"] - expected["articulation_root_pose"]).abs().max())
            if "articulation_root_pose" in expected
            else None
        ),
        "robot_q_error_rad": float(delta[..., robot].max()),
        "key_q_error_m": float(delta[..., keys].max()),
        "robot_qd_error_rad_s": float(delta[..., [nq + i for i in robot]].max()),
        "key_qd_error_m_s": float(delta[..., [nq + i for i in keys]].max()),
        "force_error_n": float(
            (actual["robot_net_normal_forces_w"] - expected["robot_net_normal_forces_w"]).abs().max()
        ),
        "key_pressed_mismatches": int((actual["key_pressed"] != expected["key_pressed"]).sum()),
        "termination_mismatches": int(
            ((actual["terminated"] != expected["terminated"]) | (actual["truncated"] != expected["truncated"])).sum()
        ),
        "typed_mismatches": int(((actual["typed_after"] != expected["typed_after"]) & valid_command[..., None]).sum()),
        "state_values_finite": bool(torch.isfinite(actual["next_states"]).all()),
        "initial_state_error": (actual["states"][:, 0] - expected["states"][:, 0]).abs().amax(dim=-1).tolist(),
        "initial_body_position_error_m": (
            actual["primary_body_q"][:, 0, :, :3] - expected["primary_body_q"][:, 0, :, :3]
        )
        .abs()
        .amax(dim=(1, 2))
        .tolist(),
        "state_errors_by_frame": delta.amax(dim=-1).tolist(),
        "joint_f_errors_by_frame": (actual["joint_f"] - expected["joint_f"]).abs().amax(dim=-1).tolist(),
    }


def replay_timeline(
    env: Any,
    adapter: DataGenerationAdapter,
    expected: dict[str, torch.Tensor],
    contexts: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Replay all worlds synchronously, restoring recorded reset outcomes only after matching terminations."""
    restore_context(env, adapter, {name: value[:, 0] for name, value in contexts.items()})
    recorder = KeyboardFrameRecorder(env, adapter)
    decimation = env.cfg.decimation
    horizon = expected["states"].shape[1]
    reset_rows = 0
    video_recorders = getattr(env, "video_recorders", [])
    if video_recorders:
        env.video_recorders = []
    try:
        for control, frame in enumerate(range(0, horizon, decimation)):
            actions = expected["actions"][:, frame : frame + decimation]
            if not torch.equal(actions, actions[:, :1].expand_as(actions)):
                raise ValueError("Recorded actions changed inside a control interval.")
            result = recorder.step(actions[:, 0].to(env.device))
            actual_done = result[2] | result[3]
            reference_done = (expected["terminated"] | expected["truncated"])[:, frame + decimation - 1].to(env.device)
            ids = torch.nonzero(actual_done & reference_done).flatten().to(torch.int32)
            if ids.numel() and frame + decimation < horizon:
                restore_context(
                    env, adapter, {name: value[:, control + 1] for name, value in contexts.items()}, env_ids=ids
                )
                reset_rows += ids.numel()
            if video_recorders:
                env.sim.render()
                for video in video_recorders:
                    video.step()
    finally:
        if video_recorders:
            env.video_recorders = video_recorders
    report = compare_keyboard_frames(recorder.tensors(), expected, metadata)
    report["reset_rows_restored"] = reset_rows
    report["ongoing_episode_state_corrections"] = 0
    return report


def _read_tensor_group(group: Any) -> dict[str, torch.Tensor]:
    if not isinstance(group, h5py.Group):
        raise ValueError("Keyboard data and trajectory context must be HDF5 groups.")
    result = {}
    for name, value in group.items():
        if not isinstance(value, h5py.Dataset):
            raise ValueError(f"Expected tensor dataset: {name}.")
        result[name] = torch.from_numpy(value[()])
    return result


def _json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def _run_replay(
    env: Any, adapter: DataGenerationAdapter, args: argparse.Namespace, metadata: dict, saved: dict
) -> None:
    fields = [
        "schema",
        "task",
        "keyboard_variant",
        "physics_dt",
        "decimation",
        "num_substeps",
        "num_q",
        "num_qd",
        "key_q_indices",
        "force_body_names",
        "joint_names",
        "body_names",
        "asset_order",
        "solver_cfg",
    ]
    fields += [
        name
        for name in (
            "use_cuda_graph",
            "deterministic_mode",
            "deterministic_workspace",
            "mjwarp_model_context_fields",
            "max_contact_tokens",
            "contact_filter",
            "contact_fixed_body_slots",
        )
        if name in saved
    ]
    for name in fields:
        if saved[name] != metadata[name]:
            raise ValueError(f"Replay contract mismatch: {name}.")
    for name, values in saved.get("physics_parameters", {}).items():
        if not np.allclose(values, metadata["physics_parameters"][name], rtol=1.0e-6, atol=1.0e-7):
            raise ValueError(f"Replay physics parameters differ: {name}.")
    synchronous = saved.get("replay_layout") == KEYBOARD_REPLAY_LAYOUT
    with h5py.File(args.dataset, "r") as file:
        data = _read_tensor_group(file["data"])
        contexts = _read_tensor_group(file["context/trajectories"])
        timeline = (
            {name: _read_tensor_group(file["replay/" + name]) for name in ("frames", "contexts")}
            if synchronous
            else None
        )
    stored_horizon = data["states"].shape[1]
    count = data["states"].shape[0]
    indices = args.trajectory_index if args.trajectory_index is not None else list(range(count))
    if not indices or min(indices) < 0 or max(indices) >= count:
        raise ValueError("Replay trajectory index outside the dataset.")
    if timeline is not None:
        if args.trajectory_index is not None or args.replay_horizon is not None:
            raise ValueError(
                "Synchronous replay validates the entire timeline; omit trajectory-index and replay-horizon."
            )
        validate_keyboard_replay(timeline, {**data, "trajectory_context": contexts}, saved)
        replay_horizon = timeline["frames"]["states"].shape[1]
        reports = [replay_timeline(env, adapter, timeline["frames"], timeline["contexts"], metadata)]
    else:
        replay_horizon = stored_horizon if args.replay_horizon is None else args.replay_horizon
        if not 0 < replay_horizon <= stored_horizon or replay_horizon % env.cfg.decimation:
            raise ValueError("Replay horizon must fit the stored clip and be a multiple of decimation.")
        data = {name: value[:, :replay_horizon] for name, value in data.items()}
        reports = []
        for batch_indices, active_worlds in keyboard_replay_batches(contexts["source_env_id"], indices, env.num_envs):
            report = replay_batch(
                env,
                adapter,
                {name: value[batch_indices] for name, value in data.items()},
                {name: value[batch_indices] for name, value in contexts.items()},
                metadata,
                active_worlds,
            )
            report["trajectory_indices"] = [batch_indices[world] for world in active_worlds]
            report["source_worlds"] = active_worlds
            reports.append(report)
    passed = all(
        r["state_values_finite"]
        and (r["articulation_root_pose_error"] is None or r["articulation_root_pose_error"] <= 1.0e-6)
        and r["robot_q_error_rad"] <= args.robot_q_atol
        and r["key_q_error_m"] <= args.key_q_atol
        and r["force_error_n"] <= args.force_atol
        and r["robot_qd_error_rad_s"] <= args.robot_qd_atol
        and r["key_qd_error_m_s"] <= args.key_qd_atol
        and r["key_pressed_mismatches"] == r["termination_mismatches"] == r["typed_mismatches"] == 0
        for r in reports
    )
    _json(
        args.report,
        {
            "mode": "synchronous_open_loop_action_replay" if synchronous else "open_loop_action_replay",
            "passed": passed,
            "trajectory_count": len(indices),
            "stored_horizon": stored_horizon,
            "replay_horizon": replay_horizon,
            "source_num_envs": env.num_envs,
            "deterministic_mode": metadata["deterministic_mode"],
            "use_cuda_graph": metadata["use_cuda_graph"],
            "robot_q_atol_rad": args.robot_q_atol,
            "key_q_atol_m": args.key_q_atol,
            "force_atol_n": args.force_atol,
            "robot_qd_atol_rad_s": args.robot_qd_atol,
            "key_qd_atol_m_s": args.key_qd_atol,
            "batches": reports,
        },
    )
    if not passed:
        raise RuntimeError(f"Keyboard replay validation failed; see {args.report}.")


def _reset_policy(runner: Any, done: torch.Tensor) -> None:
    if runner is None:
        return
    actor = getattr(runner.alg, "actor", None)
    if actor is None:
        actor = getattr(runner.alg, "policy", getattr(runner.alg, "actor_critic", None))
    if actor is not None:
        actor.reset(done)


def _resolve_scene(args: argparse.Namespace, saved: dict[str, Any] | None) -> tuple:
    from isaaclab_tasks.utils import resolve_task_config

    if args.num_envs is None:
        args.num_envs = int(saved["source_num_envs"]) if saved else 8
    if saved and args.num_envs != saved["source_num_envs"]:
        raise ValueError("Replay requires the original source_num_envs to preserve world layout.")
    variant = saved["keyboard_variant"] if saved else (args.keyboard_variant or 0)
    if args.keyboard_variant is not None and args.keyboard_variant != variant:
        raise ValueError("Keyboard variant differs from the recording.")
    if args.seed is None:
        args.seed = int(saved["source_seed"]) if saved else 0
    if saved and args.seed != saved["source_seed"]:
        raise ValueError("Replay requires the recorded seed and clone layout.")
    play_mode = saved.get("play_mode", False) if saved else True
    cfg, agent = resolve_task_config(
        KEYBOARD_TASK, "rsl_rl_cfg_entry_point", play_mode=play_mode, overrides=["physics=newton_mjwarp"]
    )
    cfg = configure_fixed_keyboard(cfg, variant=variant)
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device or "cuda:0"
    cfg.seed = args.seed
    deterministic = saved.get("deterministic", False) if saved else bool(args.deterministic_physics)
    if saved and args.deterministic_physics is not None and args.deterministic_physics != deterministic:
        raise ValueError("Replay determinism setting differs from the recording.")
    cfg.sim.physics.deterministic = deterministic
    cfg.sim.physics.use_cuda_graph = (
        saved.get("use_cuda_graph", not deterministic)
        if saved
        else (False if deterministic else cfg.sim.physics.use_cuda_graph)
    )
    cfg.commands.typing.debug_vis = False
    return cfg, agent, variant, play_mode


def _configure_replay_video(cfg: Any, args: argparse.Namespace, saved: dict | None) -> None:
    """Record a selected visual world without changing the complete physical replay."""
    if args.video_dir is None:
        return
    if args.mode != "replay" or not saved or saved.get("replay_layout") != KEYBOARD_REPLAY_LAYOUT:
        raise ValueError("Video recording requires synchronous replay of a collected dataset.")
    if not 0 <= args.video_env_index < cfg.scene.num_envs:
        raise ValueError("video-env-index must identify an existing recorded world.")
    if args.video_dir.exists():
        raise FileExistsError(args.video_dir)
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    with h5py.File(args.dataset, "r") as file:
        frames = file["replay/frames/states"]
        if not isinstance(frames, h5py.Dataset) or frames.shape[1] % cfg.decimation:
            raise ValueError("Replay video requires a complete control-step timeline.")
        steps = int(frames.shape[1]) // cfg.decimation
        origins = file["replay/contexts/env_origin"]
        if not isinstance(origins, h5py.Dataset):
            raise ValueError("Replay video requires recorded environment origins.")
        origin = np.asarray(origins[args.video_env_index, 0], dtype=float)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("Replay video environment origin must be a finite position.")
    camera = cfg.sim.default_visualizer_cfg
    if camera is None:
        raise ValueError("Replay video requires a default visualizer camera.")
    cfg.sim.visualizer_cfgs = [
        NewtonGLVisualizerCfg(
            headless=True,
            eye=tuple(float(value + offset) for value, offset in zip(camera.eye, origin)),
            lookat=tuple(float(value + offset) for value, offset in zip(camera.lookat, origin)),
            focal_length=camera.focal_length,
            window_width=1280,
            window_height=720,
            visible_env_indices=[args.video_env_index],
            randomly_sample_visible_envs=False,
            enable_live_plots=False,
        )
    ]
    cfg.video_recorders = [
        VideoRecorderCfg(
            source="visualizer:newton_gl",
            output_dir=str(args.video_dir),
            video_length=steps,
            output_filename_prefix=f"keyboard_replay_env{args.video_env_index}",
        )
    ]
    args.visualizer = ["newton_gl"]
    args.visualizer_explicit = True
    args.max_visible_envs = 1


def collect_keyboard_chunk(
    env: Any,
    adapter: DataGenerationAdapter,
    args: argparse.Namespace,
    metadata: dict,
    agent: Any,
    wrapped: Any,
    runner: Any,
    inference_policy: Any,
) -> dict:
    """Record one bounded reset-safe chunk using an already initialized environment and policy."""
    started = time.monotonic()
    metadata = dict(metadata)
    metadata["recording_id"] = uuid.uuid4().hex
    recorder = KeyboardFrameRecorder(
        env, adapter, record_model_context=bool(metadata.get("mjwarp_model_context_fields"))
    )
    completed_episodes = clean_successes = 0
    policy = (
        KeyboardPressPolicy(env, action_limit=args.script_action_limit) if args.collection_mode == "press" else None
    )
    random_action = adapter.zero_action()
    random_low, random_high = adapter.action_bounds(low=-args.random_action_limit, high=args.random_action_limit)
    packing_before = adapter.contact_truncation_summary() if "resident_worker" in metadata else None
    for step in range(args.control_steps):
        action = policy() if policy is not None else adapter.zero_action()
        if args.collection_mode == "random":
            if step % args.random_hold_controls == 0:
                random_action.uniform_(-args.random_action_limit, args.random_action_limit)
                random_action.clamp_(min=random_low, max=random_high)
            action = random_action
        if inference_policy is not None:
            assert wrapped is not None
            action = inference_policy(wrapped.get_observations())
            if agent.clip_actions is not None:
                action = action.clamp(-agent.clip_actions, agent.clip_actions)
        if not bool(torch.isfinite(action).all()):
            raise RuntimeError("Collection policy produced non-finite actions.")
        result = recorder.step(action)
        done = result[2] | result[3]
        completed_episodes += int(done.sum())
        clean_successes += int(
            (
                done
                & env.termination_manager.get_term("success")
                & ~env.termination_manager.get_term("abnormal_robot")
                & ~env.termination_manager.get_term("excessive_contact")
            ).sum()
        )
        if policy is not None:
            policy.reset(done)
        _reset_policy(runner, done)
        if (step + 1) % 25 == 0:
            print(f"[collect] {step + 1}/{args.control_steps} controls, {completed_episodes} episodes", flush=True)
    rollouts = recorder.clips(args.horizon)
    metadata.update(
        collection_mode=args.collection_mode,
        script_action_limit=args.script_action_limit,
        random_action_limit=args.random_action_limit,
        random_hold_controls=args.random_hold_controls,
        terminal_tail_clips_may_overlap=True,
        replay_layout=KEYBOARD_REPLAY_LAYOUT,
    )
    write_keyboard_dataset(args.dataset, rollouts, metadata, replay=recorder.timeline())
    packing = adapter.contact_truncation_summary()
    if packing is None:
        raise RuntimeError("Native contact statistics are unavailable.")
    if packing_before is not None:
        for key in (
            "frames",
            "raw_contacts",
            "packed_contacts",
            "dropped_contacts",
            "truncated_frames",
            "overflow_tokens_total",
        ):
            packing[key] -= packing_before[key]
        if "filtered_contacts" in packing:
            packing["filtered_contacts"] -= packing_before.get("filtered_contacts", 0)
        packing["truncated_frame_ratio"] = packing["truncated_frames"] / max(1, packing["frames"])
    report = {
        "mode": "collect",
        "basic_validation_passed": all(
            packing.get(key, -1) == 0
            for key in ("dropped_contacts", "truncated_frames", "overflow_tokens_total", "overflow_frames")
        ),
        "replay_performed": False,
        "dataset": str(args.dataset),
        "source_num_envs": env.num_envs,
        "source_seed": args.seed,
        "deterministic": metadata["deterministic"],
        "deterministic_mode": metadata["deterministic_mode"],
        "policy_checkpoint": metadata.get("policy_checkpoint"),
        "completed_episodes": completed_episodes,
        "clean_successes": clean_successes,
        "clean_success_rate": clean_successes / completed_episodes if completed_episodes else None,
        "pressed_slot_count": int(rollouts["key_pressed"].any(dim=(0, 1)).sum()),
        "trajectory_count": int(rollouts["states"].shape[0]),
        "physics_frames_per_trajectory": args.horizon,
        "total_recorded_frames_per_env": len(recorder.frames),
        "key_down_edges_inside_clips": int((rollouts["key_pressed"][:, 1:] & ~rollouts["key_pressed"][:, :-1]).sum()),
        "max_normal_force_n": float(rollouts["robot_net_normal_forces_w"].norm(dim=-1).max()),
        "terminal_clips": int((rollouts["terminated"] | rollouts["truncated"]).any(dim=1).sum()),
        "native_contact_packing": packing,
        "collection_seconds": time.monotonic() - started,
    }
    if "resident_worker" in metadata:
        report["resident_worker"] = metadata["resident_worker"]
    _json(args.report, report)
    if not report["basic_validation_passed"]:
        raise RuntimeError(f"Keyboard contact packing validation failed; see {args.report}.")
    return report


def _worker_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".pending")
    _json(temporary, value)
    os.replace(temporary, path)


def serve_collection_requests(
    env: Any,
    adapter: DataGenerationAdapter,
    args: argparse.Namespace,
    metadata: dict,
    agent: Any,
    wrapped: Any,
    runner: Any,
    inference_policy: Any,
) -> None:
    """Serve sequential chunk requests without recreating the simulator or reloading the policy."""
    state = {
        "pid": os.getpid(),
        "source_num_envs": env.num_envs,
        "boot_seed": args.seed,
        "environment_creations": 1,
        "policy_loads": 1,
        "chunks_completed": 0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    _worker_json(args.worker_status, state)
    for line in sys.stdin:
        request = json.loads(line)
        sequence = state["chunks_completed"]
        dataset, report, response = (Path(request[key]).resolve() for key in ("dataset", "report", "response"))
        if (
            request["sequence"] != sequence
            or type(request["seed"]) is not int
            or dataset.parent != args.dataset.resolve().parent
            or report.parent != args.report.resolve().parent
            or response.parent != args.worker_status.resolve().parent
            or dataset.exists()
            or report.exists()
            or response.exists()
        ):
            raise ValueError("Invalid resident collection request or existing output.")
        chunk_args = argparse.Namespace(**vars(args))
        chunk_args.dataset, chunk_args.report, chunk_args.seed = dataset, report, request["seed"]
        try:
            env.cfg.seed = chunk_args.seed
            env.reset(seed=chunk_args.seed)
            _reset_policy(runner, torch.ones(env.num_envs, dtype=torch.bool, device=env.device))
            chunk_metadata = {
                **metadata,
                "source_seed": chunk_args.seed,
                "resident_worker": {**state, "sequence": sequence},
            }
            collect_keyboard_chunk(env, adapter, chunk_args, chunk_metadata, agent, wrapped, runner, inference_policy)
            state["chunks_completed"] += 1
            _worker_json(args.worker_status, state)
            _worker_json(response, {**request, "ok": True, "pid": state["pid"], "source_num_envs": env.num_envs})
        except Exception as error:
            _worker_json(response, {**request, "ok": False, "error_type": type(error).__name__, "pid": state["pid"]})
            raise


def main() -> None:
    import gymnasium as gym

    from isaaclab.app import add_launcher_args, launch_simulation

    import isaaclab_tasks  # noqa: F401

    from isaaclab_neural.physics.nerd_solver_cfg import NerdSolverCfg

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("collect", "replay", "worker"))
    parser.add_argument("--worker-status", type=Path)
    parser.add_argument("--task", choices=(KEYBOARD_TASK,), default=KEYBOARD_TASK)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, help="Record full synchronous replay with headless Newton GL.")
    parser.add_argument(
        "--video-env-index", type=int, default=0, help="Recorded world to show; all worlds still replay."
    )
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic-physics", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--bounded-deterministic-workspace", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--keyboard-variant", type=int, default=None)
    parser.add_argument("--control-steps", type=int, default=140)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--max-contact-tokens", type=int, default=None)
    parser.add_argument("--contact-filter", choices=("none", "solver_active", "solver_active_movable"), default=None)
    parser.add_argument("--collection-mode", choices=("press", "random", "zero", "policy"), default="press")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--script-action-limit", type=float, default=1.0)
    parser.add_argument("--random-action-limit", type=float, default=1.0)
    parser.add_argument("--random-hold-controls", type=int, default=1)
    parser.add_argument("--trajectory-index", type=int, nargs="+", default=None)
    parser.add_argument("--replay-horizon", type=int, default=None)
    parser.add_argument("--robot-q-atol", type=float, default=1.0e-3)
    parser.add_argument("--key-q-atol", type=float, default=5.0e-5)
    parser.add_argument("--force-atol", type=float, default=0.1)
    parser.add_argument("--robot-qd-atol", type=float, default=1.0e-2)
    parser.add_argument("--key-qd-atol", type=float, default=1.0e-3)
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.mode in ("collect", "worker") and args.collection_mode == "policy" and args.checkpoint is None:
        parser.error("policy collection requires --checkpoint")
    if args.mode == "worker" and (args.worker_status is None or args.collection_mode != "policy"):
        parser.error("worker mode requires --worker-status and policy collection")
    if (args.num_envs is not None and args.num_envs <= 0) or args.control_steps <= 0:
        parser.error("num-envs and control-steps must be positive")
    if not np.isfinite(args.random_action_limit) or args.random_action_limit <= 0 or args.random_hold_controls <= 0:
        parser.error("random-action-limit must be finite and positive; random-hold-controls must be positive")
    if args.report.exists():
        raise FileExistsError(args.report)
    if args.mode == "collect" and args.dataset.exists():
        raise FileExistsError(args.dataset)
    saved = read_keyboard_metadata(args.dataset) if args.mode == "replay" else None
    max_contact_tokens = int(saved.get("max_contact_tokens", 256)) if saved else (args.max_contact_tokens or 256)
    contact_filter = saved.get("contact_filter", "none") if saved else (args.contact_filter or "none")
    if saved and args.contact_filter is not None and args.contact_filter != contact_filter:
        raise ValueError("Replay contact filter differs from the recording.")
    if saved and args.max_contact_tokens is not None and args.max_contact_tokens != max_contact_tokens:
        raise ValueError("Replay contact-token capacity differs from the recording.")
    if max_contact_tokens <= 0 or args.max_contact_tokens == 0:
        parser.error("max-contact-tokens must be positive")
    cfg, agent, variant, play_mode = _resolve_scene(args, saved)
    if args.horizon <= 0 or args.horizon % cfg.decimation:
        parser.error("horizon must be a positive multiple of decimation")
    _configure_replay_video(cfg, args, saved)
    from isaaclab_neural.generate.keyboard_determinism import keyboard_deterministic_workspace, workspace_contract

    bounded_workspace = (
        bool(saved.get("deterministic_workspace")) if saved else bool(args.bounded_deterministic_workspace)
    )
    if (
        saved
        and args.bounded_deterministic_workspace is not None
        and args.bounded_deterministic_workspace != bounded_workspace
    ):
        raise ValueError("Replay workspace profile differs from the recording.")
    if bounded_workspace:
        if not cfg.sim.physics.deterministic or cfg.sim.physics.use_cuda_graph:
            raise ValueError("Bounded deterministic workspaces require deterministic eager physics.")
        if saved and saved.get("deterministic_workspace") != workspace_contract():
            raise ValueError("Replay workspace source contract differs from the recording.")
    with keyboard_deterministic_workspace(bounded_workspace) as workspace, launch_simulation(cfg, args):
        env = cast("ManagerBasedRLEnv", gym.make(KEYBOARD_TASK, cfg=cfg).unwrapped)
        try:
            with torch.inference_mode():
                env.reset()
                adapter = DataGenerationAdapter(
                    env,
                    NerdSolverCfg(
                        states_frame="world",
                        num_contacts_per_env=max_contact_tokens,
                        contact_representation="contact_tokens",
                        max_contact_tokens=max_contact_tokens,
                        contact_filter=contact_filter,
                        contact_packing_policy="body_round_robin_pair_atomic",
                    ),
                )
                metadata = runtime_metadata(env, adapter, variant)
                metadata["max_contact_tokens"] = max_contact_tokens
                if workspace is not None:
                    metadata["deterministic_workspace"] = workspace
                    metadata["mjwarp_model_context_fields"] = list(MJW_MODEL_FIELDS)
                metadata["play_mode"] = play_mode
                if args.mode in ("collect", "worker"):
                    wrapped = runner = inference_policy = None
                    if args.collection_mode == "policy":
                        from isaaclab.utils.dict import class_to_dict

                        from isaaclab_rl.rsl_rl import (
                            RslRlVecEnvWrapper,
                            create_rsl_rl_runner,
                            handle_deprecated_rsl_rl_cfg,
                        )

                        agent.device, agent.seed = env.device, args.seed
                        agent = handle_deprecated_rsl_rl_cfg(agent, version("rsl-rl-lib"))
                        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
                        runner = create_rsl_rl_runner(wrapped, agent)
                        runner.load(str(args.checkpoint))
                        inference_policy = runner.get_inference_policy(device=env.device)
                        with args.checkpoint.open("rb") as stream:
                            metadata["policy_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
                        metadata["policy_checkpoint"] = str(args.checkpoint.resolve())
                        metadata["policy_agent"] = class_to_dict(agent)
                    if args.mode == "worker":
                        serve_collection_requests(
                            env, adapter, args, metadata, agent, wrapped, runner, inference_policy
                        )
                    else:
                        collect_keyboard_chunk(env, adapter, args, metadata, agent, wrapped, runner, inference_policy)
                else:
                    assert saved is not None
                    _run_replay(env, adapter, args, metadata, saved)
        finally:
            env.close()
    if args.video_dir is not None and not any(path.stat().st_size for path in args.video_dir.glob("*.mp4")):
        raise RuntimeError("Replay completed but no video was produced; check renderer and encoder output.")


if __name__ == "__main__":
    main()
