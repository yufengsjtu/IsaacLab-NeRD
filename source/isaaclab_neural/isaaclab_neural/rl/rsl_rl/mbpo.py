# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain-aware real-state buffers for manager-based G1 MBPO rollouts."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import torch

from isaaclab.managers import ManagerTermBase

from isaaclab_neural.contacts.contact_set_schema import NATIVE_CONTACT_VALUE_FIELDS
from isaaclab_neural.contacts.native_pair_projector import project_native_pairs
from isaaclab_neural.data.physics_context import WorldPhysicsContext
from isaaclab_neural.data.terrain_context import build_terrain_context
from isaaclab_neural.physics.applied_joint_f import backend_applied_effort

HISTORY_KEYS = ("states", "joint_f", "root_body_q", "gravity_dir")
MODEL_WINDOW = 25


def initialize_branch_scene(base) -> dict | None:
    """Freeze terrain provenance without changing the worker's policy random-number stream."""
    generator = base.cfg.scene.terrain.terrain_generator
    context = build_terrain_context(base.cfg, generator.seed) if generator is not None else None
    base.mbpo_terrain_context = context
    base.mbpo_physics_context = (
        WorldPhysicsContext(base.sim.physics_manager._model, base.num_envs)
        if getattr(base.cfg, "mbpo_randomized_physics", False)
        else None
    )
    return context


def capture_branch_scene(base, ids: torch.Tensor) -> dict:
    """Capture per-world terrain origins [m], terrain indices and immutable terrain provenance."""
    terrain = base.scene.terrain
    values = {"env_origins": base.scene.env_origins.index_select(0, ids).detach().cpu().clone()}
    for key in ("terrain_levels", "terrain_types"):
        value = getattr(terrain, key, None)
        if isinstance(value, torch.Tensor):
            values[key] = value.index_select(0, ids).detach().cpu().clone()
    physics = getattr(base, "mbpo_physics_context", None)
    if physics is not None:
        values.update({f"physics/{key}": value for key, value in physics.capture(ids).items()})
    return {"context": copy.deepcopy(getattr(base, "mbpo_terrain_context", None)), "values": values}


def restore_branch_scene(base, ids: torch.Tensor, snapshot: dict, *, refresh_sensors: bool = True) -> None:
    """Restore terrain identity/origins [m] and invalidate height scans after a physical-state write."""
    if snapshot["context"] != getattr(base, "mbpo_terrain_context", None):
        raise ValueError("Cannot transplant a branch into different terrain geometry or terrain seeds.")
    terrain = base.scene.terrain
    values = snapshot["values"]
    origins = values["env_origins"].to(base.scene.env_origins)
    if origins.shape != (len(ids), 3):
        raise ValueError("Branch terrain origins have an incompatible world layout.")
    for key in ("terrain_levels", "terrain_types"):
        if key in values:
            live = getattr(terrain, key)
            if values[key].shape != (len(ids),):
                raise ValueError("Branch terrain indices have an incompatible world layout.")
            live[ids] = values[key].to(live)
    physics = getattr(base, "mbpo_physics_context", None)
    parameters = {key.removeprefix("physics/"): value for key, value in values.items() if key.startswith("physics/")}
    if bool(parameters) != (physics is not None):
        raise ValueError("Branch physical-parameter context is missing or has a different scope.")
    if physics is not None:
        from newton import ModelFlags

        physics.restore(ids, parameters)
        base.sim.physics_manager.add_model_change(ModelFlags.BODY_INERTIAL_PROPERTIES | ModelFlags.SHAPE_PROPERTIES)
    base.scene.env_origins[ids] = origins
    terrain.env_origins[ids] = origins.to(terrain.env_origins)
    if refresh_sensors:
        refresh_branch_sensors(base, ids)


def refresh_branch_sensors(base, ids: torch.Tensor) -> None:
    """Invalidate and refresh terrain-height observations after changing root state or terrain origins [m]."""
    if getattr(base, "mbpo_terrain_context", None) is not None:
        sensor = base.scene["height_scanner"]
        sensor.reset(ids)
        sensor.update(0.0, force_recompute=True)


def stateful_objects(base) -> dict:
    """Return task-manager objects holding branch-dependent per-world tensors."""
    objects = {"action_manager": base.action_manager}
    for name, term in base.action_manager._terms.items():
        objects[f"action/{name}"] = term
    for name, term in base.command_manager._terms.items():
        objects[f"command/{name}"] = term
    for label, manager in (("reward", base.reward_manager), ("termination", base.termination_manager)):
        for name, cfg in zip(manager._term_names, manager._term_cfgs):
            if isinstance(cfg.func, ManagerTermBase):
                objects[f"{label}/{name}"] = cfg.func
    if getattr(getattr(base, "cfg", None), "mbpo_randomized_physics", False):
        events = base.event_manager
        for index, name in enumerate(events._mode_term_names.get("interval", [])):
            if events._mode_term_cfgs["interval"][index].is_global_time:
                raise ValueError("MBPO branch replay requires per-world rather than global interval events.")
            objects[f"event/{name}"] = SimpleNamespace(time_left=events._interval_term_time_left[index])
    return objects


def capture_manager_state(base, ids: torch.Tensor) -> dict:
    """Snapshot commands, previous actions, episode counters and stateful reward/termination terms."""
    objects = {}
    for path, obj in stateful_objects(base).items():
        objects[path] = {
            key: value.index_select(0, ids).detach().cpu().clone()
            for key, value in _per_env_tensors(obj, base.num_envs).items()
        }
    return {"episode_length_buf": base.episode_length_buf.index_select(0, ids).cpu().clone(), "objects": objects}


def restore_manager_state(base, snapshot: dict, env_ids: torch.Tensor, rows: torch.Tensor) -> None:
    """Restore selected manager rows without reinitializing commands or previous actions."""
    objects = stateful_objects(base)
    if set(objects) != set(snapshot["objects"]):
        raise ValueError("MJ and NeRD manager layouts differ.")
    for path, tensors in snapshot["objects"].items():
        live = _per_env_tensors(objects[path], base.num_envs)
        if set(live) != set(tensors):
            raise ValueError(f"MJ and NeRD manager tensor layouts differ at {path}.")
        for key, value in tensors.items():
            live[key][env_ids] = value.index_select(0, rows.to(value.device)).to(live[key])
            if (
                key == "time_left"
                and path.startswith(("event/", "command/"))
                and getattr(base, "_mbpo_inside_step", False)
            ):
                live[key][env_ids] += base.step_dt


def project_history(batch: dict, solver) -> dict[str, torch.Tensor]:
    """Project recorded native contact pairs into the existing G1 contact-token history layout."""
    device = solver.torch_device
    count, history_steps = batch["states"].shape[:2]
    counts = batch["native_counts"].to(device)
    projected = project_native_pairs(
        {key: value.to(device) for key, value in batch["native"].items()},
        num_frames=count * history_steps,
        representation="contact_tokens",
        packing_policy="body_round_robin_pair_atomic",
        frame_ids=torch.arange(count * history_steps, device=device).repeat_interleave(counts),
        max_contact_tokens=solver.max_contact_tokens,
        contact_filter=solver.contact_filter,
    )
    history = {key: batch[key].to(device) for key in HISTORY_KEYS}
    history["contact_tokens"] = projected["contact_tokens"].reshape(count, history_steps, solver.max_contact_tokens, -1)
    history["contact_token_overflow"] = projected["contact_token_overflow"].reshape(count, history_steps)
    if set(history) != {"root_body_q", "states", "joint_f", "gravity_dir", *solver.contacts.keys()}:
        raise ValueError("Recorded contact/history fields differ from the pretrained NeRD contract.")
    return history


def write_history_rows(solver, env_ids: torch.Tensor, history: dict[str, torch.Tensor]) -> None:
    """Restore raw SI state/effort histories without padding captured history frames."""
    length = solver.num_states_history
    if history["states"].shape[1] != length:
        raise ValueError("Branch history length differs from the model context window.")
    if len(solver.states_history) != length:
        snapshot = solver._history_snapshot()
        solver.states_history.clear()
        for _ in range(length):
            solver.states_history.append({key: value.clone() for key, value in snapshot.items()})
    for step, frame in enumerate(solver.states_history):
        for key, value in history.items():
            frame[key].index_copy_(0, env_ids, value[:, step].to(frame[key]))
        frame["states_embedding"].index_copy_(0, env_ids, solver.embed_states(history["states"][:, step]))
    solver._mark_history_rows_for_rebuild(env_ids)


def write_physical_rows(base, env_ids: torch.Tensor, states: torch.Tensor) -> None:
    """Write raw generalized G1 states in SI units into selected NeRD worlds."""
    adapter = base.neural_adapter
    manager = base.sim.physics_manager
    adapter.sync(update_history=False)
    full = adapter.solver.states.clone()
    full[env_ids] = states.to(full)
    adapter.solver._assign_states_from_torch(adapter.state, full)
    manager.invalidate_fk()
    manager.forward()
    base.scene.update(dt=base.physics_dt)
    adapter.sync(update_history=False)


def merge_branch_batches(blocks: list[dict]) -> dict:
    """Merge captured anchor batches, preserving their shared terrain and per-row terrain assignment."""
    if not blocks:
        raise RuntimeError("No complete real-MJ branch histories were captured.")
    scene = blocks[0]["scene"]
    if any(block["scene"]["context"] != scene["context"] for block in blocks):
        raise ValueError("Cannot mix different terrain atlases in one MBPO replay buffer.")
    merged = {
        key: torch.cat([block[key] for block in blocks]) for key in (*HISTORY_KEYS, "current_state", "native_counts")
    }
    merged["native"] = {key: torch.cat([block["native"][key] for block in blocks]) for key in blocks[0]["native"]}
    merged["manager"] = {
        "episode_length_buf": torch.cat([block["manager"]["episode_length_buf"] for block in blocks]),
        "objects": {
            path: {key: torch.cat([block["manager"]["objects"][path][key] for block in blocks]) for key in tensors}
            for path, tensors in blocks[0]["manager"]["objects"].items()
        },
    }
    merged["scene"] = {
        "context": scene["context"],
        "values": {key: torch.cat([block["scene"]["values"][key] for block in blocks]) for key in scene["values"]},
    }
    return merged


def install_branch_reset(base, buffer, statistics: dict, standing_fraction: float) -> None:
    """Mix terrain-valid ordinary resets with restored real-MJ branch starts."""
    if not 0 <= standing_fraction <= 1:
        raise ValueError("Standing reset fraction must lie in [0, 1].")
    original, original_step = base._reset_idx, base.step

    def control_step(actions):
        base._mbpo_inside_step = True
        try:
            return original_step(actions)
        finally:
            base._mbpo_inside_step = False

    def branch_reset(env_ids):
        ids = torch.as_tensor(env_ids, device=base.device, dtype=torch.long).reshape(-1)
        if ids.numel() and getattr(base.cfg, "mbpo_terrain_profile", "fixed") == "stock":
            buffer.restore_initial_scene(base, ids)
        original(env_ids)
        if ids.numel():
            selected = torch.rand(len(ids), device=buffer.device, generator=buffer.generator) >= standing_fraction
            statistics["standing_resets"] = statistics.get("standing_resets", 0) + int((~selected).sum())
            ids = ids[selected]
        if ids.numel():
            info = buffer.restore(base, ids)
            statistics["branches"] = statistics.get("branches", 0) + info["branched"]
            statistics.setdefault("source_iterations", []).extend(info["source_iterations"])

    base._reset_idx = branch_reset
    base.step = control_step


class FastHistory:
    """Record real pre/post-physics inputs and actual H10/16-frame targets without reset crossings."""

    def __init__(self, env, adapter, *, length: int = 10, capacity: int = MODEL_WINDOW):
        self.env, self.adapter, self.length = env, adapter, length
        self.capacity = max(length, capacity)
        self.manager = env.sim.physics_manager
        self.action_term = env.action_manager.get_term("joint_pos")
        self.frames: list[dict] = []
        self.valid = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.last_external_jump = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.external_jump_count = 0
        self._pending = None
        self._before_callback, self._after_callback = self.before, self.after
        self.manager.register_post_actuator_callback(self._before_callback)
        self.manager.register_post_step_callback(self._after_callback)

    @torch.no_grad()
    def before(self) -> None:
        self.adapter.sync()
        inputs = self.adapter.solver.get_raw_neural_model_inputs()
        frame = {key: inputs[key][:, 0].detach().clone() for key in HISTORY_KEYS}
        frame["joint_f"][:, 6:] = backend_applied_effort(self.env.scene["robot"]).detach()
        native = self.adapter.native_contact_pairs()
        frame["native"] = {key: native[key].detach().clone() for key in NATIVE_CONTACT_VALUE_FIELDS}
        frame["actions"] = self.action_term.raw_actions.detach().clone()
        frame["action_targets"] = self.action_term.processed_actions.detach().clone()
        self._pending = frame

    @torch.no_grad()
    def after(self) -> None:
        frame, self._pending = self._pending, None
        solver = self.adapter.solver
        state = torch.empty_like(solver.states)
        solver._acquire_states_to_torch(self.manager._state_0, state)
        solver.wrap2PI(state)
        frame["observed_next_state"] = state
        self.frames.append(frame)
        if len(self.frames) > self.capacity:
            self.frames.pop(0)
        self.valid += 1

    def finish_control_step(self, done: torch.Tensor) -> None:
        """Exclude reset/push-crossing model windows without changing real PPO transitions or done flags."""
        finished = done.bool().to(self.valid.device)
        self.last_external_jump.zero_()
        if getattr(self.env.cfg, "mbpo_randomized_physics", False) and self.frames:
            solver = self.adapter.solver
            actual = torch.empty_like(solver.states)
            solver._acquire_states_to_torch(self.manager._state_0, actual)
            solver.wrap2PI(actual)
            changed = (actual - self.frames[-1]["observed_next_state"]).abs().amax(dim=1) > 1e-5
            self.last_external_jump = changed & ~finished
            self.external_jump_count += int(self.last_external_jump.sum())
        self.valid[finished | self.last_external_jump] = 0

    def complete(self, frames: int = 10) -> torch.Tensor:
        """Return worlds with enough actual consecutive post-reset physical frames."""
        return (self.valid >= frames).nonzero().flatten()

    def gather(self, ids: torch.Tensor) -> dict:
        """Gather H10 histories, post-step branch states in SI units and terrain context."""
        if len(self.frames) < self.length or bool((self.valid[ids] < self.length).any()):
            raise ValueError("Branch starts require complete real histories.")
        frames = self.frames[-self.length :]
        batch = self._stack(frames, ids)
        batch["current_state"] = frames[-1]["observed_next_state"].index_select(0, ids).cpu()
        batch["scene"] = capture_branch_scene(self.env, ids)
        return batch

    def model_windows(self, ids: torch.Tensor, decimation: int) -> dict:
        """Gather observed MJ H10 targets and16 actual future states in SI units for online fitting."""
        if decimation != 4 or self.length != 10 or len(self.frames) != MODEL_WINDOW:
            raise ValueError("G1 MBPO model targets require H10, decimation4 and25 captured frames.")
        if bool((self.valid[ids] < MODEL_WINDOW).any()):
            raise ValueError("Model fitting windows cannot cross a reset.")
        history, future = self.frames[: self.length], self.frames[self.length - 1 :]
        batch = self._stack(history, ids)
        batch["next_states"] = torch.stack(
            [frame["observed_next_state"].index_select(0, ids) for frame in history], 1
        ).cpu()
        batch["oracle_states"] = torch.stack(
            [frame["observed_next_state"].index_select(0, ids) for frame in future], 1
        ).cpu()
        controls = future[::decimation]
        for key in ("actions", "action_targets"):
            batch[key] = torch.stack([frame[key].index_select(0, ids) for frame in controls], 1).cpu()
        for frame, target in zip(future, [control for control in controls for _ in range(decimation)]):
            if not torch.equal(frame["action_targets"], target["action_targets"]):
                raise RuntimeError("Recorded model window crossed an action-hold boundary.")
        batch["env_ids"] = ids.cpu()
        physics = getattr(self.env, "mbpo_physics_context", None)
        if physics is not None:
            batch.update({f"physics/{key}": value for key, value in physics.capture(ids).items()})
        terrain = self.env.scene.terrain
        for key in ("terrain_levels", "terrain_types"):
            value = getattr(terrain, key, None)
            if isinstance(value, torch.Tensor):
                batch[key] = value.index_select(0, ids).detach().cpu().clone()
        return batch

    def close(self) -> None:
        self.manager.unregister_post_step_callback(self._after_callback)
        self.manager._post_actuator_callbacks.remove(self._before_callback)

    def _stack(self, frames: list[dict], ids: torch.Tensor) -> dict:
        length = len(frames)
        batch = {
            key: torch.stack([frame[key].index_select(0, ids) for frame in frames], 1).cpu() for key in HISTORY_KEYS
        }
        positions = torch.full((self.env.num_envs,), -1, dtype=torch.long, device=ids.device)
        positions[ids] = torch.arange(len(ids), device=ids.device)
        parts, frame_ids = [], []
        for step, frame in enumerate(frames):
            slots = positions[frame["native"]["world_ids"].long()]
            keep = (slots >= 0).nonzero().flatten()
            parts.append({key: value.index_select(0, keep) for key, value in frame["native"].items()})
            frame_ids.append(slots.index_select(0, keep) * length + step)
        keys = torch.cat(frame_ids)
        order = torch.sort(keys, stable=True).indices
        batch["native"] = {
            key: torch.cat([part[key] for part in parts]).index_select(0, order).cpu() for key in parts[0]
        }
        batch["native_counts"] = torch.bincount(keys, minlength=len(ids) * length).cpu()
        return batch


class BranchBuffer:
    """Bounded ring buffer of real states/history, task-manager tensors and terrain assignments."""

    def __init__(self, solver, capacity: int, seed: int, scene_context: dict | None):
        self.solver, self.capacity = solver, capacity
        self.device = solver.torch_device
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.scene_context = scene_context
        self.data: dict | None = None
        self.size, self.cursor = 0, 0

    def add(self, batch: dict) -> None:
        """Insert a collector batch; reject a different terrain atlas before mutating the buffer."""
        if batch["scene"]["context"] != self.scene_context:
            raise ValueError("Collector and model worlds have different terrain geometry.")
        rows = {
            "history": project_history(batch, self.solver),
            "state": batch["current_state"].to(self.device),
            "manager": {
                path: {key: value.to(self.device) for key, value in values.items()}
                for path, values in batch["manager"]["objects"].items()
            },
            "scene": {key: value.to(self.device) for key, value in batch["scene"]["values"].items()},
        }
        count = len(rows["state"])
        if not 0 < count <= self.capacity:
            raise ValueError("Collector batch exceeds the bounded branch buffer.")
        if self.data is None:
            self._allocate(rows)
        slots = (self.cursor + torch.arange(count, device=self.device)) % self.capacity
        for key, value in rows["history"].items():
            self.data["history"][key][slots] = value
        self.data["state"][slots] = rows["state"]
        self.data["source_iteration"][slots] = batch["iteration"]
        for path, values in rows["manager"].items():
            for key, value in values.items():
                self.data["manager"][path][key][slots] = value
        for key, value in rows["scene"].items():
            self.data["scene"][key][slots] = value
        self.cursor, self.size = (self.cursor + count) % self.capacity, min(self.capacity, self.size + count)

    def restore_initial_scene(self, base, env_ids: torch.Tensor) -> None:
        """Select real curriculum/material assignments before an ordinary task reset, without copying robot state."""
        self._restore_scene_rows(base, env_ids, self._sample_rows(base, env_ids))

    def restore(self, base, env_ids: torch.Tensor) -> dict:
        """Restore real physical parameters, state, history and manager clocks on the captured terrain patch."""
        rows = self._sample_rows(base, env_ids)
        data = self.data
        self._restore_scene_rows(base, env_ids, rows)
        write_physical_rows(base, env_ids, data["state"].index_select(0, rows))
        write_history_rows(
            self.solver, env_ids, {key: value.index_select(0, rows) for key, value in data["history"].items()}
        )
        restore_manager_state(base, {"objects": data["manager"]}, env_ids, rows)
        refresh_branch_sensors(base, env_ids)
        base.episode_length_buf[env_ids] = 0
        return {"branched": len(env_ids), "source_iterations": data["source_iteration"].index_select(0, rows).tolist()}

    def _sample_rows(self, base, env_ids: torch.Tensor) -> torch.Tensor:
        if self.size == 0:
            raise RuntimeError("No real branch states are available.")
        if self.scene_context != getattr(base, "mbpo_terrain_context", None):
            raise ValueError("Branch destination terrain changed.")
        return torch.randint(0, self.size, (len(env_ids),), generator=self.generator, device=self.device)

    def _restore_scene_rows(self, base, env_ids: torch.Tensor, rows: torch.Tensor) -> None:
        restore_branch_scene(
            base,
            env_ids,
            {
                "context": self.scene_context,
                "values": {key: value.index_select(0, rows) for key, value in self.data["scene"].items()},
            },
            refresh_sensors=False,
        )

    def _allocate(self, rows: dict) -> None:
        def empty(value):
            return torch.empty((self.capacity, *value.shape[1:]), dtype=value.dtype, device=self.device)

        self.data = {
            "history": {key: empty(value) for key, value in rows["history"].items()},
            "state": empty(rows["state"]),
            "source_iteration": torch.empty(self.capacity, dtype=torch.long, device=self.device),
            "manager": {
                path: {key: empty(value) for key, value in values.items()} for path, values in rows["manager"].items()
            },
            "scene": {key: empty(value) for key, value in rows["scene"].items()},
        }


def _per_env_tensors(obj, num_envs: int) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in vars(obj).items()
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == num_envs
    }
