# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 29-DoF dataset sampler driven by scheduled BFM-Zero latents."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

import isaaclab.utils.math as math_utils

from isaaclab_neural.controllers.bfm_zero import BFM_ZERO_Z_DIM, project_bfm_zero_latent
from isaaclab_neural.generate.adapter import DataGenerationAdapter
from isaaclab_neural.generate.sampler import ActionTrajectorySampler

BFM_LATENT_SOURCES = ("tracking", "goal", "reward", "sphere", "slerp")
"""Latent sources: LAFAN1 tracking sequences, goal-reaching and reward-inference latents, uniform latents on the
sphere, and spherical interpolation from the previous latent to a library latent."""


class BfmLatentLibrary:
    """BFM-Zero latents exported to one ``.npz`` file.

    The file contains ``tracking_z`` (concatenated per-frame tracking latents), ``tracking_offsets`` (clip
    boundaries, length ``num_clips + 1``), ``tracking_holdout`` (clips reserved for evaluation), ``goal_z`` and
    ``reward_z``.
    """

    def __init__(self, path: str, device: str | torch.device, *, clips: str = "train"):
        """Load the library.

        Args:
            path: Exported ``.npz`` file.
            device: Device of the latent tensors.
            clips: Tracking clips to sample: ``train`` (excluding held-out clips), ``holdout`` or ``all``.

        Raises:
            ValueError: ``clips`` is unknown or selects no clip.
        """
        if clips not in ("train", "holdout", "all"):
            raise ValueError(f"Unknown tracking clip selection: {clips}.")
        with np.load(path) as data:
            self.tracking_z = torch.as_tensor(data["tracking_z"], dtype=torch.float32, device=device)
            offsets = data["tracking_offsets"].astype(np.int64)
            holdout = data["tracking_holdout"].astype(bool)
            self.goal_z = torch.as_tensor(data["goal_z"], dtype=torch.float32, device=device)
            self.reward_z = torch.as_tensor(data["reward_z"], dtype=torch.float32, device=device)
        selected = [
            index for index in range(len(offsets) - 1) if clips == "all" or (clips == "holdout") == bool(holdout[index])
        ]
        if not selected:
            raise ValueError(f"{path} has no '{clips}' tracking clips.")
        self.clip_start = torch.as_tensor(offsets[selected], device=device)
        self.clip_end = torch.as_tensor(offsets[[index + 1 for index in selected]], device=device)
        frames = torch.cat(
            [torch.arange(int(s), int(e), device=device) for s, e in zip(self.clip_start, self.clip_end)]
        )
        self.static_z = torch.cat([self.goal_z, self.reward_z, self.tracking_z[frames]])


class BfmLatentTrajectorySampler(ActionTrajectorySampler):
    """Record physics frames while the BFM-Zero latent action follows a per-environment schedule.

    Each environment draws segments with a source from :data:`BFM_LATENT_SOURCES` and a duration from
    ``hold_range_s``. Tracking segments play a random window of a training clip; the other sources hold or
    interpolate one latent. Horizontal force pulses are written to the floating-base generalized force between
    physics frames, so they are recorded as ``joint_f`` inputs.
    """

    def __init__(
        self,
        adapter: DataGenerationAdapter,
        *,
        latent_library: str,
        source_weights: Mapping[str, float] | None = None,
        hold_range_s: tuple[float, float] = (1.0, 3.0),
        push_force_range: tuple[float, float] = (0.0, 0.0),
        push_wait_range_s: tuple[float, float] = (2.0, 5.0),
        push_duration_s: float = 0.1,
        clips: str = "train",
        action_term: str = "bfm",
        **kwargs: Any,
    ):
        """Initialize the sampler.

        Args:
            adapter: Data-generation adapter of a BFM-Zero environment.
            latent_library: Exported latent library (see :class:`BfmLatentLibrary`).
            source_weights: Relative weights per source. Defaults to equal weights.
            hold_range_s: Segment duration range [s].
            push_force_range: Horizontal push magnitude range [N]; ``(0, 0)`` disables pushes.
            push_wait_range_s: Range of the interval between push onsets [s].
            push_duration_s: Push duration [s].
            clips: Tracking clips to sample, see :class:`BfmLatentLibrary`.
            action_term: Name of the :class:`~isaaclab_neural.controllers.bfm_zero.BfmZeroLatentAction` term.
            **kwargs: Forwarded to :class:`ActionTrajectorySampler`.
        """
        super().__init__(adapter, **kwargs)
        env = adapter.env
        self.device = torch.device(adapter.device)
        self.num_envs = adapter.num_envs
        self.decimation = max(1, int(env.cfg.decimation))
        physics_dt = float(env.cfg.sim.dt)
        control_dt = physics_dt * self.decimation
        self.library = BfmLatentLibrary(latent_library, self.device, clips=clips)
        weights = {name: 1.0 for name in BFM_LATENT_SOURCES} if source_weights is None else dict(source_weights)
        unknown = set(weights) - set(BFM_LATENT_SOURCES)
        if unknown:
            raise ValueError(f"Unknown BFM latent sources: {sorted(unknown)}.")
        self.source_probs = torch.tensor(
            [float(weights.get(name, 0.0)) for name in BFM_LATENT_SOURCES], device=self.device
        )
        if float(self.source_probs.sum()) <= 0.0:
            raise ValueError("At least one BFM latent source needs a positive weight.")
        self.hold_steps = tuple(max(1, round(value / control_dt)) for value in hold_range_s)
        self.push_force_range = push_force_range
        self.push_wait_frames = tuple(max(1, round(value / physics_dt)) for value in push_wait_range_s)
        self.push_frames = max(1, round(push_duration_s / physics_dt))
        self.action_term = env.action_manager.get_term(action_term)
        self._source = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._remaining = torch.zeros_like(self._source)
        self._duration = torch.ones_like(self._source)
        self._cursor = torch.zeros_like(self._source)
        self._cursor_end = torch.zeros_like(self._source)
        self._start_z = torch.zeros(self.num_envs, BFM_ZERO_Z_DIM, device=self.device)
        self._target_z = torch.zeros_like(self._start_z)
        self._latent = project_bfm_zero_latent(torch.randn_like(self._start_z))
        self._push_wrench = torch.zeros(self.num_envs, 6, device=self.device)
        self._push_left = torch.zeros_like(self._source)
        self._push_wait = torch.zeros_like(self._source)
        self._dof_starts: torch.Tensor | None = None
        self._joint_f: torch.Tensor | None = None
        if push_force_range[1] > 0.0:
            import warp as wp  # noqa: PLC0415
            from isaaclab_newton.physics import NewtonManager  # noqa: PLC0415

            from isaaclab_neural.envs.mdp_g1_29dof import newton_free_joint_dof_starts  # noqa: PLC0415

            self._dof_starts = newton_free_joint_dof_starts(self.num_envs, self.device)
            self._joint_f = wp.to_torch(NewtonManager.get_control().joint_f)

    def _random_static(self, count: int) -> torch.Tensor:
        return self.library.static_z[torch.randint(0, len(self.library.static_z), (count,), device=self.device)]

    def _start_segments(self, env_ids: torch.Tensor) -> None:
        count = len(env_ids)
        if count == 0:
            return
        source = torch.multinomial(self.source_probs, count, replacement=True)
        duration = torch.randint(self.hold_steps[0], self.hold_steps[1] + 1, (count,), device=self.device)
        self._source[env_ids], self._remaining[env_ids], self._duration[env_ids] = source, duration, duration
        self._start_z[env_ids] = self._latent[env_ids]
        target = torch.empty(count, BFM_ZERO_Z_DIM, device=self.device)
        for index, name in enumerate(BFM_LATENT_SOURCES):
            mask = source == index
            selected = int(mask.sum())
            if selected == 0:
                continue
            if name == "goal":
                target[mask] = self.library.goal_z[
                    torch.randint(0, len(self.library.goal_z), (selected,), device=self.device)
                ]
            elif name == "reward":
                target[mask] = self.library.reward_z[
                    torch.randint(0, len(self.library.reward_z), (selected,), device=self.device)
                ]
            elif name == "sphere":
                target[mask] = project_bfm_zero_latent(torch.randn(selected, BFM_ZERO_Z_DIM, device=self.device))
            elif name == "slerp":
                target[mask] = self._random_static(selected)
            else:
                clip = torch.randint(0, len(self.library.clip_start), (selected,), device=self.device)
                start, end = self.library.clip_start[clip], self.library.clip_end[clip]
                span = (end - start - duration[mask]).clamp_min(1)
                cursor = start + (torch.rand(selected, device=self.device) * span).long()
                ids = env_ids[mask]
                self._cursor[ids], self._cursor_end[ids] = cursor, end
                target[mask] = self.library.tracking_z[cursor]
        self._target_z[env_ids] = target

    def _next_latent(self) -> torch.Tensor:
        self._start_segments((self._remaining <= 0).nonzero().flatten())
        latent = self._target_z.clone()
        tracking = self._source == BFM_LATENT_SOURCES.index("tracking")
        if tracking.any():
            cursor = torch.minimum(self._cursor[tracking], self._cursor_end[tracking] - 1)
            latent[tracking] = self.library.tracking_z[cursor]
            self._cursor[tracking] += 1
        slerp = self._source == BFM_LATENT_SOURCES.index("slerp")
        if slerp.any():
            fraction = (1.0 - self._remaining[slerp].float() / self._duration[slerp].float())[:, None]
            start = torch.nn.functional.normalize(self._start_z[slerp], dim=-1)
            end = torch.nn.functional.normalize(self._target_z[slerp], dim=-1)
            omega = torch.arccos((start * end).sum(-1, keepdim=True).clamp(-0.9999, 0.9999))
            latent[slerp] = (
                torch.sin((1.0 - fraction) * omega) * start + torch.sin(fraction * omega) * end
            ) / torch.sin(omega)
        self._remaining -= 1
        self._latent = project_bfm_zero_latent(latent)
        return self._latent

    def _reset_schedule(self) -> None:
        self._remaining.zero_()
        self._latent = self._random_static(self.num_envs)
        self._start_segments(torch.arange(self.num_envs, device=self.device))
        self._push_left.zero_()
        self._push_wrench.zero_()
        self._push_wait = torch.randint(
            self.push_wait_frames[0], self.push_wait_frames[1] + 1, (self.num_envs,), device=self.device
        )
        self._write_push(torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))

    def _write_push(self, active: torch.Tensor) -> None:
        if self._joint_f is None or self._dof_starts is None:
            return
        from isaaclab_neural.envs.mdp_g1_29dof import write_root_wrench  # noqa: PLC0415

        write_root_wrench(self._joint_f, self._dof_starts, self._push_wrench * active[:, None])

    def _pre_step_hook(self, step: int) -> None:
        if self._joint_f is None:
            return
        self._push_wait -= 1
        start = (self._push_wait <= 0) & (self._push_left == 0)
        count = int(start.sum())
        if count:
            yaw = torch.rand(count, device=self.device) * 2.0 * math.pi
            force = math_utils.sample_uniform(*self.push_force_range, (count,), device=self.device)
            self._push_wrench[start] = 0.0
            self._push_wrench[start, 0] = force * torch.cos(yaw)
            self._push_wrench[start, 1] = force * torch.sin(yaw)
            self._push_left[start] = self.push_frames
            self._push_wait[start] = torch.randint(
                self.push_wait_frames[0], self.push_wait_frames[1] + 1, (count,), device=self.device
            )
        active = self._push_left > 0
        self._write_push(active)
        self._push_left[active] -= 1

    def sample_trajectories_latent_mode(
        self,
        num_transitions: int,
        *,
        initial_states_source: str = "env",
        render: bool = False,
    ) -> dict[str, Any]:
        """Collect trajectories driven by scheduled BFM-Zero latents.

        Args:
            num_transitions: Number of transitions to collect.
            initial_states_source: Trajectory reset source; ``env`` applies the environment reset events.
            render: Whether to render each frame.

        Returns:
            Rollouts without actions; ``joint_f`` holds the applied joint efforts and any push wrench.
        """

        def step_fn(step: int) -> torch.Tensor:
            process = step % self.decimation == 0
            latent = self._next_latent() if process else self._latent
            self.adapter.step_action_frame(latent, process_action=process)
            return latent

        return self._sample_loop(
            num_transitions,
            record_actions=False,
            initial_states_source=initial_states_source,
            initial_states_pool=None,
            render=render,
            step_fn=step_fn,
            pre_rollout_fn=self._reset_schedule,
        )
