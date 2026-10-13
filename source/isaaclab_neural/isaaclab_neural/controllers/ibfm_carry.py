# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase controller that selects I-BFM latents for picking up, carrying and placing a box.

This is a port of the base ``PhaseController`` of the I-BFM MuJoCo runtime (``carry_phased_sim2sim.py``): approach
the box, lift it, transport it towards the goal, lower and release it, and get up after falls. Latents are blended
on the latent sphere from the released Carry Box latent set. Defaults follow the released runtime, except for
``reapproach_switch_distance``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from isaaclab_neural.controllers.ibfm import IbfmPolicy

IBFM_CARRY_PHASES = ("approach", "lift", "transport", "place", "complete", "get_up", "backoff")
"""Controller phases."""


@dataclass
class IbfmCarryControllerCfg:
    """Thresholds of :class:`IbfmCarryController`; distances [m], heights [m], speeds [m/s], durations [s]."""

    heading_blend_full_deg: float = 60.0
    """Approach bearing at which the turn latent reaches its maximum weight [deg]."""
    max_turn_blend: float = 0.9
    transport_heading_blend_full_deg: float = 45.0
    transport_max_turn_blend: float = 0.45
    transport_near_max_turn_blend: float = 0.8
    """Turn-latent weight close to the goal; the runtime fades towards it inside ``transport_turn_boost_distance``."""
    transport_turn_boost_distance: float = 1.2
    approach_switch_distance: float = 0.75
    reapproach_switch_distance: float = 0.75
    """Approach-to-lift distance after a lost box or failed placement, when the robot starts from a standstill."""
    lift_timeout_seconds: float = 0.0
    """Lift duration without a grasp after which the robot backs off and re-approaches; 0 disables the timeout."""
    backoff_distance: float = 1.0
    backoff_max_seconds: float = 1.5
    backoff_blend_seconds: float = 0.4
    fall_upright_threshold: float = 0.2
    fall_root_height: float = 0.3
    fall_tilted_root_height: float = 0.5
    fall_detect_steps: int = 10
    get_up_upright_threshold: float = 0.85
    get_up_root_height: float = 0.65
    get_up_stable_steps: int = 25
    hold_switch_lift: float = 0.5
    """Normalized lift at which a bilateral grasp starts transport."""
    hold_release_lift: float = 0.25
    hold_switch_upright: float = 0.85
    phase_hold_steps: int = 5
    place_align_distance: float = 0.25
    place_enter_steps: int = 5
    place_drift_distance: float = 0.35
    place_drift_steps: int = 10
    place_abort_distance: float = 0.8
    place_success_distance: float = 0.2
    place_success_height: float = 0.05
    place_success_speed: float = 0.15
    place_success_upright: float = 0.85
    place_success_steps: int = 25
    place_release_height: float = 0.08
    place_settle_seconds: float = 0.2
    place_lower_seconds: float = 1.0
    place_release_seconds: float = 2.0
    place_retry_delay_seconds: float = 1.0
    place_max_attempt_seconds: float = 10.0
    align_goal_lookahead: float = 0.0
    """Distance by which the observed goal is moved further away during placement to cancel stopping short."""
    ground_center_height: float = 0.186028293
    """Object reference-frame height of a box resting on the ground."""
    lift_height: float = 0.35
    """Height above the resting height that counts as fully lifted."""


@dataclass
class IbfmCarryLatents:
    """Carry Box latents, each a projected tensor of shape ``[256]``; ``get_up`` is a sequence ``[T, 256]``."""

    approach: torch.Tensor
    lift: torch.Tensor
    transport: torch.Tensor
    align: torch.Tensor
    lower: torch.Tensor
    place: torch.Tensor
    get_up: torch.Tensor
    turn: dict[int, torch.Tensor] = field(default_factory=dict)
    """Left (``1``) and right (``-1``) turn latents of the approach."""
    transport_turn: dict[int, torch.Tensor] = field(default_factory=dict)
    backoff: torch.Tensor | None = None
    """Backward-walking latent of the released Push Box set; required by a positive ``lift_timeout_seconds``."""
    stand: torch.Tensor | None = None
    """Standing latent from which the backoff blends."""

    @classmethod
    def from_directory(cls, path: str, device: str | torch.device, push_path: str | None = None) -> IbfmCarryLatents:
        """Load the ``z_*.pkl`` files of the released ``tasks/carry_box/latents`` directory.

        Args:
            path: Latent directory.
            device: Tensor device.
            push_path: Optional released ``tasks/push_box/latents`` directory providing the backoff latents.

        Returns:
            The latents.
        """
        import joblib  # noqa: PLC0415

        def load(name: str):
            return joblib.load(os.path.join(path, f"z_{name}.pkl"))

        def tensor(value) -> torch.Tensor:
            return torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device).reshape(-1, 256)

        single = {name: tensor(load(name))[0] for name in ("approach", "lift", "transport", "align", "lower", "place")}
        banks = {name: {d: tensor(load(name)[d])[0] for d in (-1, 1)} for name in ("turn", "transport_turn")}
        extra = {}
        if push_path is not None:
            extra["backoff"] = tensor(joblib.load(os.path.join(push_path, "selected/backoff_replay_v1/z_backoff.pkl")))[
                0
            ]
            extra["stand"] = tensor(joblib.load(os.path.join(push_path, "get_up/z_get_up_stand.pkl")))[0]
        return cls(**single, get_up=tensor(load("get_up_reward")), **banks, **extra)


def _smooth(progress: float) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    return progress * progress * (3.0 - 2.0 * progress)


def _bearing_deg(base_rot: np.ndarray, direction_xy: np.ndarray) -> float:
    local = base_rot[:2, :2].T @ direction_xy
    return math.degrees(math.atan2(local[1], local[0]))


class IbfmCarryController:
    """Phase machine of one environment.

    Call :meth:`update` with the metrics of the current state (except on the first step), then :meth:`latent`.
    Metrics are ``horizontal_distance`` (robot to box), ``goal_distance`` (box to goal, horizontal),
    ``object_height``, ``object_speed``, ``lift`` (normalized), ``root_height``, ``pelvis_up_z``, ``robot_upright``,
    ``box_upright`` and ``bilateral_contact``.
    """

    def __init__(self, cfg: IbfmCarryControllerCfg, dt: float):
        """Initialize the controller in the approach phase.

        Args:
            cfg: Thresholds.
            dt: Control period [s].
        """
        self.cfg = cfg
        self.place_settle_steps = round(cfg.place_settle_seconds / dt)
        self.lift_timeout_steps = round(cfg.lift_timeout_seconds / dt)
        self.backoff_max_steps = max(1, round(cfg.backoff_max_seconds / dt))
        self.backoff_blend_steps = max(1, round(cfg.backoff_blend_seconds / dt))
        self.place_lower_steps = max(1, round(cfg.place_lower_seconds / dt))
        self.place_release_steps = max(1, round(cfg.place_release_seconds / dt))
        self.place_retry_steps = max(1, round(cfg.place_retry_delay_seconds / dt))
        self.place_max_attempt_steps = max(1, round(cfg.place_max_attempt_seconds / dt))
        self.phase, self.phase_steps = "approach", 0
        self.ready_steps = self.lost_steps = self.place_enter_steps = self.place_drift_steps = 0
        self.placed_steps = self.failure_steps = self.completion_lost_steps = 0
        self.release_started_step: int | None = None
        self.place_attempts = self.recoveries = self.falls = 0
        self.fall_detect_steps = self.get_up_stable_steps = 0
        self.reapproach = False

    def _set(self, phase: str) -> None:
        self.phase, self.phase_steps = phase, 0
        self.ready_steps = self.lost_steps = self.place_enter_steps = self.place_drift_steps = 0
        self.placed_steps = self.failure_steps = self.completion_lost_steps = 0
        self.release_started_step = None

    def _recover(self, metrics: dict) -> None:
        # From a standstill, the lift latent does not step towards a box outside arm reach.
        self.reapproach = True
        self._set("approach" if metrics["horizontal_distance"] > self.cfg.reapproach_switch_distance else "lift")

    def latent(
        self, latents: IbfmCarryLatents, metrics: dict, base_rot: np.ndarray, box_dir: np.ndarray, goal_dir: np.ndarray
    ) -> tuple[torch.Tensor, bool, bool]:
        """Return the command latent of the current phase.

        Args:
            latents: Carry Box latents.
            metrics: Current metrics.
            base_rot: Root rotation matrix, shape ``[3, 3]``.
            box_dir: Horizontal robot-to-box vector [m], shape ``[2]``.
            goal_dir: Horizontal box-to-goal vector [m], shape ``[2]``.

        Returns:
            Projected latent ``[256]``, whether the object context must be hidden from the policy, and whether the
            observed goal is moved ahead by ``align_goal_lookahead``.
        """
        cfg, project = self.cfg, IbfmPolicy.project_z
        if self.phase == "get_up":
            return latents.get_up[min(self.phase_steps, latents.get_up.shape[0] - 1)], True, False
        if self.phase == "approach":
            bearing = _bearing_deg(base_rot, box_dir)
            weight = min(abs(bearing) / cfg.heading_blend_full_deg, 1.0) * cfg.max_turn_blend
            turn = latents.turn[1 if bearing >= 0.0 else -1]
            return project((1 - weight) * latents.approach + weight * turn), True, False
        if self.phase == "lift":
            return latents.lift, False, False
        if self.phase == "backoff":
            blend = _smooth(self.phase_steps / self.backoff_blend_steps)
            return project((1 - blend) * latents.stand + blend * latents.backoff), True, False
        if self.phase == "transport":
            bearing = _bearing_deg(base_rot, goal_dir)
            near = _smooth(
                (cfg.transport_turn_boost_distance - metrics["goal_distance"])
                / (cfg.transport_turn_boost_distance - cfg.place_align_distance)
            )
            far = min(abs(bearing) / cfg.transport_heading_blend_full_deg, 1.0) * cfg.transport_max_turn_blend
            weight = (1 - near) * far + near * cfg.transport_near_max_turn_blend
            turn = latents.transport_turn[1 if bearing >= 0.0 else -1]
            return project((1 - weight) * latents.transport + weight * turn), False, False
        if self.phase == "place":
            if self.release_started_step is None:
                blend = _smooth((self.phase_steps - self.place_settle_steps) / self.place_lower_steps)
                return project((1 - blend) * latents.align + blend * latents.lower), False, True
            blend = _smooth((self.phase_steps - self.release_started_step) / self.place_release_steps)
            return project((1 - blend) * latents.lower + blend * latents.place), False, True
        return latents.place, False, False

    def update(self, metrics: dict) -> None:
        """Advance the phase machine.

        Args:
            metrics: Current metrics.
        """
        cfg, m = self.cfg, metrics
        self.phase_steps += 1
        fallen = m["root_height"] < cfg.fall_root_height or (
            m["pelvis_up_z"] < cfg.fall_upright_threshold and m["root_height"] < cfg.fall_tilted_root_height
        )
        self.fall_detect_steps = self.fall_detect_steps + 1 if fallen else 0
        if self.phase == "get_up":
            upright = m["pelvis_up_z"] >= cfg.get_up_upright_threshold and m["root_height"] >= cfg.get_up_root_height
            self.get_up_stable_steps = self.get_up_stable_steps + 1 if upright else 0
            if self.get_up_stable_steps >= cfg.get_up_stable_steps:
                self._recover(m)
                self.fall_detect_steps = self.get_up_stable_steps = 0
            return
        if self.fall_detect_steps >= cfg.fall_detect_steps:
            self.falls += 1
            self._set("get_up")
            self.get_up_stable_steps = 0
            return
        if self.phase == "approach":
            switch = cfg.reapproach_switch_distance if self.reapproach else cfg.approach_switch_distance
            if m["horizontal_distance"] <= switch:
                self._set("lift")
            return
        if self.phase == "lift":
            ready = (
                m["bilateral_contact"]
                and m["lift"] >= cfg.hold_switch_lift
                and m["robot_upright"] >= cfg.hold_switch_upright
            )
            self.ready_steps = self.ready_steps + 1 if ready else 0
            if self.ready_steps >= cfg.phase_hold_steps:
                self._set("transport")
            elif self.lift_timeout_steps > 0 and self.phase_steps >= self.lift_timeout_steps:
                self._set("backoff")
            return
        if self.phase == "backoff":
            if m["horizontal_distance"] >= cfg.backoff_distance or self.phase_steps >= self.backoff_max_steps:
                self.reapproach = False
                self._set("approach")
            return
        if self.phase == "transport":
            lost = not m["bilateral_contact"] or m["lift"] < cfg.hold_release_lift
            self.lost_steps = self.lost_steps + 1 if lost else 0
            inside = not lost and m["goal_distance"] <= cfg.place_align_distance
            self.place_enter_steps = self.place_enter_steps + 1 if inside else 0
            if self.lost_steps >= cfg.phase_hold_steps:
                self._recover(m)
            elif self.place_enter_steps >= cfg.place_enter_steps:
                self._set("place")
                self.place_attempts += 1
            return
        ground_error = abs(m["object_height"] - cfg.ground_center_height)
        resting = (
            m["goal_distance"] <= cfg.place_success_distance
            and ground_error <= cfg.place_success_height
            and m["object_speed"] <= cfg.place_success_speed
            and m["box_upright"] >= cfg.place_success_upright
        )
        if self.phase == "place":
            holding = self.release_started_step is None and m["bilateral_contact"]
            self.place_drift_steps = (
                self.place_drift_steps + 1 if holding and m["goal_distance"] > cfg.place_drift_distance else 0
            )
            if self.place_drift_steps >= cfg.place_drift_steps:
                self._set("transport")
                return
            if holding and ground_error <= cfg.place_release_height:
                self.release_started_step = self.phase_steps
            self.placed_steps = self.placed_steps + 1 if resting and not m["bilateral_contact"] else 0
            if self.placed_steps >= cfg.place_success_steps:
                self._set("complete")
                return
            released = not m["bilateral_contact"]
            failed = (
                (released and ground_error > cfg.place_release_height)
                or (
                    released
                    and ground_error <= cfg.place_success_height
                    and m["object_speed"] <= cfg.place_success_speed
                    and m["goal_distance"] > cfg.place_success_distance
                )
                or self.phase_steps >= self.place_max_attempt_steps
                or m["goal_distance"] > cfg.place_abort_distance
            )
            self.failure_steps = self.failure_steps + 1 if failed else 0
            if self.failure_steps >= self.place_retry_steps:
                self.recoveries += 1
                self._recover(m)
            return
        self.completion_lost_steps = 0 if resting else self.completion_lost_steps + 1
        if self.completion_lost_steps >= self.place_retry_steps:
            self.recoveries += 1
            self._recover(m)
