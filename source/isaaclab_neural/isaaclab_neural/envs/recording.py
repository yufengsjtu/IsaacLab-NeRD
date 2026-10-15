# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers shared by the native twins that the env-driven NeRD collector records."""

from __future__ import annotations

from typing import Any


def apply_frame_stepping(env_cfg: Any) -> None:
    """Make Isaac Lab advance and update the scene once per physics frame.

    CUDA graphs are disabled on the Newton MJWarp preset and explicit actuators run on the Isaac Lab path: with
    graph-safe Newton actuators the physics manager folds the whole decimation loop into one step, which hides the
    per-frame scene updates the recorder observes. Dynamics are otherwise unchanged.

    Args:
        env_cfg: A manager-based environment config whose ``sim.physics`` is a Newton preset or a Newton config.
    """
    physics = env_cfg.sim.physics
    newton_cfg = getattr(physics, "newton_mjwarp", physics)
    newton_cfg.use_cuda_graph = False
    env_cfg.sim.use_newton_actuators = False
