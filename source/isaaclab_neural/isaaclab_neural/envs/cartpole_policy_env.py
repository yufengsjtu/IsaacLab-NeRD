# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native Cartpole environments that train the teacher and record control-step NeRD data."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.cartpole.cartpole_manager_env_cfg import CartpoleEnvCfg

from .recording import apply_frame_stepping


@configclass
class CartpolePolicyEnvCfg(CartpoleEnvCfg):
    """Core manager-based Cartpole on the Newton MJWarp preset, the scene of :class:`NerdCartpoleEnvCfg`."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim.physics.default = self.sim.physics.newton_mjwarp


@configclass
class CartpoleControlStepDatasetGenEnvCfg(CartpolePolicyEnvCfg):
    """Policy environment stepped frame by frame for the env-driven collector."""

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_frame_stepping(self)
