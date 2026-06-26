# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physics manager configuration for NeRD-backed Newton simulation."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

from .nerd_solver_cfg import NerdSolverCfg


@configclass
class NerdNewtonCfg(NewtonCfg):
    """Newton physics config that uses ``NewtonNerdManager``.

    Use this instead of modifying ``isaaclab_newton.physics.NewtonManager`` when
    enabling ``solver_type="nerd"``.
    """

    solver_cfg: NerdSolverCfg = NerdSolverCfg()
    """NeRD solver configuration."""

    use_cuda_graph: bool = False
    """Whether to capture NeRD simulation in a CUDA graph."""
