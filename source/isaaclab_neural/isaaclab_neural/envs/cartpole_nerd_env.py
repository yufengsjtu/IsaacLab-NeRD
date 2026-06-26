# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Registered Cartpole environment backed by the NeRD Newton solver."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg

from isaaclab_neural.physics import NerdNewtonCfg, NerdSolverCfg

from .neural_env_wrapper import NerdManagerBasedRLEnv


DEFAULT_CARTPOLE_NERD_MODEL_PATH = "/home/rowany/workspace/pre-trained_models/Cartpole/nn/final_model.pt"


@configclass
class NerdCartpoleEnvCfg(CartpoleEnvCfg):
    """Manager-based Cartpole config using NeRD-backed Newton physics."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim.physics = NerdNewtonCfg(
            solver_cfg=NerdSolverCfg(
                name="NeuralSolver",
                neural_model_path=DEFAULT_CARTPOLE_NERD_MODEL_PATH,
                num_contacts_per_env=0,
                contact_mode="fixed_ground",
                contact_packing_policy="stable_index",
                validate_contact_fingerprint=False,
                states_frame="body",
                anchor_frame_step="every",
                prediction_type="relative",
                orientation_prediction_parameterization="quaternion",
                use_cuda_graph=False,
            ),
            num_substeps=1,
            debug_mode=False,
            use_cuda_graph=False,
        )


class NerdCartpoleEnv(NerdManagerBasedRLEnv):
    """Cartpole env that installs ``NerdNewtonCfg`` before IsaacLab initializes simulation."""
