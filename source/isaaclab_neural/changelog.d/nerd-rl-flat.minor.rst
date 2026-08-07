# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

Added
^^^^^

* Added RSL-RL policy learning support for
  ``Isaac-Velocity-Flat-Anymal-C-NeRD-v0`` via
  :class:`~isaaclab_neural.envs.agents.anymal_c_nerd_rsl_rl_ppo_cfg.AnymalCFlatNeRDPPORunnerCfg`
  and ``isaaclab_neural.rl.rsl_rl`` train/play entry points. Flat NeRD disables
  ``push_robot``, keeps the stock 48-D observation layout, and raises tracking
  reward weights to ``2.0`` / ``1.0``.
