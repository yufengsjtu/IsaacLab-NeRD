# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cartpole dataset trajectory sampler."""

from __future__ import annotations

from isaaclab_neural.generate.sampler import ActionTrajectorySampler


class CartpoleTrajectorySampler(ActionTrajectorySampler):
    """Cartpole defaults matching the original direct-force data sampler."""
