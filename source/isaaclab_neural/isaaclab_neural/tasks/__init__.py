# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-free registry of NeRD tasks for the control-step pipeline.

See :mod:`isaaclab_neural.tasks.specs` and ``docs/tasks.md``.
"""

from .specs import TASK_SPECS, NerdTaskSpec, expand_experiment, get_task_spec, register_task_spec

__all__ = ["TASK_SPECS", "NerdTaskSpec", "expand_experiment", "get_task_spec", "register_task_spec"]
