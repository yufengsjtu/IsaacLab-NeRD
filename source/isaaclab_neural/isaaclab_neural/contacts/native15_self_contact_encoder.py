# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Owner-frame native15 encoders that include directed robot self-collisions."""

from __future__ import annotations

from isaaclab_neural.contacts.active15_contact_encoder import Active15ContactEncoder


class Active15SelfContactEncoder(Active15ContactEncoder):
    """Encode solver-active contacts with two owner-directed rows per self pair."""

    _include_robot_self_collisions = True


class Raw15SelfContactEncoder(Active15SelfContactEncoder):
    """Encode all raw candidates with two owner-directed rows per self pair."""

    _solver_active_only = False
