# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pack raw Newton contacts into owner-body-frame Raw15 tokens."""

from __future__ import annotations

from isaaclab_neural.contacts.active15_contact_encoder import Active15ContactEncoder


class Raw15ContactEncoder(Active15ContactEncoder):
    """Encode every canonical raw robot contact without a distance cutoff."""

    _solver_active_only = False
