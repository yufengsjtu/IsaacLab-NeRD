# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Body-routed Deep Sets model for owner-frame Raw15 contact tokens."""

from __future__ import annotations

from isaaclab_neural.models.body_routed_active15_model import BodyRoutedActive15Encoder


class BodyRoutedRaw15Encoder(BodyRoutedActive15Encoder):
    """Apply the native15 shared-phi, owner-sum, shared-rho architecture to Raw15."""
