# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

Changed
^^^^^^^

* Enabled geometry-based ``feet_air_time``, ``undesired_contacts``, and
  ``base_contact`` terms for Anymal-C NeRD when ``contact_mode`` is
  ``newton_native``; they stay off under ``fixed_ground``. These terms use
  signed native-contact separation without synthesizing contact forces.
