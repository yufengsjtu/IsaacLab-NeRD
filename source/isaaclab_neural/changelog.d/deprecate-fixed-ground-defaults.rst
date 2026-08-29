# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

Deprecated
^^^^^^^^^^

* Deprecated ``fixed_ground`` contact mode in favor of ``newton_native``.
  Updated configurations should select ``newton_native``; legacy
  ``fixed_ground`` inputs now warn and are rejected because the contact
  representations are incompatible. Regenerate fixed-ground datasets and
  checkpoints with native contacts before upgrading.

Added
^^^^^

* Added contact-free NeRD execution with ``num_contacts_per_env=0`` for
  Cartpole. Anymal-C configurations continue to use a positive contact
  capacity.

Changed
^^^^^^^

* **Breaking:** Removed the legacy ``anymal_fixed_ground`` and
  ``cartpole_fixed_ground`` OSMO presets. Use ``anymal_newton_native`` and
  ``cartpole_newton_native`` instead.
* **Breaking:** Removed ``anymal_newton_native_shared_per_body_smoke`` and
  ``anymal_newton_native_shared_per_body_small_data_smoke``. Use
  ``anymal_newton_native_shared_per_body`` instead.
* **Breaking:** Removed the rough shared-per-body size and data-budget
  ablations. Use ``anymal_rough_formal`` instead.
* **Breaking:** Removed the rough contact-force and RoPE smoke presets. Use
  ``anymal_rough_newton_native_shared_per_body_contact_force_formal`` and
  ``anymal_rough_newton_native_shared_per_body_rope_formal`` respectively.
* **Breaking:** Removed internal benchmark modules, diagnostic OSMO presets,
  and their configurations. Use :mod:`isaaclab_neural.eval.eval` for
  evaluation, ``NERD_STEP_PROFILE=1`` for segmented timing, and the documented
  formal OSMO presets for training.
* **Breaking:** Changed the default native contact capacity from zero to 64.
  Contact-free systems must explicitly set ``num_contacts_per_env=0``.
