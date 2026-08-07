# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

Changed
^^^^^^^

* Vectorized Newton-native flat contact packing in
  :class:`~isaaclab_neural.contacts.newton_contact_adapter.NewtonContactAdapter`
  and avoided redundant clones in :meth:`~isaaclab_neural.contacts.newton_contact_adapter.NewtonContactAdapter.to_neural_inputs`
  (history snapshots still clone). Packing semantics are unchanged.

Added
^^^^^

* Added opt-in NeRD step profiling via ``NERD_STEP_PROFILE=1``
  (:mod:`isaaclab_neural.utils.step_profile`) and
  :mod:`isaaclab_neural.eval.benchmark_nerd_contact_modes` to compare
  stock ground-truth MJWarp, NeRD ``fixed_ground``, and NeRD
  ``newton_native`` step costs.
