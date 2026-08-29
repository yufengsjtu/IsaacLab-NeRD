# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import newton
import numpy as np
import warp as wp

from isaaclab_neural.contacts.kernels import collision_detection_ground_kernel


def find_ground_shape_index(model: newton.Model) -> int:
    """Find the ground shape index (shape with body == -1).

    Returns the index, or -1 if no ground shape exists.
    """
    shape_body = model.shape_body.numpy()  # type: ignore[union-attr]
    candidates = np.where(shape_body == -1)[0]
    if len(candidates) == 0:
        return -1
    return int(candidates[0])


# NOTE: only implemented for ground plane for now
# results stored in a specialized newton.Contacts that has the fixed order of contact points
def collision_detection_fixed_ground(
    model: newton.Model,
    state: newton.State,
    contacts: newton.Contacts,
    ground_shape_index: int = -1,
):
    """Reject the deprecated fixed-ground collision path.

    Args:
        model: Newton model.
        state: Newton state with populated body poses.
        contacts: Fixed-order contact buffer.
        ground_shape_index: Ground shape index, or -1 to auto-detect it.

    Raises:
        ValueError: Always, because fixed-ground contacts are unsupported.
    """
    warnings.warn(
        "collision_detection_fixed_ground() is deprecated and incompatible with the native Newton contact path.",
        DeprecationWarning,
        stacklevel=2,
    )
    raise ValueError("fixed_ground contact processing is no longer supported; use newton_native contacts.")

    # Retained until the deprecated public API can be removed in a major release.
    if ground_shape_index < 0:
        ground_shape_index = find_ground_shape_index(model)
        if ground_shape_index < 0:
            return

    wp.launch(
        collision_detection_ground_kernel,
        dim=contacts.rigid_contact_max,
        inputs=[
            state.body_q,
            model.shape_transform,
            model.shape_body,
            ground_shape_index,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_point0,
            model.up_axis.to_vec3(),
        ],
        outputs=[
            contacts.rigid_contact_shape1,
            contacts.rigid_contact_point1,
            contacts.rigid_contact_normal,
            contacts.rigid_contact_depth,
        ],
        device=model.device,
    )
