# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import newton
import numpy as np
import warp as wp

from isaaclab_neural.contacts.kernels import collision_detection_ground_kernel


def find_ground_shape_index(model: newton.Model) -> int:
    """Find the ground shape index (shape with body == -1).

    Returns the index, or -1 if no ground shape exists.
    """
    shape_body = model.shape_body.numpy()
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
    """Run collision detection against the ground plane.

    Args:
        model: Newton model.
        state: Newton state with populated body_q (run eval_fk first).
        contacts: AbstractContact's newton_contacts view.
        ground_shape_index: Index of the ground shape. If -1, auto-detect
            by searching for shape_body == -1. If no ground exists, this
            function is a no-op.
    """
    if ground_shape_index < 0:
        # Auto-detect ground shape
        ground_shape_index = find_ground_shape_index(model)
        if ground_shape_index < 0:
            return  # no ground plane

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
