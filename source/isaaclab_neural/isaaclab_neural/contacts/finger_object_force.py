# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka finger-to-object normal-force label and deployment contract."""

import h5py
import numpy as np
import torch
import warp as wp

from isaaclab.utils.math import quat_apply

LABEL_KEY = "finger_object_normal_forces_w"
TARGET_KEY = "finger_object_normal_forces_b"
SENSOR_NAMES = ("panda_rightfinger_object_s", "panda_leftfinger_object_s")
BODY_NAMES = ("panda_rightfinger", "panda_leftfinger")
METADATA = {
    "finger_object_force_schema": "franka_filtered_normal_v1",
    "finger_object_force_frame": "world",
    "finger_object_force_units": "N",
    "finger_object_force_timing": "post_solve_t_to_t_plus_1",
    "finger_object_force_body_names": ",".join(BODY_NAMES),
    "finger_object_force_filter": "Object",
}
MODEL_CONTRACT = {**METADATA, "prediction": "gated_vector_v1", "active_threshold_n": 0.01}


def validate_force_metadata(group: h5py.Group) -> None:
    """Validate fixed finger order, filtering, frame and post-solve label timing."""
    if LABEL_KEY not in group:
        if "finger_object_force_schema" in group.attrs:
            raise ValueError(f"Missing {LABEL_KEY} despite declared metadata.")
        return
    field = group[LABEL_KEY]
    states = group["states"]
    if not isinstance(field, h5py.Dataset) or not isinstance(states, h5py.Dataset):
        raise ValueError("Finger force labels and states must be HDF5 datasets.")
    if field.shape != (*states.shape[:2], 2, 3) or not np.issubdtype(field.dtype, np.floating):
        raise ValueError(f"{LABEL_KEY} must have shape [B, T, 2, 3] and floating dtype.")
    if any(group.attrs.get(key) != value for key, value in METADATA.items()):
        raise ValueError("Finger-object normal-force metadata does not match the supported contract.")


def validate_sensors(scene, names) -> None:
    """Require the same filtered normal-force sources as the core Lift policy."""
    if tuple(names) != SENSOR_NAMES:
        raise ValueError(f"Finger force sensor order must be {SENSOR_NAMES}.")
    if not scene["robot"].is_fixed_base:
        raise ValueError("Franka finger force labels currently require a fixed robot root frame.")
    for name, body in zip(names, BODY_NAMES):
        sensor = scene.sensors[name]
        if not sensor.cfg.prim_path.endswith("/" + body):
            raise ValueError(f"{name} does not refer to {body}.")
        if list(sensor.cfg.filter_prim_paths_expr) != [scene["object"].cfg.prim_path]:
            raise ValueError(f"{name} must filter only the Lift object.")


def read_normal_forces(scene, names=SENSOR_NAMES) -> torch.Tensor:
    """Read post-solve normal forces [N], shape [B, right/left finger, xyz], in world axes."""
    values = []
    for name in names:
        force = scene.sensors[name].data.normal_force_matrix_w
        if force is None or tuple(force.torch.shape) != (scene.num_envs, 1, 1, 3):
            raise RuntimeError(f"{name} must provide one finger and one object force vector.")
        values.append(force.torch.reshape(scene.num_envs, 3))
    result = torch.stack(values, dim=1)
    if not torch.isfinite(result).all():
        raise ValueError("Finger-object normal-force labels contain non-finite values.")
    return result


def publish_normal_forces(scene, forces_b: torch.Tensor, names=SENSOR_NAMES) -> None:
    """Publish predicted normal forces [N], shape [B, 2, 3], to Newton sensor source buffers.

    The source buffers feed the existing sensor history, policy clipping, frame
    conversion and reward terms. Only normal_force_matrix_w and its history are
    supported predictions; friction, aggregate forces and contact points are not.
    """
    if forces_b.shape != (scene.num_envs, 2, 3) or not torch.isfinite(forces_b).all():
        raise ValueError("Predicted finger-object normal forces must be finite [B, 2, 3].")
    rotation = scene["robot"].data.root_link_quat_w.torch[:, None].expand(-1, 2, -1)
    forces_w = quat_apply(rotation, forces_b)
    for i, name in enumerate(names):
        sensor = scene.sensors[name]
        matrix = wp.to_torch(sensor._newton_force_matrix_view)
        matrix.copy_(forces_w[:, i].reshape_as(matrix))
        friction = sensor._newton_force_matrix_friction_view
        if friction is not None:
            wp.to_torch(friction).zero_()
        total = wp.to_torch(sensor._newton_total_force_view)
        total.zero_()
        total_friction = sensor._newton_total_force_friction_view
        if total_friction is not None:
            wp.to_torch(total_friction).zero_()
