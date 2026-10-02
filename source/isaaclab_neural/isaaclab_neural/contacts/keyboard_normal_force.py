# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SO101 robot normal-force labels, auxiliary loss, and sensor publication.

Only normal forces are predicted. Friction buffers are cleared; reconstructed
aggregate force therefore contains the normal component alone. Keyboard tasks
consume the explicit normal-force history, not an inferred friction signal.
"""

from __future__ import annotations

import json
from typing import Any

import h5py
import numpy as np
import torch
import warp as wp

LABEL_KEY = "robot_net_normal_forces_w"
RAW_KEY = "robot_net_normal_forces_raw_w"
LOGITS_KEY = "robot_normal_contact_logits"
BODY_NAMES = ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1")
BODY_NAME_ORDERS = (BODY_NAMES, tuple(name + "_link" for name in BODY_NAMES))
SENSOR_NAME = "robot_contact"
MODEL_CONTRACT: dict[str, Any] = {
    "schema": "so101_keyboard_robot_normal_v1",
    "frame": "world",
    "units": "N",
    "timing": "post_solve_before_reset",
    "body_names": BODY_NAMES,
    "prediction": "gated_vector_v1",
    "active_threshold_n": 0.01,
}


def validate_force_metadata(group: h5py.Group) -> None:
    """Check the ordered, world-frame robot normal-force dataset contract."""
    if LABEL_KEY not in group:
        return
    from isaaclab_neural.data.keyboard import validate_keyboard_metadata

    encoded = group.attrs.get("keyboard_contract")
    if not isinstance(encoded, (str, bytes)):
        raise ValueError("Robot normal-force labels require keyboard contract metadata.")
    metadata = json.loads(encoded)
    if not isinstance(metadata, dict):
        raise ValueError("Invalid keyboard metadata.")
    validate_keyboard_metadata(metadata)
    if tuple(metadata["force_body_names"]) not in BODY_NAME_ORDERS:
        raise ValueError("Robot normal-force dataset body order differs from the SO101 contract.")
    forces, states = group[LABEL_KEY], group["states"]
    if not isinstance(forces, h5py.Dataset) or not isinstance(states, h5py.Dataset):
        raise ValueError("Normal-force labels and states must be datasets.")
    if forces.shape != (*states.shape[:2], len(BODY_NAMES), 3) or not np.issubdtype(forces.dtype, np.floating):
        raise ValueError("Robot normal-force labels must be floating [B, T, 7, 3].")
    if "contacts/native_pairs" not in group.file:
        raise ValueError("Robot normal-force labels require native contact storage.")


def validate_sensor(scene: Any, sensor_name: str = SENSOR_NAME) -> None:
    """Require the unfiltered SO101 sensor in the trained body order."""
    sensor = scene.sensors[sensor_name]
    names = tuple(str(name).rstrip("/").rsplit("/", 1)[-1] for name in sensor.sensor_names)
    if names not in BODY_NAME_ORDERS:
        raise ValueError(f"Robot normal-force sensor body order must be {BODY_NAMES}, got {names}.")
    if sensor.cfg.filter_prim_paths_expr:
        raise ValueError("Robot normal-force publication requires an unfiltered sensor.")


def publish_normal_forces(scene: Any, forces_w: torch.Tensor, sensor_name: str = SENSOR_NAME) -> None:
    """Write predicted normal forces [N], shape [B, 7, 3], into Newton source views."""
    validate_sensor(scene, sensor_name)
    if forces_w.shape != (scene.num_envs, len(BODY_NAMES), 3) or not bool(torch.isfinite(forces_w).all()):
        raise ValueError("Robot normal-force predictions must be finite [B, 7, 3].")
    sensor = scene.sensors[sensor_name]
    total = wp.to_torch(sensor._newton_total_force_view)
    total.copy_(forces_w.reshape_as(total))
    wp.to_torch(sensor._newton_total_force_friction_view).zero_()


def normal_force_loss(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    rms: Any,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return normalized vector regression plus active-contact BCE and physical metrics."""
    if target.ndim != 4 or target.shape[-2:] != (len(BODY_NAMES), 3) or not bool(torch.isfinite(target).all()):
        raise ValueError("Robot normal-force targets must be finite [B, T, 7, 3].")
    if rms is None:
        raise RuntimeError("Robot normal-force training requires train-set RMS statistics.")
    active = target.norm(dim=-1) > MODEL_CONTRACT["active_threshold_n"]
    vector_loss = torch.nn.functional.mse_loss(rms.normalize(outputs[RAW_KEY]), rms.normalize(target))
    contact_loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs[LOGITS_KEY], active.to(target.dtype))
    with torch.no_grad():
        prediction = outputs[LABEL_KEY]
        error_sq = (prediction - target).square().sum(-1)
        predicted_active = outputs[LOGITS_KEY] > 0
        true_positive = (active & predicted_active).sum()
        active_count = active.sum()
        metrics = {
            "robot_normal_normalized_MSE": vector_loss.detach(),
            "robot_normal_contact_BCE": contact_loss.detach(),
            "robot_normal_RMSE_N": error_sq.mean().sqrt(),
            # Zero when no contact is active (the sum is zero and the count is clamped).
            "robot_normal_active_RMSE_N": ((error_sq * active).sum() / active_count.clamp_min(1)).sqrt(),
            "robot_normal_active_fraction": active.float().mean(),
            "robot_normal_contact_precision": true_positive / predicted_active.sum().clamp_min(1),
            "robot_normal_contact_recall": true_positive / active_count.clamp_min(1),
            "robot_normal_overload_disagreement": (
                (prediction.norm(dim=-1).amax(-1) > 20.0) != (target.norm(dim=-1).amax(-1) > 20.0)
            )
            .float()
            .mean(),
        }
        # One device-to-host transfer for all metrics.
        metrics = dict(zip(metrics, torch.stack([value.float() for value in metrics.values()]).tolist(), strict=True))
    return vector_loss + contact_loss, metrics
