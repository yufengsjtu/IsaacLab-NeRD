# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""I-BFM interactive behavioural foundation model as a latent-conditioned controller for the G1 29-DoF robot.

I-BFM (Han et al., 2026) extends BFM-Zero with box state and hand-box contact. Its actor keeps the BFM-Zero layout
with a 98-dimensional state (BFM-Zero state plus a 34-dimensional object context) and is prompted with the
spherical log map from the backward embedding of the current state to the commanded latent. This module
re-implements the actor, the backward map and the observation convention of the I-BFM MuJoCo runtime in plain
PyTorch. The weights are licensed CC BY-NC 4.0 and are loaded from a user-provided file at runtime.

Quaternions follow the Isaac Lab ``(x, y, z, w)`` convention.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

import isaaclab.utils.math as math_utils
from isaaclab.utils import string as string_utils

from isaaclab_neural.controllers.bfm_zero import BFM_ZERO_Z_DIM, BfmZeroActor
from isaaclab_neural.robots.g1_29dof import (
    G1_29DOF_IBFM_ACTION_SCALE,
    G1_29DOF_IBFM_EFFORT_LIMIT,
    G1_29DOF_IBFM_STIFFNESS,
)

IBFM_POLICY_PATH_ENV_VAR = "IBFM_POLICY_PATH"
"""Environment variable naming the exported policy file."""

IBFM_OBJECT_CONTEXT_DIM = 34
"""Object context: 18 object features, 2 hand contacts, 12 hand-spatial features, object presence and task id."""

IBFM_PRIVILEGED_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
)
"""Bodies of the privileged state, root first; a virtual head above ``torso_link`` is appended."""

IBFM_HEAD_OFFSET = (0.0, 0.0, 0.35)
"""Offset of the virtual head body in the torso frame [m]."""

IBFM_HAND_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
"""Left and right hand bodies."""

IBFM_HAND_CONTACT_OFFSETS = ((0.11, -0.01, 0.0), (0.11, 0.01, 0.0))
"""Midpoints of the left and right hand collision capsules in the hand body frames [m]."""

IBFM_HAND_CAPSULE_SEGMENTS = (((0.07, 0.0, 0.0), (0.15, -0.02, 0.0)), ((0.07, 0.0, 0.0), (0.15, 0.02, 0.0)))
"""Centre lines of the left and right hand collision capsules in the hand body frames [m]."""

IBFM_HAND_CAPSULE_RADIUS = 0.035
"""Radius of the hand collision capsules [m]."""


def ibfm_quat_to_tan_norm(quat: torch.Tensor) -> torch.Tensor:
    """Encode rotations by the rotated x and z axes.

    Args:
        quat: Quaternions ``(x, y, z, w)``, shape ``[..., 4]``.

    Returns:
        Rotated x axis followed by the rotated z axis, shape ``[..., 6]``.
    """
    flat = quat.reshape(-1, 4)
    axes = torch.zeros(flat.shape[0], 2, 3, device=quat.device, dtype=quat.dtype)
    axes[:, 0, 0] = 1.0
    axes[:, 1, 2] = 1.0
    rotated = math_utils.quat_apply(flat[:, None].expand(-1, 2, -1), axes)
    return rotated.reshape(*quat.shape[:-1], 6)


def ibfm_sphere_log_map(base: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Map ``target`` to the tangent space of the latent sphere at ``base``, preserving arc length.

    Args:
        base: Base latents, shape ``[..., D]``.
        target: Target latents, shape ``[..., D]``.
        eps: Normalization epsilon.

    Returns:
        Tangent vectors of norm ``sqrt(D) * angle(base, target)``, shape ``[..., D]``.
    """
    radius = math.sqrt(base.shape[-1])
    x = F.normalize(base, dim=-1, eps=eps)
    y = F.normalize(target, dim=-1, eps=eps)
    cosine = (x * y).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    tangent = y - cosine * x
    sine = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    theta = torch.atan2(sine, cosine)
    basis = torch.zeros_like(x).scatter_(-1, x.abs().argmin(dim=-1, keepdim=True), 1.0)
    antipodal = F.normalize(basis - (basis * x).sum(dim=-1, keepdim=True) * x, dim=-1, eps=eps)
    direction = torch.where((sine <= eps) & (cosine < 0.0), antipodal, tangent / sine.clamp_min(eps))
    return radius * theta * direction


class IbfmPolicy(nn.Module):
    """Deterministic I-BFM actor prompted through the backward map.

    The actor is a :class:`~isaaclab_neural.controllers.bfm_zero.BfmZeroActor` over ``[state, last_action,
    history_actor]``. Its latent input is the spherical log map from the backward embedding of the normalized
    ``[state, privileged_state]`` to the projected command latent.
    """

    def __init__(
        self,
        state_dim: int = 98,
        privileged_dim: int = 497,
        action_dim: int = 29,
        history_dim: int = 372,
        z_dim: int = BFM_ZERO_Z_DIM,
        hidden_dim: int = 2048,
        hidden_layers: int = 6,
        embedding_layers: int = 2,
        backward_hidden_dim: int = 256,
        norm_eps: float = 1e-5,
    ):
        """Initialize the policy.

        Args:
            state_dim: Actor state dimension.
            privileged_dim: Privileged state dimension, an input of the backward map only.
            action_dim: Number of actuated joints.
            history_dim: Actor history dimension.
            z_dim: Latent dimension.
            hidden_dim: Width of the actor trunk.
            hidden_layers: Number of residual blocks in the actor head.
            embedding_layers: Number of blocks in each actor embedding.
            backward_hidden_dim: Hidden width of the backward map.
            norm_eps: Variance epsilon of the observation normalizers.
        """
        super().__init__()
        self.state_dim = state_dim
        self.actor = BfmZeroActor(
            obs_dim=state_dim + action_dim + history_dim,
            z_dim=z_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            embedding_layers=embedding_layers,
            norm_eps=norm_eps,
        )
        self.backward = nn.Sequential(
            nn.Linear(state_dim + privileged_dim, backward_hidden_dim),
            nn.LayerNorm(backward_hidden_dim),
            nn.Tanh(),
            nn.Linear(backward_hidden_dim, z_dim),
        )
        self.register_buffer("backward_mean", torch.zeros(state_dim + privileged_dim))
        self.register_buffer("backward_var", torch.ones(state_dim + privileged_dim))
        self.norm_eps = norm_eps

    @staticmethod
    def project_z(z: torch.Tensor) -> torch.Tensor:
        """Project latents onto the sphere of radius ``sqrt(D)``.

        Args:
            z: Latents, shape ``[..., D]``.

        Returns:
            Projected latents with the same shape.
        """
        return math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)

    def embed(self, state: torch.Tensor, privileged_state: torch.Tensor) -> torch.Tensor:
        """Compute the backward embedding of the current state.

        Args:
            state: Actor states, shape ``[N, state_dim]``.
            privileged_state: Privileged states, shape ``[N, privileged_dim]``.

        Returns:
            Projected embeddings, shape ``[N, z_dim]``.
        """
        obs = torch.cat([state, privileged_state], dim=-1)
        obs = (obs - self.backward_mean) / torch.sqrt(self.backward_var + self.norm_eps)
        return self.project_z(self.backward(obs))

    def forward(
        self,
        state: torch.Tensor,
        privileged_state: torch.Tensor,
        last_action: torch.Tensor,
        history_actor: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the mean action.

        Args:
            state: Actor states, shape ``[N, state_dim]``.
            privileged_state: Privileged states, shape ``[N, privileged_dim]``.
            last_action: Previous rescaled actions, shape ``[N, action_dim]``.
            history_actor: Actor histories, shape ``[N, history_dim]``.
            z: Command latents, shape ``[N, z_dim]``; projected internally.

        Returns:
            Actions in ``[-1, 1]``, shape ``[N, action_dim]``.
        """
        prompt = ibfm_sphere_log_map(self.embed(state, privileged_state), self.project_z(z))
        return self.actor(torch.cat([state, last_action, history_actor], dim=-1), prompt)


def load_ibfm_policy(path: str, device: str | torch.device) -> IbfmPolicy:
    """Load an exported I-BFM policy.

    The file is a ``torch.save`` dictionary with ``"actor"`` (state dictionary of the I-BFM ``_actor`` module),
    ``"backward"`` (state dictionary of the backward map ``net``), ``"normalizer"`` (``mean`` and ``var`` per
    observation key ``state``, ``privileged_state``, ``last_action`` and ``history_actor``) and ``"meta"``.

    Args:
        path: Exported policy file.
        device: Device of the returned module.

    Returns:
        Frozen policy in evaluation mode.

    Raises:
        KeyError: The file lacks a required entry.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    missing = {"actor", "backward", "normalizer", "meta"} - set(payload)
    if missing:
        raise KeyError(f"{path} is not an exported I-BFM policy (missing {sorted(missing)}).")
    meta, stats = payload["meta"], payload["normalizer"]
    dims = meta["obs_dims"]
    policy = IbfmPolicy(
        state_dim=dims["state"],
        privileged_dim=dims["privileged_state"],
        action_dim=dims["last_action"],
        history_dim=dims["history_actor"],
        z_dim=meta["z_dim"],
        hidden_dim=meta["hidden_dim"],
        hidden_layers=meta["hidden_layers"],
        embedding_layers=meta["embedding_layers"],
        backward_hidden_dim=meta["backward_hidden_dim"],
        norm_eps=meta["norm_eps"],
    )
    actor_keys, backward_keys = ("state", "last_action", "history_actor"), ("state", "privileged_state")
    policy.actor.load_state_dict(
        {
            **payload["actor"],
            "obs_mean": torch.cat([stats[key]["mean"] for key in actor_keys]),
            "obs_var": torch.cat([stats[key]["var"] for key in actor_keys]),
        }
    )
    policy.backward.load_state_dict({key.replace("net.", ""): value for key, value in payload["backward"].items()})
    policy.backward_mean.copy_(torch.cat([stats[key]["mean"] for key in backward_keys]))
    policy.backward_var.copy_(torch.cat([stats[key]["var"] for key in backward_keys]))
    return policy.to(device).eval().requires_grad_(False)


def ibfm_action_scale(joint_names: Sequence[str]) -> list[float]:
    """Return the per-joint target scale ``action_scale * effort_limit / stiffness`` of the I-BFM runtime.

    Args:
        joint_names: Joint names in action order.

    Returns:
        Target offset per unit rescaled action [rad].
    """

    def resolve(values: dict[str, float]) -> list[float]:
        return string_utils.resolve_matching_names_values(values, joint_names, preserve_order=False)[2]

    scale, effort = resolve(G1_29DOF_IBFM_ACTION_SCALE), resolve(G1_29DOF_IBFM_EFFORT_LIMIT)
    return [s * e / k for s, e, k in zip(scale, effort, resolve(G1_29DOF_IBFM_STIFFNESS))]


def ibfm_object_observation(
    base_pos: torch.Tensor,
    base_quat: torch.Tensor,
    object_pos: torch.Tensor,
    object_quat: torch.Tensor,
    object_lin_vel: torch.Tensor,
    object_ang_vel: torch.Tensor,
    target_pos: torch.Tensor,
) -> torch.Tensor:
    """Return the root-local object pose, velocity and object-to-target displacement.

    Args:
        base_pos: Root positions in the world frame [m], shape ``[N, 3]``.
        base_quat: Root orientations, shape ``[N, 4]``.
        object_pos: Object reference-frame positions [m], shape ``[N, 3]``.
        object_quat: Object reference-frame orientations, shape ``[N, 4]``.
        object_lin_vel: Object linear velocities in the world frame [m/s], shape ``[N, 3]``.
        object_ang_vel: Object angular velocities in the world frame [rad/s], shape ``[N, 3]``.
        target_pos: Target object positions in the world frame [m], shape ``[N, 3]``.

    Returns:
        Position (3), tangent-normal rotation (6), linear (3) and angular (3) velocity, and target displacement (3),
        all in the root frame, shape ``[N, 18]``.
    """
    rotate = lambda value: math_utils.quat_apply_inverse(base_quat, value)  # noqa: E731
    rotation = ibfm_quat_to_tan_norm(math_utils.quat_mul(math_utils.quat_conjugate(base_quat), object_quat))
    return torch.cat(
        [
            rotate(object_pos - base_pos),
            rotation,
            rotate(object_lin_vel),
            rotate(object_ang_vel),
            rotate(target_pos - object_pos),
        ],
        dim=-1,
    )


def ibfm_hand_contact_points(hand_pos: torch.Tensor, hand_quat: torch.Tensor) -> torch.Tensor:
    """Return the world positions of the hand capsule midpoints.

    Args:
        hand_pos: Left and right hand body positions [m], shape ``[N, 2, 3]``.
        hand_quat: Left and right hand body orientations, shape ``[N, 2, 4]``.

    Returns:
        Capsule midpoints [m], shape ``[N, 2, 3]``.
    """
    offsets = torch.tensor(IBFM_HAND_CONTACT_OFFSETS, device=hand_pos.device, dtype=hand_pos.dtype)
    return hand_pos + math_utils.quat_apply(hand_quat, offsets.expand_as(hand_pos))


def ibfm_hand_spatial_observation(
    base_pos: torch.Tensor, base_quat: torch.Tensor, hand_points: torch.Tensor, object_pos: torch.Tensor
) -> torch.Tensor:
    """Return root-local hand points and hand-minus-object vectors.

    Args:
        base_pos: Root positions [m], shape ``[N, 3]``.
        base_quat: Root orientations, shape ``[N, 4]``.
        hand_points: Hand capsule midpoints [m], shape ``[N, 2, 3]``.
        object_pos: Object reference-frame positions [m], shape ``[N, 3]``.

    Returns:
        Left and right hand points followed by left and right hand-minus-object vectors, shape ``[N, 12]``.
    """
    quat = base_quat[:, None].expand(-1, 2, -1)
    local = math_utils.quat_apply_inverse(quat, hand_points - base_pos[:, None])
    relative = math_utils.quat_apply_inverse(quat, hand_points - object_pos[:, None])
    return torch.cat([local.flatten(1), relative.flatten(1)], dim=-1)


def ibfm_object_context(
    object_observation: torch.Tensor,
    object_present: torch.Tensor,
    hand_contact: torch.Tensor,
    hand_spatial: torch.Tensor,
) -> torch.Tensor:
    """Assemble the object context; all object features are zero when no object is present.

    Args:
        object_observation: Output of :func:`ibfm_object_observation`, shape ``[N, 18]``.
        object_present: Object presence in ``{0, 1}``, shape ``[N]``; also used as task id.
        hand_contact: Left and right hand-object contact in ``{0, 1}``, shape ``[N, 2]``.
        hand_spatial: Output of :func:`ibfm_hand_spatial_observation`, shape ``[N, 12]``.

    Returns:
        Object context, shape ``[N, IBFM_OBJECT_CONTEXT_DIM]``.
    """
    present = object_present.to(object_observation.dtype)[:, None]
    features = torch.cat([object_observation, hand_contact.to(object_observation.dtype), hand_spatial], dim=-1)
    return torch.cat([features * present, present, present], dim=-1)


def ibfm_privileged_state(
    body_pos: torch.Tensor,
    body_quat: torch.Tensor,
    body_lin_vel: torch.Tensor,
    body_ang_vel: torch.Tensor,
    object_context: torch.Tensor,
    torso_index: int = IBFM_PRIVILEGED_BODY_NAMES.index("torso_link"),
) -> torch.Tensor:
    """Return the heading-local whole-body state of the I-BFM backward map.

    The virtual head is placed :data:`IBFM_HEAD_OFFSET` above the torso. As in the I-BFM runtime, its velocity adds
    the cross product of the torso angular velocity with the torso-frame offset.

    Args:
        body_pos: Positions of :data:`IBFM_PRIVILEGED_BODY_NAMES` [m], shape ``[N, B, 3]``.
        body_quat: Orientations of the bodies, shape ``[N, B, 4]``.
        body_lin_vel: Body-origin linear velocities in the world frame [m/s], shape ``[N, B, 3]``.
        body_ang_vel: Angular velocities in the world frame [rad/s], shape ``[N, B, 3]``.
        object_context: Output of :func:`ibfm_object_context`, shape ``[N, C]``.
        torso_index: Index of ``torso_link`` among the bodies.

    Returns:
        Root height, local body positions without the root, local tangent-normal rotations, and local linear and
        angular velocities, followed by the object context, shape ``[N, 15 * (B + 1) - 2 + C]``.
    """
    offset = torch.tensor(IBFM_HEAD_OFFSET, device=body_pos.device, dtype=body_pos.dtype)
    torso_quat, torso_ang_vel = body_quat[:, torso_index], body_ang_vel[:, torso_index]
    head_pos = body_pos[:, torso_index] + math_utils.quat_apply(torso_quat, offset.expand_as(body_pos[:, 0]))
    head_vel = body_lin_vel[:, torso_index] + torch.cross(torso_ang_vel, offset.expand_as(torso_ang_vel), dim=-1)
    pos = torch.cat([body_pos, head_pos[:, None]], dim=1)
    quat = torch.cat([body_quat, torso_quat[:, None]], dim=1)
    lin_vel = torch.cat([body_lin_vel, head_vel[:, None]], dim=1)
    ang_vel = torch.cat([body_ang_vel, torso_ang_vel[:, None]], dim=1)
    heading_inv = math_utils.quat_conjugate(math_utils.yaw_quat(quat[:, 0]))[:, None].expand_as(quat)
    rotate = lambda value: math_utils.quat_apply(heading_inv, value).flatten(1)  # noqa: E731
    rotation = ibfm_quat_to_tan_norm(math_utils.quat_mul(heading_inv, quat)).flatten(1)
    local_pos = rotate(pos - pos[:, :1])[:, 3:]
    return torch.cat([pos[:, 0, 2:3], local_pos, rotation, rotate(lin_vel), rotate(ang_vel), object_context], dim=-1)


def ibfm_hand_box_gap(
    hand_pos: torch.Tensor,
    hand_quat: torch.Tensor,
    box_pos: torch.Tensor,
    box_quat: torch.Tensor,
    box_half_size: Sequence[float],
    num_samples: int = 17,
) -> torch.Tensor:
    """Return the gap between each hand capsule and an oriented box.

    The capsule centre line is sampled uniformly; the error relative to the exact segment distance is below
    ``segment_length / (2 * (num_samples - 1))``.

    Args:
        hand_pos: Left and right hand body positions [m], shape ``[N, 2, 3]``.
        hand_quat: Left and right hand body orientations, shape ``[N, 2, 4]``.
        box_pos: Box centre positions [m], shape ``[N, 3]``.
        box_quat: Box orientations, shape ``[N, 4]``.
        box_half_size: Box half extents in the box frame [m].
        num_samples: Number of samples along each centre line.

    Returns:
        Capsule-surface to box-surface distance, negative when penetrating [m], shape ``[N, 2]``.
    """
    segments = torch.tensor(IBFM_HAND_CAPSULE_SEGMENTS, device=hand_pos.device, dtype=hand_pos.dtype)
    weights = torch.linspace(0.0, 1.0, num_samples, device=hand_pos.device, dtype=hand_pos.dtype)[:, None]
    points = segments[:, None, 0] + weights * (segments[:, None, 1] - segments[:, None, 0])
    num_envs = hand_pos.shape[0]
    points = points.expand(num_envs, -1, -1, -1)
    world = hand_pos[:, :, None] + math_utils.quat_apply(hand_quat[:, :, None].expand(-1, -1, num_samples, -1), points)
    local = math_utils.quat_apply_inverse(
        box_quat[:, None, None].expand(-1, 2, num_samples, -1), world - box_pos[:, None, None]
    )
    half = torch.tensor(box_half_size, device=hand_pos.device, dtype=hand_pos.dtype)
    outside = (local.abs() - half).clamp_min(0.0).norm(dim=-1)
    inside = (local.abs() - half).amax(dim=-1).clamp_max(0.0)
    return (outside + inside).amin(dim=-1) - IBFM_HAND_CAPSULE_RADIUS
