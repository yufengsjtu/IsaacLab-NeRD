# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Copy implicit-actuator shadow PD into Newton ``control.joint_f``.

Isaac Lab implicit actuators apply PD through solver stiffness/damping and
leave Newton ``control.joint_f`` as feed-forward effort, typically zero.
They still publish clipped PD telemetry on ``actuators.applied_effort`` in
public (user) joint order.

NeRD records and consumes ``joint_f`` in Newton backend order. These helpers
scatter that telemetry into each articulation's ``joint_f`` slice. Call them
**after** the ground-truth Newton step when recording, and **after** actuator
telemetry but **before** the neural solver when NeRD is active. Writing the
same values into ``control.joint_f`` *before* an analytic MJWarp step applies
PD twice.
"""

from __future__ import annotations

from typing import Any

import torch


def as_torch_tensor(value: Any) -> torch.Tensor:
    """Return a detached torch view of a ProxyArray, Warp array, or tensor."""
    tensor = getattr(value, "torch", value)
    if isinstance(tensor, torch.Tensor):
        return tensor.detach()
    return torch.as_tensor(tensor)


def reorder_user_effort_to_backend(
    applied_effort: torch.Tensor,
    *,
    user_to_backend: torch.Tensor | None = None,
    has_ordering: bool = False,
) -> torch.Tensor:
    """Scatter user-order applied effort into backend joint order.

    Args:
        applied_effort: Effort [N or N·m, depending on joint type], shape
            ``[num_envs, num_joints]``.
        user_to_backend: Map from each public joint index to its backend index,
            shape ``[num_joints]``. Ignored when ``has_ordering`` is false.
        has_ordering: Whether ``user_to_backend`` is a non-identity permutation.

    Returns:
        Backend-order effort [N or N·m], same shape as ``applied_effort``.
    """
    if applied_effort.ndim != 2:
        raise ValueError(f"applied_effort must be 2-D [num_envs, num_joints], got {tuple(applied_effort.shape)}.")
    if not has_ordering or user_to_backend is None:
        return applied_effort
    index = user_to_backend.long().reshape(-1)
    if index.numel() != applied_effort.shape[-1]:
        raise ValueError(
            "user_to_backend length must match num_joints: "
            f"got {int(index.numel())} vs {int(applied_effort.shape[-1])}."
        )
    backend = torch.empty_like(applied_effort)
    backend[:, index] = applied_effort
    return backend


def backend_applied_effort(articulation: Any) -> torch.Tensor | None:
    """Return ``applied_effort`` in Newton backend joint order, or ``None``.

    Args:
        articulation: Isaac Lab articulation with actuator telemetry.

    Returns:
        Effort [N or N·m, depending on joint type], shape
        ``[num_envs, num_joints]``, or ``None`` when the articulation has no
        actuator telemetry.
    """
    actuators = getattr(articulation, "actuators", None)
    if actuators is None:
        return None
    applied = getattr(actuators, "applied_effort", None)
    if applied is None:
        return None
    applied_t = as_torch_tensor(applied)
    if applied_t.ndim != 2 or applied_t.numel() == 0:
        return None
    data = getattr(articulation, "data", None)
    has_ordering = bool(getattr(data, "has_joint_ordering", False))
    user_to_backend = None
    if has_ordering:
        mapper = getattr(articulation, "_joint_user_to_backend_map", None)
        if callable(mapper):
            user_to_backend = as_torch_tensor(mapper()).to(dtype=torch.long, device=applied_t.device)
    return reorder_user_effort_to_backend(applied_t, user_to_backend=user_to_backend, has_ordering=has_ordering)


def copy_applied_effort_into_sim_joint_f(articulation: Any) -> bool:
    """Write backend-order ``applied_effort`` into the articulation ``joint_f`` slice.

    Args:
        articulation: Isaac Lab articulation bound to Newton ``control.joint_f``.

    Returns:
        True when a write occurred.
    """
    effort = backend_applied_effort(articulation)
    if effort is None:
        return False
    data = getattr(articulation, "data", None)
    sim_effort = getattr(data, "_sim_bind_joint_effort", None)
    if sim_effort is None:
        return False
    _assign_joint_f(sim_effort, effort)
    return True


def copy_scene_applied_effort_into_sim_joint_f(scene: Any) -> int:
    """Copy ``applied_effort`` into every scene articulation's Newton ``joint_f`` slice.

    Args:
        scene: Interactive scene exposing ``articulations``.

    Returns:
        Number of articulations that received a write.
    """
    written = 0
    for articulation in iter_scene_articulations(scene):
        if copy_applied_effort_into_sim_joint_f(articulation):
            written += 1
    return written


def iter_scene_articulations(scene: Any):
    """Yield articulations registered on an InteractiveScene-like object."""
    if scene is None:
        return
    articulations = getattr(scene, "articulations", None)
    if articulations is None:
        return
    values = articulations.values() if hasattr(articulations, "values") else articulations
    yield from values


def _assign_joint_f(destination: Any, values: torch.Tensor) -> None:
    """Copy a 2-D effort tensor into a Warp or torch ``joint_f`` buffer."""
    payload = values.detach().contiguous()
    if isinstance(destination, torch.Tensor):
        if tuple(destination.shape) != tuple(payload.shape):
            raise RuntimeError(
                "applied_effort shape does not match sim joint_f: "
                f"{tuple(payload.shape)} vs {tuple(destination.shape)}."
            )
        destination.copy_(payload.to(device=destination.device, dtype=destination.dtype))
        return

    import warp as wp

    if not hasattr(destination, "assign"):
        raise TypeError(f"Unsupported joint_f destination type: {type(destination)!r}.")
    dest_shape = tuple(int(dim) for dim in getattr(destination, "shape", ()))
    if dest_shape and dest_shape != tuple(payload.shape):
        raise RuntimeError(f"applied_effort shape does not match sim joint_f: {tuple(payload.shape)} vs {dest_shape}.")
    destination.assign(wp.from_torch(payload))
