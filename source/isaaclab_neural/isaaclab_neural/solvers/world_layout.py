# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""World-major generalized-state layout for :class:`~isaaclab_neural.solvers.neural_solver.NeuralSolver`."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NeuralSolverLayout:
    """Per-environment generalized-state layout derived from a Newton model.

    Attributes:
        num_envs: Number of Newton worlds.
        dof_q_per_env: Generalized position coordinates in one world.
        dof_qd_per_env: Generalized velocity coordinates in one world.
        num_joints_per_env: Joint count in one world, including ``FREE``/``FIXED``.
        num_bodies_per_env: Body count in one world.
        joint_f_dim: Joint-force coordinates in one world.
        articulations_per_env: Articulation count in one world.
        state_world_ids: World id for each environment row, shape ``[num_envs]``.
        root_body_ids: Root body id of the first articulation in each world.
    """

    num_envs: int
    dof_q_per_env: int
    dof_qd_per_env: int
    num_joints_per_env: int
    num_bodies_per_env: int
    joint_f_dim: int
    articulations_per_env: int
    state_world_ids: np.ndarray
    root_body_ids: np.ndarray


def _as_numpy(value) -> np.ndarray:
    """Return a numpy array from a Newton/Warp buffer or ndarray."""
    if value is None:
        raise ValueError("Newton model is missing a required buffer for NeRD layout.")
    numpy_fn = getattr(value, "numpy", None)
    if numpy_fn is not None:
        return np.asarray(numpy_fn())
    return np.asarray(value)


def resolve_neural_solver_layout(model) -> NeuralSolverLayout:
    """Build a uniform per-world layout for articulated NeRD state.

    The solver stores one generalized-state row per Newton world. Worlds may
    contain multiple articulations (for example a fixed-base Franka plus a
    ``FREE`` cube) when every world is a replica with contiguous joints.

    Args:
        model: Finalized Newton model.

    Returns:
        The layout used to size solver state caches.

    Raises:
        ValueError: If the model is empty, worlds are non-uniform, or joints
            are not packed world-major.
    """
    world_count = int(model.world_count)
    art_count = int(model.articulation_count)
    joint_count = int(model.joint_count)
    body_count = int(model.body_count)
    if art_count <= 0:
        raise ValueError(
            "NeuralSolver only supports articulated rigid body dynamics, "
            "so there has to be at least one articulation in the provided Warp sim model."
        )
    if world_count <= 0:
        raise ValueError("NeuralSolver requires a positive Newton world_count.")
    if art_count % world_count != 0:
        raise ValueError(
            "NeuralSolver requires the same number of articulations in every world, "
            f"got articulation_count={art_count} and world_count={world_count}."
        )
    if joint_count % world_count != 0 or body_count % world_count != 0:
        raise ValueError("NeuralSolver requires uniform body and joint counts across environments.")

    articulations_per_env = art_count // world_count
    joints_per_env = joint_count // world_count
    bodies_per_env = body_count // world_count
    joint_f = getattr(model, "joint_f", None)
    if joint_f is None:
        raise ValueError("NeuralSolver requires model.joint_f.")
    joint_f_count = int(joint_f.shape[0])
    if joint_f_count % world_count != 0:
        raise ValueError("NeuralSolver requires uniform joint-force counts across environments.")
    joint_f_dim = joint_f_count // world_count

    q_starts = _as_numpy(model.joint_q_start)
    qd_starts = _as_numpy(model.joint_qd_start)
    if q_starts.shape[0] < joint_count + 1 or qd_starts.shape[0] < joint_count + 1:
        raise ValueError("Newton joint_q_start/joint_qd_start must include a sentinel at joint_count.")

    joint_world_array = getattr(model, "joint_world", None)
    if joint_world_array is None:
        if articulations_per_env != 1:
            raise ValueError(
                "NeuralSolver requires one Newton world per articulation when joint_world is missing, "
                f"got world_count={world_count} and articulation_count={art_count}."
            )
    else:
        joint_world = _as_numpy(joint_world_array)
        if joint_world.shape[0] != joint_count:
            raise ValueError(
                f"model.joint_world length {joint_world.shape[0]} does not match joint_count {joint_count}."
            )
        for world_id in range(world_count):
            start = world_id * joints_per_env
            end = start + joints_per_env
            expected = np.full(joints_per_env, world_id, dtype=joint_world.dtype)
            if not np.array_equal(joint_world[start:end], expected):
                raise ValueError(
                    "NeuralSolver requires world-contiguous joints (replicated environments). "
                    f"World {world_id} is not a contiguous joint block."
                )

    dof_q_per_env = int(q_starts[joints_per_env] - q_starts[0])
    dof_qd_per_env = int(qd_starts[joints_per_env] - qd_starts[0])
    if dof_q_per_env <= 0 or dof_qd_per_env <= 0:
        raise ValueError("NeuralSolver requires a positive generalized-state size per world.")
    for world_id in range(1, world_count):
        start = world_id * joints_per_env
        end = start + joints_per_env
        if int(q_starts[end] - q_starts[start]) != dof_q_per_env:
            raise ValueError("Uneven generalized-position counts across Newton worlds.")
        if int(qd_starts[end] - qd_starts[start]) != dof_qd_per_env:
            raise ValueError("Uneven generalized-velocity counts across Newton worlds.")

    root_body_ids = _root_body_ids_per_world(model, world_count, articulations_per_env)
    if np.unique(root_body_ids).size != world_count:
        raise ValueError(f"Articulation root body ids are not unique: {root_body_ids}.")

    return NeuralSolverLayout(
        num_envs=world_count,
        dof_q_per_env=dof_q_per_env,
        dof_qd_per_env=dof_qd_per_env,
        num_joints_per_env=joints_per_env,
        num_bodies_per_env=bodies_per_env,
        joint_f_dim=joint_f_dim,
        articulations_per_env=articulations_per_env,
        state_world_ids=np.arange(world_count, dtype=np.int64),
        root_body_ids=root_body_ids.astype(np.int64, copy=False),
    )


def _root_body_ids_per_world(model, world_count: int, articulations_per_env: int) -> np.ndarray:
    """Return the root-link body id of the first articulation in each world."""
    art_starts = _as_numpy(model.articulation_start)
    joint_child = getattr(model, "joint_child", None)
    if joint_child is None:
        raise ValueError("Cannot resolve articulation root bodies from Newton joint_child metadata.")
    child = _as_numpy(joint_child)
    first_art = np.arange(world_count, dtype=np.int64) * int(articulations_per_env)
    if np.any(first_art < 0) or np.any(first_art >= art_starts.shape[0]):
        raise ValueError("articulation_start does not cover every world's first articulation.")
    root_joints = art_starts[first_art].astype(np.int64, copy=False)
    if np.any(root_joints < 0) or np.any(root_joints >= child.shape[0]):
        raise ValueError("Articulation root joint indices are out of range.")
    root_body_ids = child[root_joints]
    if np.any(root_body_ids < 0):
        raise ValueError("Cannot resolve articulation root bodies from Newton joint_child metadata.")
    return root_body_ids
