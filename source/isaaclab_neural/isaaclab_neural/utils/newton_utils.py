"""Helpers wrapping Newton-API operations on articulations."""
from __future__ import annotations

from typing import Optional

import numpy as np
import newton
import warp as wp

from isaaclab_neural.utils import warp_utils


def base_joint_type(joint_types: np.ndarray) -> Optional[int]:
    """Return the type code of the first non-FIXED joint, or None if all are FIXED.

    Newton models may begin with FIXED joints (e.g. Cartpole's world-attach for
    fixed-base robots, or sensor/shell mounts on floating-base robots). FIXED
    joints contribute 0 q-coords and 0 qd-coords, so the first non-FIXED joint's
    q/qd always starts at index 0 in the per-env state vector. Use this helper
    instead of ``joint_types[0]`` whenever the intent is "what kind of base does
    this articulation have".
    """
    for t in joint_types:
        if int(t) != int(newton.JointType.FIXED):
            return int(t)
    return None

def eval_fk(model, state):
    """Apply generalized coordinates to maximal coordinates."""
    
    newton.eval_fk(
        model=model,
        joint_q=state.joint_q,
        joint_qd=state.joint_qd,
        state=state,
        mask=None,
    )

def eval_ik(model, state):
    """Convert new maximal coordinates back to generalized coordinates."""
    
    newton.eval_ik(
        model=model,
        state=state,
        joint_q=state.joint_q,
        joint_qd=state.joint_qd
    )

def assign_states_from_torch(newton_env, torch_states):
    """Assign states from torch to newton environment."""
    assert torch_states.shape[0] <= newton_env.num_envs
    wp.launch(
        warp_utils.assign_states,
        dim = torch_states.shape[0],
        inputs = [
            wp.from_torch(torch_states),
            newton_env.dof_q_per_env,
            newton_env.dof_qd_per_env
        ],
        outputs = [
            newton_env.state.joint_q,
            newton_env.state.joint_qd
        ],
        device = newton_env.device
    )

def acquire_states_to_torch(newton_env, torch_states):
    """Acquire states from newton environment to torch."""

    assert torch_states.shape[0] == newton_env.num_envs
    wp.launch(
        warp_utils.acquire_states,
        dim = torch_states.shape[0],
        inputs = [
            newton_env.state.joint_q,
            newton_env.state.joint_qd,
            newton_env.dof_q_per_env,
            newton_env.dof_qd_per_env
        ],
        outputs = [wp.from_torch(torch_states)],
        device = newton_env.device
    )