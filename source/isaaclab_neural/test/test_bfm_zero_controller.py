# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BFM-Zero actor, observation history and action-scale contracts."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from isaaclab_neural.controllers.bfm_zero import (
    BFM_ZERO_HISTORY_KEYS,
    BfmZeroActor,
    BfmZeroObservationHistory,
    bfm_zero_action_scale,
    load_bfm_zero_actor,
    project_bfm_zero_latent,
)
from isaaclab_neural.robots.g1_29dof import G1_29DOF_JOINT_NAMES


def _reference_actor(actor: BfmZeroActor, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Forward pass written directly from the BFM-Zero residual actor definition."""
    params = dict(actor.named_parameters())

    def block(prefix: str, x: torch.Tensor, activation: bool) -> torch.Tensor:
        x = F.layer_norm(x, x.shape[-1:], params[f"{prefix}.mlp.0.weight"], params[f"{prefix}.mlp.0.bias"])
        x = F.linear(x, params[f"{prefix}.mlp.1.weight"], params[f"{prefix}.mlp.1.bias"])
        return F.mish(x) if activation else x

    normalized = (obs - actor.obs_mean) / torch.sqrt(actor.obs_var + actor.norm_eps)
    s = block("embed_s.1", block("embed_s.0", normalized, True), True)
    e = block("embed_z.1", block("embed_z.0", torch.cat([normalized, z], dim=-1), True), True)
    h = torch.cat([s, e], dim=-1)
    h = h + block("policy.0", h, True)
    return torch.tanh(block("policy.1", h, False))


def test_actor_matches_reference_forward_and_round_trips(tmp_path):
    torch.manual_seed(0)
    meta = {"obs_dim": 6, "z_dim": 4, "action_dim": 3, "hidden_dim": 8, "hidden_layers": 1, "embedding_layers": 2}
    actor = BfmZeroActor(**meta)
    actor.obs_mean.uniform_(-1.0, 1.0)
    actor.obs_var.uniform_(0.5, 2.0)
    obs, z = torch.randn(5, 6), project_bfm_zero_latent(torch.randn(5, 4))
    with torch.no_grad():
        expected = _reference_actor(actor, obs, z)
        torch.testing.assert_close(actor(obs, z), expected)
    path = tmp_path / "actor.pt"
    torch.save({"actor": actor.state_dict(), "meta": meta}, path)
    loaded = load_bfm_zero_actor(str(path), "cpu")
    torch.testing.assert_close(loaded(obs, z), expected)
    assert not any(p.requires_grad for p in loaded.parameters())


def test_loader_rejects_files_without_metadata(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"actor": {}}, path)
    with pytest.raises(KeyError):
        load_bfm_zero_actor(str(path), "cpu")


def test_latent_projection_has_norm_sixteen():
    z = project_bfm_zero_latent(torch.randn(7, 256) * 3.0)
    torch.testing.assert_close(z.norm(dim=-1), torch.full((7,), 16.0))


def test_history_excludes_the_current_step_and_uses_sorted_keys():
    joints = 2
    history = BfmZeroObservationHistory(num_envs=1, num_joints=joints, device="cpu")

    def frame(value: float):
        return (
            torch.full((1, joints), value),
            torch.full((1, joints), 10 * value),
            torch.full((1, 3), 100 * value),
            torch.full((1, 3), 4 * value),
        )

    first = history.observe(*frame(1.0))
    history.record(torch.full((1, joints), 0.5))
    second = history.observe(*frame(2.0))
    state_dim = 2 * joints + 6
    # current step: state, then the previous action
    torch.testing.assert_close(second[0, :joints], torch.full((joints,), 2.0))
    torch.testing.assert_close(second[0, state_dim - 3 : state_dim], torch.full((3,), 2.0))
    torch.testing.assert_close(second[0, state_dim : state_dim + joints], torch.full((joints,), 0.5))
    # history holds only the first step, newest first, in alphabetical key order
    dims = {"actions": joints, "base_ang_vel": 3, "dof_pos": joints, "dof_vel": joints, "projected_gravity": 3}
    first_values = {"actions": 0.0, "base_ang_vel": 1.0, "dof_pos": 1.0, "dof_vel": 10.0, "projected_gravity": 100.0}
    offset = state_dim + joints
    for key in BFM_ZERO_HISTORY_KEYS:
        segment = second[0, offset : offset + 4 * dims[key]].reshape(4, dims[key])
        torch.testing.assert_close(segment[0], torch.full((dims[key],), first_values[key]))
        torch.testing.assert_close(segment[1:], torch.zeros(3, dims[key]))
        offset += 4 * dims[key]
    assert offset == second.shape[1] == first.shape[1]
    torch.testing.assert_close(first[0, state_dim:], torch.zeros(first.shape[1] - state_dim))


def test_history_reset_clears_selected_environments():
    history = BfmZeroObservationHistory(num_envs=2, num_joints=1, device="cpu")
    history.observe(torch.ones(2, 1), torch.ones(2, 1), torch.ones(2, 3), torch.ones(2, 3))
    history.record(torch.ones(2, 1))
    history.reset([0])
    obs = history.observe(torch.zeros(2, 1), torch.zeros(2, 1), torch.zeros(2, 3), torch.zeros(2, 3))
    assert torch.count_nonzero(obs[0]) == 0
    assert torch.count_nonzero(obs[1]) > 0


def test_action_scale_matches_bfm_zero_deployment_config():
    # action_scale entries of the BFM-Zero deploy policy config (motivo_newG1.yaml)
    deployed = {
        "left_hip_yaw_joint": 0.54754699,
        "left_hip_pitch_joint": 0.222001498914,
        "left_knee_joint": 0.35066156,
        "left_ankle_roll_joint": 0.43857802,
        "waist_yaw_joint": 0.54754699,
        "waist_pitch_joint": 0.43857802,
        "left_shoulder_pitch_joint": 0.43857802,
        "right_wrist_pitch_joint": 0.07450086,
    }
    scale = dict(zip(G1_29DOF_JOINT_NAMES, bfm_zero_action_scale(G1_29DOF_JOINT_NAMES)))
    for name, value in deployed.items():
        assert scale[name] == pytest.approx(value, rel=1e-5)
