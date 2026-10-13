# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""I-BFM policy, observation and hand-contact contracts."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from isaaclab_neural.controllers.ibfm import (
    IBFM_HAND_CAPSULE_RADIUS,
    IBFM_OBJECT_CONTEXT_DIM,
    IBFM_PRIVILEGED_BODY_NAMES,
    IbfmPolicy,
    ibfm_action_scale,
    ibfm_hand_box_gap,
    ibfm_object_context,
    ibfm_object_observation,
    ibfm_privileged_state,
    ibfm_sphere_log_map,
    load_ibfm_policy,
)
from isaaclab_neural.robots.g1_29dof import G1_29DOF_JOINT_NAMES

import isaaclab.utils.math as math_utils


def _yaw(angle: float, count: int = 1) -> torch.Tensor:
    """Quaternions ``(x, y, z, w)`` of a rotation about the world z axis."""
    return torch.tensor([[0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)]]).repeat(count, 1)


def test_log_map_preserves_arc_length_and_direction():
    dim = 16
    x = torch.zeros(3, dim)
    x[:, 0] = 1.0
    y = torch.zeros(3, dim)
    y[0, 0] = 1.0  # identical
    y[1, 1] = 1.0  # orthogonal
    y[2, 0] = -1.0  # antipodal
    log = ibfm_sphere_log_map(x, y)
    radius = math.sqrt(dim)
    torch.testing.assert_close(log.norm(dim=-1), torch.tensor([0.0, radius * math.pi / 2, radius * math.pi]))
    torch.testing.assert_close(F.normalize(log[1], dim=-1), y[1])
    assert abs(float((log[2] * x[2]).sum())) < 1e-6


def test_exported_policy_round_trips_through_loader(tmp_path):
    torch.manual_seed(0)
    dims = {"state": 5, "privileged_state": 4, "last_action": 3, "history_actor": 6}
    meta = {
        "obs_dims": dims,
        "z_dim": 8,
        "hidden_dim": 16,
        "hidden_layers": 1,
        "embedding_layers": 2,
        "backward_hidden_dim": 12,
        "norm_eps": 1e-5,
    }
    source = IbfmPolicy(5, 4, 3, 6, z_dim=8, hidden_dim=16, hidden_layers=1, backward_hidden_dim=12)
    stats = {key: {"mean": torch.randn(dim), "var": torch.rand(dim) + 0.5} for key, dim in dims.items()}
    actor_state = {k: v for k, v in source.actor.state_dict().items() if k not in ("obs_mean", "obs_var")}
    payload = {
        "actor": actor_state,
        "backward": {f"net.{k}": v for k, v in source.backward.state_dict().items()},
        "normalizer": stats,
        "meta": meta,
    }
    path = tmp_path / "policy.pt"
    torch.save(payload, path)
    policy = load_ibfm_policy(str(path), "cpu")

    state, priv, last, hist = (torch.randn(4, dim) for dim in dims.values())
    z = torch.randn(4, 8)
    norm = lambda key, value: (value - stats[key]["mean"]) / torch.sqrt(stats[key]["var"] + 1e-5)  # noqa: E731
    embedding = source.backward(torch.cat([norm("state", state), norm("privileged_state", priv)], dim=-1))
    prompt = ibfm_sphere_log_map(IbfmPolicy.project_z(embedding), IbfmPolicy.project_z(z))
    actor_obs = torch.cat([norm("state", state), norm("last_action", last), norm("history_actor", hist)], dim=-1)
    with torch.no_grad():
        source.actor.obs_mean.zero_()
        source.actor.obs_var.fill_(1.0 - 1e-5)
        expected = source.actor(actor_obs, prompt)
        torch.testing.assert_close(policy(state, priv, last, hist, z), expected)
    assert not any(p.requires_grad for p in policy.parameters())


def test_loader_rejects_incomplete_files(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"actor": {}, "meta": {}}, path)
    with pytest.raises(KeyError):
        load_ibfm_policy(str(path), "cpu")


def test_object_observation_is_root_local_and_context_masks_absent_objects():
    base_pos = torch.tensor([[1.0, 2.0, 0.8]])
    base_quat = _yaw(math.pi / 2)
    object_quat = _yaw(math.pi / 2)
    obs = ibfm_object_observation(
        base_pos,
        base_quat,
        torch.tensor([[1.0, 3.0, 0.2]]),
        object_quat,
        torch.tensor([[0.0, 1.0, 0.0]]),
        torch.zeros(1, 3),
        torch.tensor([[1.0, 5.0, 0.2]]),
    )
    # one metre ahead of a robot facing +y, aligned with it, moving forward, target two metres ahead
    torch.testing.assert_close(obs[0, :3], torch.tensor([1.0, 0.0, -0.6]))
    torch.testing.assert_close(obs[0, 3:9], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(obs[0, 9:12], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(obs[0, 15:18], torch.tensor([2.0, 0.0, 0.0]))

    hand_spatial = torch.randn(2, 12)
    context = ibfm_object_context(obs.repeat(2, 1), torch.tensor([1.0, 0.0]), torch.ones(2, 2), hand_spatial)
    assert context.shape == (2, IBFM_OBJECT_CONTEXT_DIM)
    torch.testing.assert_close(context[0, 20:32], hand_spatial[0])
    torch.testing.assert_close(context[0, -2:], torch.ones(2))
    torch.testing.assert_close(context[1], torch.zeros(IBFM_OBJECT_CONTEXT_DIM))


def test_privileged_state_is_heading_invariant():
    torch.manual_seed(0)
    bodies = len(IBFM_PRIVILEGED_BODY_NAMES)
    pos = torch.randn(1, bodies, 3)
    quat = math_utils.random_orientation(bodies, "cpu").unsqueeze(0)
    lin_vel, ang_vel = torch.randn(1, bodies, 3), torch.randn(1, bodies, 3)
    context = torch.randn(1, IBFM_OBJECT_CONTEXT_DIM)
    state = ibfm_privileged_state(pos, quat, lin_vel, ang_vel, context)
    assert state.shape == (1, 15 * (bodies + 1) - 2 + IBFM_OBJECT_CONTEXT_DIM)

    turn = _yaw(0.7, bodies).unsqueeze(0)
    shift = torch.tensor([3.0, -2.0, 0.0])
    moved = ibfm_privileged_state(
        math_utils.quat_apply(turn, pos) + shift,
        math_utils.quat_mul(turn, quat),
        math_utils.quat_apply(turn, lin_vel),
        math_utils.quat_apply(turn, ang_vel),
        context,
    )
    torch.testing.assert_close(moved, state, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(state[0, 0], pos[0, 0, 2])


def test_hand_box_gap_measures_capsule_surface_distance():
    hand_pos = torch.zeros(1, 2, 3)
    hand_quat = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(1, 2, 1)
    half = (0.1, 0.1, 0.1)
    # box face at y = 0.4 over x in [0.01, 0.21]; the closest centre-line points are y = 0 (left) and 0.02 (right)
    gap = ibfm_hand_box_gap(hand_pos, hand_quat, torch.tensor([[0.11, 0.5, 0.0]]), _yaw(0.0), half)
    torch.testing.assert_close(gap, torch.tensor([[0.4, 0.38]]) - IBFM_HAND_CAPSULE_RADIUS)
    inside = ibfm_hand_box_gap(hand_pos, hand_quat, torch.tensor([[0.11, 0.0, 0.0]]), _yaw(0.0), half)
    assert bool((inside < -IBFM_HAND_CAPSULE_RADIUS).all())


def test_action_scale_matches_ibfm_runtime_config():
    scale = dict(zip(G1_29DOF_JOINT_NAMES, ibfm_action_scale(G1_29DOF_JOINT_NAMES)))
    # control.action_scale * effort_limit / stiffness of the I-BFM g1_29dof.yaml, with waist overrides
    assert scale["left_hip_pitch_joint"] == pytest.approx(0.25 * 139.0 / 99.09843)
    assert scale["right_hip_yaw_joint"] == pytest.approx(0.25 * 88.0 / 40.17924)
    assert scale["waist_pitch_joint"] == pytest.approx(0.08 * 50.0 / 40.0)
    assert scale["waist_yaw_joint"] == pytest.approx(0.0334827 * 88.0 / 40.17924)
    assert scale["left_wrist_yaw_joint"] == pytest.approx(0.25 * 13.4 / 8.61103)


def _carry_latents():
    from isaaclab_neural.controllers.ibfm_carry import IbfmCarryLatents

    torch.manual_seed(0)
    z = lambda: IbfmPolicy.project_z(torch.randn(256))  # noqa: E731
    return IbfmCarryLatents(
        approach=z(),
        lift=z(),
        transport=z(),
        align=z(),
        lower=z(),
        place=z(),
        get_up=IbfmPolicy.project_z(torch.randn(3, 256)),
        turn={-1: z(), 1: z()},
        transport_turn={-1: z(), 1: z()},
        backoff=z(),
        stand=z(),
    )


def _metrics(**overrides):
    metrics = {
        "horizontal_distance": 1.5,
        "goal_distance": 2.0,
        "object_height": 0.186,
        "object_speed": 0.0,
        "lift": 0.0,
        "root_height": 0.78,
        "pelvis_up_z": 1.0,
        "robot_upright": 1.0,
        "box_upright": 1.0,
        "bilateral_contact": False,
    }
    return {**metrics, **overrides}


def test_carry_controller_walks_lifts_and_transports():
    from isaaclab_neural.controllers.ibfm_carry import IbfmCarryController, IbfmCarryControllerCfg

    latents, cfg = _carry_latents(), IbfmCarryControllerCfg()
    controller = IbfmCarryController(cfg, dt=0.02)
    rot, ahead = torch.eye(3).numpy(), torch.tensor([1.0, 0.0]).numpy()
    latent, masked, _ = controller.latent(latents, _metrics(), rot, ahead, ahead)
    # facing the box: pure approach latent, object hidden from the policy
    torch.testing.assert_close(latent, latents.approach)
    assert masked
    controller.update(_metrics(horizontal_distance=0.7))
    assert controller.phase == "lift"
    for _ in range(cfg.phase_hold_steps):
        controller.update(_metrics(horizontal_distance=0.4, lift=0.6, bilateral_contact=True))
    assert controller.phase == "transport"
    for _ in range(cfg.phase_hold_steps):
        controller.update(_metrics(horizontal_distance=0.4, lift=0.0))
    assert controller.phase == "lift" and controller.reapproach


def test_carry_controller_backs_off_after_lift_timeout_and_gets_up_after_falls():
    from isaaclab_neural.controllers.ibfm_carry import IbfmCarryController, IbfmCarryControllerCfg

    latents = _carry_latents()
    cfg = IbfmCarryControllerCfg(lift_timeout_seconds=0.1)
    controller = IbfmCarryController(cfg, dt=0.02)
    controller.update(_metrics(horizontal_distance=0.7))
    for _ in range(5):
        controller.update(_metrics(horizontal_distance=0.5))
    assert controller.phase == "backoff"
    rot, ahead = torch.eye(3).numpy(), torch.tensor([1.0, 0.0]).numpy()
    latent, masked, _ = controller.latent(latents, _metrics(), rot, ahead, ahead)
    torch.testing.assert_close(latent, latents.stand)
    assert masked
    controller.update(_metrics(horizontal_distance=cfg.backoff_distance))
    assert controller.phase == "approach"
    for _ in range(cfg.fall_detect_steps):
        controller.update(_metrics(root_height=0.2))
    assert controller.phase == "get_up" and controller.falls == 1


def test_collision_layer_replaces_asset_colliders_with_mjcf_geometry(tmp_path):
    pytest.importorskip("mujoco")
    from isaaclab_neural.eval.ibfm_box_demo import write_ibfm_collision_usd

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    asset = Usd.Stage.CreateNew(str(tmp_path / "robot.usda"))
    root = UsdGeom.Xform.Define(asset, "/robot")
    asset.SetDefaultPrim(root.GetPrim())
    UsdGeom.Xform.Define(asset, "/robot/link")
    mesh = UsdGeom.Cube.Define(asset, "/robot/link/collisions/mesh")
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    asset.GetRootLayer().Save()
    (tmp_path / "robot.xml").write_text(
        '<mujoco><worldbody><body name="link"><freejoint/>'
        '<geom name="arm" type="capsule" size="0.03" fromto="0 0 0 0.2 0 0"/>'
        '<geom name="ghost" type="sphere" size="0.1" contype="0" conaffinity="0"/>'
        "</body></worldbody></mujoco>"
    )
    path = write_ibfm_collision_usd(
        str(tmp_path / "robot.usda"), str(tmp_path / "layer.usda"), str(tmp_path / "robot.xml")
    )
    stage = Usd.Stage.Open(path)
    assert not stage.GetPrimAtPath("/robot/link/collisions").IsActive()
    capsule = UsdGeom.Capsule(stage.GetPrimAtPath("/robot/link/ibfm_arm"))
    assert capsule.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
    assert capsule.GetRadiusAttr().Get() == pytest.approx(0.03)
    assert capsule.GetHeightAttr().Get() == pytest.approx(0.2)
    # the ends of the capsule axis (local z) are the MJCF segment ends
    transform = UsdGeom.Xformable(capsule).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    ends = sorted(tuple(round(v, 6) for v in transform.Transform(Gf.Vec3d(0, 0, z))) for z in (-0.1, 0.1))
    assert ends == [(0.0, 0.0, 0.0), (0.2, 0.0, 0.0)]
    assert not stage.GetPrimAtPath("/robot/link/ibfm_ghost").IsValid()
