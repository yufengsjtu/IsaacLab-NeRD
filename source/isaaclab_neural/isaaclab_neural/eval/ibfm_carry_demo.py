# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pick up, carry and place a box with the I-BFM controller on the G1 29-DoF robot in Newton MJWarp.

Each environment starts with the robot facing a box on the ground; the goal lies ``--goal-x``/``--goal-y`` beyond the
box. :class:`~isaaclab_neural.controllers.ibfm_carry.IbfmCarryController` selects the latents. The hand collision
capsules of the I-BFM MJCF are added to the wrist-yaw links, since the 29-DoF asset without hands has none.

Success follows the I-BFM Carry Box protocol: the box reference point stays within 0.2 m of the goal (3D) for
0.5 s. The weights and latents are CC BY-NC 4.0 and are not distributed.

Example::

    PYTHONPATH="$PWD/source/isaaclab_neural:$PWD/source/isaaclab" python -m isaaclab_neural.eval.ibfm_carry_demo \\
      --policy ibfm_policy_isaaclab.pt --latent-dir I-BFM-Code/ibfm/tasks/carry_box/latents \\
      --num-envs 32 --yaw-range 0.6 --out carry.json --record carry.npz
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os

from isaaclab.app import add_launcher_args, launch_simulation

BOX_HALF_SIZE = (0.1675266, 0.1688563, 0.1800429)
"""Half extents of the Carry Box geometry [m]."""

BOX_GEOM_POS = (0.00102584, -0.00160677, 0.00637628)
"""Box geometry position in the object reference frame [m]."""

BOX_GEOM_QUAT = (0.97153219, 0.22227911, 0.04236142, 0.07016202)
"""Box geometry orientation ``(x, y, z, w)`` in the object reference frame."""

SUCCESS_DISTANCE = 0.2
"""Carry Box success radius around the goal [m]."""

SUCCESS_HOLD_S = 0.5
"""Duration the success radius must hold [s]."""


def parse_args() -> argparse.Namespace:
    """Parse demo and launcher arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", default=os.environ.get("IBFM_POLICY_PATH"), help="Exported I-BFM policy.")
    parser.add_argument("--latent-dir", required=True, help="Released tasks/carry_box/latents directory.")
    parser.add_argument(
        "--push-latent-dir",
        default=None,
        help="Released tasks/push_box/latents directory; enables backing off after a failed lift.",
    )
    parser.add_argument("--usd", default=None, help="Local G1 29-DoF root layer; defaults to the robot config.")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--distance", type=float, default=1.5, help="Robot-to-box distance [m].")
    parser.add_argument("--distance-range", type=float, default=0.0, help="Uniform +- range of --distance [m].")
    parser.add_argument("--yaw-range", type=float, default=0.0, help="Uniform +- robot yaw towards the box [rad].")
    parser.add_argument("--goal-x", type=float, default=2.0, help="Goal offset from the box along x [m].")
    parser.add_argument("--goal-y", type=float, default=0.0, help="Goal offset from the box along y [m].")
    parser.add_argument("--box-mass", type=float, default=0.7, help="[kg]")
    parser.add_argument("--box-friction", type=float, default=0.9)
    parser.add_argument("--contact-enter", type=float, default=0.01, help="Hand-box gap that starts contact [m].")
    parser.add_argument("--contact-exit", type=float, default=0.02, help="Hand-box gap that ends contact [m].")
    parser.add_argument("--param", action="append", default=[], help="Controller override KEY=VALUE.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="JSON summary.")
    parser.add_argument("--record", default=None, help="NPZ with robot and box trajectories.")
    add_launcher_args(parser)
    return parser.parse_args()


def demo_controller_cfg(overrides: list[str], backoff: bool):
    """Return the demo controller configuration with ``KEY=VALUE`` overrides applied.

    The demo defaults widen the placement window, look ahead at the goal while placing, soften near-goal turning,
    re-approach after a failed placement and, when ``backoff`` is set, back off after a lift without a grasp; these
    raised Carry Box success in Newton MJWarp.
    """
    from isaaclab_neural.controllers.ibfm_carry import IbfmCarryControllerCfg  # noqa: PLC0415

    cfg = IbfmCarryControllerCfg(
        place_align_distance=0.35,
        align_goal_lookahead=0.15,
        transport_near_max_turn_blend=0.45,
        transport_turn_boost_distance=0.8,
        reapproach_switch_distance=0.5,
        lift_timeout_seconds=3.0 if backoff else 0.0,
    )
    types = {f.name: f.type for f in dataclasses.fields(cfg)}
    for item in overrides:
        key, value = item.split("=", 1)
        if key not in types:
            raise ValueError(f"Unknown controller parameter: {key}.")
        setattr(cfg, key, int(value) if types[key] in (int, "int") else float(value))
    return cfg


def main() -> None:
    """Run the demo."""
    args = parse_args()

    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

    import isaaclab.sim as sim_utils

    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005,
        device=args.device,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=800,
                nconmax=250,
                cone="pyramidal",
                impratio=1.0,
                integrator="implicitfast",
                use_mujoco_contacts=False,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
            num_substeps=2,
            default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
        ),
    )
    with launch_simulation(sim_cfg, args):
        run(args, sim_cfg)


def add_hand_capsules(num_envs: int, robot_path: str = "/World/envs/env_{}/Robot") -> None:
    """Author the I-BFM MJCF hand collision capsules on the wrist-yaw links of every environment."""
    import numpy as np
    from isaaclab_neural.controllers.ibfm import IBFM_HAND_CAPSULE_RADIUS, IBFM_HAND_CAPSULE_SEGMENTS

    from pxr import Gf, UsdGeom, UsdPhysics

    import isaaclab.sim as sim_utils

    stage = sim_utils.get_current_stage()
    for env in range(num_envs):
        for side, (start, end) in zip(("left", "right"), IBFM_HAND_CAPSULE_SEGMENTS):
            start, end = np.asarray(start), np.asarray(end)
            capsule = UsdGeom.Capsule.Define(stage, f"{robot_path.format(env)}/{side}_wrist_yaw_link/ibfm_hand_capsule")
            capsule.CreateAxisAttr("X")
            capsule.CreateRadiusAttr(IBFM_HAND_CAPSULE_RADIUS)
            capsule.CreateHeightAttr(float(np.linalg.norm(end - start)))
            capsule.CreatePurposeAttr("guide")
            xform = UsdGeom.Xformable(capsule)
            xform.AddTranslateOp().Set(Gf.Vec3d(*((start + end) / 2.0).tolist()))
            xform.AddRotateZOp().Set(math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])))
            UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())


def run(args: argparse.Namespace, sim_cfg) -> None:
    """Build the scene, roll out the controller and report Carry Box success."""
    import numpy as np
    import torch
    from isaaclab_neural.controllers import ibfm
    from isaaclab_neural.controllers.bfm_zero import BfmZeroObservationHistory
    from isaaclab_neural.controllers.ibfm_carry import IbfmCarryController, IbfmCarryLatents
    from isaaclab_neural.robots.g1_29dof import G1_29DOF_DEFAULT_JOINT_POS, G1_29DOF_IBFM_CFG, G1_29DOF_JOINT_NAMES

    import isaaclab.sim as sim_utils
    import isaaclab.utils.math as math_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.utils import string as string_utils
    from isaaclab.utils.configclass import configclass

    if not args.policy:
        raise ValueError("Pass --policy or set IBFM_POLICY_PATH.")
    torch.manual_seed(args.seed)
    controller_cfg = demo_controller_cfg(args.param, args.push_latent_dir is not None)
    control_dt, decimation = 0.02, 4
    spawn = G1_29DOF_IBFM_CFG.spawn if args.usd is None else G1_29DOF_IBFM_CFG.spawn.replace(usd_path=args.usd)

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        )
        robot = G1_29DOF_IBFM_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot", spawn=spawn, soft_joint_pos_limit_factor=1.0
        )
        box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            spawn=sim_utils.CuboidCfg(
                size=tuple(2.0 * h for h in BOX_HALF_SIZE),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=args.box_mass),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=args.box_friction, dynamic_friction=args.box_friction
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.5, 0.2)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(args.distance, 0.0, 0.2), rot=BOX_GEOM_QUAT),
        )

    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(SceneCfg(num_envs=args.num_envs, env_spacing=8.0))
    add_hand_capsules(args.num_envs)
    sim.reset()
    robot, box, device, n = scene["robot"], scene["box"], sim.device, args.num_envs
    if list(robot.joint_names) != list(G1_29DOF_JOINT_NAMES):
        raise RuntimeError(f"Unexpected joint order {robot.joint_names}.")
    body_ids = [robot.body_names.index(name) for name in ibfm.IBFM_PRIVILEGED_BODY_NAMES]
    hand_ids = [robot.body_names.index(name) for name in ibfm.IBFM_HAND_BODY_NAMES]
    policy = ibfm.load_ibfm_policy(args.policy, device)
    latents = IbfmCarryLatents.from_directory(args.latent_dir, device, args.push_latent_dir)
    default = torch.zeros(len(G1_29DOF_JOINT_NAMES), device=device)
    ids, _, values = string_utils.resolve_matching_names_values(G1_29DOF_DEFAULT_JOINT_POS, G1_29DOF_JOINT_NAMES)
    default[ids] = torch.tensor(values, device=device)
    scale = torch.tensor(ibfm.ibfm_action_scale(G1_29DOF_JOINT_NAMES), device=device)
    limits = robot.data.joint_pos_limits.torch[0]
    geom_pos = torch.tensor(BOX_GEOM_POS, device=device).expand(n, 3)
    geom_quat = torch.tensor(BOX_GEOM_QUAT, device=device).expand(n, 4)

    origins = scene.env_origins
    distance = args.distance + (2 * torch.rand(n, device=device) - 1) * args.distance_range
    yaw = (2 * torch.rand(n, device=device) - 1) * args.yaw_range
    root = torch.cat(
        [origins + torch.tensor([0.0, 0.0, 0.8], device=device), math_utils.quat_from_euler_xyz(0 * yaw, 0 * yaw, yaw)],
        -1,
    )
    robot.write_root_pose_to_sim_index(root_pose=root)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=device))
    robot.write_joint_position_to_sim_index(position=default.expand(n, -1).clone())
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros(n, len(default), device=device))
    marker = origins + torch.stack(
        [distance, torch.zeros_like(distance), torch.full_like(distance, controller_cfg.ground_center_height)], -1
    )
    box.write_root_pose_to_sim_index(root_pose=torch.cat([marker + geom_pos, geom_quat], -1))
    box.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=device))
    goal = marker + torch.tensor([args.goal_x, args.goal_y, 0.0], device=device)
    robot.reset()
    box.reset()
    scene.update(0.0)

    history = BfmZeroObservationHistory(n, len(default), device)
    contact = torch.zeros(n, 2, dtype=torch.bool, device=device)
    controllers = [IbfmCarryController(controller_cfg, control_dt) for _ in range(n)]
    gravity = torch.tensor([0.0, 0.0, -1.0], device=device).expand(n, 3)
    hold_steps = round(SUCCESS_HOLD_S / control_dt)
    hold, success_time = np.zeros(n, dtype=int), [None] * n
    record = {key: [] for key in ("root_pos", "root_quat_xyzw", "joint_pos", "box_pos", "box_quat_xyzw", "phase")}
    for step in range(int(args.seconds / control_dt)):
        data = robot.data
        base_pos, base_quat = data.root_link_pos_w.torch, data.root_link_quat_w.torch
        box_pos, box_quat = box.data.root_link_pos_w.torch, box.data.root_link_quat_w.torch
        object_quat = math_utils.quat_mul(box_quat, math_utils.quat_conjugate(geom_quat))
        object_pos = box_pos - math_utils.quat_apply(object_quat, geom_pos)
        box_vel = box.data.root_link_lin_vel_w.torch
        hand_pos, hand_quat = data.body_link_pos_w.torch[:, hand_ids], data.body_link_quat_w.torch[:, hand_ids]
        gap = ibfm.ibfm_hand_box_gap(hand_pos, hand_quat, box_pos, box_quat, BOX_HALF_SIZE)
        contact = torch.where(gap < args.contact_enter, True, torch.where(gap > args.contact_exit, False, contact))
        context = ibfm.ibfm_object_context(
            ibfm.ibfm_object_observation(
                base_pos, base_quat, object_pos, object_quat, box_vel, box.data.root_link_ang_vel_w.torch, goal
            ),
            torch.ones(n, device=device),
            contact.float(),
            ibfm.ibfm_hand_spatial_observation(
                base_pos, base_quat, ibfm.ibfm_hand_contact_points(hand_pos, hand_quat), object_pos
            ),
        )

        base_rot = math_utils.matrix_from_quat(base_quat).cpu().numpy()
        box_up = math_utils.matrix_from_quat(box_quat)[:, 2, 2].clamp(0.0, 1.0).cpu().numpy()
        cpu = {k: v.cpu().numpy() for k, v in (("base", base_pos), ("object", object_pos), ("goal", goal))}
        speed, both = box_vel.norm(dim=-1).cpu().numpy(), contact.all(dim=-1).cpu().numpy()
        commands, masks, lookahead = [], [], []
        for i, controller in enumerate(controllers):
            box_dir = cpu["object"][i, :2] - cpu["base"][i, :2]
            goal_dir = cpu["goal"][i, :2] - cpu["object"][i, :2]
            metrics = {
                "horizontal_distance": float(np.linalg.norm(box_dir)),
                "goal_distance": float(np.linalg.norm(goal_dir)),
                "object_height": float(cpu["object"][i, 2]),
                "object_speed": float(speed[i]),
                "lift": float(
                    np.clip(
                        (cpu["object"][i, 2] - controller_cfg.ground_center_height) / controller_cfg.lift_height, 0, 1
                    )
                ),
                "root_height": float(cpu["base"][i, 2]),
                "pelvis_up_z": float(base_rot[i, 2, 2]),
                "robot_upright": float(np.clip(base_rot[i, 2, 2], 0.0, 1.0)),
                "box_upright": float(box_up[i]),
                "bilateral_contact": bool(both[i]),
            }
            if step > 0:
                controller.update(metrics)
                inside = np.linalg.norm(cpu["goal"][i] - cpu["object"][i]) < SUCCESS_DISTANCE
                hold[i] = hold[i] + 1 if inside else 0
                if hold[i] >= hold_steps and success_time[i] is None:
                    success_time[i] = step * control_dt
            command, mask, ahead = controller.latent(latents, metrics, base_rot[i], box_dir, goal_dir)
            commands.append(command)
            masks.append(mask)
            lookahead.append(ahead)
        mask = torch.tensor(masks, device=device)[:, None]
        context = torch.where(mask, torch.zeros_like(context), context)
        if controller_cfg.align_goal_lookahead > 0.0:
            ahead = torch.tensor(lookahead, device=device)[:, None]
            delta = context[:, 15:17]
            moved = context.clone()
            moved[:, 15:17] = delta + controller_cfg.align_goal_lookahead * delta / delta.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            context = torch.where(ahead, moved, context)

        obs = history.observe(
            data.joint_pos.torch - default,
            data.joint_vel.torch,
            math_utils.quat_apply_inverse(base_quat, gravity),
            math_utils.quat_apply_inverse(base_quat, data.root_ang_vel_w.torch),
        )
        joints = len(default)
        privileged = ibfm.ibfm_privileged_state(
            data.body_link_pos_w.torch[:, body_ids],
            data.body_link_quat_w.torch[:, body_ids],
            data.body_link_lin_vel_w.torch[:, body_ids],
            data.body_link_ang_vel_w.torch[:, body_ids],
            context,
        )
        state = torch.cat([obs[:, : 2 * joints + 6], context], dim=-1)
        with torch.no_grad():
            action = (
                policy(
                    state,
                    privileged,
                    obs[:, 2 * joints + 6 : 3 * joints + 6],
                    obs[:, 3 * joints + 6 :],
                    torch.stack(commands),
                ).clamp(-1.0, 1.0)
                * 5.0
            )
        history.record(action)
        target = torch.clamp(default + scale * action, limits[:, 0], limits[:, 1])
        for _ in range(decimation):
            robot.set_joint_position_target_index(target=target)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())
        if args.record:
            for key, value in (
                ("root_pos", base_pos - origins),
                ("root_quat_xyzw", base_quat),
                ("joint_pos", data.joint_pos.torch),
                ("box_pos", box_pos - origins),
                ("box_quat_xyzw", box_quat),
            ):
                record[key].append(value.cpu().numpy())
            record["phase"].append([c.phase for c in controllers])
        if step % 250 == 0:
            phases = [c.phase for c in controllers]
            print(
                f"t={step * control_dt:5.1f}s phases={ {p: phases.count(p) for p in sorted(set(phases))} } "
                f"success={np.mean([t is not None for t in success_time]):.2f}",
                flush=True,
            )

    summary = {
        "success_rate": float(np.mean([t is not None for t in success_time])),
        "success_time_s": success_time,
        "placement_attempted": float(np.mean([c.place_attempts > 0 for c in controllers])),
        "falls_per_env": float(np.mean([c.falls for c in controllers])),
        "final_phases": [c.phase for c in controllers],
        "controller": dataclasses.asdict(controller_cfg),
        "args": {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, list, type(None)))},
    }
    print(json.dumps({k: summary[k] for k in ("success_rate", "placement_attempted", "falls_per_env")}))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(summary, handle, indent=1)
    if args.record:
        np.savez_compressed(
            args.record,
            **{k: np.asarray(v) for k, v in record.items()},
            goal=(goal - origins).cpu().numpy(),
            box_half=np.asarray(BOX_HALF_SIZE),
        )


if __name__ == "__main__":
    main()
