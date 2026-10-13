# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Carry, push or kick a box with the I-BFM controller on the G1 29-DoF robot in Newton MJWarp.

``carry`` starts with the robot facing a box on the ground and uses
:class:`~isaaclab_neural.controllers.ibfm_carry.IbfmCarryController`. ``push`` and ``kick`` start with the robot
behind the box and run the released I-BFM phase controllers in a subprocess of the I-BFM ``uv`` environment
(``--released-controller``, see :mod:`isaaclab_neural.eval.ibfm_released_controller_server`). The goal lies
``--goal-x``/``--goal-y`` beyond the box. By default the asset colliders are replaced
with the capsules and spheres of the I-BFM training MJCF, and the box starts flat with its object reference frame
turned over as in the expert data; both were needed for the released policy to behave as in MuJoCo.

Success holds the box reference point within :data:`SUCCESS_RADIUS` of the goal for 0.5 s: 3D for carry, as in the
I-BFM protocol, and horizontal for push and kick, with the radii of the released interactive runtimes. The weights
and latents are CC BY-NC 4.0 and are not distributed.

Example::

    PYTHONPATH="$PWD/source/isaaclab_neural:$PWD/source/isaaclab" python -m isaaclab_neural.eval.ibfm_box_demo \\
      --task carry --policy ibfm_policy_isaaclab.pt --latent-root I-BFM-Code/ibfm/tasks \\
      --num-envs 64 --distance-range 0.25 --yaw-range 0.6 --out carry.json --video carry.mp4
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

BOX_REST_QUAT = (-5.7e-05, 2.65e-04, -0.919011, 0.394231)
"""Box geometry orientation ``(x, y, z, w)`` at rest relative to the robot heading in the released Carry, Push and
Kick Box start state (motion 67, frame 0): the geometry lies flat and the object reference frame is turned over, as
in the expert data. Starting with the object frame upright instead puts the box observation out of distribution."""

SUCCESS_RADIUS = {"carry": 0.2, "push": 0.4, "kick": 0.2}
"""Success radius around the goal per task [m]."""

SUCCESS_HOLD_S = 0.5
"""Duration the success radius must hold [s]."""

TASK_DEFAULTS = {"carry": {"distance": 1.5, "box_mass": 0.7}, "push": {"distance": 1.0, "box_mass": 1.25}}
TASK_DEFAULTS["kick"] = TASK_DEFAULTS["push"]
"""Robot-to-box distance [m] and box mass [kg] of the released task runtimes."""


def parse_args() -> argparse.Namespace:
    """Parse demo and launcher arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=("carry", "push", "kick"), default="carry")
    parser.add_argument("--policy", default=os.environ.get("IBFM_POLICY_PATH"), help="Exported I-BFM policy.")
    parser.add_argument("--latent-root", required=True, help="Released I-BFM-Code/ibfm/tasks directory.")
    parser.add_argument("--usd", default=None, help="Local G1 29-DoF root layer; defaults to the robot config.")
    parser.add_argument("--mjcf", default=None, help="I-BFM MJCF whose colliders replace the asset colliders.")
    parser.add_argument(
        "--usd-collision", action="store_true", help="Keep the asset colliders and add only the hand capsules."
    )
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--distance", type=float, default=None, help="Robot-to-box distance [m]; task default.")
    parser.add_argument("--distance-range", type=float, default=0.0, help="Uniform +- range of --distance [m].")
    parser.add_argument("--yaw-range", type=float, default=0.0, help="Uniform +- robot yaw towards the box [rad].")
    parser.add_argument("--goal-x", type=float, default=2.0, help="Goal offset from the box along x [m].")
    parser.add_argument("--goal-y", type=float, default=0.0, help="Goal offset from the box along y [m].")
    parser.add_argument("--box-mass", type=float, default=None, help="[kg]; task default.")
    parser.add_argument("--box-friction", type=float, default=0.9)
    parser.add_argument("--contact-enter", type=float, default=0.01, help="Hand-box gap that starts contact [m].")
    parser.add_argument("--contact-exit", type=float, default=0.02, help="Hand-box gap that ends contact [m].")
    parser.add_argument("--param", action="append", default=[], help="Controller override KEY=VALUE.")
    parser.add_argument(
        "--released-controller",
        default=None,
        metavar="PYTHON",
        help="Push/kick: run the released I-BFM phase controller with this Python of the I-BFM uv environment.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="JSON summary.")
    parser.add_argument("--video", default=None, help="Record a Newton GL mp4 following --video-env.")
    parser.add_argument("--video-env", type=int, default=0)
    parser.add_argument("--video-size", type=int, nargs=2, default=(960, 540), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--video-distance", type=float, default=3.2, help="Camera distance [m].")
    parser.add_argument("--video-focal-length", type=float, default=20.0, help="Camera focal length.")
    parser.add_argument("--video-azimuth", type=float, default=-60.0, help="Camera azimuth [deg].")
    parser.add_argument("--trace", default=None, help="CSV of the --video-env state per control step.")
    parser.add_argument("--stop-on-success", action="store_true", help="End once every environment succeeded.")
    add_launcher_args(parser)
    args = parser.parse_args()
    for key, value in TASK_DEFAULTS[args.task].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def _apply_overrides(cfg, overrides: list[str]):
    types = {f.name: f.type for f in dataclasses.fields(cfg)}
    for item in overrides:
        key, value = item.split("=", 1)
        if key not in types:
            raise ValueError(f"Unknown controller parameter: {key}.")
        setattr(cfg, key, int(value) if types[key] in (int, "int") else float(value))
    return cfg


def demo_controller_cfg(task: str, overrides: list[str]):
    """Return the demo controller configuration of a task with ``KEY=VALUE`` overrides applied.

    The carry defaults widen the placement window while keeping the drift threshold above it, move the observed goal
    0.3 m further during placement, release faster, soften near-goal turning, re-approach after a failed placement
    and back off after a lift without a grasp. Push and kick run the released controllers and take no overrides.
    """
    from isaaclab_neural.controllers.ibfm_carry import IbfmCarryControllerCfg  # noqa: PLC0415

    if task == "carry":
        cfg = IbfmCarryControllerCfg(
            place_align_distance=0.3,
            place_drift_distance=0.5,
            align_goal_lookahead=0.3,
            transport_near_max_turn_blend=0.45,
            transport_turn_boost_distance=0.8,
            place_release_seconds=1.0,
            reapproach_switch_distance=0.5,
            lift_timeout_seconds=3.0,
        )
        return _apply_overrides(cfg, overrides)
    if overrides:
        raise ValueError("Controller overrides apply to carry only.")
    return None


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
    if args.video:
        from isaaclab_neural.eval.newton_video import newton_video_visualizer_cfg  # noqa: PLC0415

        sim_cfg.visualizer_cfgs = [newton_video_visualizer_cfg(*args.video_size, args.video_focal_length)]
    with launch_simulation(sim_cfg, args):
        run(args, sim_cfg)


def write_ibfm_collision_usd(
    usd_path: str, out_path: str, mjcf_path: str | None = None, variants: dict[str, str] | None = None
) -> str:
    """Write a USD layer that references the G1 asset and adds the I-BFM MJCF collision geometry.

    With ``mjcf_path``, the asset's own colliders are deactivated and replaced by the capsules and spheres of the
    I-BFM training MJCF (``humanoidverse/data/robots/g1_mjlab/g1_29dof.xml``), whose body frames match the asset links.
    Without it, only the hand capsules are added, since the asset without hands has none. Colliders authored on the
    stage after spawning are not imported into the Newton model, so the geometry is written to a referencing layer
    that is spawned instead of the asset.

    Args:
        usd_path: Local G1 29-DoF root layer.
        out_path: Output ``.usda`` path.
        mjcf_path: I-BFM training MJCF; ``None`` adds only the hand capsules.
        variants: Variant selections of the spawned asset, used to find its colliders.

    Returns:
        ``out_path``.
    """
    import numpy as np
    from isaaclab_neural.controllers.ibfm import IBFM_HAND_CAPSULE_RADIUS, IBFM_HAND_CAPSULE_SEGMENTS

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    source = Usd.Stage.Open(usd_path)
    root = source.GetDefaultPrim()
    for name, selection in (variants or {}).items():
        root.GetVariantSets().GetVariantSet(name).SetVariantSelection(selection)
    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim(f"/{root.GetName()}", "Xform")
    prim.GetReferences().AddReference(os.path.abspath(usd_path))
    stage.SetDefaultPrim(prim)

    def capsule(path: str, radius: float, half_length: float, pos, quat_wxyz) -> None:
        geom = UsdGeom.Capsule.Define(stage, Sdf.Path(path))
        geom.CreateAxisAttr("Z")
        geom.CreateRadiusAttr(radius)
        geom.CreateHeightAttr(2.0 * half_length)
        xform = UsdGeom.Xformable(geom)
        xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
        xform.AddOrientOp().Set(Gf.Quatf(*map(float, quat_wxyz)))
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())

    if mjcf_path is None:
        for side, (start, end) in zip(("left", "right"), IBFM_HAND_CAPSULE_SEGMENTS):
            start, end = np.asarray(start), np.asarray(end)
            axis = (end - start) / np.linalg.norm(end - start)
            # rotate +z onto the segment direction (which lies in the x-y plane)
            half = math.atan2(axis[1], axis[0]) / 2.0
            quat = Gf.Quatf(math.cos(half), 0.0, 0.0, math.sin(half)) * Gf.Quatf(
                math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0
            )
            capsule(
                f"/{root.GetName()}/{side}_wrist_yaw_link/ibfm_hand_capsule",
                IBFM_HAND_CAPSULE_RADIUS,
                float(np.linalg.norm(end - start)) / 2.0,
                (start + end) / 2.0,
                (quat.GetReal(), *quat.GetImaginary()),
            )
    else:
        import mujoco  # noqa: PLC0415

        for link in root.GetChildren():
            for child in link.GetChildren():
                if child.GetName() == "collisions" or child.HasAPI(UsdPhysics.CollisionAPI):
                    stage.OverridePrim(f"/{root.GetName()}/{link.GetName()}/{child.GetName()}").SetActive(False)
        model = mujoco.MjModel.from_xml_path(mjcf_path)
        for geom in range(model.ngeom):
            if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom])
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or f"geom_{geom}"
            path = f"/{root.GetName()}/{body}/ibfm_{name}"
            size, pos, quat = model.geom_size[geom], model.geom_pos[geom], model.geom_quat[geom]
            if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                capsule(path, size[0], size[1], pos, quat)
            elif model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_SPHERE:
                sphere = UsdGeom.Sphere.Define(stage, Sdf.Path(path))
                sphere.CreateRadiusAttr(float(size[0]))
                UsdGeom.Xformable(sphere).AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
                UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
            else:
                raise ValueError(f"Unsupported MJCF collision geom type {model.geom_type[geom]} for {name}.")
    stage.GetRootLayer().Export(out_path)
    return out_path


class ReleasedControllerBridge:
    """Drive the released Push Box or Kick Box controller in a subprocess of the I-BFM environment."""

    def __init__(self, python: str, ibfm_root: str, task: str, num_envs: int):
        import subprocess  # noqa: PLC0415

        server = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ibfm_released_controller_server.py")
        self.process = subprocess.Popen(
            [python, server, ibfm_root, task, str(num_envs)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        self.phase = self._read()["phase"]

    def _read(self) -> dict:
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("The released controller server exited.")
        return json.loads(line)

    def request(self, payload: dict) -> dict:
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        reply = self._read()
        self.phase = reply["phase"]
        return reply

    def close(self) -> None:
        self.process.stdin.close()
        self.process.wait()


def released_push_metrics(cpu: dict, i: int, base_rot, box_rot, contact, standoff: float = 0.8) -> dict:
    """Return the metrics of the released Push Box runtime for environment ``i``."""
    import numpy as np  # noqa: PLC0415

    base, box, goal = cpu["base"][i], cpu["object"][i], cpu["goal"][i]
    robot_box = box[:2] - base[:2]
    box_goal = goal[:2] - box[:2]
    u = box_goal / max(float(np.linalg.norm(box_goal)), 1e-8)
    lateral = np.array([-u[1], u[0]])
    hands = cpu["hands"][i] - box
    rot = base_rot[i]
    local = rot[:2, :2].T @ robot_box
    left, right = bool(contact[i, 0]), bool(contact[i, 1])
    return {
        "horizontal_distance": float(np.linalg.norm(robot_box)),
        "goal_distance": float(np.linalg.norm(box_goal)),
        "object_height": float(box[2]),
        "object_speed": float(np.linalg.norm(cpu["box_vel"][i])),
        "object_vertical_speed": float(cpu["box_vel"][i, 2]),
        "lift": float(np.clip((box[2] - 0.186028293) / 0.35, 0.0, 1.0)),
        "root_height": float(base[2]),
        "pelvis_up_z": float(rot[2, 2]),
        "robot_upright": float(np.clip(rot[2, 2], 0.0, 1.0)),
        "box_upright": float(np.clip(box_rot[i, 2, 2], 0.0, 1.0)),
        "any_contact": left or right,
        "bilateral_contact": left and right,
        "left_hand_rear_depth": float(-hands[0, :2] @ u),
        "right_hand_rear_depth": float(-hands[1, :2] @ u),
        "left_hand_lateral_offset": float(hands[0, :2] @ lateral),
        "right_hand_lateral_offset": float(hands[1, :2] @ lateral),
        "left_hand_height_from_box": float(hands[0, 2]),
        "right_hand_height_from_box": float(hands[1, 2]),
        "nonhand_robot_box_contact": False,
        "nonhand_robot_box_contact_count": 0.0,
        "approach_boxward_speed": float(
            cpu["base_vel"][i, :2] @ (robot_box / max(float(np.linalg.norm(robot_box)), 1e-8))
        ),
        "transport_goalward_speed": float(cpu["box_vel"][i, :2] @ u),
        "push_staging_distance": float(np.linalg.norm(box[:2] - u * standoff - base[:2])),
        "push_behind_projection": float(robot_box @ u),
        "push_lateral_error": float(abs(robot_box @ lateral)),
        "push_box_bearing_deg": float(math.degrees(math.atan2(local[1], local[0]))),
        "root_heading_deg": float(math.degrees(math.atan2(rot[1, 0], rot[0, 0]))),
        "box_x": float(box[0]),
        "box_y": float(box[1]),
    }


def run(args: argparse.Namespace, sim_cfg) -> None:
    """Build the scene, roll out the controller and report task success."""
    import numpy as np
    import torch
    from isaaclab_neural.controllers import ibfm
    from isaaclab_neural.controllers.bfm_zero import BfmZeroObservationHistory
    from isaaclab_neural.controllers.ibfm_carry import IbfmCarryController, IbfmCarryLatents
    from isaaclab_neural.robots.g1_29dof import G1_29DOF_DEFAULT_JOINT_POS, G1_29DOF_IBFM_CFG, G1_29DOF_JOINT_NAMES

    import isaaclab.sim as sim_utils
    import isaaclab.utils.math as math_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.utils import string as string_utils
    from isaaclab.utils.configclass import configclass

    if not args.policy:
        raise ValueError("Pass --policy or set IBFM_POLICY_PATH.")
    if args.task != "carry" and not args.released_controller:
        raise ValueError("Push and kick need --released-controller.")
    torch.manual_seed(args.seed)
    carry = args.task == "carry"
    controller_cfg = demo_controller_cfg(args.task, args.param)
    control_dt, decimation = 0.02, 4
    ground_height = 0.186028293
    import tempfile

    from isaaclab_neural.robots.g1_29dof import resolve_g1_29dof_usd_path

    base_usd = resolve_g1_29dof_usd_path(args.usd or G1_29DOF_IBFM_CFG.spawn.usd_path)
    mjcf = (
        None
        if args.usd_collision
        else (
            args.mjcf
            or os.path.join(args.latent_root, os.pardir, "humanoidverse", "data", "robots", "g1_mjlab", "g1_29dof.xml")
        )
    )
    hand_usd = write_ibfm_collision_usd(
        base_usd,
        os.path.join(tempfile.mkdtemp(prefix="ibfm_robot_"), "g1_29dof_ibfm.usda"),
        mjcf,
        G1_29DOF_IBFM_CFG.spawn.variants,
    )
    spawn = G1_29DOF_IBFM_CFG.spawn.replace(usd_path=hand_usd)

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
            init_state=RigidObjectCfg.InitialStateCfg(pos=(args.distance, 0.0, 0.2), rot=BOX_REST_QUAT),
        )

    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(SceneCfg(num_envs=args.num_envs, env_spacing=8.0))
    goal_marker = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/ibfm_goal",
            markers={
                "goal": sim_utils.CylinderCfg(
                    radius=SUCCESS_RADIUS[args.task],
                    height=0.01,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.25)),
                )
            },
        )
    )
    sim.reset()
    robot, box, device, n = scene["robot"], scene["box"], sim.device, args.num_envs
    if list(robot.joint_names) != list(G1_29DOF_JOINT_NAMES):
        raise RuntimeError(f"Unexpected joint order {robot.joint_names}.")
    body_ids = [robot.body_names.index(name) for name in ibfm.IBFM_PRIVILEGED_BODY_NAMES]
    hand_ids = [robot.body_names.index(name) for name in ibfm.IBFM_HAND_BODY_NAMES]
    policy = ibfm.load_ibfm_policy(args.policy, device)
    if carry:
        latents = IbfmCarryLatents.from_directory(
            os.path.join(args.latent_root, "carry_box", "latents"),
            device,
            os.path.join(args.latent_root, "push_box", "latents"),
        )
        controllers = [IbfmCarryController(controller_cfg, control_dt) for _ in range(n)]
        bridge = None
    else:
        controllers = []
        bridge = ReleasedControllerBridge(
            args.released_controller, os.path.join(args.latent_root, os.pardir), args.task, n
        )
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
    box_quat0 = torch.tensor(BOX_REST_QUAT, device=device).expand(n, 4)
    box_pos0 = origins + torch.stack(
        [distance, torch.zeros_like(distance), torch.full_like(distance, BOX_HALF_SIZE[2] + 0.002)], -1
    )
    box.write_root_pose_to_sim_index(root_pose=torch.cat([box_pos0, box_quat0], -1))
    marker_quat0 = math_utils.quat_mul(box_quat0, math_utils.quat_conjugate(geom_quat))
    marker = box_pos0 - math_utils.quat_apply(marker_quat0, geom_pos)
    marker[:, 2] = ground_height
    box.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=device))
    goal = marker + torch.tensor([args.goal_x, args.goal_y, 0.0], device=device)
    goal_marker.visualize(
        translations=goal * torch.tensor([1.0, 1.0, 0.0], device=device) + torch.tensor([0, 0, 0.005], device=device)
    )
    robot.reset()
    box.reset()
    scene.update(0.0)

    history = BfmZeroObservationHistory(n, len(default), device)
    contact = torch.zeros(n, 2, dtype=torch.bool, device=device)
    gravity = torch.tensor([0.0, 0.0, -1.0], device=device).expand(n, 3)
    hold_steps = round(SUCCESS_HOLD_S / control_dt)
    hold, success_time = np.zeros(n, dtype=int), [None] * n
    video = None
    if args.video:
        from isaaclab_neural.eval.newton_video import NewtonFollowVideo  # noqa: PLC0415

        video = NewtonFollowVideo(
            sim, args.video, fps=1.0 / control_dt, distance=args.video_distance, azimuth_deg=args.video_azimuth
        )
    trace = []
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
        box_rot = math_utils.matrix_from_quat(box_quat).cpu().numpy()
        cpu = {
            k: v.cpu().numpy()
            for k, v in (
                ("base", base_pos),
                ("object", object_pos),
                ("goal", goal),
                ("box", box_pos),
                ("hands", hand_pos),
                ("box_vel", box_vel),
                ("base_vel", data.root_link_lin_vel_w.torch),
            )
        }
        speed, both = box_vel.norm(dim=-1).cpu().numpy(), contact.all(dim=-1).cpu().numpy()
        commands, masks, lookahead = [], [], []
        for i, controller in enumerate([] if bridge else controllers):
            box_dir = cpu["object"][i, :2] - cpu["base"][i, :2]
            goal_dir = cpu["goal"][i, :2] - cpu["object"][i, :2]
            metrics = {
                "horizontal_distance": float(np.linalg.norm(box_dir)),
                "goal_distance": float(np.linalg.norm(goal_dir)),
                "object_height": float(cpu["object"][i, 2]),
                "object_speed": float(speed[i]),
                "lift": float(np.clip((cpu["object"][i, 2] - ground_height) / controller_cfg.lift_height, 0, 1)),
                "root_height": float(cpu["base"][i, 2]),
                "pelvis_up_z": float(base_rot[i, 2, 2]),
                "robot_upright": float(np.clip(base_rot[i, 2, 2], 0.0, 1.0)),
                "box_upright": float(np.clip(box_rot[i, 2, 2], 0.0, 1.0)),
                "bilateral_contact": bool(both[i]),
            }
            if step > 0:
                controller.update(metrics)
                inside = float(np.linalg.norm(cpu["goal"][i] - cpu["object"][i])) < SUCCESS_RADIUS[args.task]
                hold[i] = hold[i] + 1 if inside else 0
                if hold[i] >= hold_steps and success_time[i] is None:
                    success_time[i] = step * control_dt
            command, mask, ahead = controller.latent(latents, metrics, base_rot[i], box_dir, goal_dir)
            commands.append(command)
            masks.append(mask)
            lookahead.append(ahead)
        if bridge is not None:
            contact_np = contact.cpu().numpy()
            released = [released_push_metrics(cpu, i, base_rot, box_rot, contact_np) for i in range(n)]
            if step > 0:
                bridge.request({"cmd": "update", "metrics": released})
                for i in range(n):
                    inside = released[i]["goal_distance"] < SUCCESS_RADIUS[args.task]
                    hold[i] = hold[i] + 1 if inside else 0
                    if hold[i] >= hold_steps and success_time[i] is None:
                        success_time[i] = step * control_dt
            masks = lookahead = [False] * n
        mask = torch.tensor(masks, device=device)[:, None]
        context = torch.where(mask, torch.zeros_like(context), context)
        if carry and controller_cfg.align_goal_lookahead > 0.0:
            ahead = torch.tensor(lookahead, device=device)[:, None]
            delta = context[:, 15:17]
            moved = context.clone()
            direction = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            moved[:, 15:17] = delta + controller_cfg.align_goal_lookahead * direction
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
        if bridge is not None:
            reply = bridge.request(
                {
                    "cmd": "latent",
                    "metrics": released,
                    "state": state.cpu().tolist(),
                    "privileged": privileged.cpu().tolist(),
                    "last_action": obs[:, 2 * joints + 6 : 3 * joints + 6].cpu().tolist(),
                    "history": obs[:, 3 * joints + 6 :].cpu().tolist(),
                    "base_rot": base_rot.reshape(n, 9).tolist(),
                    "box_dir": (cpu["object"][:, :2] - cpu["base"][:, :2]).tolist(),
                    "goal_dir": (cpu["goal"][:, :2] - cpu["object"][:, :2]).tolist(),
                }
            )
            state = torch.tensor(reply["state"], device=device)
            privileged = torch.tensor(reply["privileged"], device=device)
            commands = list(torch.tensor(reply["latent"], device=device))
        with torch.no_grad():
            action = policy(
                state,
                privileged,
                obs[:, 2 * joints + 6 : 3 * joints + 6],
                obs[:, 3 * joints + 6 :],
                torch.stack(commands),
            ).clamp(-1.0, 1.0)
        action = action * 5.0
        history.record(action)
        target = torch.clamp(default + scale * action, limits[:, 0], limits[:, 1])
        for _ in range(decimation):
            robot.set_joint_position_target_index(target=target)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())
        if args.trace:
            e = args.video_env
            trace.append(
                [
                    step * control_dt,
                    bridge.phase[e] if bridge else controllers[e].phase,
                    *cpu["base"][e].round(3),
                    *cpu["object"][e].round(3),
                ]
                + [
                    round(float(np.linalg.norm(cpu["goal"][e, :2] - cpu["object"][e, :2])), 3),
                    round(float(speed[e]), 3),
                ]
                + [int(v) for v in contact[e].tolist()]
            )
        if video is not None:
            video.capture(robot.data.root_link_pos_w.torch[args.video_env].cpu().numpy())
        if step % 250 == 0:
            phases = bridge.phase if bridge else [c.phase for c in controllers]
            print(
                f"t={step * control_dt:5.1f}s phases={ {p: phases.count(p) for p in sorted(set(phases))} } "
                f"success={np.mean([t is not None for t in success_time]):.2f}",
                flush=True,
            )
        if (
            args.stop_on_success
            and all(t is not None for t in success_time)
            and step * control_dt >= max(success_time) + 2.0
        ):
            break

    if video is not None:
        video.close()
    if bridge is not None:
        final_phases = list(bridge.phase)
        bridge.close()
    else:
        final_phases = [c.phase for c in controllers]
    if args.trace:
        import csv  # noqa: PLC0415

        with open(args.trace, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["t", "phase", "root_x", "root_y", "root_z", "box_x", "box_y", "box_z", "goal_xy", "box_speed"]
                + ["contact_l", "contact_r"]
            )
            writer.writerows(trace)
    times = [t for t in success_time if t is not None]
    summary = {
        "task": args.task,
        "success_rate": float(np.mean([t is not None for t in success_time])),
        "success_time_median_s": float(np.median(times)) if times else None,
        "success_time_s": success_time,
        "falls_per_env": None if bridge is not None else float(np.mean([c.falls for c in controllers])),
        "final_phases": final_phases,
        "controller": dataclasses.asdict(controller_cfg) if controller_cfg is not None else None,
        "args": {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, list, type(None)))},
    }
    print(json.dumps({k: summary[k] for k in ("task", "success_rate", "success_time_median_s", "falls_per_env")}))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(summary, handle, indent=1)


if __name__ == "__main__":
    main()
