# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree G1 29-DoF asset configuration matched to the BFM-Zero controller.

The articulation uses the Isaac Lab ``g1_29dof.usd`` asset without hands and the PhysX collision variant. Actuator
gains, armature, effort limits and joint friction follow the humanoidverse ``g1/g1_29dof.yaml`` robot configuration
used by the released BFM-Zero deployment policy.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable
from urllib.parse import urlparse

from isaaclab_physx.sim.schemas import PhysxArticulationRootPropertiesCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.configclass import configclass

G1_29DOF_USD_URL = "omniverse://isaac-dev.ov.nvidia.com/Isaac/IsaacLab/Robots/Unitree/G1/g1_29dof.usd"
"""Isaac Lab G1 29-DoF asset. Its root layer references payloads through absolute HTTPS URLs."""

G1_29DOF_USD_ENV_VAR = "ISAACLAB_NEURAL_G1_29DOF_USD"
"""Environment variable that overrides :data:`G1_29DOF_USD_URL`, e.g. with a pre-localized copy."""

G1_29DOF_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
"""Joint order shared by the BFM-Zero policy and the Newton articulation."""

G1_29DOF_DEFAULT_JOINT_POS = {".*_hip_pitch_joint": -0.1, ".*_knee_joint": 0.3, ".*_ankle_pitch_joint": -0.2}
"""Default joint positions [rad]; unspecified joints default to zero."""

G1_29DOF_WALKING_JOINT_POS = {
    ".*_hip_pitch_joint": -0.20,
    ".*_knee_joint": 0.42,
    ".*_ankle_pitch_joint": -0.23,
    ".*_elbow_joint": 0.87,
    "left_shoulder_roll_joint": 0.16,
    "left_shoulder_pitch_joint": 0.35,
    "right_shoulder_roll_joint": -0.16,
    "right_shoulder_pitch_joint": 0.35,
}
"""Standing pose of the Isaac Lab 29-DoF G1 locomotion configuration (isaac-sim/IsaacLab PR #8055) [rad]."""

G1_29DOF_WALKING_ROOT_HEIGHT = 0.793
"""Root height of :data:`G1_29DOF_WALKING_JOINT_POS` above flat ground [m]."""

G1_29DOF_STIFFNESS = {
    ".*_hip_yaw_joint": 40.17924,
    ".*_hip_roll_joint": 99.09843,
    ".*_hip_pitch_joint": 99.09843,
    ".*_knee_joint": 99.09843,
    ".*_ankle_.*_joint": 28.50125,
    "waist_yaw_joint": 40.17924,
    "waist_(roll|pitch)_joint": 28.50125,
    ".*_shoulder_.*_joint": 14.25062,
    ".*_elbow_joint": 14.25062,
    ".*_wrist_roll_joint": 14.25062,
    ".*_wrist_(pitch|yaw)_joint": 16.77833,
}
"""Implicit PD stiffness [N·m/rad]."""

G1_29DOF_DAMPING = {
    ".*_hip_yaw_joint": 2.55789,
    ".*_hip_roll_joint": 6.3088,
    ".*_hip_pitch_joint": 6.3088,
    ".*_knee_joint": 6.3088,
    ".*_ankle_.*_joint": 1.81445,
    "waist_yaw_joint": 2.55789,
    "waist_(roll|pitch)_joint": 1.81445,
    ".*_shoulder_.*_joint": 0.90722,
    ".*_elbow_joint": 0.90722,
    ".*_wrist_roll_joint": 0.90722,
    ".*_wrist_(pitch|yaw)_joint": 1.06814,
}
"""Implicit PD damping [N·m·s/rad]."""

G1_29DOF_ARMATURE = {
    ".*_hip_(pitch|roll)_joint": 0.025101925,
    ".*_hip_yaw_joint": 0.01017752004,
    ".*_knee_joint": 0.025101925,
    ".*_ankle_.*_joint": 0.00721945,
    "waist_yaw_joint": 0.01017752004,
    "waist_(roll|pitch)_joint": 0.00721945,
    ".*_shoulder_.*_joint": 0.003609725,
    ".*_elbow_joint": 0.003609725,
    ".*_wrist_roll_joint": 0.003609725,
    ".*_wrist_(pitch|yaw)_joint": 0.00425,
}
"""Reflected rotor inertia [kg·m²]."""

G1_29DOF_EFFORT_LIMIT = {
    ".*_hip_.*_joint": 88.0,
    ".*_knee_joint": 139.0,
    ".*_ankle_.*_joint": 50.0,
    "waist_yaw_joint": 88.0,
    "waist_(roll|pitch)_joint": 50.0,
    ".*_shoulder_.*_joint": 25.0,
    ".*_elbow_joint": 25.0,
    ".*_wrist_roll_joint": 25.0,
    ".*_wrist_(pitch|yaw)_joint": 5.0,
}
"""Solver effort limits [N·m]; also the effort limits used for the BFM-Zero action scale."""

G1_29DOF_VELOCITY_LIMIT = {
    ".*_hip_.*_joint": 32.0,
    ".*_knee_joint": 20.0,
    ".*_ankle_.*_joint": 37.0,
    "waist_yaw_joint": 32.0,
    "waist_(roll|pitch)_joint": 37.0,
    ".*_shoulder_.*_joint": 37.0,
    ".*_elbow_joint": 37.0,
    ".*_wrist_roll_joint": 37.0,
    ".*_wrist_(pitch|yaw)_joint": 22.0,
}
"""Requested solver velocity limits [rad/s]. MJWarp does not enforce this clamp."""

G1_29DOF_JOINT_FRICTION = 0.03
"""Static joint friction effort [N·m]."""

G1_29DOF_WRIST_YAW_LINK_MASS = 0.254576
"""Unitree rev_1_0 wrist-yaw link mass including the rubber hand [kg]; the asset without hands has 0.0846 kg."""


def g1_29dof_mirror_path(url: str, download_dir: str) -> str:
    """Return where Isaac Lab's recursive asset retrieval mirrors a remote URL.

    The layout matches :func:`isaaclab.utils.assets.retrieve_file_path`:
    ``<download_dir>/<scheme>/<netloc>/<path>``.

    Args:
        url: Remote asset URL.
        download_dir: Root of the local mirror.

    Returns:
        Local mirror path, or an empty string when ``url`` is not a remote URL.
    """
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc or not parsed.path:
        return ""
    return os.path.join(
        download_dir, parsed.scheme, parsed.netloc.replace(":", "_"), *parsed.path.lstrip("/").split("/")
    )


def localize_usd_root_layer(root_path: str, download_dir: str) -> str:
    """Copy a mirrored root layer and point its absolute remote references at the local mirror.

    Without Kit, OpenUSD cannot resolve ``http(s)://`` asset paths. Isaac Lab mirrors every dependency, but the root
    layer still names remote URLs; nested layers of the G1 29-DoF asset use relative paths and need no rewrite.

    Args:
        root_path: Local copy of the root layer.
        download_dir: Root of the local mirror.

    Returns:
        Path of the rewritten root layer next to ``root_path``, or ``root_path`` when nothing needed rewriting.
    """
    from pxr import Sdf, UsdUtils  # noqa: PLC0415

    def _local(path: str) -> str:
        mirrored = g1_29dof_mirror_path(path, download_dir)
        return mirrored if mirrored and os.path.isfile(mirrored) else path

    references: set[str] = set()
    UsdUtils.ModifyAssetPaths(Sdf.Layer.FindOrOpen(root_path), lambda path: (references.add(path), path)[1])
    if all(_local(path) == path for path in references):
        return root_path
    stem, extension = os.path.splitext(root_path)
    localized = f"{stem}.localized{extension}"
    shutil.copyfile(root_path, localized)
    layer = Sdf.Layer.FindOrOpen(localized)
    layer.Reload(force=True)
    UsdUtils.ModifyAssetPaths(layer, _local)
    layer.Save()
    return localized


def resolve_g1_29dof_usd_path(usd_path: str, download_dir: str | None = None) -> str:
    """Return a local G1 29-DoF root layer, downloading and localizing remote assets when needed.

    Args:
        usd_path: Local path or remote URL. :data:`G1_29DOF_USD_ENV_VAR` takes precedence when set.
        download_dir: Local mirror directory. Defaults to the system temporary directory.

    Returns:
        Local path of a root layer whose references resolve without Kit.
    """
    from isaaclab.utils.assets import check_file_path, retrieve_file_path  # noqa: PLC0415

    usd_path = os.environ.get(G1_29DOF_USD_ENV_VAR) or usd_path
    if check_file_path(usd_path) == 1:
        return os.path.abspath(usd_path)
    download_dir = os.path.abspath(download_dir or tempfile.gettempdir())
    return localize_usd_root_layer(retrieve_file_path(usd_path, download_dir=download_dir), download_dir)


def spawn_g1_29dof_usd(prim_path: str, cfg: G129DofUsdFileCfg, *args, **kwargs):
    """Spawn :class:`G129DofUsdFileCfg` from a localized root layer.

    Args:
        prim_path: Prim path or expression to spawn at.
        cfg: Spawner configuration.
        *args: Positional arguments forwarded to :func:`isaaclab.sim.spawn_from_usd`.
        **kwargs: Keyword arguments forwarded to :func:`isaaclab.sim.spawn_from_usd`.

    Returns:
        The spawned prim.
    """
    local_cfg = cfg.replace(usd_path=resolve_g1_29dof_usd_path(cfg.usd_path, cfg.download_dir))
    return sim_utils.spawn_from_usd(prim_path, local_cfg, *args, **kwargs)


@configclass
class G129DofUsdFileCfg(sim_utils.UsdFileCfg):
    """USD spawner that localizes absolute remote references before spawning."""

    func: Callable | str = spawn_g1_29dof_usd

    download_dir: str | None = None
    """Local mirror directory. Defaults to the system temporary directory."""


G1_29DOF_BFM_CFG = ArticulationCfg(
    spawn=G129DofUsdFileCfg(
        usd_path=G1_29DOF_USD_URL,
        variants={"Physics": "PhysX", "left_hand": "None", "right_hand": "None"},
        activate_contact_sensors=False,
        articulation_props=PhysxArticulationRootPropertiesCfg(enabled_self_collisions=False),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8), joint_pos=G1_29DOF_DEFAULT_JOINT_POS, joint_vel={".*": 0.0}
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=G1_29DOF_STIFFNESS,
            damping=G1_29DOF_DAMPING,
            armature=G1_29DOF_ARMATURE,
            friction=G1_29DOF_JOINT_FRICTION,
            joint_effort_limit=G1_29DOF_EFFORT_LIMIT,
            joint_velocity_limit=G1_29DOF_VELOCITY_LIMIT,
        )
    },
)
"""Unitree G1 29-DoF articulation without hands, with BFM-Zero actuator parameters."""
