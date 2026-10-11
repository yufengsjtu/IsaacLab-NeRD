# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Registered Unitree G1 environments backed by the NeRD Newton solver."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import TYPE_CHECKING

import torch
from isaaclab_newton.physics import NewtonCollisionPipelineCfg, NewtonShapeCfg

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from isaaclab_tasks.core.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg
from isaaclab_tasks.utils.hydra import resolve_presets

from isaaclab_neural.physics import NerdNewtonCfg, NerdSolverCfg
from isaaclab_neural.physics.nerd_solver_cfg import normalize_contact_mode

from ..contacts.geometry_mdp import GeometryFeetAirTimePositiveBiped, GeometryFeetSlide, geometry_illegal_contact
from .neural_env_wrapper import NerdManagerBasedRLEnv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

DEFAULT_G1_FLAT_NERD_MODEL_PATH = "./pre-trained_models/G1/nn/final_model.pt"
DEFAULT_G1_ROUGH_NERD_MODEL_PATH = "./pre-trained_models/G1-Rough-Native/nn/final_model.pt"


def g1_invalid_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    maximum_height: float | None = 1.5,
    maximum_linear_velocity: float = 20.0,
    maximum_angular_velocity: float = 50.0,
    maximum_joint_velocity: float = 100.0,
) -> torch.Tensor:
    """Terminate extreme finite G1 states with an optional flat-height guard; reject non-finite states.

    Args:
        env: Environment containing the robot and environment origins.
        asset_cfg: Robot to inspect.
        maximum_height: Root height above the environment origin [m], or None to disable the flat-height guard.
        maximum_linear_velocity: Maximum absolute root velocity component [m/s].
        maximum_angular_velocity: Maximum absolute root angular velocity component [rad/s].
        maximum_joint_velocity: Maximum absolute joint velocity [rad/s].

    Returns:
        Failure flags for each environment, not timeouts.

    Raises:
        FloatingPointError: State contains NaN or infinity. Resetting after reward
            computation would otherwise allow non-finite rewards into PPO.
    """
    data = env.scene[asset_cfg.name].data
    root = data.root_state_w.torch
    joints = data.joint_pos.torch
    joint_velocity = data.joint_vel.torch
    if not (torch.isfinite(root).all() & torch.isfinite(joints).all() & torch.isfinite(joint_velocity).all()):
        raise FloatingPointError("Non-finite G1 state; stopping before reward computation.")
    height = torch.zeros(len(root), dtype=torch.bool, device=root.device)
    if maximum_height is not None:
        height = root[:, 2] - env.scene.env_origins[:, 2] > maximum_height
    return (
        height
        | (root[:, 7:10].abs().amax(dim=1) > maximum_linear_velocity)
        | (root[:, 10:13].abs().amax(dim=1) > maximum_angular_velocity)
        | (joint_velocity.abs().amax(dim=1) > maximum_joint_velocity)
    )


def g1_replayed_terrain_levels(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    """Report inherited real-MJ terrain levels without advancing curriculum on short imagined episodes."""
    del env_ids
    return env.scene.terrain.terrain_levels.float().mean()


def configure_g1_mbpo_rough(cfg: G1RoughEnvCfg, settings: dict) -> None:
    """Configure a shared rough atlas, explicit physical events and real-episode curriculum authority.

    Args:
        cfg: Rough manager-based configuration before scene construction.
        settings: Terrain/physics profiles, shared seed, worker role, observation-noise settings and the
            geometry ``feet_contact_separation`` [m] used by the gait rewards (default 0).

    The ``stock`` profile preserves default G1 terrain geometry and real-MJ curriculum. Imagined
    branches inherit captured terrain assignments instead of advancing difficulty on short timeouts.
    Omitting the profile retains the previous fixed-atlas, nominal-physics behavior.
    """
    from isaaclab.utils.dict import class_to_dict

    if isinstance(cfg, G1FlatEnvCfg) or cfg.scene.height_scanner is None:
        raise ValueError("Rough MBPO requires a rough terrain scene and its height scanner.")
    profile = settings.get("terrain_profile", "fixed")
    events = settings.get("physics_events", "nominal")
    if profile not in {"fixed", "stock"} or events not in {"nominal", "stock"}:
        raise ValueError("Unsupported rough MBPO terrain or physics-event profile.")
    stock = G1RoughEnvCfg()
    defaults = stock.scene.terrain.terrain_generator
    rows = settings.get("terrain_rows", defaults.num_rows if profile == "stock" else 4)
    cols = settings.get("terrain_cols", defaults.num_cols if profile == "stock" else 4)
    seed = settings.get("terrain_seed", 42)
    low = settings.get("terrain_difficulty_min", defaults.difficulty_range[0] if profile == "stock" else 0.0)
    high = settings.get("terrain_difficulty_max", defaults.difficulty_range[1] if profile == "stock" else 0.35)
    if (
        type(rows) is not int
        or type(cols) is not int
        or min(rows, cols) < 1
        or max(rows, cols) > 20
        or type(seed) is not int
        or seed < 0
        or not 0 <= low <= high <= 1
    ):
        raise ValueError("Invalid rough-terrain seed, grid or difficulty range.")
    generator = deepcopy(defaults if profile == "stock" else cfg.scene.terrain.terrain_generator)
    if generator is None:
        raise ValueError("Rough MBPO requires a reproducible terrain generator.")
    generator.seed, generator.num_rows, generator.num_cols = seed, rows, cols
    generator.difficulty_range = (float(low), float(high))
    generator.curriculum = profile == "stock"
    if profile == "stock":
        if (rows, cols, (low, high)) != (defaults.num_rows, defaults.num_cols, defaults.difficulty_range):
            raise ValueError("Stock rough alignment cannot override default terrain geometry or difficulty.")
        cfg.curriculum = deepcopy(stock.curriculum)
        cfg.scene.terrain.max_init_terrain_level = stock.scene.terrain.max_init_terrain_level
        if settings.get("role") in {"nerd_train", "model_fit", "probe_nerd"}:
            cfg.curriculum.terrain_levels.func = g1_replayed_terrain_levels
        cfg.rewards.feet_slide.weight = stock.rewards.feet_slide.weight
    else:
        cfg.scene.terrain.max_init_terrain_level = rows - 1
        cfg.curriculum.terrain_levels = None
    cfg.scene.terrain.terrain_generator = generator
    cfg.observations.policy.enable_corruption = settings.get("policy_observation_corruption", True)
    cfg.mbpo_randomized_physics = events == "stock"
    cfg.mbpo_terrain_profile = profile
    event_names = ("physics_material", "add_base_mass", "push_robot", "base_external_force_torque")
    for name in event_names:
        setattr(cfg.events, name, deepcopy(getattr(stock.events, name)) if events == "stock" else None)
    event_config = class_to_dict(cfg.events)
    cfg.mbpo_event_contract = json.loads(json.dumps({name: event_config[name] for name in event_names}, default=str))
    cfg.mbpo_asset_contract = json.loads(json.dumps(class_to_dict(cfg.scene.robot), default=str))
    cfg.terminations.invalid_state = (
        None
        if settings.get("contact_mdp") == "stock"
        else DoneTerm(func=g1_invalid_state, params={"maximum_height": None})
    )
    # Geometry gait rewards count a foot as touching within this separation [m]; the torso termination keeps
    # strict contact. A small positive margin removes millimetre contact chatter from NeRD foot predictions.
    separation = settings.get("feet_contact_separation", 0.0)
    if not isinstance(separation, (int, float)) or not 0.0 <= separation <= 0.01:
        raise ValueError("feet_contact_separation must lie in [0, 0.01] m.")
    if settings.get("contact_mdp") != "stock":
        for name in ("feet_air_time", "feet_slide"):
            term = getattr(cfg.rewards, name)
            if term is not None:
                term.params["maximum_separation"] = float(separation)


def apply_g1_contact_mode_mdp(env_cfg) -> None:
    """Configure geometry-based contact MDP terms for G1 Newton-native contacts.

    G1 stock rewards use contact-force sensors (``feet_air_time``, ``feet_slide``,
    ``base_contact``). NeRD has no constraint-solver contact force, so those
    terms are remapped or disabled under ``newton_native``.
    """
    mode = env_cfg.sim.physics.solver_cfg.contact_mode
    slide_cfg = env_cfg.rewards.feet_slide
    env_cfg.scene.contact_forces = None
    # Force-sensor feet slide is unavailable without contact_forces.
    env_cfg.rewards.feet_slide = None
    if mode == "newton_native":
        air_time_cfg = env_cfg.rewards.feet_air_time
        if air_time_cfg is None:
            stock_cfg = G1FlatEnvCfg() if isinstance(env_cfg, G1FlatEnvCfg) else G1RoughEnvCfg()
            air_time_cfg = stock_cfg.rewards.feet_air_time
        slide_weight = 0.0
        if isinstance(env_cfg, G1FlatEnvCfg):
            if slide_cfg is None:
                slide_cfg = G1FlatEnvCfg().rewards.feet_slide
            slide_weight = slide_cfg.weight
        env_cfg.rewards.feet_air_time = RewTerm(
            func=GeometryFeetAirTimePositiveBiped,
            weight=air_time_cfg.weight,
            params={
                "body_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
                "command_name": "base_velocity",
                "threshold": air_time_cfg.params["threshold"],
            },
        )
        env_cfg.rewards.feet_slide = RewTerm(
            func=GeometryFeetSlide,
            weight=slide_weight,
            params={
                "body_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
                "history_length": 3,
            },
        )
        env_cfg.terminations.base_contact = DoneTerm(
            func=geometry_illegal_contact,
            params={"body_cfg": SceneEntityCfg("robot", body_names="torso_link")},
        )
    else:
        env_cfg.rewards.feet_air_time = None
        env_cfg.terminations.base_contact = None


@configclass
class NerdG1FlatEnvCfg(G1FlatEnvCfg):
    """Manager-based G1 flat config using NeRD-backed Newton physics."""

    def __post_init__(self) -> None:
        super().__post_init__()
        base_physics = resolve_presets(deepcopy(self.sim.physics))
        collision_cfg = deepcopy(getattr(base_physics, "collision_cfg", None))
        if collision_cfg is None:
            collision_cfg = NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000)
        default_shape_cfg = deepcopy(getattr(base_physics, "default_shape_cfg", None))
        if default_shape_cfg is None:
            default_shape_cfg = NewtonShapeCfg(margin=0.01)
        self.events.push_robot = None  # type: ignore[assignment]
        self.sim.physics = NerdNewtonCfg(
            solver_cfg=NerdSolverCfg(
                name="TransformerNeuralSolver",
                neural_model_path=DEFAULT_G1_FLAT_NERD_MODEL_PATH,
                num_states_history=10,
                num_contacts_per_env=64,
                max_contact_tokens=64,
                contact_mode="newton_native",
                contact_representation="contact_tokens",
                contact_filter="solver_active",
                contact_packing_policy="body_round_robin_pair_atomic",
                min_contact_event_threshold=0.12,
                states_frame="body",
                anchor_frame_step="every",
                states_embedding_type="identical",
                prediction_type="relative",
                orientation_prediction_parameterization="quaternion",
            ),
            collision_cfg=collision_cfg,
            default_shape_cfg=default_shape_cfg,
            num_substeps=1,
            debug_mode=False,
            use_cuda_graph=False,
        )
        apply_g1_contact_mode_mdp(self)
        self.terminations.invalid_state = DoneTerm(func=g1_invalid_state)


class NerdG1FlatEnv(NerdManagerBasedRLEnv):
    """G1 flat env that installs ``NerdNewtonCfg`` before simulation initialization."""

    def _configure_solver_dependent_mdp(self, cfg) -> None:
        normalize_contact_mode(cfg.sim.physics.solver_cfg.contact_mode)


@configclass
class NerdG1RoughEnvCfg(G1RoughEnvCfg):
    """Manager-based G1 rough-terrain config using NeRD-backed Newton physics."""

    def __post_init__(self) -> None:
        super().__post_init__()
        base_physics = resolve_presets(deepcopy(self.sim.physics))
        collision_cfg = deepcopy(getattr(base_physics, "collision_cfg", None))
        if collision_cfg is None:
            collision_cfg = NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000)
        default_shape_cfg = deepcopy(getattr(base_physics, "default_shape_cfg", None))
        if default_shape_cfg is None:
            default_shape_cfg = NewtonShapeCfg(margin=0.01)

        self.events.push_robot = None  # type: ignore[assignment]
        self.sim.physics = NerdNewtonCfg(
            solver_cfg=NerdSolverCfg(
                name="TransformerNeuralSolver",
                neural_model_path=DEFAULT_G1_ROUGH_NERD_MODEL_PATH,
                num_states_history=10,
                num_contacts_per_env=64,
                max_contact_tokens=64,
                contact_mode="newton_native",
                contact_representation="contact_tokens",
                contact_filter="solver_active",
                contact_packing_policy="body_round_robin_pair_atomic",
                min_contact_event_threshold=0.12,
                states_frame="body",
                anchor_frame_step="every",
                states_embedding_type="identical",
                prediction_type="relative",
                orientation_prediction_parameterization="quaternion",
            ),
            collision_cfg=collision_cfg,
            default_shape_cfg=default_shape_cfg,
            num_substeps=1,
            debug_mode=False,
            use_cuda_graph=False,
        )
        apply_g1_contact_mode_mdp(self)


class NerdG1RoughEnv(NerdManagerBasedRLEnv):
    """G1 rough-terrain env that installs ``NerdNewtonCfg`` before simulation initialization."""

    def _configure_solver_dependent_mdp(self, cfg) -> None:
        normalize_contact_mode(cfg.sim.physics.solver_cfg.contact_mode)
