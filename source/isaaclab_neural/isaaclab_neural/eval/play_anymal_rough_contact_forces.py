# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate per-body contact-force predictions during an Anymal rough-terrain policy rollout.

The rollout uses the ground-truth Newton MJWarp solver. A separate NeRD solver
runs as an observer, so the Newton ContactSensor remains a meaningful source of
post-solve force targets while the neural prediction does not affect dynamics.

Example:
    PYTHONPATH=./source/isaaclab_neural ./isaaclab.sh -p \
      -m isaaclab_neural.eval.play_anymal_rough_contact_forces \
      --nerd-checkpoint ./data/trained_models/anymal-force/nn/final_model.pt \
      --policy-checkpoint ./logs/rsl_rl/anymal_c_rough/<run>/model_1499.pt \
      --num-steps 500 --visualizer newton
"""

from __future__ import annotations

# The runtime APIs below are selected dynamically by Hydra/backend presets, and
# the IDE may resolve their installed main-worktree stubs instead of this feature worktree.
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false
# pyright: reportOptionalMemberAccess=false, reportPrivateUsage=false
import argparse
import copy
import importlib.metadata as metadata
import sys
from collections.abc import Mapping
from pathlib import Path

import torch


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse validation and Isaac Lab launcher arguments."""
    parser = argparse.ArgumentParser(description="Validate Anymal per-body contact-force predictions.")
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Velocity-Rough-Anymal-C-Play-v0",
        help="Ground-truth Newton rough-terrain task.",
    )
    parser.add_argument(
        "--action-mode",
        choices=["policy", "random"],
        default="policy",
        help="Drive Anymal with an RSL-RL policy or uniformly sampled actions.",
    )
    parser.add_argument("--policy-agent", type=str, default="rsl_rl_cfg_entry_point")
    parser.add_argument("--policy-checkpoint", type=str, default=None, help="RSL-RL policy checkpoint.")
    parser.add_argument("--nerd-checkpoint", type=str, default=None, help="Force-enabled NeRD checkpoint.")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel Newton environments.")
    parser.add_argument("--num-steps", type=int, default=500, help="Number of policy steps.")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Physics steps excluded from plots and metrics. Defaults to the checkpoint history length minus one.",
    )
    parser.add_argument(
        "--active-force-threshold",
        type=float,
        default=1.0,
        help="Ground-truth force magnitude above which a body is active [N].",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot-env-id", type=int, default=0, help="Environment row recorded in the force plot.")
    parser.add_argument(
        "--force-plot-path",
        type=str,
        default="./videos/contact_force_validation/contact_sensor_forces.png",
        help="Output PNG containing one ContactSensor force-magnitude curve per body.",
    )
    parser.add_argument("--video", action="store_true", default=False, help="Record the validation rollout.")
    parser.add_argument(
        "--video-path",
        type=str,
        default="./videos/contact_force_validation/contact_force_validation.mp4",
        help="Output MP4 path.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=None,
        help="Output video frame rate. Defaults to the policy update rate.",
    )
    parser.add_argument(
        "--marker-force-scale",
        type=float,
        default=0.002,
        help="Arrow length per force magnitude [m/N].",
    )
    parser.add_argument(
        "--marker-offset",
        type=float,
        default=0.015,
        help="Opposite vertical offset applied to sensor and prediction arrows [m].",
    )
    parser.add_argument(
        "--marker-threshold",
        type=float,
        default=1.0,
        help="Hide force arrows below this magnitude [N].",
    )

    from isaaclab_tasks.utils import add_launcher_args

    add_launcher_args(parser)
    return parser.parse_known_args()


def requested_newton_visualizer(args: argparse.Namespace) -> bool:
    """Return whether the launcher requested a Newton visualizer."""
    visualizers = getattr(args, "visualizer", None)
    if isinstance(visualizers, str):
        visualizers = visualizers.split(",")
    return bool(visualizers) and "newton" in {str(value).strip().lower() for value in visualizers}


def rotate_body_forces_to_world(forces_b: torch.Tensor, root_body_q: torch.Tensor) -> torch.Tensor:
    """Rotate root-body-frame forces into the world frame.

    Args:
        forces_b: Per-body linear forces [N], shape ``[B, num_bodies, 3]``.
        root_body_q: Root poses, shape ``[B, 7]``, with xyzw quaternion order.

    Returns:
        Per-body linear forces in the world frame [N].
    """
    if forces_b.ndim != 3 or forces_b.shape[-1] != 3:
        raise ValueError(f"forces_b must have shape [B, num_bodies, 3], got {tuple(forces_b.shape)}.")
    if root_body_q.shape != (forces_b.shape[0], 7):
        raise ValueError(f"root_body_q must have shape {(forces_b.shape[0], 7)}, got {tuple(root_body_q.shape)}.")

    from isaaclab_neural.utils import torch_utils

    quaternions = root_body_q[:, None, 3:7].expand(*forces_b.shape[:-1], 4)
    return torch_utils.quat_rotate(quaternions.reshape(-1, 4), forces_b.reshape(-1, 3)).view_as(forces_b)


def contact_force_metrics(
    sensor_forces_w: torch.Tensor,
    predicted_forces_w: torch.Tensor,
    *,
    active_threshold_n: float,
) -> dict[str, float]:
    """Compute vector-force RMSE metrics matching the training evaluator."""
    if sensor_forces_w.shape != predicted_forces_w.shape or sensor_forces_w.shape[-1] != 3:
        raise ValueError("Sensor and predicted force histories must have matching [..., num_bodies, 3] shapes.")
    if sensor_forces_w.numel() == 0:
        raise ValueError("Cannot compute contact-force metrics from empty histories.")
    if active_threshold_n < 0.0:
        raise ValueError("active_threshold_n must be non-negative.")
    error_norm_sq = (predicted_forces_w - sensor_forces_w).square().sum(dim=-1)
    active_mask = sensor_forces_w.norm(dim=-1) > active_threshold_n
    active_rmse = error_norm_sq[active_mask].mean().sqrt() if active_mask.any() else error_norm_sq.new_zeros(())
    return {
        "body_force_RMSE_N": float(error_norm_sq.mean().sqrt()),
        "body_force_active_RMSE_N": float(active_rmse),
        "body_force_active_fraction": float(active_mask.float().mean()),
    }


def save_contact_force_plot(
    sensor_force_magnitudes_n: torch.Tensor,
    predicted_force_magnitudes_n: torch.Tensor,
    body_names: list[str] | tuple[str, ...],
    physics_dt: float,
    output_path: str | Path,
) -> Path:
    """Plot ContactSensor and neural force magnitudes for every primary body."""
    if sensor_force_magnitudes_n.ndim != 2:
        raise ValueError(
            "sensor_force_magnitudes_n must have shape [num_physics_steps, num_bodies], "
            f"got {tuple(sensor_force_magnitudes_n.shape)}."
        )
    if predicted_force_magnitudes_n.shape != sensor_force_magnitudes_n.shape:
        raise ValueError(
            "Predicted and ContactSensor force histories must have the same shape, "
            f"got {tuple(predicted_force_magnitudes_n.shape)} and {tuple(sensor_force_magnitudes_n.shape)}."
        )
    if sensor_force_magnitudes_n.shape[0] == 0:
        raise ValueError("Cannot plot an empty contact-force history.")
    if sensor_force_magnitudes_n.shape[1] != len(body_names):
        raise ValueError(
            f"Force history has {sensor_force_magnitudes_n.shape[1]} bodies but {len(body_names)} names were provided."
        )
    if physics_dt <= 0.0:
        raise ValueError("physics_dt must be positive.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    time_s = torch.arange(sensor_force_magnitudes_n.shape[0], dtype=torch.float32) * physics_dt
    figure, axes = plt.subplots(5, 4, figsize=(18, 16), sharex=True)
    flat_axes = axes.reshape(-1)
    for body_index, body_name in enumerate(body_names):
        axis = flat_axes[body_index]
        axis.plot(
            time_s.numpy(),
            sensor_force_magnitudes_n[:, body_index].cpu().numpy(),
            linewidth=1.0,
            label="ContactSensor",
        )
        axis.plot(
            time_s.numpy(),
            predicted_force_magnitudes_n[:, body_index].cpu().numpy(),
            linewidth=1.0,
            alpha=0.8,
            label="NeRD estimate",
        )
        axis.set_title(f"{body_index:02d}: {body_name}")
        axis.set_ylabel("||F|| [N]")
        axis.grid(alpha=0.3)
        axis.legend(fontsize="small")
    for axis in flat_axes[len(body_names) :]:
        axis.set_visible(False)
    for axis in flat_axes[-4:]:
        if axis.get_visible():
            axis.set_xlabel("Time [s]")
    figure.suptitle("Anymal per-body resultant contact-force magnitudes")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"[force-plot] saved to: {path}")
    return path


def save_foot_contact_force_plots(
    sensor_force_magnitudes_n: torch.Tensor,
    predicted_force_magnitudes_n: torch.Tensor,
    body_names: list[str] | tuple[str, ...],
    physics_dt: float,
    output_path: str | Path,
) -> list[Path]:
    """Save one truth-versus-estimate force plot for each Anymal foot."""
    if sensor_force_magnitudes_n.shape != predicted_force_magnitudes_n.shape:
        raise ValueError("Predicted and ContactSensor force histories must have the same shape.")
    if sensor_force_magnitudes_n.ndim != 2 or sensor_force_magnitudes_n.shape[1] != len(body_names):
        raise ValueError("Force histories and body names have incompatible shapes.")

    foot_indices = [index for index, name in enumerate(body_names) if str(name).upper().endswith("FOOT")]
    if len(foot_indices) != 4:
        raise ValueError(f"Expected four Anymal foot bodies, found {len(foot_indices)}: {foot_indices}.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    combined_path = Path(output_path).expanduser()
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    time_s = torch.arange(sensor_force_magnitudes_n.shape[0], dtype=torch.float32) * physics_dt
    saved_paths = []
    for body_index in foot_indices:
        body_name = str(body_names[body_index])
        foot_path = combined_path.with_name(f"{combined_path.stem}_{body_index:02d}_{body_name}{combined_path.suffix}")
        figure, axis = plt.subplots(figsize=(12, 5))
        axis.plot(
            time_s.numpy(),
            sensor_force_magnitudes_n[:, body_index].cpu().numpy(),
            linewidth=1.2,
            label="ContactSensor",
        )
        axis.plot(
            time_s.numpy(),
            predicted_force_magnitudes_n[:, body_index].cpu().numpy(),
            linewidth=1.2,
            alpha=0.8,
            label="NeRD estimate",
        )
        axis.set(
            xlabel="Time [s]",
            ylabel="||F|| [N]",
            title=f"{body_index:02d}: {body_name} resultant contact-force magnitude",
        )
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(foot_path, dpi=150)
        plt.close(figure)
        saved_paths.append(foot_path)
        print(f"[force-plot] saved foot plot to: {foot_path}")
    return saved_paths


def _load_saved_agent_cfg(checkpoint_path: str, fallback_agent_cfg, *, device: str, seed: int):
    """Load the RSL-RL runner config saved beside a policy checkpoint when available."""
    agent_cfg_path = Path(checkpoint_path).expanduser().resolve().parent / "params" / "agent.yaml"
    if not agent_cfg_path.exists():
        return fallback_agent_cfg

    import yaml

    with agent_cfg_path.open("r") as file:
        agent_cfg = yaml.safe_load(file)
    if not isinstance(agent_cfg, Mapping):
        raise ValueError(f"Expected a mapping in saved RSL-RL agent config: {agent_cfg_path}")
    agent_cfg = copy.deepcopy(dict(agent_cfg))
    agent_cfg["device"] = device
    agent_cfg["seed"] = seed
    print(f"[policy] using saved RSL-RL agent config: {agent_cfg_path}")
    return agent_cfg


def _build_policy(env, agent_cfg, checkpoint_path: str, device: str):
    """Wrap the environment and load an RSL-RL inference policy."""
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

    if isinstance(agent_cfg, Mapping):
        runner_cfg = copy.deepcopy(dict(agent_cfg))
        runner_cfg["device"] = device
        class_name = runner_cfg.get("class_name", "OnPolicyRunner")
        clip_actions = runner_cfg.get("clip_actions")
        runner_device = runner_cfg.get("device", device)
    else:
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        agent_cfg.device = device
        runner_cfg = agent_cfg.to_dict()
        class_name = getattr(agent_cfg, "class_name", "OnPolicyRunner")
        clip_actions = getattr(agent_cfg, "clip_actions", None)
        runner_device = getattr(agent_cfg, "device", device)

    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)
    runner_cls = {"OnPolicyRunner": OnPolicyRunner, "DistillationRunner": DistillationRunner}.get(class_name)
    if runner_cls is None:
        raise ValueError(f"Unsupported RSL-RL runner class: {class_name}")
    runner = runner_cls(wrapped_env, runner_cfg, log_dir=None, device=runner_device)
    print(f"[policy] loading checkpoint: {checkpoint_path}")
    runner.load(checkpoint_path, map_location=device)
    policy = runner.get_inference_policy(device=wrapped_env.unwrapped.device)
    return wrapped_env, policy


class _ActionSource:
    """Produce held policy-rate actions for policy or random validation."""

    def __init__(self, env, adapter, agent_cfg, args: argparse.Namespace):
        self.mode = args.action_mode
        self.rollout_env = env
        self.policy = None
        self.observations = None
        self.random_action_low = None
        self.random_action_high = None
        if self.mode == "policy":
            policy_agent_cfg = _load_saved_agent_cfg(
                args.policy_checkpoint,
                agent_cfg,
                device=args.device,
                seed=args.seed,
            )
            if policy_agent_cfg is None:
                raise ValueError(f"No RSL-RL agent config found for entry point: {args.policy_agent}")
            self.rollout_env, self.policy = _build_policy(env, policy_agent_cfg, args.policy_checkpoint, args.device)
            self.observations = self.rollout_env.get_observations()
        else:
            self.random_action_low, self.random_action_high = adapter.action_bounds()
            print("[actions] using uniformly sampled random actions")

    def sample(self, adapter) -> torch.Tensor:
        """Return the next policy-rate action batch."""
        if self.mode == "random":
            return adapter.sample_action(self.random_action_low, self.random_action_high)
        actions = self.policy(self.observations)
        if self.rollout_env.clip_actions is not None:
            actions = torch.clamp(actions, -self.rollout_env.clip_actions, self.rollout_env.clip_actions)
        return actions

    def update_observations(self) -> None:
        """Refresh policy observations after one environment-rate step."""
        if self.mode == "policy":
            self.observations = self.rollout_env.get_observations()

    def close(self) -> None:
        """Close the underlying rollout environment."""
        self.rollout_env.close()


def _build_observer_solver_cfg(checkpoint_path: str):
    """Build the shadow NeRD solver config embedded in a checkpoint."""
    from isaaclab_neural.physics import NerdSolverCfg
    from isaaclab_neural.utils.checkpoint import get_cfg_from_checkpoint, load_checkpoint

    checkpoint = load_checkpoint(checkpoint_path, device="cpu")
    cfg = get_cfg_from_checkpoint(checkpoint, checkpoint_path)
    solver_values = dict(cfg["env"]["neural_solver_cfg"])
    solver_values.pop("use_cuda_graph", None)
    solver_values["neural_model_path"] = None
    return checkpoint, NerdSolverCfg(**solver_values)


class _ForceArrowMarkers:
    """Display sensor and predicted per-body force vectors with separate colors."""

    def __init__(self, device: torch.device | str, *, force_scale: float, threshold_n: float, offset_m: float):
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG

        self.sensor = VisualizationMarkers(
            RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/ContactForceValidation/sensor")
        )
        self.prediction = VisualizationMarkers(
            BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/ContactForceValidation/prediction")
        )
        self.device = torch.device(device)
        self.force_scale = float(force_scale)
        self.threshold_n = float(threshold_n)
        self.offset_m = float(offset_m)

    def update(
        self, positions_w: torch.Tensor, sensor_forces_w: torch.Tensor, predicted_forces_w: torch.Tensor
    ) -> None:
        """Update red ContactSensor and blue neural force arrows."""
        sensor_origins = positions_w.clone()
        predicted_origins = positions_w.clone()
        sensor_origins[:, 2] += self.offset_m
        predicted_origins[:, 2] -= self.offset_m
        sensor_quat, sensor_scale = self._arrow_transforms(sensor_origins, sensor_forces_w)
        predicted_quat, predicted_scale = self._arrow_transforms(predicted_origins, predicted_forces_w)
        self.sensor.visualize(sensor_origins, sensor_quat, sensor_scale)
        self.prediction.visualize(predicted_origins, predicted_quat, predicted_scale)

    def _arrow_transforms(self, origins_w: torch.Tensor, forces_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from isaaclab.utils import math as math_utils

        magnitudes = torch.linalg.norm(forces_w, dim=-1)
        active = magnitudes > self.threshold_n
        orientations = torch.zeros((forces_w.shape[0], 4), device=self.device)
        orientations[:, 0] = 1.0
        scales = torch.zeros_like(forces_w)
        if active.any():
            active_origins = origins_w[active]
            rotations = math_utils.create_rotation_matrix_from_view(
                active_origins,
                active_origins + forces_w[active],
                up_axis="Z",
                device=str(self.device),
            )
            quat_opengl = math_utils.quat_from_matrix(rotations)
            orientations[active] = math_utils.convert_camera_frame_orientation_convention(
                quat_opengl, "opengl", "world"
            )
            scales[active, 0] = magnitudes[active] * self.force_scale
            scales[active, 1:] = 0.1
        return orientations, scales


def run(env_cfg, agent_cfg, args: argparse.Namespace) -> None:
    """Run policy-controlled ground-truth Newton and shadow NeRD force inference."""
    if args.nerd_checkpoint is None:
        raise ValueError("--nerd-checkpoint is required.")
    if args.action_mode == "policy" and args.policy_checkpoint is None:
        raise ValueError("--policy-checkpoint is required when --action-mode=policy.")
    if args.num_envs <= 0 or args.num_steps <= 0:
        raise ValueError("--num-envs and --num-steps must be positive.")
    if not 0 <= args.plot_env_id < args.num_envs:
        raise ValueError(f"--plot-env-id must be in [0, {args.num_envs}), got {args.plot_env_id}.")
    if args.marker_force_scale <= 0.0 or args.marker_threshold < 0.0 or args.marker_offset < 0.0:
        raise ValueError("Marker scale must be positive; threshold and offset must be non-negative.")
    if args.video_fps is not None and args.video_fps <= 0:
        raise ValueError("--video-fps must be positive.")
    if args.warmup_steps is not None and args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative.")
    if args.active_force_threshold < 0.0:
        raise ValueError("--active-force-threshold must be non-negative.")

    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed
    terrain_cfg = getattr(getattr(env_cfg.scene, "terrain", None), "terrain_generator", None)
    if terrain_cfg is not None:
        terrain_cfg.seed = args.seed

    checkpoint, solver_cfg = _build_observer_solver_cfg(args.nerd_checkpoint)
    if solver_cfg.contact_mode != "newton_native":
        raise ValueError("Contact-force validation requires a newton_native NeRD checkpoint.")
    if getattr(solver_cfg, "contact_representation", "flat") != "contact_tokens":
        raise ValueError("Contact-force validation requires a contact_tokens NeRD checkpoint.")

    from isaaclab_newton.physics import MJWarpSolverCfg
    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

    if not isinstance(env_cfg.sim.physics, NewtonCfg) or not isinstance(
        env_cfg.sim.physics.solver_cfg, MJWarpSolverCfg
    ):
        raise ValueError("Contact-force validation requires the presets=newton_mjwarp physics preset.")
    env_cfg.sim.physics.solver_cfg.use_mujoco_contacts = False

    from isaaclab_neural.utils.usd_utils import newton_material_binding_api_autofix

    from isaaclab_tasks.utils import launch_simulation

    with launch_simulation(env_cfg, args):
        import gymnasium as gym
        import warp as wp
        from isaaclab_neural.generate.adapter import DataGenerationAdapter
        from isaaclab_neural.utils.checkpoint import reconstruct_model_from_checkpoint

        with newton_material_binding_api_autofix():
            gym_kwargs = {"cfg": env_cfg, "device": args.device}
            if args.video:
                gym_kwargs["render_mode"] = "rgb_array"
            env = gym.make(args.task, **gym_kwargs).unwrapped
            env.reset()

        adapter = DataGenerationAdapter(env, solver_cfg, record_body_net_contact_forces=True)
        neural_model = reconstruct_model_from_checkpoint(checkpoint, adapter.solver, device=args.device)
        if not getattr(neural_model, "predict_body_net_contact_forces", False):
            raise ValueError("The NeRD checkpoint does not contain an enabled body contact-force head.")
        adapter.solver.set_neural_solver_model(neural_model)
        adapter.solver.eval()
        adapter.solver.reset()

        action_source = _ActionSource(env, adapter, agent_cfg, args)

        body_names = adapter.body_net_contact_force_body_names
        if len(body_names) != 17:
            raise RuntimeError(f"Expected 17 Anymal primary bodies, got {len(body_names)}: {body_names}")
        primary_body_ids = adapter.contact_adapter.primary_body_ids_by_world[args.plot_env_id]
        markers = (
            _ForceArrowMarkers(
                args.device,
                force_scale=args.marker_force_scale,
                threshold_n=args.marker_threshold,
                offset_m=args.marker_offset,
            )
            if requested_newton_visualizer(args)
            else None
        )
        if markers is not None:
            print("[markers] ContactSensor=red, neural prediction=blue")

        decimation = max(1, int(env.cfg.decimation))
        physics_dt = float(env.physics_dt)
        warmup_steps = (
            max(0, int(adapter.solver.num_states_history) - 1) if args.warmup_steps is None else args.warmup_steps
        )
        print(f"[metrics] excluding first {warmup_steps} physics step(s) as transformer warmup")
        sensor_history: list[torch.Tensor] = []
        predicted_history: list[torch.Tensor] = []
        physics_step = 0
        video_writer = None
        if args.video:
            import imageio.v2 as imageio

            video_path = Path(args.video_path).expanduser()
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_fps = args.video_fps or max(1, round(1.0 / (physics_dt * decimation)))
            video_writer = imageio.get_writer(video_path, fps=video_fps, codec="libx264")
            print(f"[video] recording {video_fps} FPS to: {video_path}")
        try:
            with torch.inference_mode():
                for policy_step in range(args.num_steps):
                    actions = action_source.sample(adapter)
                    env.action_manager.process_action(actions)
                    for _ in range(decimation):
                        env.action_manager.apply_action()
                        env.scene.write_data_to_sim()

                        contacts = adapter._prepare_contacts()
                        adapter.solver.sync_from_newton(
                            adapter.backend.state,
                            contacts,
                            adapter.backend.control.joint_f,
                            update_history=True,
                        )
                        model_inputs = adapter.solver.get_neural_model_inputs()
                        adapter.solver._forward_neural_model(model_inputs)
                        predicted_forces_b = adapter.solver.predicted_body_net_contact_forces_b
                        if predicted_forces_b is None:
                            raise RuntimeError("NeRD solver did not expose predicted body contact forces.")
                        predicted_forces_w = rotate_body_forces_to_world(
                            predicted_forces_b,
                            adapter.solver.root_body_q,
                        )

                        with adapter.backend.single_physics_step():
                            env.sim.step(render=False)
                        env._sim_step_counter += 1
                        env.scene.update(dt=physics_dt)

                        sensor_forces_w = adapter.body_net_contact_forces_w()
                        if physics_step >= warmup_steps:
                            sensor_history.append(sensor_forces_w[args.plot_env_id].detach().cpu())
                            predicted_history.append(predicted_forces_w[args.plot_env_id].detach().cpu())
                        if markers is not None:
                            body_poses_w = wp.to_torch(adapter.backend.state.body_q).index_select(0, primary_body_ids)
                            markers.update(
                                body_poses_w[:, :3],
                                sensor_forces_w[args.plot_env_id],
                                predicted_forces_w[args.plot_env_id],
                            )
                            env.render()
                        physics_step += 1

                    if video_writer is not None:
                        frame = env.render(recompute=True)
                        if frame is None:
                            raise RuntimeError("Video recording requested, but env.render() returned no RGB frame.")
                        video_writer.append_data(frame[..., :3])
                    action_source.update_observations()
                    if policy_step % 25 == 0 or policy_step == args.num_steps - 1:
                        sensor_magnitudes = sensor_forces_w[args.plot_env_id].norm(dim=-1)
                        print(
                            f"[step {policy_step:05d}] "
                            f"sensor_mean={sensor_magnitudes.mean().item():.3f} N, "
                            f"sensor_max={sensor_magnitudes.max().item():.3f} N"
                        )
        finally:
            if video_writer is not None:
                video_writer.close()
                print(f"[video] saved to: {Path(args.video_path).expanduser()}")
            if sensor_history:
                sensor_force_vectors = torch.stack(sensor_history)
                predicted_force_vectors = torch.stack(predicted_history)
                metrics = contact_force_metrics(
                    sensor_force_vectors,
                    predicted_force_vectors,
                    active_threshold_n=args.active_force_threshold,
                )
                print("[metrics] " + ", ".join(f"{name}={value:.6f}" for name, value in metrics.items()))
                sensor_history_tensor = sensor_force_vectors.norm(dim=-1)
                predicted_history_tensor = predicted_force_vectors.norm(dim=-1)
                save_contact_force_plot(
                    sensor_history_tensor,
                    predicted_history_tensor,
                    body_names,
                    physics_dt,
                    args.force_plot_path,
                )
                save_foot_contact_force_plots(
                    sensor_history_tensor,
                    predicted_history_tensor,
                    body_names,
                    physics_dt,
                    args.force_plot_path,
                )
            action_source.close()


def main() -> None:
    """Parse Hydra configuration and run validation."""
    args, hydra_args = parse_args()
    if not any(value.startswith("presets=") for value in hydra_args):
        hydra_args = ["presets=newton_mjwarp", *hydra_args]
    sys.argv = [sys.argv[0], *hydra_args]

    import isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c  # noqa: F401
    from isaaclab_tasks.utils.hydra import hydra_task_config

    agent_entry_point = args.policy_agent if args.action_mode == "policy" else ""

    @hydra_task_config(args.task, agent_entry_point)
    def hydra_main(env_cfg, agent_cfg) -> None:
        run(env_cfg, agent_cfg, args)

    hydra_main()


if __name__ == "__main__":
    main()
