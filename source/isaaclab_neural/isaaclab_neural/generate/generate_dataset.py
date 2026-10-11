# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate NeRD HDF5 datasets from registered IsaacLab tasks."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_neural.envs  # noqa: F401 - registers NeRD dataset-generation tasks
from isaaclab_neural.generate.arguments import get_parser
from isaaclab_neural.utils.python_utils import set_random_seed


def _default_env_name(task: str) -> str:
    """Derive a compact dataset metadata name from a gym task id."""
    name = task
    if name.startswith("Isaac-"):
        name = name[len("Isaac-") :]
    if name.endswith("-v0"):
        name = name[: -len("-v0")]
    return name


def _default_robot_name(name: str) -> str:
    """Infer the default sampling-range key from a task or dataset name."""
    if "Anymal-C" in name:
        return "Anymal-C"
    if "Cartpole" in name:
        return "Cartpole"
    if "G1" in name:
        return "G1"
    return name


def _sampler_spec(
    task: str,
    env_name: str,
    robot_name: str,
    randomize_pd_gains: bool,
    kp_range: tuple[float, float],
    kd_range: tuple[float, float],
    kp_scale_range: tuple[float, float] = (0.8, 1.2),
    kd_scale_range: tuple[float, float] = (0.8, 1.2),
    hold_hand_actions: bool = True,
    bfm_latent_kwargs: dict | None = None,
):
    """Return sampler class and kwargs for a dataset-generation task.

    ``bfm_latent_kwargs`` selects :class:`~isaaclab_neural.generate.samplers.g1_29dof_bfm.BfmLatentTrajectorySampler`
    with those keyword arguments, independent of the robot name.
    """
    from isaaclab_neural.generate.sampler import ActionTrajectorySampler
    from isaaclab_neural.generate.samplers import AnymalCTrajectorySampler, CartpoleTrajectorySampler
    from isaaclab_neural.generate.samplers.g1 import G1TrajectorySampler

    if bfm_latent_kwargs is not None:
        from isaaclab_neural.generate.samplers.g1_29dof_bfm import BfmLatentTrajectorySampler

        return BfmLatentTrajectorySampler, bfm_latent_kwargs

    if robot_name == "Cartpole":
        return CartpoleTrajectorySampler, {}

    if robot_name == "G1":
        return G1TrajectorySampler, {
            "randomize_pd_gains": randomize_pd_gains,
            "kp_scale_range": kp_scale_range,
            "kd_scale_range": kd_scale_range,
            "hold_hand_actions": hold_hand_actions,
        }

    if robot_name == "Anymal-C":
        dataset_style_task = "Dataset" in task or "Dataset" in env_name
        use_pd_randomization = randomize_pd_gains or dataset_style_task
        sampler_cls = AnymalCTrajectorySampler if use_pd_randomization else ActionTrajectorySampler
        sampler_kwargs = {}
        if sampler_cls is AnymalCTrajectorySampler:
            sampler_kwargs = {
                "randomize_pd_gains": use_pd_randomization,
                "kp_range": kp_range,
                "kd_range": kd_range,
            }
        return sampler_cls, sampler_kwargs

    return ActionTrajectorySampler, {}


def bfm_latent_sampler_kwargs(args) -> dict | None:
    """Translate ``--bfm-*`` CLI options into BFM latent sampler keyword arguments.

    Args:
        args: Parsed dataset-generation arguments.

    Returns:
        Sampler keyword arguments, or None unless ``--sample-mode bfm_latent``.

    Raises:
        ValueError: The latent library is missing or a source weight is malformed.
    """
    if args.sample_mode != "bfm_latent":
        return None
    if not args.bfm_latent_library:
        raise ValueError("--bfm-latent-library is required when --sample-mode bfm_latent.")
    weights = {}
    for item in args.bfm_source_weights.split(","):
        name, _, value = item.partition("=")
        if not value:
            raise ValueError(f"Malformed --bfm-source-weights entry: '{item}'.")
        weights[name.strip()] = float(value)
    return {
        "latent_library": args.bfm_latent_library,
        "source_weights": weights,
        "hold_range_s": (args.bfm_hold_min_s, args.bfm_hold_max_s),
        "push_force_range": (args.bfm_push_force_min, args.bfm_push_force_max),
        "push_wait_range_s": (args.bfm_push_wait_min_s, args.bfm_push_wait_max_s),
        "push_duration_s": args.bfm_push_duration_s,
        "clips": "holdout" if args.bfm_holdout_only else "all" if args.bfm_include_holdout else "train",
    }


def dataset_path(dataset_dir: str, env_name: str, dataset_name: str) -> Path:
    """Return output dataset path."""
    return Path(dataset_dir).expanduser() / env_name / dataset_name


def configure_env(env_cfg, args) -> None:
    """Apply data-generation CLI overrides to an IsaacLab env config."""
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"):
        env_cfg.scene.num_envs = args.num_envs
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = args.seed
    terrain_cfg = getattr(getattr(getattr(env_cfg, "scene", None), "terrain", None), "terrain_generator", None)
    if terrain_cfg is not None:
        terrain_cfg.seed = args.seed
    if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "device"):
        env_cfg.sim.device = args.device

    from isaaclab_newton.physics import MJWarpSolverCfg
    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

    from isaaclab_neural.physics.nerd_solver_cfg import normalize_contact_mode

    if not isinstance(env_cfg.sim.physics, NewtonCfg):
        raise ValueError("Dataset generation requires a Newton physics preset.")
    args.contact_mode = normalize_contact_mode(args.contact_mode)
    if args.contact_mode == "newton_native":
        physics_solver_cfg = env_cfg.sim.physics.solver_cfg
        if not isinstance(physics_solver_cfg, MJWarpSolverCfg):
            raise ValueError("Newton-native dataset generation requires the MJWarp solver.")
        physics_solver_cfg.use_mujoco_contacts = False


def build_solver_cfg(args):
    """Build a NeRD solver config used only for dataset recording."""
    from isaaclab_neural.contacts import resolve_contact_packing_policy
    from isaaclab_neural.physics import NerdSolverCfg

    packing_policy = resolve_contact_packing_policy(args.contact_mode, args.contact_packing_policy)

    return NerdSolverCfg(
        name="NeuralSolver",
        states_frame=args.states_frame,
        anchor_frame_step=args.anchor_frame_step,
        states_embedding_type=args.states_embedding_type,
        prediction_type=args.prediction_type,
        orientation_prediction_parameterization=args.orientation_prediction_parameterization,
        min_contact_event_threshold=args.min_contact_event_threshold,
        num_contacts_per_env=args.num_contacts_per_env,
        contact_mode=args.contact_mode,
        contact_packing_policy=packing_policy,
        contact_representation=args.contact_representation,
        max_contact_tokens=args.max_contact_tokens,
    )


args_cli, hydra_args = get_parser().parse_known_args()
sys.argv = [sys.argv[0], "presets=newton_mjwarp"] + hydra_args


@hydra_task_config(args_cli.task, args_cli.policy_agent if args_cli.sample_mode == "policy" else "")
def main(env_cfg, agent_cfg=None) -> None:
    """Generate and write one HDF5 dataset."""
    import gymnasium as gym
    import torch

    from isaaclab.app import launch_simulation

    from isaaclab_neural.data import build_terrain_context, write_rollouts_to_hdf5
    from isaaclab_neural.generate.adapter import DataGenerationAdapter
    from isaaclab_neural.utils.commons import JOINT_F_LIM, JOINT_Q_MAX, JOINT_Q_MIN, JOINT_QD_LIM
    from isaaclab_neural.utils.usd_utils import newton_material_binding_api_autofix

    env_name = args_cli.env_name or _default_env_name(args_cli.task)
    robot_name = args_cli.robot_name or _default_robot_name(env_name)
    output_path = dataset_path(args_cli.dataset_dir, env_name, args_cli.dataset_name)
    if output_path.exists() and not args_cli.force_overwrite:
        raise FileExistsError(f"Dataset already exists: {output_path}. Pass --force-overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    configure_env(env_cfg, args_cli)
    solver_cfg = build_solver_cfg(args_cli)
    # Ragged native contacts are serialized on CPU. Staging them there avoids
    # retaining every frame's variable pair tensors on CUDA until a write chunk
    # completes.
    has_contacts = int(getattr(args_cli, "num_contacts_per_env", 0) or 0) > 0
    if args_cli.data_device is not None:
        data_device = args_cli.data_device
    elif has_contacts:
        data_device = "cpu"
    else:
        data_device = args_cli.device
    if str(data_device).startswith("cpu") and has_contacts:
        print("[dataset] staging ragged native contacts on CPU to avoid GPU OOM during chunk merge.")

    with launch_simulation(env_cfg, args_cli):
        with newton_material_binding_api_autofix():
            env = gym.make(args_cli.task, cfg=env_cfg, device=args_cli.device).unwrapped
            env.reset()
        terrain_context = build_terrain_context(env_cfg, args_cli.seed)
        adapter = DataGenerationAdapter(
            env,
            solver_cfg,
            record_body_net_contact_forces=args_cli.record_body_net_contact_forces,
        )
        body_net_contact_force_body_names = adapter.body_net_contact_force_body_names or None
        sampler_cls, sampler_kwargs = _sampler_spec(
            args_cli.task,
            env_name,
            robot_name,
            args_cli.randomize_pd_gains,
            (args_cli.kp_min, args_cli.kp_max),
            (args_cli.kd_min, args_cli.kd_max),
            (args_cli.kp_scale_min, args_cli.kp_scale_max),
            (args_cli.kd_scale_min, args_cli.kd_scale_max),
            hold_hand_actions=not bool(getattr(args_cli, "no_hold_g1_hand_actions", False)),
            bfm_latent_kwargs=bfm_latent_sampler_kwargs(args_cli),
        )
        sampler = sampler_cls(
            adapter,
            trajectory_length=args_cli.trajectory_length,
            data_device=data_device,
            action_low=args_cli.action_low,
            action_high=args_cli.action_high,
            joint_q_min=JOINT_Q_MIN.get(robot_name),
            joint_q_max=JOINT_Q_MAX.get(robot_name),
            joint_qd_lim=JOINT_QD_LIM.get(robot_name),
            joint_f_lim=JOINT_F_LIM.get(robot_name),
            **sampler_kwargs,
        )
        initial_states_pool = None
        if args_cli.initial_states_source == "input_states_pool" and not args_cli.initial_states_pool:
            raise ValueError("--initial-states-pool is required when --initial-states-source=input_states_pool.")
        if args_cli.initial_states_pool:
            from isaaclab_neural.generate.initial_states import load_initial_states_pool

            initial_states_pool = load_initial_states_pool(args_cli.initial_states_pool, args_cli.device)
        policy = None
        env_wrapped = None
        runner = None
        if args_cli.policy_action_noise_std < 0.0:
            raise ValueError("--policy-action-noise-std must be non-negative.")
        if args_cli.policy_action_noise_std > 0.0 and args_cli.sample_mode != "policy":
            raise ValueError("--policy-action-noise-std requires --sample-mode policy.")
        if args_cli.sample_mode == "policy":
            if args_cli.policy_checkpoint is None:
                raise ValueError("--policy-checkpoint is required when --sample-mode policy.")
            from isaaclab_neural.eval.eval import build_policy_env, load_saved_agent_cfg_for_policy

            policy_agent_cfg = load_saved_agent_cfg_for_policy(
                args_cli.policy_checkpoint,
                agent_cfg,
                device=args_cli.device,
                seed=args_cli.seed,
            )
            env_wrapped, policy, runner = build_policy_env(
                env, policy_agent_cfg, args_cli.policy_checkpoint, args_cli.device
            )

        def sample_rollouts(num_transitions: int):
            if args_cli.sample_mode == "bfm_latent":
                return sampler.sample_trajectories_latent_mode(
                    num_transitions,
                    initial_states_source=args_cli.initial_states_source,
                    render=args_cli.render,
                )
            if args_cli.sample_mode == "joint_f":
                return sampler.sample_trajectories_joint_f_mode(
                    num_transitions,
                    passive=args_cli.zero_actions,
                    initial_states_source=args_cli.initial_states_source,
                    initial_states_pool=initial_states_pool,
                    render=args_cli.render,
                )
            if args_cli.sample_mode == "policy":
                return sampler.sample_trajectories_policy_mode(
                    num_transitions,
                    policy=policy,
                    env_wrapped=env_wrapped,
                    initial_states_source=args_cli.initial_states_source,
                    render=args_cli.render,
                    action_noise_std=args_cli.policy_action_noise_std,
                    clip_action_noise=not getattr(args_cli, "no_clip_policy_action_noise", False),
                )
            return sampler.sample_trajectories_action_mode(
                num_transitions,
                zero_actions=args_cli.zero_actions,
                initial_states_source=args_cli.initial_states_source,
                initial_states_pool=initial_states_pool,
                step_granularity=args_cli.step_granularity,
                resample_actions_every_frame=bool(args_cli.resample_actions_every_frame),
                action_hold_frames=args_cli.action_hold_frames,
                render=args_cli.render,
            )

        if args_cli.write_chunk_transitions > 0:
            from isaaclab_neural.data import append_rollouts_to_hdf5

            if output_path.exists():
                output_path.unlink()
            total_transitions = 0
            while total_transitions < args_cli.num_transitions:
                chunk_transitions = min(args_cli.write_chunk_transitions, args_cli.num_transitions - total_transitions)
                rollouts = sample_rollouts(chunk_transitions)
                append_rollouts_to_hdf5(
                    output_path,
                    rollouts,
                    env_name,
                    terrain_context=terrain_context,
                    body_net_contact_force_body_names=body_net_contact_force_body_names,
                )
                written_transitions = rollouts["states"].shape[0] * rollouts["states"].shape[1]
                total_transitions += written_transitions
                print(
                    f"[dataset] appended {written_transitions} transitions ({total_transitions} total) to {output_path}"
                )
                del rollouts
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            rollouts = sample_rollouts(args_cli.num_transitions)
            write_rollouts_to_hdf5(
                output_path,
                rollouts,
                env_name,
                terrain_context=terrain_context,
                body_net_contact_force_body_names=body_net_contact_force_body_names,
            )
        truncation_summary = adapter.contact_truncation_summary()
        if truncation_summary is not None:
            print(f"[contacts] truncation summary: {truncation_summary}")
        del runner
        print(f"[dataset] wrote {output_path}")
        env.close()


if __name__ == "__main__":
    set_random_seed(args_cli.seed)
    main()  # type: ignore[call-arg]
