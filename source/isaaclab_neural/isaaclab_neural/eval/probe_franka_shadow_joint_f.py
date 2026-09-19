# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Probe implicit Panda shadow PD as a NeRD ``joint_f`` stand-in.

Runs analytic Newton ``Isaac-Reach-Franka-Dataset-Gen-v0`` at physics-frame
rate with the Reach RL joint-position action. It does not load a NeRD
checkpoint.

Checks:

1. ``applied_effort`` matches ``clip(kp (q* − q) + kd (qd* − qd))`` from the
   pre-step state.
2. Newton ``control.joint_f`` stays ~0 *during* the analytic solve (implicit
   feed-forward). Dataset-Gen overlays ``applied_effort`` only after the step.
3. Reconstructing PD from the post-step state is worse (timing).
4. ``M qdd + g(q)`` vs shadow PD on arm joints (Coriolis omitted).
5. Copying shadow PD into ``control.joint_f`` before the solver (double PD).

Example::

    PYTHONPATH="$(pwd)/source/isaaclab_neural${PYTHONPATH:+:$PYTHONPATH}" \\
      uv run python -m isaaclab_neural.eval.probe_franka_shadow_joint_f \\
      --policy-checkpoint ./pretrained/control_policy/franka_reach_menagerie/model_999.pt \\
      --num-envs 8 --num-steps 240 --visualizer none presets=newton_mjwarp
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import isaaclab_neural.envs  # noqa: F401
import torch
from isaaclab_neural.eval.shadow_joint_f import (
    evaluate_shadow_joint_f_probe,
    inverse_dynamics_inertial_gravity,
    reconstruct_shadow_pd,
)
from isaaclab_neural.generate.sampler import policy_control_interval_dt, should_refresh_policy_mdp

from isaaclab_tasks.utils.hydra import hydra_task_config


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse probe and Isaac Lab launcher arguments.

    ``--policy-checkpoint`` is optional at parse time because Hydra re-enters this
    module after stripping argparse flags from ``sys.argv``. The first parse
    stores the checkpoint in ``ISAACLAB_PROBE_POLICY_CHECKPOINT``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, default="Isaac-Reach-Franka-Dataset-Gen-v0")
    parser.add_argument(
        "--policy-checkpoint",
        type=str,
        default=os.environ.get("ISAACLAB_PROBE_POLICY_CHECKPOINT") or None,
        help="Reach RSL-RL checkpoint. Required on the first process entry.",
    )
    parser.add_argument("--policy-agent", type=str, default="rsl_rl_cfg_entry_point")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=240, help="Physics frames per rollout phase.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-inject", action="store_true", help="Skip the double-PD writeback rollout.")
    parser.add_argument("--metrics-json", type=str, default=None)
    from isaaclab.app import add_launcher_args

    add_launcher_args(parser)
    return parser.parse_known_args()


def _prepare_hydra_argv(hydra_args: list[str]) -> list[str]:
    """Keep a Newton MJWarp preset override for Hydra without duplicating it."""
    if not any(arg.startswith("presets=") for arg in hydra_args):
        return ["presets=newton_mjwarp", *hydra_args]
    return hydra_args


def configure_env(env_cfg, args: argparse.Namespace) -> None:
    """Apply CLI overrides, restore the Reach RL action, and disable CUDA graphs."""
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"):
        env_cfg.scene.num_envs = args.num_envs
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = args.seed
    if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "device"):
        if getattr(args, "device", None) is None:
            args.device = env_cfg.sim.device
        else:
            env_cfg.sim.device = args.device
    restore = getattr(env_cfg, "restore_reach_rl_arm_action", None)
    if not callable(restore):
        raise RuntimeError(f"{args.task} does not expose restore_reach_rl_arm_action().")
    restore()
    # Python post-actuator inject and per-frame tensor reads must not be graph-captured.
    env_cfg.sim.physics.use_cuda_graph = False
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "ee_pose"):
        env_cfg.commands.ee_pose.debug_vis = False


def _to_torch(value) -> torch.Tensor:
    tensor = getattr(value, "torch", value)
    return torch.as_tensor(tensor).detach().float().clone()


def _control_joint_f(control, num_envs: int) -> torch.Tensor:
    import warp as wp

    if control is None or getattr(control, "joint_f", None) is None:
        return torch.zeros((num_envs, 0), dtype=torch.float32)
    return wp.to_torch(control.joint_f).float().view(num_envs, -1).clone()


def _arm_joint_mask(joint_names: list[str], device: torch.device) -> torch.Tensor:
    return torch.tensor([name.startswith("panda_joint") for name in joint_names], device=device, dtype=torch.bool)


def refresh_policy_mdp(env, dt: float) -> None:
    """Advance command and observation managers without stepping or resetting."""
    command_manager = getattr(env, "command_manager", None)
    observation_manager = getattr(env, "observation_manager", None)
    if command_manager is not None:
        command_manager.compute(dt=dt)
    if observation_manager is not None:
        env.obs_buf = observation_manager.compute(update_history=True)


def step_physics_frame(env, action: torch.Tensor, *, process_action: bool) -> None:
    """Apply the held joint-position action and advance one Newton physics frame."""
    from isaaclab_newton.physics import NewtonManager

    if process_action:
        env.action_manager.process_action(action)
    env.action_manager.apply_action()
    env.scene.write_data_to_sim()
    old_decimation = NewtonManager._decimation
    NewtonManager._decimation = 1
    try:
        env.sim.step(render=False)
    finally:
        NewtonManager._decimation = old_decimation
    if hasattr(env, "_sim_step_counter"):
        env._sim_step_counter += 1
    env.scene.update(dt=float(getattr(env, "physics_dt", 0.0)))


def capture_pre_step(robot, control, num_envs: int) -> dict[str, torch.Tensor]:
    """Snapshot PD inputs, mass matrix, and gravity at the current configuration."""
    return {
        "joint_pos": _to_torch(robot.data.joint_pos),
        "joint_vel": _to_torch(robot.data.joint_vel),
        "joint_pos_target": _to_torch(robot.data.joint_pos_target),
        "joint_vel_target": _to_torch(robot.data.joint_vel_target),
        "stiffness": _to_torch(robot.data.joint_stiffness),
        "damping": _to_torch(robot.data.joint_damping),
        "effort_limit": _to_torch(robot.data.joint_effort_limits),
        "mass_matrix": _to_torch(robot.data.mass_matrix),
        "gravity": _to_torch(robot.data.gravity_compensation_forces),
        "joint_f": _control_joint_f(control, num_envs),
    }


def _stack(records: list[dict[str, torch.Tensor]], key: str) -> torch.Tensor:
    return torch.stack([row[key] for row in records], dim=0)


def run_rollout(
    *,
    env,
    env_wrapped,
    policy,
    runner,
    control,
    num_steps: int,
    inject: bool,
) -> dict[str, torch.Tensor | float]:
    """Roll out the Reach policy for ``num_steps`` physics frames."""
    from isaaclab_neural.eval.eval import reset_policy

    robot = env.scene["robot"]
    num_envs = int(env.num_envs)
    decimation = max(1, int(getattr(env.cfg, "decimation", 1)))
    physics_dt = float(getattr(env, "physics_dt", env.cfg.sim.dt))
    control_dt = policy_control_interval_dt(env, decimation=decimation)
    env.reset()
    reset_policy(policy, runner, torch.ones(num_envs, dtype=torch.bool, device=env.device))
    obs = env_wrapped.get_observations()
    with torch.inference_mode():
        action = policy(obs)
    rows: list[dict[str, torch.Tensor]] = []

    inject_state = {"enabled": inject}

    def _inject_shadow_pd() -> None:
        if not inject_state["enabled"]:
            return
        import warp as wp

        applied = _to_torch(robot.actuators.applied_effort).reshape(-1).contiguous()
        joint_f_n = int(control.joint_f.size)
        if applied.numel() != joint_f_n:
            raise RuntimeError(
                f"applied_effort size {applied.numel()} does not match control.joint_f size {joint_f_n}."
            )
        control.joint_f.assign(wp.from_torch(applied))

    from isaaclab_newton.physics import NewtonManager

    if inject:
        NewtonManager.register_post_actuator_callback(_inject_shadow_pd)

    try:
        for step_id in range(num_steps):
            process_action = step_id == 0 or step_id % decimation == 0
            if process_action:
                if should_refresh_policy_mdp(step_id, decimation=decimation):
                    refresh_policy_mdp(env, control_dt)
                    obs = env_wrapped.get_observations()
                with torch.inference_mode():
                    action = policy(obs)
            pre = capture_pre_step(robot, control, num_envs)
            step_physics_frame(env, action, process_action=process_action)
            joint_pos_post = _to_torch(robot.data.joint_pos)
            joint_vel_post = _to_torch(robot.data.joint_vel)
            joint_pos_target = _to_torch(robot.data.joint_pos_target)
            joint_vel_target = _to_torch(robot.data.joint_vel_target)
            applied = _to_torch(robot.actuators.applied_effort)
            computed = pre["stiffness"] * (joint_pos_target - pre["joint_pos"]) + pre["damping"] * (
                joint_vel_target - pre["joint_vel"]
            )
            reconstructed = reconstruct_shadow_pd(
                pre["joint_pos"],
                pre["joint_vel"],
                joint_pos_target,
                joint_vel_target,
                pre["stiffness"],
                pre["damping"],
                pre["effort_limit"],
            )
            reconstructed_post = reconstruct_shadow_pd(
                joint_pos_post,
                joint_vel_post,
                joint_pos_target,
                joint_vel_target,
                pre["stiffness"],
                pre["damping"],
                pre["effort_limit"],
            )
            joint_acc = (joint_vel_post - pre["joint_vel"]) / physics_dt
            mass = pre["mass_matrix"]
            gravity = pre["gravity"]
            if mass.shape[-1] != reconstructed.shape[-1]:
                mass = mass[:, -reconstructed.shape[-1] :, -reconstructed.shape[-1] :]
                gravity = gravity[:, -reconstructed.shape[-1] :]
            id_force = inverse_dynamics_inertial_gravity(mass, joint_acc, gravity)
            rows.append(
                {
                    "applied": applied,
                    "reconstructed": reconstructed,
                    "reconstructed_post": reconstructed_post,
                    "recorded_joint_f": _control_joint_f(control, num_envs)[:, : applied.shape[-1]],
                    "id_force": id_force,
                    "computed_unclipped": computed,
                    "abs_qd": joint_vel_post.abs().mean(dim=-1),
                    "state_finite": torch.isfinite(joint_pos_post).all(dim=-1)
                    & torch.isfinite(joint_vel_post).all(dim=-1),
                }
            )
            if step_id == 0 or (step_id + 1) % 60 == 0:
                identity = float((applied - reconstructed).abs().max())
                print(
                    f"[probe{' inject' if inject else ''}] step={step_id} "
                    f"|tau|={float(applied.abs().mean()):.3f} "
                    f"identity_max={identity:.3e} "
                    f"|joint_f|={float(rows[-1]['recorded_joint_f'].abs().max()):.3e}"
                )
    finally:
        if inject:
            _unregister_post_actuator_callback(_inject_shadow_pd)

    applied = _stack(rows, "applied")
    state_finite = _stack(rows, "state_finite")
    abs_qd = _stack(rows, "abs_qd")
    finite_qd = abs_qd[state_finite]
    return {
        "applied": applied,
        "reconstructed": _stack(rows, "reconstructed"),
        "reconstructed_post": _stack(rows, "reconstructed_post"),
        "recorded_joint_f": _stack(rows, "recorded_joint_f"),
        "id_force": _stack(rows, "id_force"),
        "computed_unclipped": _stack(rows, "computed_unclipped"),
        "abs_qd_mean": float(finite_qd.mean()) if finite_qd.numel() > 0 else float("nan"),
        "nan_frac": float((~state_finite).float().mean()),
    }


def _unregister_post_actuator_callback(callback) -> None:
    """Remove a post-actuator callback if Newton still has it registered."""
    from isaaclab_newton.physics import NewtonManager

    callbacks = getattr(NewtonManager, "_post_actuator_callbacks", [])
    if callback in callbacks:
        callbacks.remove(callback)


args_cli, hydra_args = parse_args()
if args_cli.policy_checkpoint:
    os.environ["ISAACLAB_PROBE_POLICY_CHECKPOINT"] = args_cli.policy_checkpoint
if args_cli.metrics_json:
    os.environ["ISAACLAB_PROBE_METRICS_JSON"] = args_cli.metrics_json
sys.argv = [sys.argv[0]] + _prepare_hydra_argv(hydra_args)


@hydra_task_config(args_cli.task, args_cli.policy_agent)
def main(env_cfg, agent_cfg) -> None:
    """Launch Dataset-Gen Newton and probe implicit shadow PD."""
    from isaaclab_neural.utils.usd_utils import newton_material_binding_api_autofix
    from isaaclab_newton.physics import NewtonManager

    from isaaclab.app import launch_simulation

    if not args_cli.policy_checkpoint:
        args_cli.policy_checkpoint = os.environ.get("ISAACLAB_PROBE_POLICY_CHECKPOINT")
    if not args_cli.metrics_json:
        args_cli.metrics_json = os.environ.get("ISAACLAB_PROBE_METRICS_JSON")
    if not args_cli.policy_checkpoint:
        raise ValueError("--policy-checkpoint is required.")

    configure_env(env_cfg, args_cli)
    actuator_names = {name: type(cfg).__name__ for name, cfg in env_cfg.scene.robot.actuators.items()}
    print(f"[probe] actuators={actuator_names} use_cuda_graph={env_cfg.sim.physics.use_cuda_graph}")

    from isaaclab_neural.eval.eval import build_policy_env, load_saved_agent_cfg_for_policy

    with launch_simulation(env_cfg, args_cli):
        import gymnasium as gym

        with newton_material_binding_api_autofix():
            env = gym.make(args_cli.task, cfg=env_cfg, device=args_cli.device)

        agent_cfg = load_saved_agent_cfg_for_policy(
            args_cli.policy_checkpoint, agent_cfg, device=args_cli.device, seed=args_cli.seed
        )
        env_wrapped, policy, runner = build_policy_env(env, agent_cfg, args_cli.policy_checkpoint, args_cli.device)
        unwrapped = env_wrapped.unwrapped
        control = NewtonManager._control
        robot = unwrapped.scene["robot"]
        joint_names = list(robot.joint_names)
        print(
            f"[probe] task={args_cli.task} num_envs={args_cli.num_envs} steps={args_cli.num_steps} joints={joint_names}"
        )
        baseline = run_rollout(
            env=unwrapped,
            env_wrapped=env_wrapped,
            policy=policy,
            runner=runner,
            control=control,
            num_steps=args_cli.num_steps,
            inject=False,
        )
        inject = None
        if not args_cli.skip_inject:
            inject = run_rollout(
                env=unwrapped,
                env_wrapped=env_wrapped,
                policy=policy,
                runner=runner,
                control=control,
                num_steps=args_cli.num_steps,
                inject=True,
            )
        report = evaluate_shadow_joint_f_probe(
            applied=baseline["applied"],
            reconstructed=baseline["reconstructed"],
            reconstructed_post=baseline["reconstructed_post"],
            recorded_joint_f=baseline["recorded_joint_f"],
            id_force=baseline["id_force"],
            computed_unclipped=baseline["computed_unclipped"],
            abs_qd_baseline=float(baseline["abs_qd_mean"]),
            abs_qd_inject=None if inject is None else float(inject["abs_qd_mean"]),
            inject_nan_frac=None if inject is None else float(inject["nan_frac"]),
            arm_joint_mask=_arm_joint_mask(joint_names, baseline["applied"].device),
        )
        report["joint_names"] = joint_names
        report["num_envs"] = args_cli.num_envs
        report["num_steps"] = args_cli.num_steps
        print(f"[probe] summary={json.dumps(report, indent=2)}")
        if args_cli.metrics_json:
            path = Path(args_cli.metrics_json).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(f"[probe] wrote {path}")
        env_wrapped.close()
        if not report["verdict"]["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
