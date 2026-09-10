# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare the Reach RL policy on implicit vs explicit IdealPD Panda actuators.

Runs analytic Newton ``Isaac-Reach-Franka-Dataset-Gen-v0`` with the Reach RL
joint-position action. Use this before switching Dataset-Gen / NeRD-v0 off
implicit PD. It does not load a NeRD checkpoint.

Example::

    PYTHONPATH="$(pwd)/source/isaaclab_neural${PYTHONPATH:+:$PYTHONPATH}" \\
      uv run python -m isaaclab_neural.eval.compare_franka_reach_actuators \\
      --actuator implicit --policy-checkpoint ./pretrained/control_policy/franka_reach/model_999.pt \\
      --num-envs 16 --num-steps 360 --visualizer none presets=newton_mjwarp
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata as metadata
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import isaaclab_neural.envs  # noqa: F401
import torch

from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg

from isaaclab_tasks.utils.hydra import hydra_task_config


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse comparison and Isaac Lab launcher arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, default="Isaac-Reach-Franka-Dataset-Gen-v0")
    parser.add_argument("--policy-checkpoint", type=str, required=True)
    parser.add_argument("--policy-agent", type=str, default="rsl_rl_cfg_entry_point")
    parser.add_argument("--actuator", choices=["implicit", "ideal_pd"], required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=360)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metrics-json", type=str, default=None)
    from isaaclab.app import add_launcher_args

    add_launcher_args(parser)
    return parser.parse_known_args()


def replace_panda_actuators_with_ideal_pd(robot_cfg) -> None:
    """Swap implicit Panda drives for IdealPD with the same gains and limits."""
    converted = {}
    for name, cfg in robot_cfg.actuators.items():
        if not isinstance(cfg, ImplicitActuatorCfg):
            converted[name] = cfg
            continue
        converted[name] = IdealPDActuatorCfg(
            joint_names_expr=list(cfg.joint_names_expr),
            stiffness=cfg.stiffness,
            damping=cfg.damping,
            armature=cfg.armature,
            joint_effort_limit=cfg.joint_effort_limit,
            actuator_effort_limit=cfg.joint_effort_limit,
        )
    robot_cfg.actuators = converted


def configure_env(env_cfg, args: argparse.Namespace) -> None:
    """Apply CLI overrides and restore the Reach RL arm action."""
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
    if args.actuator == "ideal_pd":
        replace_panda_actuators_with_ideal_pd(env_cfg.scene.robot)


def _proxy_to_torch(value) -> torch.Tensor:
    tensor = getattr(value, "torch", value)
    return torch.as_tensor(tensor).float()


def collect_step_metrics(env) -> dict[str, torch.Tensor]:
    """Read EE tracking, success, and effort telemetry for the current step."""
    unwrapped = env.unwrapped
    command = unwrapped.command_manager.get_term("ee_pose")
    robot = unwrapped.scene["robot"]
    success = unwrapped.termination_manager.get_term("success")
    applied = _proxy_to_torch(robot.data.applied_torque)
    return {
        "position_error": command.metrics["position_error"].detach(),
        "orientation_error": command.metrics["orientation_error"].detach(),
        "success": torch.as_tensor(success, device=applied.device, dtype=torch.float32),
        "applied_effort_abs": applied.abs().mean(dim=-1).detach(),
    }


def summarize(records: dict[str, list[float]], *, num_success_resets: int, num_timeouts: int) -> dict[str, float]:
    """Average recorded per-step means into a JSON-serializable dict."""
    summary = {key: float(sum(values) / max(len(values), 1)) for key, values in records.items()}
    summary["success_resets"] = float(num_success_resets)
    summary["timeouts"] = float(num_timeouts)
    return summary


def load_saved_agent_cfg_for_policy(checkpoint_path: str, fallback_agent_cfg, *, device: str, seed: int):
    """Load the RSL-RL runner config saved next to a policy checkpoint when available."""
    agent_cfg_path = Path(checkpoint_path).expanduser().resolve().parent / "params" / "agent.yaml"
    if not agent_cfg_path.exists():
        return fallback_agent_cfg

    import yaml

    with agent_cfg_path.open("r") as f:
        agent_cfg = yaml.safe_load(f)
    if not isinstance(agent_cfg, Mapping):
        raise ValueError(f"Expected a mapping in saved RSL-RL agent config: {agent_cfg_path}")

    agent_cfg = copy.deepcopy(dict(agent_cfg))
    agent_cfg["device"] = device
    agent_cfg["seed"] = seed
    print(f"[policy] using saved RSL-RL agent config: {agent_cfg_path}")
    return agent_cfg


def normalize_runner_cfg(agent_cfg, *, device: str):
    """Return runner kwargs from a saved YAML mapping or an RSL-RL config object."""
    if isinstance(agent_cfg, Mapping):
        runner_cfg = copy.deepcopy(dict(agent_cfg))
        runner_cfg["device"] = device
        return (
            runner_cfg,
            runner_cfg.get("class_name", "OnPolicyRunner"),
            runner_cfg.get("clip_actions"),
            runner_cfg.get("device", device),
        )

    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = device
    runner_cfg = agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else dict(vars(agent_cfg))
    return (
        runner_cfg,
        getattr(agent_cfg, "class_name", "OnPolicyRunner"),
        getattr(agent_cfg, "clip_actions", None),
        getattr(agent_cfg, "device", device),
    )


def build_policy_env(env, agent_cfg, checkpoint_path: str, device: str):
    """Wrap ``env`` and load the Reach RSL-RL checkpoint for inference."""
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    runner_cfg, class_name, clip_actions, runner_device = normalize_runner_cfg(agent_cfg, device=device)
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)
    runner_cls = {"OnPolicyRunner": OnPolicyRunner, "DistillationRunner": DistillationRunner}.get(class_name)
    if runner_cls is None:
        raise ValueError(f"Unsupported RSL-RL runner class: {class_name}")

    print(f"[policy] loading RSL-RL checkpoint: {checkpoint_path}")
    runner = runner_cls(wrapped_env, runner_cfg, log_dir=None, device=runner_device)
    runner.load(checkpoint_path, map_location=device)
    return wrapped_env, runner.get_inference_policy(device=wrapped_env.unwrapped.device), runner


def reset_policy(policy, runner, dones: torch.Tensor) -> None:
    """Reset recurrent policy state on environment dones."""
    if hasattr(policy, "reset"):
        policy.reset(dones)
        return

    policy_state = getattr(getattr(runner, "alg", None), "policy", None)
    if policy_state is None:
        policy_state = getattr(getattr(runner, "alg", None), "actor_critic", None)
    if policy_state is not None and hasattr(policy_state, "reset"):
        policy_state.reset(dones)


args_cli, hydra_args = parse_args()
sys.argv = [sys.argv[0]] + hydra_args


@hydra_task_config(args_cli.task, args_cli.policy_agent)
def main(env_cfg, agent_cfg) -> None:
    """Launch Dataset-Gen Newton and roll out the Reach policy."""
    configure_env(env_cfg, args_cli)
    actuator_names = {name: type(cfg).__name__ for name, cfg in env_cfg.scene.robot.actuators.items()}
    print(f"[compare] actuator={args_cli.actuator} groups={actuator_names}")

    from isaaclab_neural.utils.usd_utils import newton_material_binding_api_autofix

    from isaaclab.app import launch_simulation

    with launch_simulation(env_cfg, args_cli):
        import gymnasium as gym

        with newton_material_binding_api_autofix():
            env = gym.make(args_cli.task, cfg=env_cfg, device=args_cli.device)

        agent_cfg = load_saved_agent_cfg_for_policy(
            args_cli.policy_checkpoint, agent_cfg, device=args_cli.device, seed=args_cli.seed
        )
        env, policy, runner = build_policy_env(env, agent_cfg, args_cli.policy_checkpoint, args_cli.device)
        obs = env.get_observations()
        records = {
            "reward_mean": [],
            "position_error_m": [],
            "orientation_error_rad": [],
            "success_frac": [],
            "applied_effort_abs_nm": [],
        }
        num_success_resets = 0
        num_timeouts = 0
        print(
            f"[compare] task={args_cli.task} num_envs={args_cli.num_envs} "
            f"steps={args_cli.num_steps} seed={args_cli.seed}"
        )
        for step_id in range(args_cli.num_steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, rew, dones, extras = env.step(actions)
                reset_policy(policy, runner, dones)
            metrics = collect_step_metrics(env)
            records["reward_mean"].append(float(torch.as_tensor(rew).float().mean()))
            records["position_error_m"].append(float(metrics["position_error"].mean()))
            records["orientation_error_rad"].append(float(metrics["orientation_error"].mean()))
            records["success_frac"].append(float(metrics["success"].mean()))
            records["applied_effort_abs_nm"].append(float(metrics["applied_effort_abs"].mean()))
            dones_t = torch.as_tensor(dones).bool()
            timeout = extras.get("time_outs")
            if timeout is None:
                timeout = env.unwrapped.termination_manager.get_term("time_out")
            timeout_t = torch.as_tensor(timeout).bool()
            num_success_resets += int((dones_t & metrics["success"].bool()).sum())
            num_timeouts += int((dones_t & timeout_t).sum())
            if step_id == 0 or (step_id + 1) % 60 == 0:
                print(
                    f"[step {step_id}] reward={records['reward_mean'][-1]:.4f} "
                    f"pos_err={records['position_error_m'][-1]:.4f} "
                    f"ori_err={records['orientation_error_rad'][-1]:.4f} "
                    f"success={records['success_frac'][-1]:.3f} "
                    f"|tau|={records['applied_effort_abs_nm'][-1]:.3f}"
                )
        summary = summarize(records, num_success_resets=num_success_resets, num_timeouts=num_timeouts)
        summary["actuator"] = args_cli.actuator
        summary["num_envs"] = args_cli.num_envs
        summary["num_steps"] = args_cli.num_steps
        print(f"[compare] summary={json.dumps(summary, indent=2)}")
        if args_cli.metrics_json:
            path = Path(args_cli.metrics_json).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(summary, indent=2) + "\n")
            print(f"[compare] wrote {path}")
        env.close()


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
