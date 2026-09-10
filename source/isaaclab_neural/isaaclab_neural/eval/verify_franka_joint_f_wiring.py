# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify Franka Dataset-Gen records shadow PD as ``joint_f`` after the GT step.

Runs analytic Newton ``Isaac-Reach-Franka-Dataset-Gen-v0``. It does not load a
NeRD checkpoint and does not write shadow PD into ``control.joint_f`` before
MJWarp.

Checks:

1. During the analytic solve, ``control.joint_f`` stays ~0.
2. After :meth:`DataGenerationAdapter.step_action_frame`, recorded ``joint_f``
   matches backend-order ``applied_effort``.
3. Arm joints carry a non-zero shadow-PD signal.

Example::

    ./source/isaaclab_neural/isaaclab_neural/eval/scripts/verify_franka_joint_f_wiring.sh
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import isaaclab_neural.envs  # noqa: F401
import torch
from isaaclab_neural.eval.shadow_joint_f import ARM_SHADOW_RMS_MIN_NM, IDENTITY_MAX_ABS_NM, RECORDED_JOINT_F_MAX_ABS_NM
from isaaclab_neural.physics.applied_joint_f import backend_applied_effort

from isaaclab_tasks.utils.hydra import hydra_task_config


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse wiring-verify and Isaac Lab launcher arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, default="Isaac-Reach-Franka-Dataset-Gen-v0")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=32, help="Physics frames to record.")
    parser.add_argument("--seed", type=int, default=0)
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
    """Apply CLI overrides and disable CUDA graphs so Python snapshots stay live."""
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"):
        env_cfg.scene.num_envs = args.num_envs
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = args.seed
    if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "device"):
        if getattr(args, "device", None) is None:
            args.device = env_cfg.sim.device
        else:
            env_cfg.sim.device = args.device
    env_cfg.sim.physics.use_cuda_graph = False
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "ee_pose"):
        env_cfg.commands.ee_pose.debug_vis = False


def _control_joint_f(control, num_envs: int) -> torch.Tensor:
    import warp as wp

    if control is None or getattr(control, "joint_f", None) is None:
        return torch.zeros((num_envs, 0), dtype=torch.float32)
    return wp.to_torch(control.joint_f).float().view(num_envs, -1).clone()


def _arm_joint_mask(joint_names: list[str], device: torch.device) -> torch.Tensor:
    return torch.tensor([name.startswith("panda_joint") for name in joint_names], device=device, dtype=torch.bool)


args_cli, hydra_args = parse_args()
if args_cli.metrics_json:
    os.environ["ISAACLAB_JOINT_F_WIRING_METRICS_JSON"] = args_cli.metrics_json
os.environ.setdefault("ISAACLAB_JOINT_F_WIRING_TASK", args_cli.task)
os.environ.setdefault("ISAACLAB_JOINT_F_WIRING_NUM_ENVS", str(args_cli.num_envs))
os.environ.setdefault("ISAACLAB_JOINT_F_WIRING_NUM_STEPS", str(args_cli.num_steps))
os.environ.setdefault("ISAACLAB_JOINT_F_WIRING_SEED", str(args_cli.seed))
sys.argv = [sys.argv[0]] + _prepare_hydra_argv(hydra_args)


@hydra_task_config(args_cli.task, "")
def main(env_cfg, agent_cfg=None) -> None:
    """Launch Dataset-Gen Newton and verify post-step ``joint_f`` overlay."""
    del agent_cfg
    from isaaclab_neural.generate.adapter import DataGenerationAdapter
    from isaaclab_neural.physics import NerdSolverCfg
    from isaaclab_neural.utils.usd_utils import newton_material_binding_api_autofix
    from isaaclab_newton.physics import NewtonManager

    from isaaclab.app import launch_simulation

    if not args_cli.metrics_json:
        args_cli.metrics_json = os.environ.get("ISAACLAB_JOINT_F_WIRING_METRICS_JSON")
    args_cli.task = os.environ.get("ISAACLAB_JOINT_F_WIRING_TASK", args_cli.task)
    args_cli.num_envs = int(os.environ.get("ISAACLAB_JOINT_F_WIRING_NUM_ENVS", args_cli.num_envs))
    args_cli.num_steps = int(os.environ.get("ISAACLAB_JOINT_F_WIRING_NUM_STEPS", args_cli.num_steps))
    args_cli.seed = int(os.environ.get("ISAACLAB_JOINT_F_WIRING_SEED", args_cli.seed))

    configure_env(env_cfg, args_cli)
    print(f"[wiring] use_cuda_graph={env_cfg.sim.physics.use_cuda_graph}")

    with launch_simulation(env_cfg, args_cli):
        import gymnasium as gym

        with newton_material_binding_api_autofix():
            env = gym.make(args_cli.task, cfg=env_cfg, device=args_cli.device).unwrapped
            env.reset()

        solver_cfg = NerdSolverCfg(
            name="NeuralSolver",
            states_frame="world",
            num_contacts_per_env=0,
            contact_mode="newton_native",
        )
        adapter = DataGenerationAdapter(env, solver_cfg)
        robot = env.scene["robot"]
        control = NewtonManager._control
        num_envs = int(env.num_envs)
        joint_names = list(robot.joint_names)
        print(f"[wiring] task={args_cli.task} num_envs={num_envs} steps={args_cli.num_steps} joints={joint_names}")

        during_solve: list[torch.Tensor] = []

        def _snapshot_joint_f_during_solve() -> None:
            during_solve.append(_control_joint_f(control, num_envs))

        NewtonManager.register_post_actuator_callback(_snapshot_joint_f_during_solve)
        try:
            adapter.reset()
            recorded_rows: list[torch.Tensor] = []
            applied_rows: list[torch.Tensor] = []
            action_low, action_high = adapter.action_bounds()
            for step_id in range(args_cli.num_steps):
                action = adapter.sample_action(action_low, action_high)
                adapter.step_action_frame(action, process_action=True)
                recorded = adapter.raw_neural_inputs()["joint_f"]
                while recorded.ndim > 2 and recorded.shape[1] == 1:
                    recorded = recorded.squeeze(1)
                applied = backend_applied_effort(robot)
                if applied is None:
                    raise RuntimeError("Robot applied_effort telemetry is unavailable.")
                recorded_rows.append(recorded[:, : applied.shape[-1]].clone())
                applied_rows.append(applied.clone())
                if step_id == 0 or (step_id + 1) % 16 == 0:
                    during = during_solve[-1] if during_solve else torch.zeros(1)
                    print(
                        f"[wiring] step={step_id} "
                        f"|tau|={float(applied.abs().mean()):.3f} "
                        f"|joint_f_solve|={float(during.abs().max()):.3e} "
                        f"|joint_f_record|={float(recorded.abs().mean()):.3f}"
                    )
        finally:
            callbacks = getattr(NewtonManager, "_post_actuator_callbacks", [])
            if _snapshot_joint_f_during_solve in callbacks:
                callbacks.remove(_snapshot_joint_f_during_solve)

        recorded = torch.stack(recorded_rows, dim=0)
        applied = torch.stack(applied_rows, dim=0)
        during = torch.stack(during_solve[-args_cli.num_steps :], dim=0)
        residual = (recorded - applied).abs()
        arm_mask = _arm_joint_mask(joint_names, recorded.device)
        arm_rms = float(torch.sqrt(torch.mean(recorded[:, :, arm_mask] ** 2)))
        report = {
            "joint_names": joint_names,
            "num_envs": num_envs,
            "num_steps": args_cli.num_steps,
            "during_solve_max_abs_nm": float(during.abs().max()),
            "record_vs_applied_max_abs_nm": float(residual.max()),
            "recorded_mean_abs_nm": float(recorded.abs().mean()),
            "arm_recorded_rms_nm": arm_rms,
            "verdict": {
                "gt_joint_f_empty_ok": float(during.abs().max()) <= RECORDED_JOINT_F_MAX_ABS_NM,
                "record_matches_applied_ok": float(residual.max()) <= IDENTITY_MAX_ABS_NM,
                "arm_shadow_signal_ok": arm_rms >= ARM_SHADOW_RMS_MIN_NM,
            },
        }
        report["verdict"]["passed"] = all(report["verdict"].values())
        print(f"[wiring] summary={json.dumps(report, indent=2)}")
        if args_cli.metrics_json:
            path = Path(args_cli.metrics_json).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(f"[wiring] wrote {path}")
        env.close()
        if not report["verdict"]["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
