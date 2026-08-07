# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark NeRD and ground-truth solver step costs.

Enables :mod:`isaaclab_neural.utils.step_profile` and reports ms/step for each
hot-path section. Use ``--compare`` to run ``fixed_ground``, ``newton_native``,
and stock ground-truth MJWarp in isolated subprocesses (PhysicsManager is
process-global).

Ground-truth workers only report wall-clock / ``env_step_total`` (NeRD section
timers such as ``model_forward`` do not apply).

Example:
    ./isaaclab.sh -p -m isaaclab_neural.eval.benchmark_nerd_contact_modes \\
      --compare \\
      --neural-model-path-fg /path/to/fg/final_model.pt \\
      --neural-model-path-native /path/to/native/final_model.pt \\
      --num_envs 256 --warmup 20 --steps 50 --headless \\
      presets=newton_mjwarp
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

RESULT_PREFIX = "__NERD_CONTACT_BENCH__ "
DEFAULT_NERD_TASK = "Isaac-Velocity-Flat-Anymal-C-NeRD-v0"
DEFAULT_GT_TASK = "Isaac-Velocity-Flat-Anymal-C-v0"
CONTACT_MODES = ("fixed_ground", "newton_native", "ground_truth")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Benchmark NeRD / GT solver stepping cost.")
    parser.add_argument("--task", type=str, default=DEFAULT_NERD_TASK, help="NeRD task id for NeRD workers.")
    parser.add_argument(
        "--gt-task",
        type=str,
        default=DEFAULT_GT_TASK,
        help="Stock GT task id used when --contact-mode ground_truth or --compare.",
    )
    parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=20, help="Warmup env steps before timing.")
    parser.add_argument("--steps", type=int, default=100, help="Timed env steps.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--contact-mode",
        choices=list(CONTACT_MODES),
        default=None,
        help="Single-mode run. Required unless --compare.",
    )
    parser.add_argument("--neural-model-path", type=str, default=None, help="Dynamics ckpt for NeRD single-mode.")
    parser.add_argument("--neural-model-path-fg", type=str, default=None, help="FG ckpt for --compare.")
    parser.add_argument(
        "--neural-model-path-native",
        type=str,
        default=None,
        help="newton_native ckpt for --compare.",
    )
    parser.add_argument("--num-contacts-per-env", type=int, default=64)
    parser.add_argument(
        "--compare",
        action="store_true",
        default=False,
        help="Run fixed_ground, newton_native, and ground_truth via subprocess.",
    )
    parser.add_argument(
        "--skip-gt",
        action="store_true",
        default=False,
        help="With --compare, skip the stock ground-truth worker.",
    )
    parser.add_argument("--output-json", type=str, default=None, help="Write results JSON to this path.")
    parser.add_argument("--worker", action="store_true", default=False, help=argparse.SUPPRESS)

    from isaaclab_tasks.utils import add_launcher_args

    add_launcher_args(parser)
    return parser.parse_known_args()


def _repo_root() -> Path:
    """Return the IsaacLab worktree root that contains ``isaaclab.sh``."""
    return Path(__file__).resolve().parents[4]


def _zero_action(env) -> Any:
    import torch

    shape = env.action_space.shape
    if len(shape) == 1:
        shape = (getattr(env, "num_envs", 1), shape[0])
    return torch.zeros(shape, device=getattr(env, "device", "cpu"), dtype=torch.float32)


def _print_compare(results: dict[str, dict[str, Any]]) -> None:
    """Print a wall-clock and section table for the collected worker results."""
    order = [name for name in ("ground_truth", "fixed_ground", "newton_native") if name in results]
    if not order:
        return

    print("\n=== Solver step-cost compare ===")
    for name in order:
        ms = float(results[name]["wall_ms_per_step"])
        task = results[name].get("task", "")
        print(f"{name:<14}: {ms:.3f} ms/step  ({task})")

    baseline_name = "ground_truth" if "ground_truth" in results else order[0]
    baseline_ms = float(results[baseline_name]["wall_ms_per_step"])
    print(f"\nWall-clock vs {baseline_name}:")
    for name in order:
        ms = float(results[name]["wall_ms_per_step"])
        ratio = ms / max(baseline_ms, 1e-9)
        print(f"  {name:<14}: {ratio:.2f}x")

    section_names: set[str] = set()
    for payload in results.values():
        section_names.update(payload.get("sections_ms_per_step", {}))
    if not section_names:
        return

    header = f"{'section':<24}" + "".join(f" {name:>12}" for name in order)
    print("\n" + header)
    for section in sorted(section_names):
        row = f"{section:<24}"
        for name in order:
            value = float(results[name].get("sections_ms_per_step", {}).get(section, 0.0))
            row += f" {value:12.3f}"
        print(row)


def _spawn_worker(
    *,
    contact_mode: str,
    neural_model_path: str | None,
    args: argparse.Namespace,
    hydra_args: list[str],
) -> dict[str, Any]:
    repo = _repo_root()
    isaaclab_sh = repo / "isaaclab.sh"
    if not isaaclab_sh.is_file():
        raise RuntimeError(f"Could not find isaaclab.sh at {isaaclab_sh}")

    cmd = [
        str(isaaclab_sh),
        "-p",
        "-m",
        "isaaclab_neural.eval.benchmark_nerd_contact_modes",
        "--worker",
        "--contact-mode",
        contact_mode,
        "--num_envs",
        str(args.num_envs),
        "--warmup",
        str(args.warmup),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--num-contacts-per-env",
        str(args.num_contacts_per_env),
        "--task",
        args.task,
        "--gt-task",
        args.gt_task,
        "--agent",
        args.agent,
    ]
    if neural_model_path is not None:
        cmd.extend(["--neural-model-path", neural_model_path])
    if getattr(args, "headless", False):
        cmd.append("--headless")
    if getattr(args, "device", None):
        cmd.extend(["--device", str(args.device)])
    cmd.extend(hydra_args)

    env = os.environ.copy()
    env["NERD_STEP_PROFILE"] = "1"

    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env, cwd=str(repo))
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        raise RuntimeError(
            f"Worker failed for {contact_mode} (exit={proc.returncode}).\n"
            f"stdout:\n{stdout[-4000:]}\nstderr:\n{stderr[-4000:]}"
        )

    payload = None
    for line in stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            payload = json.loads(line[len(RESULT_PREFIX) :])
    if payload is None:
        raise RuntimeError(f"No result payload from {contact_mode} worker.\nstdout:\n{stdout[-4000:]}")
    return payload


args_cli, hydra_args = parse_args()

# Compare mode never starts SimulationApp in the parent process.
if args_cli.compare and not args_cli.worker:
    if not args_cli.neural_model_path_fg or not args_cli.neural_model_path_native:
        raise SystemExit("--compare requires --neural-model-path-fg and --neural-model-path-native.")

    combined: dict[str, dict[str, Any]] = {}
    if not args_cli.skip_gt:
        combined["ground_truth"] = _spawn_worker(
            contact_mode="ground_truth",
            neural_model_path=None,
            args=args_cli,
            hydra_args=hydra_args,
        )
    combined["fixed_ground"] = _spawn_worker(
        contact_mode="fixed_ground",
        neural_model_path=args_cli.neural_model_path_fg,
        args=args_cli,
        hydra_args=hydra_args,
    )
    combined["newton_native"] = _spawn_worker(
        contact_mode="newton_native",
        neural_model_path=args_cli.neural_model_path_native,
        args=args_cli,
        hydra_args=hydra_args,
    )
    _print_compare(combined)
    if args_cli.output_json:
        Path(args_cli.output_json).write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args_cli.output_json}")
    raise SystemExit(0)

if args_cli.contact_mode is None:
    raise SystemExit("Provide --contact-mode, or use --compare.")
if args_cli.contact_mode != "ground_truth" and args_cli.neural_model_path is None:
    raise SystemExit("NeRD modes require --neural-model-path (or use --compare).")

# Resolve which gym task this worker should instantiate.
worker_task = args_cli.gt_task if args_cli.contact_mode == "ground_truth" else args_cli.task
is_ground_truth = args_cli.contact_mode == "ground_truth"

# Register tasks then Hydra-launch like train/play.
import isaaclab_tasks  # noqa: E402,F401

if not is_ground_truth:
    import isaaclab_neural.envs  # noqa: E402,F401
    from isaaclab_neural.rl.rsl_rl import cli_args as nerd_cli  # noqa: E402
else:
    nerd_cli = None  # type: ignore[assignment]

from isaaclab_neural.utils import step_profile  # noqa: E402
from isaaclab_neural.utils.usd_utils import newton_material_binding_api_autofix  # noqa: E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

sys.argv = [sys.argv[0]] + hydra_args
os.environ["NERD_STEP_PROFILE"] = "1"
step_profile.enable(cuda_synchronize=True)


@hydra_task_config(worker_task, args_cli.agent)
def main(env_cfg, agent_cfg) -> None:
    """Run a single-mode timed stepping benchmark (NeRD or ground truth)."""
    del agent_cfg
    import torch
    from isaaclab_tasks.utils import launch_simulation

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device

    solver_cfg = None
    if is_ground_truth:
        launch_cfg = env_cfg
    else:
        # Restrict contact-count override to native mode.
        if args_cli.contact_mode != "newton_native":
            args_cli.num_contacts_per_env = None
        assert nerd_cli is not None
        solver_cfg = nerd_cli.apply_neural_model_to_env_cfg(env_cfg, args_cli)
        launch_cfg = nerd_cli.build_launch_cfg(env_cfg)

    step_profile.reset()
    result: dict[str, Any]

    with launch_simulation(launch_cfg, args_cli):
        import gymnasium as gym

        gym_kwargs = {"cfg": env_cfg, "device": args_cli.device}
        if solver_cfg is not None:
            gym_kwargs["solver_cfg"] = solver_cfg
        with newton_material_binding_api_autofix():
            env = gym.make(worker_task, **gym_kwargs)

        action = _zero_action(env)
        env.reset()

        for _ in range(args_cli.warmup):
            with step_profile.section("env_step_total"):
                env.step(action)
        step_profile.reset()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args_cli.steps):
            with step_profile.section("env_step_total"):
                env.step(action)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        sections = step_profile.summary_dict()
        ms_per_step = {name: (total * 1000.0 / args_cli.steps) for name, total in sections.items() if total > 0}
        result = {
            "contact_mode": args_cli.contact_mode,
            "task": worker_task,
            "neural_model_path": args_cli.neural_model_path,
            "num_envs": args_cli.num_envs,
            "warmup": args_cli.warmup,
            "steps": args_cli.steps,
            "wall_seconds": wall,
            "wall_ms_per_step": wall * 1000.0 / args_cli.steps,
            "sections_seconds": sections,
            "sections_ms_per_step": ms_per_step,
        }
        step_profile.print_summary(steps=args_cli.steps)
        env.close()

    if args_cli.worker:
        print(RESULT_PREFIX + json.dumps(result), flush=True)
    else:
        print(json.dumps(result, indent=2))
        if args_cli.output_json:
            Path(args_cli.output_json).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote {args_cli.output_json}")


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
