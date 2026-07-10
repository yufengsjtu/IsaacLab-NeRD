# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark upstream Newton CUDA graph speedups across stepping action modes.

This targets pure Newton tasks (e.g. ``Isaac-Cartpole-v0`` with ``newton_mjwarp``)
where physics steps contain no PyTorch and can be captured in a Warp CUDA graph.
NeRD neural solvers always run eagerly, so their physics steps are not covered here.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import torch

ACTION_MODES = ("action", "joint_f", "action_frame")
WORKER_RESULT_PREFIX = "__BENCHMARK_RESULT__ "


def _assign_joint_forces(manager: Any, joint_f: torch.Tensor) -> None:
    """Write direct joint forces into Newton control without running actuators."""
    import warp as wp

    control = manager._control
    if control is None:
        raise RuntimeError("Newton control is not initialized; cannot apply direct joint forces.")
    control_joint_f = control.joint_f
    if control_joint_f is None:
        raise RuntimeError("Newton control.joint_f is not allocated; cannot apply direct joint forces.")
    control_joint_f.assign(wp.from_torch(joint_f.reshape(-1)))


def _launch_physics_step(manager: Any) -> None:
    """Advance one physics frame without actuators, honoring upstream CUDA graphs."""
    import warp as wp
    from newton import eval_fk

    from isaaclab.physics import PhysicsManager

    sim = PhysicsManager._sim
    if sim is None or not sim.is_playing():
        return

    if manager._model_changes:
        with wp.ScopedDevice(PhysicsManager._device):
            for change in manager._model_changes:
                manager._solver.notify_model_changed(change)
            manager._model_changes = set()

    cfg = PhysicsManager._cfg
    device = PhysicsManager._device
    if (
        getattr(manager, "_graph_capture_pending", False)
        and cfg is not None
        and cfg.use_cuda_graph
        and "cuda" in device
    ):  # type: ignore[union-attr]
        manager._graph_capture_pending = False
        manager._graph = manager._capture_relaxed_graph(device)
        if manager._graph is not None:
            print("Newton CUDA graph captured (deferred relaxed mode, RTX-compatible)", flush=True)
        else:
            print("Newton CUDA graph capture failed; falling back to eager stepping", flush=True)

    if manager._needs_collision_pipeline:
        eval_fk(
            manager._model,
            manager._state_0.joint_q,
            manager._state_0.joint_qd,
            manager._state_0,
            manager._fk_reset_mask,
        )

    manager._world_reset_mask.zero_()
    manager._fk_reset_mask.zero_()

    physics_dt = manager._solver_dt * manager._num_substeps
    use_graph = cfg is not None and cfg.use_cuda_graph and manager._graph is not None and "cuda" in device  # type: ignore[union-attr]
    if use_graph:
        wp.capture_launch(manager._graph)
    else:
        with wp.ScopedDevice(device):
            manager._simulate_physics_only()
    PhysicsManager._sim_time += physics_dt

    if manager._usdrt_stage is not None:
        manager._mark_state_dirty()

    manager._log_solver_debug()


def register_task_module(task: str) -> None:
    """Register only the task package needed by this benchmark run."""
    if task == "Isaac-Cartpole-v0":
        import isaaclab_tasks.manager_based.classic.cartpole  # noqa: F401
    elif "NeRD" in task or "Dataset-Gen" in task:
        import isaaclab_neural.envs  # noqa: F401
    else:
        raise ValueError(
            f"Task {task!r} is not registered by this benchmark. Use Isaac-Cartpole-v0 or an isaaclab_neural task."
        )


def _inject_preset(hydra_args: list[str], preset: str) -> list[str]:
    """Prepend a ``presets=<preset>`` Hydra override unless one is already provided.

    Kitless environments only ship the Newton backend, so tasks whose default preset
    is PhysX (e.g. ``Isaac-Cartpole-v0``) must be switched to a Newton preset to run.
    """
    if not preset:
        return hydra_args
    if any(arg.split("=", 1)[0] == "presets" for arg in hydra_args):
        return hydra_args
    return [f"presets={preset}", *hydra_args]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse CLI arguments and return remaining Hydra overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, default="Isaac-Cartpole-v0", help="Registered gym task id.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional NeRD checkpoint for NeRD tasks.")
    parser.add_argument(
        "--use-cuda-graph",
        choices=("true", "false"),
        default=None,
        help="Run a single configuration instead of comparing eager vs graph in subprocesses.",
    )
    parser.add_argument("--num-envs", type=int, default=512, help="Number of parallel environments.")
    parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
    parser.add_argument(
        "--preset",
        type=str,
        default="newton_mjwarp",
        help=(
            "Physics backend preset to select via Hydra (e.g. newton_mjwarp, newton_kamino). "
            "Ignored if a 'presets=' override is passed explicitly, or set to '' to disable."
        ),
    )
    parser.add_argument(
        "--action-modes",
        nargs="+",
        choices=ACTION_MODES,
        default=list(ACTION_MODES),
        help="Stepping modes to benchmark.",
    )
    parser.add_argument("--warmup-steps", type=int, default=20, help="Warmup steps before timing.")
    parser.add_argument("--measure-steps", type=int, default=200, help="Timed steps per benchmark.")
    parser.add_argument("--measure-runs", type=int, default=3, help="Repeated timed runs per mode.")
    parser.add_argument(
        "--check-equivalence",
        action="store_true",
        help="Compare eager vs CUDA graph trajectories and report max abs joint-state diffs.",
    )
    parser.add_argument(
        "--equivalence-steps",
        type=int,
        default=50,
        help="Number of rollout steps per action mode when --check-equivalence is set.",
    )

    from isaaclab_tasks.utils import add_launcher_args

    add_launcher_args(parser)
    return parser.parse_known_args()


def configure_env(env_cfg, args: argparse.Namespace) -> None:
    """Apply launch-time environment overrides."""
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"):
        env_cfg.scene.num_envs = args.num_envs
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = args.seed
    if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "device"):
        if getattr(args, "device", None) is None:
            args.device = env_cfg.sim.device
        else:
            env_cfg.sim.device = args.device


def build_launch_cfg(env_cfg):
    """Build a launch-only config that ``launch_simulation`` recognizes as Newton."""
    from isaaclab_neural.physics import NerdNewtonCfg

    physics_cfg = getattr(getattr(env_cfg, "sim", None), "physics", None)
    if not isinstance(physics_cfg, NerdNewtonCfg):
        return env_cfg

    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

    launch_cfg = copy.deepcopy(env_cfg)
    launch_cfg.sim.physics = NewtonCfg(
        num_substeps=physics_cfg.num_substeps,
        debug_mode=physics_cfg.debug_mode,
        use_cuda_graph=physics_cfg.use_cuda_graph,
    )
    return launch_cfg


def zero_action(env):
    """Return a zero action tensor for the active env batch."""
    import torch

    shape = env.action_space.shape
    if len(shape) == 1:
        shape = (getattr(env.unwrapped, "num_envs", getattr(env, "num_envs", 1)), shape[0])
    device = getattr(env.unwrapped, "device", getattr(env, "device", "cpu"))
    return torch.zeros(shape, device=device, dtype=torch.float32)


def zero_joint_f(env):
    """Return a zero joint-force tensor for the active env batch."""
    import torch
    from isaaclab_newton.physics import NewtonManager

    num_envs = int(getattr(env, "num_envs", 1))
    device = getattr(env, "device", "cpu")
    if NewtonManager._control is None or NewtonManager._control.joint_f is None:
        raise RuntimeError("Newton joint-force control is not initialized.")
    joint_f_dim = int(NewtonManager._control.joint_f.shape[0] // num_envs)
    return torch.zeros((num_envs, joint_f_dim), device=device, dtype=torch.float32)


def active_manager(env):
    """Return the active Newton manager class for the current simulation."""
    if hasattr(env, "neural_adapter"):
        return env.neural_adapter.manager
    from isaaclab_newton.physics import NewtonManager

    return NewtonManager


def _cuda_sync(device: str) -> None:
    import torch

    if "cuda" in device and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _read_newton_state(env) -> dict[str, torch.Tensor]:
    """Return a CPU copy of the current Newton generalized coordinates."""
    import warp as wp

    manager = active_manager(env)
    return {
        "joint_q": wp.to_torch(manager._state_0.joint_q).detach().clone().cpu(),
        "joint_qd": wp.to_torch(manager._state_0.joint_qd).detach().clone().cpu(),
    }


def _deterministic_control(env, step: int, *, kind: str) -> torch.Tensor:
    """Return a deterministic per-step control tensor for equivalence rollouts."""
    import torch

    if kind == "action":
        tensor = zero_action(env)
    elif kind == "joint_f":
        tensor = zero_joint_f(env)
    else:
        raise ValueError(f"Unsupported control kind: {kind}")

    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(step * 1_000_003 + 17)
    return torch.randn(tensor.shape, generator=generator, device=tensor.device, dtype=tensor.dtype)


def _apply_equivalence_step(env, action_mode: str, step: int) -> None:
    """Advance the environment by one deterministic step for equivalence checks."""
    if action_mode == "action":
        env.step(_deterministic_control(env, step, kind="action"))
        return

    if action_mode == "joint_f":
        manager = active_manager(env)
        _assign_joint_forces(manager, _deterministic_control(env, step, kind="joint_f"))
        _launch_physics_step(manager)
        unwrapped = getattr(env, "unwrapped", env)
        if hasattr(unwrapped, "scene"):
            unwrapped.scene.update(dt=getattr(unwrapped, "physics_dt", 0.0))
        return

    if action_mode == "action_frame":
        unwrapped = env.unwrapped
        action = _deterministic_control(env, step, kind="action")
        unwrapped.action_manager.process_action(action)
        unwrapped.action_manager.apply_action()
        unwrapped.scene.write_data_to_sim()
        unwrapped.sim.step(render=False)
        if hasattr(unwrapped, "_sim_step_counter"):
            unwrapped._sim_step_counter += 1
        unwrapped.scene.update(dt=getattr(unwrapped, "physics_dt", 0.0))
        return

    raise ValueError(f"Unsupported action_mode: {action_mode}")


def _collect_equivalence_trajectory(env, action_mode: str, *, steps: int) -> dict[str, torch.Tensor]:
    """Roll out one action mode and record Newton joint states after every step."""
    import torch

    env.reset()
    joint_q_steps = []
    joint_qd_steps = []
    for step in range(steps):
        _apply_equivalence_step(env, action_mode, step)
        state = _read_newton_state(env)
        joint_q_steps.append(state["joint_q"])
        joint_qd_steps.append(state["joint_qd"])

    return {
        "joint_q": torch.stack(joint_q_steps),
        "joint_qd": torch.stack(joint_qd_steps),
    }


def _compare_equivalence_trajectories(
    eager: dict[str, torch.Tensor],
    graph: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Compute max and per-step abs diffs between two recorded trajectories."""

    joint_q_diff = (eager["joint_q"] - graph["joint_q"]).abs()
    joint_qd_diff = (eager["joint_qd"] - graph["joint_qd"]).abs()
    per_step_joint_q = joint_q_diff.reshape(joint_q_diff.shape[0], -1).amax(dim=1)
    per_step_joint_qd = joint_qd_diff.reshape(joint_qd_diff.shape[0], -1).amax(dim=1)

    worst_q_step = int(per_step_joint_q.argmax().item())
    worst_qd_step = int(per_step_joint_qd.argmax().item())
    return {
        "max_joint_q_diff": float(joint_q_diff.max().item()),
        "max_joint_qd_diff": float(joint_qd_diff.max().item()),
        "worst_joint_q_step": worst_q_step,
        "worst_joint_qd_step": worst_qd_step,
        "worst_joint_q_diff": float(per_step_joint_q[worst_q_step].item()),
        "worst_joint_qd_diff": float(per_step_joint_qd[worst_qd_step].item()),
        "per_step_joint_q_diff": [float(value) for value in per_step_joint_q],
        "per_step_joint_qd_diff": [float(value) for value in per_step_joint_qd],
    }


def _print_equivalence_result(action_mode: str, result: dict[str, Any]) -> None:
    print(f"\n[{action_mode}]")
    print(f"  max |joint_q diff|:   {result['max_joint_q_diff']:.6e}")
    print(f"  max |joint_qd diff|:  {result['max_joint_qd_diff']:.6e}")
    print(f"  worst joint_q step:   {result['worst_joint_q_step']} ({result['worst_joint_q_diff']:.6e})")
    print(f"  worst joint_qd step:  {result['worst_joint_qd_step']} ({result['worst_joint_qd_diff']:.6e})")


def _time_action_steps(env, *, warmup_steps: int, measure_steps: int) -> float:
    action = zero_action(env)
    device = str(getattr(env.unwrapped, "device", getattr(env, "device", "cpu")))
    for _ in range(warmup_steps):
        env.step(action)
    _cuda_sync(device)
    start = time.perf_counter()
    for _ in range(measure_steps):
        env.step(action)
    _cuda_sync(device)
    return time.perf_counter() - start


def _time_joint_f_steps(env, *, warmup_steps: int, measure_steps: int) -> float:
    manager = active_manager(env)
    joint_f = zero_joint_f(env)
    device = str(getattr(env, "device", "cpu"))
    for _ in range(warmup_steps):
        _assign_joint_forces(manager, joint_f)
        _launch_physics_step(manager)
    _cuda_sync(device)
    start = time.perf_counter()
    for _ in range(measure_steps):
        _assign_joint_forces(manager, joint_f)
        _launch_physics_step(manager)
    _cuda_sync(device)
    return time.perf_counter() - start


def _time_action_frame_steps(env, *, warmup_steps: int, measure_steps: int) -> float:
    unwrapped = env.unwrapped
    action = zero_action(env)
    device = str(getattr(unwrapped, "device", getattr(env, "device", "cpu")))
    physics_dt = getattr(unwrapped, "physics_dt", 0.0)

    def step_once() -> None:
        unwrapped.action_manager.process_action(action)
        unwrapped.action_manager.apply_action()
        unwrapped.scene.write_data_to_sim()
        unwrapped.sim.step(render=False)
        if hasattr(unwrapped, "_sim_step_counter"):
            unwrapped._sim_step_counter += 1
        unwrapped.scene.update(dt=physics_dt)

    for _ in range(warmup_steps):
        step_once()
    _cuda_sync(device)
    start = time.perf_counter()
    for _ in range(measure_steps):
        step_once()
    _cuda_sync(device)
    return time.perf_counter() - start


def _benchmark_action_mode(
    env,
    action_mode: str,
    *,
    warmup_steps: int,
    measure_steps: int,
    measure_runs: int,
) -> dict[str, float]:
    import torch

    timings = []
    for _ in range(measure_runs):
        if action_mode == "action":
            timings.append(_time_action_steps(env, warmup_steps=warmup_steps, measure_steps=measure_steps))
        elif action_mode == "joint_f":
            timings.append(_time_joint_f_steps(env, warmup_steps=warmup_steps, measure_steps=measure_steps))
        elif action_mode == "action_frame":
            timings.append(_time_action_frame_steps(env, warmup_steps=warmup_steps, measure_steps=measure_steps))
        else:
            raise ValueError(f"Unsupported action_mode: {action_mode}")

    mean_seconds = float(sum(timings) / len(timings))
    manager = active_manager(env)
    from isaaclab.physics import PhysicsManager

    physics_cfg = PhysicsManager._cfg
    return {
        "seconds_mean": mean_seconds,
        "seconds_std": float(torch.tensor(timings).std(unbiased=False).item()) if len(timings) > 1 else 0.0,
        "steps_per_second": float(measure_steps / mean_seconds),
        "use_cuda_graph": bool(getattr(physics_cfg, "use_cuda_graph", False)),
        "graph_captured": getattr(manager, "_graph", None) is not None,
    }


def _run_worker(payload: dict[str, object]) -> dict[str, Any]:
    """Run one benchmark or equivalence collection configuration in an isolated process."""
    worker_kind = str(payload.get("worker_kind", "benchmark"))
    if worker_kind == "equivalence":
        return _run_equivalence_collect_worker(payload)
    return _run_benchmark_worker(payload)


def _configure_cuda_graph(env_cfg, args: argparse.Namespace, use_cuda_graph: bool):
    """Apply CUDA graph overrides and return an optional NeRD solver config."""
    from isaaclab_neural.physics import NerdNewtonCfg, NerdSolverCfg

    physics_cfg = getattr(getattr(env_cfg, "sim", None), "physics", None)
    solver_cfg = None
    if isinstance(physics_cfg, NerdNewtonCfg):
        if args.checkpoint is not None:
            from isaaclab_neural.utils.checkpoint import get_cfg_from_checkpoint, load_checkpoint

            checkpoint = load_checkpoint(args.checkpoint, device=args.device)
            checkpoint_cfg = get_cfg_from_checkpoint(checkpoint, args.checkpoint)
            neural_solver_cfg = dict(checkpoint_cfg["env"]["neural_solver_cfg"])
            neural_solver_cfg.pop("use_cuda_graph", None)
            solver_cfg = NerdSolverCfg(**neural_solver_cfg)
            solver_cfg.neural_model_path = args.checkpoint
        else:
            solver_cfg = copy.deepcopy(physics_cfg.solver_cfg)
        physics_cfg.use_cuda_graph = use_cuda_graph
    elif physics_cfg is not None and hasattr(physics_cfg, "use_cuda_graph"):
        physics_cfg.use_cuda_graph = use_cuda_graph
    return solver_cfg


def _make_benchmark_env(env_cfg, args: argparse.Namespace, solver_cfg):
    """Create and reset a gym environment inside an active simulation context."""
    import gymnasium as gym

    gym_kwargs = {"cfg": env_cfg, "device": args.device}
    if solver_cfg is not None:
        gym_kwargs["solver_cfg"] = solver_cfg
    env = gym.make(args.task, **gym_kwargs).unwrapped
    env.reset()
    return env


def _run_benchmark_worker(payload: dict[str, object]) -> dict[str, dict[str, float]]:
    """Run one benchmark configuration in an isolated process."""
    from isaaclab_tasks.utils.hydra import hydra_task_config

    sys.argv = [sys.argv[0], *list(cast(list[str], payload.get("hydra_args", [])))]
    args = argparse.Namespace(**cast(dict[str, object], payload["args"]))
    action_modes = list(cast(list[str], payload["action_modes"]))
    use_cuda_graph = bool(payload["use_cuda_graph"])
    register_task_module(args.task)
    benchmark_results: dict[str, dict[str, float]] | None = None

    @hydra_task_config(args.task, "")
    def _worker(env_cfg, _agent_cfg=None) -> None:
        nonlocal benchmark_results
        configure_env(env_cfg, args)
        solver_cfg = _configure_cuda_graph(env_cfg, args, use_cuda_graph)

        from isaaclab_tasks.utils import launch_simulation

        with launch_simulation(build_launch_cfg(env_cfg), args):
            env = _make_benchmark_env(env_cfg, args, solver_cfg)
            results = {}
            for action_mode in action_modes:
                results[action_mode] = _benchmark_action_mode(
                    env,
                    action_mode,
                    warmup_steps=args.warmup_steps,
                    measure_steps=args.measure_steps,
                    measure_runs=args.measure_runs,
                )
            env.close()
            benchmark_results = results

    _worker()  # type: ignore[call-arg]
    if benchmark_results is None:
        raise RuntimeError("Benchmark worker completed without collecting results.")
    return benchmark_results


def _run_equivalence_collect_worker(payload: dict[str, object]) -> dict[str, str]:
    """Collect deterministic trajectories for later eager-vs-graph comparison."""
    import torch

    from isaaclab_tasks.utils.hydra import hydra_task_config

    sys.argv = [sys.argv[0], *list(cast(list[str], payload.get("hydra_args", [])))]
    args = argparse.Namespace(**cast(dict[str, object], payload["args"]))
    action_modes = list(cast(list[str], payload["action_modes"]))
    use_cuda_graph = bool(payload["use_cuda_graph"])
    output_path = str(payload["output_path"])
    register_task_module(args.task)
    collect_status: dict[str, str] | None = None

    @hydra_task_config(args.task, "")
    def _worker(env_cfg, _agent_cfg=None) -> None:
        nonlocal collect_status
        configure_env(env_cfg, args)
        solver_cfg = _configure_cuda_graph(env_cfg, args, use_cuda_graph)

        from isaaclab_tasks.utils import launch_simulation

        with launch_simulation(build_launch_cfg(env_cfg), args):
            env = _make_benchmark_env(env_cfg, args, solver_cfg)
            trajectories = {}
            for action_mode in action_modes:
                trajectories[action_mode] = _collect_equivalence_trajectory(
                    env,
                    action_mode,
                    steps=args.equivalence_steps,
                )
            env.close()
            torch.save(trajectories, output_path)
            collect_status = {"output_path": output_path, "use_cuda_graph": str(use_cuda_graph)}

    _worker()  # type: ignore[call-arg]
    if collect_status is None:
        raise RuntimeError("Equivalence worker completed without collecting trajectories.")
    return collect_status


def _benchmark_in_subprocess(payload: dict[str, object]) -> dict[str, Any]:
    """Spawn a clean process so CUDA graph capture state does not leak across modes."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-json",
        json.dumps(payload),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True, env=os.environ.copy())
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, file=sys.stderr)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        completed.check_returncode()
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(WORKER_RESULT_PREFIX):
            return json.loads(line.removeprefix(WORKER_RESULT_PREFIX))
    if completed.stdout:
        print(completed.stdout, file=sys.stderr)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    raise RuntimeError("Benchmark worker completed without emitting a parseable result.")


def _print_mode_result(label: str, result: dict[str, float]) -> None:
    print(f"\n[{label}]")
    print(f"  cuda graph:      {result['use_cuda_graph']}")
    print(f"  graph captured:  {result['graph_captured']}")
    print(f"  seconds:         {result['seconds_mean']:.4f} +/- {result['seconds_std']:.4f}")
    print(f"  steps/s:         {result['steps_per_second']:.1f}")


def _run_equivalence_check(payload: dict[str, object]) -> dict[str, dict[str, Any]]:
    """Collect eager and graph trajectories in separate processes and compare them."""
    import torch

    with tempfile.TemporaryDirectory(prefix="cuda_graph_equivalence_") as tmp_dir:
        eager_path = str(Path(tmp_dir) / "eager.pt")
        graph_path = str(Path(tmp_dir) / "graph.pt")
        _benchmark_in_subprocess(
            {
                **payload,
                "worker_kind": "equivalence",
                "use_cuda_graph": False,
                "output_path": eager_path,
            }
        )
        _benchmark_in_subprocess(
            {
                **payload,
                "worker_kind": "equivalence",
                "use_cuda_graph": True,
                "output_path": graph_path,
            }
        )
        eager_trajectories = torch.load(eager_path, weights_only=False)
        graph_trajectories = torch.load(graph_path, weights_only=False)

    comparison: dict[str, dict[str, Any]] = {}
    for action_mode in payload["action_modes"]:
        comparison[action_mode] = _compare_equivalence_trajectories(
            eager_trajectories[action_mode],
            graph_trajectories[action_mode],
        )
    return comparison


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker-json":
        print(f"{WORKER_RESULT_PREFIX}{json.dumps(_run_worker(json.loads(sys.argv[2])))}")
        return

    args, hydra_args = parse_args()
    hydra_args = _inject_preset(hydra_args, args.preset)
    sys.argv = [sys.argv[0]] + hydra_args
    if "cuda" not in str(args.device):
        raise RuntimeError("CUDA graph stepping benchmarks require a CUDA device.")

    payload = {
        "args": vars(args),
        "action_modes": args.action_modes,
        "hydra_args": hydra_args,
    }

    if args.check_equivalence:
        comparison = _run_equivalence_check(payload)
        print(f"[equivalence] task={args.task}, num_envs={args.num_envs}, steps={args.equivalence_steps}")
        for action_mode in args.action_modes:
            _print_equivalence_result(action_mode, comparison[action_mode])
        return

    if args.use_cuda_graph is not None:
        label = "graph" if args.use_cuda_graph == "true" else "eager"
        result = _run_worker({**payload, "use_cuda_graph": args.use_cuda_graph == "true"})
        print(f"[benchmark] task={args.task}, num_envs={args.num_envs}, mode={label}")
        for action_mode in args.action_modes:
            _print_mode_result(f"{action_mode} / {label}", result[action_mode])
        return

    eager = _benchmark_in_subprocess({**payload, "use_cuda_graph": False})
    graph = _benchmark_in_subprocess({**payload, "use_cuda_graph": True})

    print(f"[benchmark] task={args.task}, num_envs={args.num_envs}, measure_steps={args.measure_steps}")
    for action_mode in args.action_modes:
        _print_mode_result(f"{action_mode} / eager", eager[action_mode])
        _print_mode_result(f"{action_mode} / graph", graph[action_mode])
        speedup = eager[action_mode]["seconds_mean"] / max(graph[action_mode]["seconds_mean"], 1e-6)
        print(f"  graph speedup:   {speedup:.2f}x")


if __name__ == "__main__":
    main()
