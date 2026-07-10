# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test for NeRD reset synchronization.

The test verifies that resetting to the same explicit generalized state produces
the same neural-model inputs, even after intermediate steps have changed solver
history and caches.
"""

from __future__ import annotations

import argparse
import copy
import sys
from collections.abc import Mapping
from typing import Any

import isaaclab_neural.envs  # noqa: F401 - registers built-in NeRD eval tasks
import torch
from isaaclab_neural.physics import NerdNewtonCfg, NerdSolverCfg
from isaaclab_neural.utils.checkpoint import get_cfg_from_checkpoint, load_checkpoint

from isaaclab_tasks.utils.hydra import hydra_task_config


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse CLI arguments and return remaining Hydra overrides."""
    parser = argparse.ArgumentParser(description="Run a reset determinism smoke test for a NeRD task.")
    parser.add_argument("--task", type=str, default="Isaac-Cartpole-NeRD-v0", help="Registered gym task id.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional NeRD checkpoint override.")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel envs.")
    parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
    parser.add_argument("--repeat-steps", type=int, default=5, help="Steps to run between repeated resets.")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance for tensor comparisons.")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for tensor comparisons.")
    parser.add_argument("--contact-mode", choices=["fixed_ground", "newton_native"], default=None)
    parser.add_argument("--num-contacts-per-env", type=int, default=None)

    from isaaclab_tasks.utils import add_launcher_args

    add_launcher_args(parser)
    return parser.parse_known_args()


def apply_solver_overrides(solver_cfg: NerdSolverCfg, args: argparse.Namespace) -> None:
    """Apply CLI solver overrides to an existing solver config."""
    if args.contact_mode is not None:
        solver_cfg.contact_mode = args.contact_mode
    if args.num_contacts_per_env is not None:
        solver_cfg.num_contacts_per_env = args.num_contacts_per_env


def build_solver_cfg(args: argparse.Namespace) -> NerdSolverCfg | None:
    """Build a NeRD solver config from ``--checkpoint`` when provided."""
    if args.checkpoint is None:
        return None

    checkpoint = load_checkpoint(args.checkpoint, device="cpu")
    cfg = get_cfg_from_checkpoint(checkpoint, args.checkpoint)
    neural_solver_cfg = dict(cfg["env"]["neural_solver_cfg"])
    neural_solver_cfg.pop("use_cuda_graph", None)
    neural_solver_cfg["neural_model_path"] = args.checkpoint
    solver_cfg = NerdSolverCfg(**neural_solver_cfg)
    apply_solver_overrides(solver_cfg, args)
    return solver_cfg


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

    physics_cfg = getattr(getattr(env_cfg, "sim", None), "physics", None)
    if args.checkpoint is None and isinstance(physics_cfg, NerdNewtonCfg):
        apply_solver_overrides(physics_cfg.solver_cfg, args)


def build_launch_cfg(env_cfg):
    """Build a launch-only config that ``launch_simulation`` recognizes as Newton."""
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


def clone_tensors(value: Any) -> Any:
    """Clone tensors in a nested dict/list/tuple structure."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: clone_tensors(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(clone_tensors(item) for item in value)
    if isinstance(value, list):
        return [clone_tensors(item) for item in value]
    return copy.deepcopy(value)


def assert_tensors_close(actual: Any, expected: Any, *, path: str = "value", atol: float, rtol: float) -> None:
    """Recursively compare tensor structures."""
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
            diff = (actual - expected).abs().max().item()
            raise AssertionError(f"{path} mismatch: max_diff={diff:.6e}, shape={tuple(actual.shape)}")
        return

    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        if set(actual) != set(expected):
            raise AssertionError(f"{path} keys mismatch: actual={sorted(actual)}, expected={sorted(expected)}")
        for key in actual:
            assert_tensors_close(actual[key], expected[key], path=f"{path}.{key}", atol=atol, rtol=rtol)
        return

    if isinstance(actual, (tuple, list)) and isinstance(expected, type(actual)):
        if len(actual) != len(expected):
            raise AssertionError(f"{path} length mismatch: actual={len(actual)}, expected={len(expected)}")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            assert_tensors_close(actual_item, expected_item, path=f"{path}[{index}]", atol=atol, rtol=rtol)
        return

    if actual != expected:
        raise AssertionError(f"{path} mismatch: actual={actual!r}, expected={expected!r}")


def zero_action(env) -> torch.Tensor:
    """Return a zero action tensor with the shape expected by ``env.step``."""
    shape = env.action_space.shape
    if len(shape) == 1:
        shape = (getattr(env, "num_envs", 1), shape[0])
    return torch.zeros(shape, device=getattr(env, "device", "cpu"), dtype=torch.float32)


def print_contact_summary(env) -> None:
    """Print fixed-ground abstract contact counts when available."""
    manager = env.neural_adapter.manager
    abstract_contact = getattr(manager, "_nerd_abstract_contacts", None)
    if abstract_contact is None:
        return

    model = getattr(manager, "_model", None)
    counts = abstract_contact.count_contacts_by_body_world_per_env(model)
    print(
        "[contacts] "
        f"mode={getattr(manager, '_nerd_contact_mode', None)}, "
        f"num_contacts_per_env={abstract_contact.num_contacts_per_env}, "
        f"per_env_counts={counts}"
    )


def run_reset_smoke(env, args: argparse.Namespace) -> None:
    """Run the reset determinism smoke test."""
    adapter = env.neural_adapter

    env.reset()
    adapter.sync(update_history=False)
    initial_states = adapter.states_torch.detach().clone()

    adapter.reset(initial_states=initial_states)
    inputs_a = clone_tensors(adapter.get_neural_model_inputs())
    root_a = adapter.root_body_q_torch.detach().clone()

    action = zero_action(env)
    for _ in range(args.repeat_steps):
        env.step(action)

    adapter.reset(initial_states=initial_states)
    inputs_b = clone_tensors(adapter.get_neural_model_inputs())
    root_b = adapter.root_body_q_torch.detach().clone()

    assert_tensors_close(root_b, root_a, path="root_body_q", atol=args.atol, rtol=args.rtol)
    assert_tensors_close(inputs_b, inputs_a, path="neural_inputs", atol=args.atol, rtol=args.rtol)

    run_partial_history_reset_smoke(env, args)

    print(
        "[OK] reset smoke passed: "
        f"task={args.task}, num_envs={args.num_envs}, repeat_steps={args.repeat_steps}, "
        f"atol={args.atol}, rtol={args.rtol}"
    )


def run_partial_history_reset_smoke(env, args: argparse.Namespace) -> None:
    """Check that per-env history reset leaves non-reset env rows untouched."""
    solver = env.neural_adapter.solver
    if not hasattr(solver, "states_history"):
        print("[SKIP] partial history reset smoke: active solver has no states_history.")
        return
    if env.num_envs < 2:
        print("[SKIP] partial history reset smoke: requires num_envs >= 2.")
        return

    action = zero_action(env)
    for _ in range(max(1, args.repeat_steps)):
        env.step(action)

    if len(solver.states_history) == 0:
        print("[SKIP] partial history reset smoke: states_history is empty.")
        return

    history_before = clone_tensors(list(solver.states_history))
    reset_env_ids = torch.arange(0, env.num_envs, 2, device=env.device, dtype=torch.long)
    keep_env_ids = torch.arange(1, env.num_envs, 2, device=env.device, dtype=torch.long)

    env.neural_adapter.reset_history(reset_env_ids)
    history_after = list(solver.states_history)

    for step_idx, (before_entry, after_entry) in enumerate(zip(history_before, history_after)):
        for key, before_value in before_entry.items():
            after_value = after_entry[key]
            if keep_env_ids.numel() > 0:
                assert_tensors_close(
                    after_value[keep_env_ids],
                    before_value[keep_env_ids],
                    path=f"partial_history[{step_idx}].{key}.kept_envs",
                    atol=args.atol,
                    rtol=args.rtol,
                )
            expected_reset = torch.zeros_like(after_value[reset_env_ids])
            assert_tensors_close(
                after_value[reset_env_ids],
                expected_reset,
                path=f"partial_history[{step_idx}].{key}.reset_envs",
                atol=args.atol,
                rtol=args.rtol,
            )

    print(
        "[OK] partial history reset smoke passed: "
        f"reset_env_ids={reset_env_ids.detach().cpu().tolist()}, "
        f"kept_env_ids={keep_env_ids.detach().cpu().tolist()}"
    )


args_cli, hydra_args = parse_args()
sys.argv = [sys.argv[0], "presets=newton"] + hydra_args


@hydra_task_config(args_cli.task, "")
def main(env_cfg, agent_cfg) -> None:
    """Launch the task and run reset determinism checks."""
    del agent_cfg
    configure_env(env_cfg, args_cli)
    solver_cfg = build_solver_cfg(args_cli)

    from isaaclab_tasks.utils import launch_simulation

    with launch_simulation(build_launch_cfg(env_cfg), args_cli):
        import gymnasium as gym

        gym_kwargs = {"cfg": env_cfg, "device": args_cli.device}
        if solver_cfg is not None:
            gym_kwargs["solver_cfg"] = solver_cfg

        env = gym.make(args_cli.task, **gym_kwargs).unwrapped
        try:
            print_contact_summary(env)
            run_reset_smoke(env, args_cli)
        finally:
            env.close()


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
