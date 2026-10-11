# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based MBPO worker dispatch and terrain/observation restoration smoke checks."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from isaaclab_neural.data.opd import file_sha256
from isaaclab_neural.eval.dynamics_consistency import state_errors
from isaaclab_neural.generate import opd as launcher
from isaaclab_neural.rl.rsl_rl.matched_mj import MatchedGeometryBridge, _evaluate_study_policy, mdp_contract
from isaaclab_neural.rl.rsl_rl.mbpo import (
    FastHistory,
    capture_branch_scene,
    capture_manager_state,
    initialize_branch_scene,
    project_history,
    refresh_branch_sensors,
    restore_branch_scene,
    restore_manager_state,
    write_history_rows,
    write_physical_rows,
)
from isaaclab_neural.rl.rsl_rl.mbpo_loop import _build, atomic_json, run_role
from isaaclab_neural.rl.rsl_rl.mbpo_model import run_model_fit
from isaaclab_neural.utils.python_utils import set_random_seed


def run_worker(env, adapter, args) -> None:
    """Dispatch a portable worker using the shared manager-based G1 factory."""
    filename = args.active_fit_config if args.stage == "active-stage-fit" else args.rl_study_config
    config = json.loads(Path(filename).read_text())
    if file_sha256(args.dynamics) != config["dynamics_sha256"]:
        raise ValueError("MBPO worker's flat pretrained dynamics changed.")
    if config["terrain"] != args.terrain:
        raise ValueError("Worker terrain selection differs from the shared configuration.")
    if config["role"] == "probe_mj":
        _probe_mj(env, adapter, args, config)
    elif config["role"] == "probe_nerd":
        _probe_nerd(env, adapter, args, config)
    elif config["role"] == "model_fit":
        run_model_fit(env, args, config)
    elif config["role"] == "evaluate":
        _run_evaluation(env, adapter, args, config)
    else:
        run_role(env, adapter, args, config)


@torch.no_grad()
def _run_evaluation(env, adapter, args, config: dict) -> None:
    base = env.unwrapped
    if (
        args.backend != "mjwarp"
        or config.get("updates") != 0
        or config.get("mode") != "eval"
        or config["contact_mdp"] not in {"stock", "geometry"}
        or not 1 <= config["eval_steps"] <= 1000
        or not 4 <= base.num_envs <= 512
        or config["num_envs"] != base.num_envs
    ):
        raise ValueError("Rough policy evaluation requires an explicit bounded MJWarp/zero-update configuration.")
    if file_sha256(args.policy_snapshot) != config["policy_sha256"]:
        raise ValueError("Evaluation policy changed after its checkpoint was selected.")
    if config["contact_mdp"] == "stock" and base.cfg.scene.contact_forces is None:
        raise ValueError("Stock rough evaluation must retain the contact-force sensor.")
    initialize_branch_scene(base)
    wrapped, runner, runner_cfg = _build(env, args, require_initial=False)
    bridge = MatchedGeometryBridge(base, adapter) if config["contact_mdp"] == "geometry" else None
    set_random_seed(config["seed"])
    try:
        obs, _ = wrapped.reset()
        if bridge is not None:
            bridge.refresh()
        results = _evaluate_study_policy(base, runner, obs, config["eval_steps"])
        atomic_json(
            {
                "evaluation": results,
                "contact_mdp": config["contact_mdp"],
                "physics_events": config["physics_events"],
                "ppo_updates": 0,
                "policy_sha256": config["policy_sha256"],
                "ppo_cfg": runner_cfg,
                "mdp": mdp_contract(base.cfg),
                "terrain_context": base.mbpo_terrain_context,
            },
            args.output / "report.json",
        )
    finally:
        if bridge is not None:
            bridge.close()


@torch.no_grad()
def _probe_mj(env, adapter, args, config) -> None:
    base = env.unwrapped
    initialize_branch_scene(base)
    bridge = MatchedGeometryBridge(base, adapter)
    recorder = FastHistory(base, adapter)
    wrapped, runner, _ = _build(env, args)
    set_random_seed(config["seed"])
    obs, _ = wrapped.reset()
    bridge.refresh()
    policy = runner.get_inference_policy(device=str(base.device))
    try:
        for _ in range(8):
            obs, _, done, _ = wrapped.step(policy(obs, stochastic_output=False))
            recorder.finish_control_step(done)
            policy.reset(done)
        ids = recorder.complete()
        if len(ids) < 2:
            raise RuntimeError("Rough smoke could not obtain two valid complete real histories.")
        batch = recorder.gather(ids)
        batch["manager"] = capture_manager_state(base, ids)
        batch["obs"] = obs["policy"].index_select(0, ids).cpu().clone()
        batch["source_env_ids"] = ids.cpu()
        batch["policy_sha256"] = file_sha256(args.policy_snapshot)
        batch["purpose"] = "branch_restore_smoke_only"
        actions, active = [], torch.ones(len(ids), dtype=torch.bool, device=base.device)
        for _ in range(4):
            action = policy(obs, stochastic_output=False)
            actions.append(action.index_select(0, ids).cpu().clone())
            obs, _, done, _ = wrapped.step(action)
            active &= ~done.index_select(0, ids).bool()
            recorder.finish_control_step(done)
            policy.reset(done)
        batch["actions"] = torch.stack(actions, 1)
        batch["oracle_states"] = torch.stack(
            [frame["observed_next_state"].index_select(0, ids).cpu() for frame in recorder.frames[-16:]], 1
        )
        batch["valid_future"] = active.cpu()
        torch.save(batch, args.output / "capture.pt")
        push = _verify_stock_push(base, adapter, recorder)
        atomic_json(
            {
                "captured_worlds": len(ids),
                "complete_future_worlds": int(active.sum()),
                "terrain_context": batch["scene"]["context"],
                "push_check": push,
                "training_data": False,
            },
            args.output / "report.json",
        )
    finally:
        recorder.close()
        bridge.close()


@torch.no_grad()
def _probe_nerd(env, adapter, args, config) -> None:
    base = env.unwrapped
    initialize_branch_scene(base)
    batch = torch.load(config["capture"], map_location="cpu", weights_only=False)
    if batch["purpose"] != "branch_restore_smoke_only" or batch["policy_sha256"] != file_sha256(args.policy_snapshot):
        raise ValueError("Rough restore probe changed policy or data role.")
    count = len(batch["current_state"])
    if count > base.num_envs:
        raise ValueError("Rough restore smoke needs enough destination worlds.")
    wrapped, _, _ = _build(env, args)
    solver = base.neural_adapter.solver
    history = project_history(batch, solver)
    original = base._reset_idx

    def branch_reset(env_ids):
        original(env_ids)
        ids = torch.arange(count, device=base.device)
        restore_branch_scene(base, ids, batch["scene"], refresh_sensors=False)
        write_physical_rows(base, ids, batch["current_state"].to(base.device))
        write_history_rows(solver, ids, history)
        restore_manager_state(base, batch["manager"], ids, torch.arange(count))
        refresh_branch_sensors(base, ids)
        base.episode_length_buf[ids] = 0

    base._reset_idx = branch_reset
    obs, _ = wrapped.reset()
    base._reset_idx = original
    state_error = float((solver.states[:count].cpu() - batch["current_state"]).abs().max())
    observation_error = float((obs["policy"][:count].cpu() - batch["obs"]).abs().max())
    ids = torch.arange(count, device=base.device)
    restored_scene = capture_branch_scene(base, ids)
    restored_manager = capture_manager_state(base, ids)
    scene_restored = all(
        torch.equal(value, restored_scene["values"][key]) for key, value in batch["scene"]["values"].items()
    )
    managers_restored = all(
        torch.equal(value, restored_manager["objects"][path][key])
        for path, tensors in batch["manager"]["objects"].items()
        for key, value in tensors.items()
    )
    history_restored = (
        all(
            torch.equal(frame[key][:count], value[:, step].to(frame[key]))
            for step, frame in enumerate(solver.states_history)
            for key, value in history.items()
        )
        and len(solver.states_history) == solver.num_states_history
    )
    report = {
        "worlds": count,
        "state_absolute_max": state_error,
        "observation_absolute_max": observation_error,
        "terrain_context_matches": base.mbpo_terrain_context == batch["scene"]["context"],
        "state_restored": state_error <= 1e-5,
        "observations_restored": observation_error <= 1e-4,
        "history_restored": history_restored,
        "scene_and_physics_restored": scene_restored,
        "manager_and_event_state_restored": managers_restored,
        "model_updates": 0,
        "policy_updates": 0,
        "training_data": False,
    }
    atomic_json(report, args.output / "restore_report.json")
    if not all(
        report[key]
        for key in (
            "state_restored",
            "observations_restored",
            "history_restored",
            "terrain_context_matches",
            "scene_and_physics_restored",
            "manager_and_event_state_restored",
        )
    ):
        raise RuntimeError("Rough branch state/history/height-scan restoration failed; training was not started.")
    predictions = []
    keep = batch["valid_future"].bool().clone()
    for control in range(4):
        action = torch.zeros((base.num_envs, batch["actions"].shape[-1]), device=base.device)
        action[:count] = batch["actions"][:, control].to(base.device)
        _, _, done, _ = wrapped.step(action)
        keep &= ~done[:count].cpu().bool()
        predictions.append(base.neural_adapter.solver.states[:count].cpu().clone())
    report["paired_complete_future_worlds"] = int(keep.sum())
    if keep.any():
        report["zero_shot_80ms_errors"] = {
            key: float(value.mean())
            for key, value in state_errors(predictions[-1][keep], batch["oracle_states"][keep, 15], 44).items()
        }
    report["push_check"] = _verify_stock_push(base, adapter)
    report["note"] = (
        "State/observation/physics/event restoration is gated; "
        "flat-to-rough prediction errors are measured, not presumed accurate."
    )
    atomic_json(report, args.output / "report.json")


@torch.no_grad()
def _verify_stock_push(base, adapter, recorder: FastHistory | None = None) -> dict:
    if not getattr(base.cfg, "mbpo_randomized_physics", False):
        return {"enabled": False}
    manager = base.event_manager
    index = manager._mode_term_names["interval"].index("push_robot")
    valid = recorder.complete() if recorder is not None else torch.arange(base.num_envs, device=base.device)
    if not len(valid):
        raise RuntimeError("Push verification requires a complete pre-push history.")
    target = int(valid[0])
    solver = adapter.solver
    before, after = torch.empty_like(solver.states), torch.empty_like(solver.states)
    solver._acquire_states_to_torch(base.sim.physics_manager._state_0, before)
    manager._interval_term_time_left[index][target] = 0.0
    manager.apply(mode="interval", dt=base.step_dt)
    solver._acquire_states_to_torch(base.sim.physics_manager._state_0, after)
    velocity = slice(solver.dof_q_per_env, solver.dof_q_per_env + 3)
    changed = float((after[target, velocity] - before[target, velocity]).abs().max())
    if not torch.isfinite(after).all() or changed <= 1e-5:
        raise RuntimeError("Stock push event did not update a finite physical root velocity.")
    report = {"enabled": True, "root_velocity_change_mps": changed, "training_data": False}
    if recorder is not None:
        recorder.finish_control_step(torch.zeros(base.num_envs, dtype=torch.bool, device=base.device))
        report["crossing_model_window_excluded"] = bool(
            recorder.last_external_jump[target] and recorder.valid[target] == 0
        )
        if not report["crossing_model_window_excluded"]:
            raise RuntimeError("Unrecorded push discontinuity was retained as a dynamics target.")
    return report


if __name__ == "__main__":
    launcher._run_extended_worker = run_worker
    launcher.main()
