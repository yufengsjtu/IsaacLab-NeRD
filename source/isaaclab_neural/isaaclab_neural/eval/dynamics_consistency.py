# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical-unit metrics for matched-input G1 dynamics queries."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch


def state_errors(predicted: torch.Tensor, target: torch.Tensor, num_q: int) -> dict[str, torch.Tensor]:
    """Return per-sample SI errors, without mixing positions and velocities in one MSE.

    Inputs have shape [..., state_dim], positions [m or rad], velocities [m/s or rad/s],
    and a free-root quaternion in xyzw order. Quaternion sign is immaterial.
    """
    q1 = torch.nn.functional.normalize(predicted[..., 3:7], dim=-1)
    q2 = torch.nn.functional.normalize(target[..., 3:7], dim=-1)
    q2 = q2 * torch.where((q1 * q2).sum(-1, keepdim=True) < 0, -1, 1)
    joint_delta = predicted[..., 7:num_q] - target[..., 7:num_q]
    joint_delta = torch.atan2(torch.sin(joint_delta), torch.cos(joint_delta))
    return {
        "root_position_m": (predicted[..., :3] - target[..., :3]).norm(dim=-1),
        "root_orientation_rad": 4 * torch.atan2((q1 - q2).norm(dim=-1), (q1 + q2).norm(dim=-1)),
        "root_linear_velocity_mps": (predicted[..., num_q : num_q + 3] - target[..., num_q : num_q + 3]).norm(dim=-1),
        "root_angular_velocity_radps": (
            predicted[..., num_q + 3 : num_q + 6] - target[..., num_q + 3 : num_q + 6]
        ).norm(dim=-1),
        "joint_position_rmse_rad": joint_delta.square().mean(dim=-1).sqrt(),
        "joint_velocity_rmse_radps": (predicted[..., num_q + 6 :] - target[..., num_q + 6 :])
        .square()
        .mean(dim=-1)
        .sqrt(),
    }


def summarize_errors(values: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    """Summarize error vectors, explicitly counting non-finite predictions."""
    result = {}
    for key, value in values.items():
        flat = value.detach().float().flatten()
        finite = torch.isfinite(flat)
        result[key] = {"count": flat.numel(), "non_finite": int((~finite).sum())}
        if finite.any():
            valid = flat[finite]
            result[key].update(
                mean=float(valid.mean()), p95=float(torch.quantile(valid, 0.95)), maximum=float(valid.max())
            )
    return result


def checkpoint_update_audit(reference_path: str, candidate_path: str) -> dict:
    """Check that a candidate updated weights, not frozen statistics or old optimizer progress."""
    from isaaclab_neural.utils.checkpoint import load_checkpoint

    reference = load_checkpoint(reference_path, device="cpu")
    candidate = load_checkpoint(candidate_path, device="cpu")
    old, new = reference["model_state_dict"], candidate["model_state_dict"]
    if set(old) != set(new):
        raise ValueError("Dynamics checkpoint structures differ.")
    stats = [key for key in old if key.startswith(("input_rms.", "output_rms.", "body_net_contact_force_rms."))]
    changed = [key for key in old if key not in stats and not torch.equal(old[key], new[key])]
    steps = [int(state["step"]) for state in candidate["optimizer_state_dict"]["state"].values() if "step" in state]
    return {
        "changed_weight_tensors": len(changed),
        "rms_tensor_count": len(stats),
        "rms_unchanged": bool(stats) and all(torch.equal(old[key], new[key]) for key in stats),
        "loss_weights_unchanged": torch.equal(reference["loss_weights"], candidate["loss_weights"]),
        "epoch": candidate["epoch"],
        "optimizer_min_step": min(steps),
        "optimizer_max_step": max(steps),
        "optimizer_learning_rates": [group["lr"] for group in candidate["optimizer_state_dict"]["param_groups"]],
    }


def compare_evaluations(
    baseline: dict, candidate: dict, *, improvement: float, regression: float, tolerances: dict
) -> dict:
    """Apply a fixed mean/P95 gate to identical oracle queries, not training loss."""
    before = sorted((s["sha256"], s["queries"], s["horizon_physics_steps"]) for s in baseline["shards"])
    after = sorted((s["sha256"], s["queries"], s["horizon_physics_steps"]) for s in candidate["shards"])
    if before != after:
        raise ValueError("Gate requires identical held-out query shards and horizons.")
    result = {}
    for horizon in ("one_step", "held_target_rollout"):
        result[horizon] = {}
        for metric, absolute in tolerances.items():
            old, new = baseline["aggregate"][horizon][metric], candidate["aggregate"][horizon][metric]
            if old["non_finite"] or new["non_finite"] or not old["count"] or not new["count"]:
                result[horizon][metric] = {"pass": False, "reason": "Non-finite or empty error population"}
                continue
            required_gain = improvement if horizon == "held_target_rollout" else 0.0
            ceiling = (
                old["mean"] * (1 - required_gain)
                if required_gain > 0
                else old["mean"] + max(old["mean"] * regression, absolute)
            )
            p95_ceiling = old["p95"] + max(old["p95"] * regression, absolute)
            result[horizon][metric] = {
                "baseline_mean": old["mean"],
                "candidate_mean": new["mean"],
                "baseline_p95": old["p95"],
                "candidate_p95": new["p95"],
                "mean_change_fraction": (new["mean"] - old["mean"]) / max(old["mean"], 1e-12),
                "pass": new["non_finite"] == 0 and new["mean"] <= ceiling and new["p95"] <= p95_ceiling,
            }
    return {"metrics": result, "passed": all(v["pass"] for h in result.values() for v in h.values())}


@torch.no_grad()
def evaluate_contexts(env, adapter, paths: list[str], output: Path, *, dataset_type=None, on_context=None) -> dict:
    """Compare NeRD with continuous MJ reference rollouts under a held recorded target.

    Histories are restored without padding. The rollout is a matched-input dynamics
    experiment, not a closed-loop policy evaluation or a replay of future PPO actions.
    """
    import newton
    import warp as wp
    from isaaclab_neural.data.opd import ContextDataset, file_sha256
    from isaaclab_neural.generate.opd import simulation_protocol

    solver = env.neural_adapter.solver
    protocol = simulation_protocol(env)
    model = env.sim.physics_manager._model
    fk_state = model.state()
    foot_ids = [
        i
        for i, name in enumerate(model.body_label)
        if str(name).endswith(("left_ankle_roll_link", "right_ankle_roll_link"))
    ]
    if len(foot_ids) != env.num_envs * 2:
        raise ValueError("Expected two G1 ankle frames per environment.")
    feet = torch.tensor(foot_ids, device=env.device)
    leg_ids = torch.tensor(
        [i for i, name in enumerate(protocol["joint_names"]) if any(x in name for x in ("hip", "knee", "ankle"))],
        device="cpu",
    )

    def foot_positions(states):
        adapter.solver._assign_states_from_torch(fk_state, states)
        newton.eval_fk(model, fk_state.joint_q, fk_state.joint_qd, fk_state)
        return wp.to_torch(fk_state.body_q).index_select(0, feet)[:, :3].reshape(env.num_envs, 2, 3).clone()

    all_errors = {"one_step": {}, "held_target_rollout": {}}
    reports = []
    for filename in paths:
        with h5py.File(filename, "r") as file:
            metadata = json.loads(file.attrs["opd_metadata"])
            if metadata["protocol"] != protocol or metadata.get("oracle_action_protocol") != "hold_recorded_target":
                raise ValueError("Evaluation requires a matching held-target oracle protocol.")
            truth = torch.from_numpy(np.asarray(file["opd/oracle_rollout_states"]))
            actions = torch.from_numpy(np.asarray(file["data/actions"][:, -1]))
            old_student = torch.from_numpy(np.asarray(file["opd/student_next_states"]))
        dataset = (dataset_type or ContextDataset)(
            [filename],
            split=metadata["split"],
            physics_dt=env.physics_dt,
            contact_filter=solver.contact_filter,
            max_contact_tokens=solver.max_contact_tokens,
        )
        count, horizon, _ = truth.shape
        predicted, foot_error = [], []
        for start in range(0, count, env.num_envs):
            end = min(start + env.num_envs, count)
            ids = list(range(start, end)) + [end - 1] * (env.num_envs - (end - start))
            samples = [dataset[i] for i in ids]
            history = {
                key: torch.stack([sample[key] for sample in samples]).to(env.device)
                for key in samples[0]
                if key not in {"next_states", "supervision_mask"}
            }
            state = history["states"][:, -1]
            env.scene["robot"].actuators.reset()
            env.action_manager.reset()
            adapter.backend.state = env.sim.physics_manager._state_0
            adapter.control.joint_f.zero_()
            adapter.reset(state)
            solver.reset_states_history()
            if history["states"].shape[1] > 1:
                solver.preload_states_history({key: value[:, :-1] for key, value in history.items()})
            if on_context is not None:
                on_context(history)
            action = actions[ids].to(env.device)
            trajectory = torch.full((env.num_envs, horizon, state.shape[-1]), float("nan"))
            foot_distances = torch.full((env.num_envs, horizon), float("nan"))
            for step in range(horizon):
                adapter.step_action_frame(action, process_action=step == 0, overlay_applied_effort=False)
                future = adapter.solver.states.detach().clone()
                trajectory[:, step] = future.cpu()
                if not torch.isfinite(future).all():
                    break
                delta = foot_positions(future) - foot_positions(truth[ids, step].to(env.device))
                foot_distances[:, step] = delta.norm(dim=-1).mean(dim=-1).cpu()
            predicted.append(trajectory[: end - start])
            foot_error.append(foot_distances[: end - start])
        predicted = torch.cat(predicted)
        foot_error = torch.cat(foot_error)
        report = {
            "source": str(filename),
            "sha256": file_sha256(filename),
            "queries": count,
            "history_length": dataset.lengths[0],
            "horizon_physics_steps": horizon,
            "action_protocol": "hold_recorded_target",
            "metrics": {},
        }
        for name, step in (("one_step", 0), ("held_target_rollout", horizon - 1)):
            metrics = state_errors(predicted[:, step], truth[:, step], solver.dof_q_per_env)
            delta_vel = predicted[:, step, solver.dof_q_per_env + 6 :] - truth[:, step, solver.dof_q_per_env + 6 :]
            metrics["leg_velocity_rmse_radps"] = delta_vel.index_select(-1, leg_ids).square().mean(-1).sqrt()
            metrics["ankle_frame_position_m"] = foot_error[:, step]
            report["metrics"][name] = summarize_errors(metrics)
            for key, value in metrics.items():
                all_errors[name].setdefault(key, []).append(value)
        report["student_reproduction_state_abs_max"] = float((predicted[:, 0] - old_student).abs().max())
        np.savez_compressed(
            output / (Path(filename).stem + "_predictions.npz"), predicted=predicted.numpy(), target=truth.numpy()
        )
        reports.append(report)
    return {
        "shards": reports,
        "aggregate": {
            name: summarize_errors({key: torch.cat(value) for key, value in metrics.items()})
            for name, metrics in all_errors.items()
        },
    }


def main() -> None:
    """Audit a candidate checkpoint or apply a previously specified evaluation gate."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("checkpoint", "gate"), required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--gate-config", type=Path)
    parser.add_argument("--kind", choices=("current", "reference"), default="current")
    parser.add_argument("--expected-updates", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.stage == "checkpoint":
        report = checkpoint_update_audit(str(args.baseline), str(args.candidate))
        report["passed"] = (
            report["rms_unchanged"]
            and report["loss_weights_unchanged"]
            and report["changed_weight_tensors"] > 0
            and report["optimizer_max_step"] == args.expected_updates
        )
    else:
        if args.gate_config is None:
            parser.error("--gate-config is required for an evaluation gate.")
        gate = json.loads(args.gate_config.read_text())
        report = compare_evaluations(
            json.loads(args.baseline.read_text()),
            json.loads(args.candidate.read_text()),
            improvement=gate["current_distribution_horizon_mean_improvement"] if args.kind == "current" else 0,
            regression=gate["current_distribution_p95_relative_regression_limit"]
            if args.kind == "current"
            else gate["reference_relative_regression_limit"],
            tolerances=gate["absolute_numeric_tolerances"],
        )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
