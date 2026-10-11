# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded synchronous PPO/OPD pilot with persistent environments and isolated oracle workers."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from isaaclab_neural.data.opd import ContextDataset, file_sha256
from isaaclab_neural.eval.dynamics_consistency import checkpoint_update_audit, compare_evaluations
from isaaclab_neural.eval.learnability import _clone_batch, _prepared
from isaaclab_neural.train.trainers import masked_dynamics_loss, validate_finetune_checkpoint
from isaaclab_neural.utils.checkpoint import (
    get_cfg_from_checkpoint,
    load_checkpoint,
    load_opd_policy_snapshot,
    reconstruct_model_from_checkpoint,
    save_checkpoint,
    save_opd_policy_snapshot,
)


def decide_candidate(current_before, current_after, reference_before, reference_after, config):
    """Apply current-distribution gates and explicit enforced/monitored reference retention."""
    reference_mode = config.get("reference_mode", "enforce")
    if reference_mode not in {"enforce", "monitor"}:
        raise ValueError("reference_mode must be 'enforce' or 'monitor'.")
    current = compare_evaluations(
        current_before,
        current_after,
        improvement=0,
        regression=config["current_regression"],
        tolerances=config["numeric_tolerances"],
    )
    reference = compare_evaluations(
        reference_before,
        reference_after,
        improvement=0,
        regression=config["reference_regression"],
        tolerances=config["numeric_tolerances"],
    )
    gain = any(
        row.get("mean_change_fraction", 0) <= -config["minimum_gain"]
        for row in current["metrics"]["held_target_rollout"].values()
    )
    reference_valid = all(
        np.isfinite(row.get(key, float("nan")))
        for horizon in reference["metrics"].values()
        for row in horizon.values()
        for key in ("baseline_mean", "candidate_mean", "baseline_p95", "candidate_p95")
    )
    reference_allowed = reference["passed"] if reference_mode == "enforce" else reference_valid
    return {
        "accepted": bool(current["passed"] and reference_allowed and gain),
        "strict_accepted": bool(current["passed"] and reference["passed"] and gain),
        "reference_mode": reference_mode,
        "reference_metrics_valid": bool(reference_valid),
        "current": current,
        "reference_vs_F_pre": reference,
        "meaningful_current_gain": gain,
    }


def _write_json(path: Path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def _write_status(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    temporary.replace(path)


def recover_runner_config(saved: dict, initial: dict) -> tuple[dict, list[str]]:
    """Restore constructor-consumed metadata from a verified original configuration, without changing weights."""
    result = copy.deepcopy(saved)
    recovered = []

    def fill(current, original, prefix=""):
        for key, value in original.items():
            path = f"{prefix}{key}"
            if key not in current:
                current[key] = copy.deepcopy(value)
                recovered.append(path)
            elif isinstance(value, dict) and isinstance(current[key], dict):
                fill(current[key], value, path + ".")
            elif current[key] != value and key != "device":
                raise ValueError(f"Saved runner config differs from original at {path}; cannot recover blindly.")

    fill(result, initial)
    return result, recovered


def _read_json(path):
    return json.loads(Path(path).read_text())


def _child(arguments: list[str], log: Path):
    command = [sys.executable, "-m", "isaaclab_neural.generate.opd", *arguments]
    _write_json(log.with_suffix(".command.json"), command)
    with log.open("x") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def _evaluation(model_path, queries, output, device):
    _child(
        [
            "--stage",
            "evaluate",
            "--backend",
            "nerd",
            "--dynamics",
            str(model_path),
            "--queries",
            *map(str, queries),
            "--output",
            str(output),
            "--num-envs",
            "4",
            "--device",
            str(device),
        ],
        output.with_suffix(".log"),
    )
    return _read_json(output / "report.json")


def _dataset(paths, solver):
    return ContextDataset(
        paths,
        split="train",
        physics_dt=0.005,
        state_dim=solver.state_dim,
        joint_f_dim=solver.joint_f_dim,
        contact_filter=solver.contact_filter,
        max_contact_tokens=solver.max_contact_tokens,
    )


def fit_online_candidate(env, adapter, dynamics: str, config_path: str, output: Path):
    """Take ten fresh-Adam steps from the active model using reference/current/stage-balanced replay."""
    config = _read_json(config_path)
    if not 1 <= config["updates"] <= 10 or not 0 < config["lr"] <= 1e-5:
        raise ValueError("Online pilot is bounded to ten gradient updates at LR <= 1e-5.")
    solver = adapter.solver
    checkpoint = load_checkpoint(dynamics, device=env.device)
    model = reconstruct_model_from_checkpoint(checkpoint, solver, device=env.device)
    validate_finetune_checkpoint(checkpoint, model, update_statistics=False)
    pools = [_dataset(config["reference"], solver), _dataset(config["current"], solver)]
    replay = [_dataset(stage, solver) for stage in config["replay_stages"]]
    if any(set(pool.lengths) != {10} for pool in [*pools, *replay]):
        raise ValueError("This pilot trains exact H10 anchors; it does not implicitly reweight short histories.")
    weights = checkpoint["loss_weights"].to(env.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    rng = np.random.default_rng(config["seed"])
    trace = hashlib.sha256()
    curve = []
    model.train()
    for step in range(config["updates"]):
        batches = []
        for pool_id, (pool, size) in enumerate(zip(pools, (128, 64))):
            ids = rng.integers(len(pool), size=size)
            batches.append((_prepared(pool, ids, solver, env.device), size))
            trace.update(np.asarray([pool_id, *ids], dtype="<i8").tobytes())
        stage_counts = np.bincount(np.arange(64) % len(replay), minlength=len(replay))
        for stage_id, (pool, size) in enumerate(zip(replay, stage_counts)):
            if not size:
                continue
            ids = rng.integers(len(pool), size=int(size))
            batches.append((_prepared(pool, ids, solver, env.device), int(size)))
            trace.update(np.asarray([stage_id + 2, *ids], dtype="<i8").tobytes())
        optimizer.zero_grad()
        loss_sum = 0.0
        for batch, size in batches:
            prediction = model(_clone_batch(batch))
            loss = masked_dynamics_loss(
                prediction * weights,
                batch["prediction_target"] * weights,
                torch.nn.MSELoss(),
                batch["supervision_mask"],
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite candidate dynamics loss.")
            (loss * (size / 256)).backward()
            loss_sum += float(loss.detach()) * size / 256
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        curve.append({"update": step + 1, "loss": loss_sum, "gradient_before_clip": float(gradient)})
    cfg = copy.deepcopy(get_cfg_from_checkpoint(checkpoint, dynamics))
    cfg["online_opd_candidate"] = config
    candidate = output / "candidate.pt"
    save_checkpoint(
        candidate,
        model,
        "G1-Native-ContactTokens",
        cfg,
        optimizer_state_dict=optimizer.state_dict(),
        epoch=0,
        loss_weights=weights,
    )
    audit = checkpoint_update_audit(dynamics, str(candidate))
    if (
        not audit["rms_unchanged"]
        or not audit["loss_weights_unchanged"]
        or audit["optimizer_max_step"] != config["updates"]
    ):
        raise RuntimeError("Candidate optimizer/statistics audit failed.")
    _write_json(
        output / "report.json",
        {
            "audit": audit,
            "curve": curve,
            "source_counts_per_update": [128, 64, 64],
            "replay_stage_count": len(replay),
            "sample_trace": trace.hexdigest(),
            "config": config,
        },
    )


def _pause_fingerprint(base, obs):
    import warp as wp

    digest = hashlib.sha256()
    for value in (
        wp.to_torch(base.sim.physics_manager._state_0.joint_q),
        wp.to_torch(base.sim.physics_manager._state_0.joint_qd),
        base.episode_length_buf,
        base.action_manager.action,
        *obs.values(),
    ):
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def _promote_without_reset(solver, candidate, device):
    if solver.use_feature_cache or solver.use_kv_cache:
        raise ValueError("The bounded pilot requires uncached NeRD inference.")
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
    try:
        checkpoint = load_checkpoint(str(candidate), device=device)
        solver.neural_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        solver.neural_model.eval()
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)


def _ppo_update(runner, obs, recorder, iteration: int, *, diagnostics=None):
    from rsl_rl.utils import check_nan

    recorder.policy_update_id = iteration
    runner.alg.train_mode()
    reward_sum = 0.0
    done_count = 0
    with torch.inference_mode():
        for _ in range(runner.cfg["num_steps_per_env"]):
            action = runner.alg.act(obs)
            obs, reward, done, extras = runner.env.step(action.to(runner.env.device))
            check_nan(obs, reward, done)
            obs, reward, done = obs.to(runner.device), reward.to(runner.device), done.to(runner.device)
            recorder.finish_control_step(done)
            runner.alg.process_env_step(obs, reward, done, extras)
            reward_sum += float(reward.mean())
            done_count += int(done.sum())
        runner.alg.compute_returns(obs)
    if diagnostics is None:
        losses = runner.alg.update()
    else:
        consistency = diagnostics.check_consistency(runner.alg)
        if not consistency["passed"]:
            raise RuntimeError(
                "PPO pre-update distribution/log-probability consistency failed; no optimizer step taken."
            )
        with diagnostics.observe_update(runner.alg):
            losses = runner.alg.update()
    if not all(np.isfinite(value) for value in losses.values()):
        raise FloatingPointError("Non-finite PPO loss.")
    runner.current_learning_iteration = iteration + 1
    return obs, {
        "ppo_updates_completed": iteration + 1,
        "reward_mean_per_step": reward_sum / runner.cfg["num_steps_per_env"],
        "done_count": done_count,
        "losses": losses,
        "learning_rate": runner.alg.learning_rate,
    }


def run_online_arm(env, adapter, args):
    """Run one arm without rebuilding/resetting the environment at PPO/OPD block boundaries."""
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_neural.generate.opd import ContextRecorder, simulation_protocol

    base = env.unwrapped
    config = _read_json(args.online_config)
    if config.get("gate", {}).get("reference_mode", "enforce") not in {"enforce", "monitor"}:
        raise ValueError("reference_mode must be 'enforce' or 'monitor'.")
    if config["rounds"] > 3 or config["ppo_updates_per_round"] > 5 or base.num_envs != 8:
        raise ValueError("The pilot is bounded to three rounds, five PPO updates each and eight environments.")
    snapshot = torch.load(args.policy_snapshot, map_location="cpu", weights_only=False)
    protocol = simulation_protocol(base)
    if snapshot["infos"]["opd"]["provenance"]["protocol"] != protocol:
        raise ValueError("Policy snapshot and online simulation protocol differ.")
    runner_cfg = copy.deepcopy(snapshot["infos"]["opd"]["runner_cfg"])
    runner_cfg["device"] = str(base.device)
    del snapshot
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg["clip_actions"])
    runner = OnPolicyRunner(wrapped, copy.deepcopy(runner_cfg), log_dir=None, device=str(base.device))
    load_opd_policy_snapshot(args.policy_snapshot, runner, require_initial=True)
    torch.manual_seed(config["policy_rng_seed"])
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(config["policy_rng_seed"])
    obs = wrapped.get_observations().to(runner.device)
    collector = ContextRecorder(
        base, adapter, capacity=256, selected_envs=base.num_envs, seed=config["sampler_seed"], verify_frames=2
    )
    active = Path(args.dynamics).resolve()
    initial = active
    solver = base.neural_adapter.solver
    solver.neural_model.eval()
    if solver.use_feature_cache or solver.use_kv_cache:
        raise ValueError("Online pilot requires uncached inference.")
    collection_id = f"online-pilot:{args.opd_mode}:{config['env_seed']}"
    replay = [config["initial_replay"]]
    records = []
    policy_paths = {0: str(Path(args.policy_snapshot).resolve())}
    save_opd_policy_snapshot(
        args.output / "policy_start.pt",
        runner,
        runner_cfg,
        updates_completed=0,
        provenance={"protocol": protocol, "source_snapshot_sha256": file_sha256(args.policy_snapshot)},
    )
    try:
        for round_id in range(config["rounds"]):
            folder = args.output / f"round_{round_id:02d}"
            folder.mkdir()
            record = {
                "round": round_id,
                "mode": args.opd_mode,
                "active_before": str(active),
                "active_before_sha256": file_sha256(active),
                "ppo": [],
                "artificial_resets": 0,
            }
            for _ in range(config["ppo_updates_per_round"]):
                iteration = runner.current_learning_iteration
                obs, row = _ppo_update(runner, obs, collector, iteration)
                record["ppo"].append(row)
                path = folder / f"policy_{iteration + 1:04d}.pt"
                save_opd_policy_snapshot(
                    path,
                    runner,
                    runner_cfg,
                    updates_completed=iteration + 1,
                    provenance={
                        "protocol": protocol,
                        "dynamics_hash": file_sha256(active),
                        "arm": args.opd_mode,
                        "round": round_id,
                    },
                )
                policy_paths[iteration + 1] = str(path.resolve())
            meta = {
                "protocol": protocol,
                "physics_dt": base.physics_dt,
                "source_kind": "nerd",
                "pool": "online",
                "collection_id": collection_id,
                "seed": config["env_seed"],
                "stochastic_policy": True,
                "policy_hash": file_sha256(policy_paths[round_id * config["ppo_updates_per_round"]]),
                "policy_snapshots": {str(k): {"path": v, "sha256": file_sha256(v)} for k, v in policy_paths.items()},
                "dynamics_hash": file_sha256(active),
                "label_origin": "unlabelled",
            }
            train_queries = collector.export(
                folder / "train_queries", {**meta, "split": "train"}, env_ids=set(range(6)), lengths={10}
            )
            valid_queries = collector.export(
                folder / "validation_queries", {**meta, "split": "validation"}, env_ids={6, 7}, lengths={10}
            )
            record["query_counts"] = {
                "retained_h10": len(collector.rows[10]),
                "seen_h10": collector.seen[10],
                "train_shards": train_queries,
                "validation_shards": valid_queries,
            }
            collector.rows.clear()
            collector.seen.clear()
            collector.failures.clear()
            before_pause = _pause_fingerprint(base, obs)
            if args.opd_mode == "online":
                for split, queries in (("train", train_queries), ("validation", valid_queries)):
                    _child(
                        [
                            "--stage",
                            "oracle",
                            "--dynamics",
                            str(active),
                            "--queries",
                            *queries,
                            "--output",
                            str(folder / f"{split}_oracle"),
                            "--num-envs",
                            "4",
                            "--device",
                            str(base.device),
                        ],
                        folder / f"{split}_oracle.log",
                    )
                train_paths = [str(p.resolve()) for p in sorted((folder / "train_oracle").glob("context_h*.hdf5"))]
                valid_paths = [str(p.resolve()) for p in sorted((folder / "validation_oracle").glob("context_h*.hdf5"))]
                fit_config = {
                    "updates": config["dynamics_updates"],
                    "lr": config["dynamics_lr"],
                    "seed": config["sampler_seed"] + round_id,
                    "reference": config["reference_train"],
                    "current": train_paths,
                    "replay_stages": replay,
                }
                _write_json(folder / "fit_config.json", fit_config)
                _child(
                    [
                        "--stage",
                        "online-fit",
                        "--dynamics",
                        str(active),
                        "--online-config",
                        str(folder / "fit_config.json"),
                        "--output",
                        str(folder / "fit"),
                        "--num-envs",
                        "4",
                        "--device",
                        str(base.device),
                    ],
                    folder / "fit.log",
                )
                candidate = (folder / "fit/candidate.pt").resolve()
                before = _evaluation(active, valid_paths, folder / "current_before", base.device)
                after = _evaluation(candidate, valid_paths, folder / "current_after", base.device)
                ref_after = _evaluation(
                    candidate, config["reference_validation"], folder / "reference_after", base.device
                )
                reference = _read_json(config["reference_baseline_report"])
                decision = decide_candidate(before, after, reference, ref_after, config["gate"])
                if active != initial:
                    origin = _evaluation(initial, valid_paths, folder / "current_F_pre", base.device)
                    cumulative = compare_evaluations(
                        origin,
                        after,
                        improvement=0,
                        regression=config["gate"]["current_regression"],
                        tolerances=config["gate"]["numeric_tolerances"],
                    )
                    decision["current_vs_F_pre"] = cumulative
                    decision["accepted"] = decision["accepted"] and cumulative["passed"]
                    decision["strict_accepted"] = decision["strict_accepted"] and cumulative["passed"]
                _write_json(folder / "decision.json", decision)
                if decision["accepted"]:
                    _promote_without_reset(solver, candidate, base.device)
                    active = candidate
                replay.append(train_paths)
                record["decision"] = decision
            record["pause_state_and_observations_unchanged"] = before_pause == _pause_fingerprint(base, obs)
            if not record["pause_state_and_observations_unchanged"]:
                raise RuntimeError("Paused state/observations changed during OPD workers or promotion.")
            record["active_after"] = str(active)
            record["active_after_sha256"] = file_sha256(active)
            record["ppo_updates_completed"] = runner.current_learning_iteration
            record["physics_frames_total"] = collector.frame
            record["last_policy"] = policy_paths[runner.current_learning_iteration]
            _write_json(folder / "report.json", record)
            records.append(record)
            print(
                json.dumps(
                    {
                        "round": round_id,
                        "ppo_updates": runner.current_learning_iteration,
                        "accepted": record.get("decision", {}).get("accepted", False),
                    }
                ),
                flush=True,
            )
    finally:
        collector.close()
        _write_json(
            args.output / "report.json",
            {
                "rounds": records,
                "ppo_updates_completed": runner.current_learning_iteration,
                "mode": args.opd_mode,
                "active_model": str(active),
                "verified_callbacks": collector.verified_callbacks,
                "config": config,
            },
        )


@torch.no_grad()
def evaluate_online_policy(env, args):
    """Evaluate the initial or updated policy in MJWarp with fixed commands and no learning."""
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_neural.generate.opd import simulation_protocol
    from isaaclab_neural.utils.python_utils import set_random_seed

    base = env.unwrapped
    snapshot = torch.load(args.policy_snapshot, map_location="cpu", weights_only=False)
    config = copy.deepcopy(snapshot["infos"]["opd"]["runner_cfg"])
    recovery = None
    if "class_name" not in config["algorithm"]:
        if not args.online_config:
            raise ValueError(
                "Constructor-consumed runner metadata requires --online-config with the original snapshot."
            )
        original_path = _read_json(args.online_config)["initial_policy"]
        original = torch.load(original_path, map_location="cpu", weights_only=False)
        config, fields = recover_runner_config(config, original["infos"]["opd"]["runner_cfg"])
        recovery = {"original_snapshot": original_path, "sha256": file_sha256(original_path), "fields": fields}
        repaired = copy.deepcopy(snapshot)
        repaired["infos"]["opd"]["runner_cfg"] = copy.deepcopy(config)
        repaired["infos"]["opd"]["provenance"]["configuration_recovery"] = recovery
        with (args.output / "policy_with_recovered_config.pt").open("xb") as stream:
            torch.save(repaired, stream)
    if snapshot["infos"]["opd"]["provenance"]["protocol"] != simulation_protocol(base):
        raise ValueError("Policy evaluation physics protocol mismatch.")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=config["clip_actions"])
    runner = OnPolicyRunner(wrapped, copy.deepcopy(config), log_dir=None, device=str(base.device))
    info = load_opd_policy_snapshot(args.policy_snapshot, runner)
    policy = runner.get_inference_policy(device=str(base.device))
    set_random_seed(args.seed)
    obs = wrapped.get_observations()
    errors_xy, errors_yaw = [], []
    completed_lengths = []
    lengths = torch.zeros(base.num_envs, device=base.device)
    terminations = defaultdict(int)
    for _ in range(args.steps):
        actions = policy(obs, stochastic_output=False)
        command = base.command_manager.get_command("base_velocity").clone()
        obs, reward, done, extras = wrapped.step(actions)
        lengths += base.step_dt
        completed_lengths.extend(lengths[done.bool()].cpu().tolist())
        lengths[done.bool()] = 0
        velocity = base.scene["robot"].data.root_lin_vel_b.torch
        yaw_rate = base.scene["robot"].data.root_ang_vel_b.torch[:, 2]
        valid = ~done.bool()
        errors_xy.extend((velocity[valid, :2] - command[valid, :2]).norm(dim=-1).cpu().tolist())
        errors_yaw.extend((yaw_rate[valid] - command[valid, 2]).abs().cpu().tolist())
        for name in base.termination_manager.active_terms:
            terminations[name] += int(base.termination_manager.get_term(name).sum())
        runner.alg.get_policy().reset(done)
    _write_json(
        args.output / "report.json",
        {
            "policy_sha256": file_sha256(args.policy_snapshot),
            "ppo_updates": info["ppo_updates_completed"],
            "seed": args.seed,
            "control_steps": args.steps,
            "num_envs": base.num_envs,
            "xy_tracking_error_mean_mps": float(np.mean(errors_xy)),
            "yaw_tracking_error_mean_radps": float(np.mean(errors_yaw)),
            "completed_episode_lengths_s": completed_lengths,
            "right_censored_episode_lengths_s": lengths.cpu().tolist(),
            "termination_counts": dict(terminations),
            "deterministic_policy": True,
            "runner_config_recovery": recovery,
            "note": (
                "Done steps are excluded from post-step tracking because the vector environment auto-resets; "
                "failures are reported separately."
            ),
        },
    )


def summarize_pilot(root: Path):
    """Audit initialization, progress and gates without attributing benefit to rejected models."""
    protocol = _read_json(root / "protocol.json")
    initial = torch.load(protocol["initial_policy"], map_location="cpu", weights_only=False)
    result = {"protocol": protocol, "arms": {}, "policy_evaluation": {}, "final_test_seeds_used": False}
    for arm in ("fixed", "online"):
        arm_report = _read_json(root / arm / "report.json")
        start = torch.load(root / arm / "policy_start.pt", map_location="cpu", weights_only=False)
        initialization_equal = all(
            torch.equal(v, start[group][key])
            for group in ("actor_state_dict", "critic_state_dict")
            for key, v in initial[group].items()
        )
        if not initialization_equal or start["optimizer_state_dict"]["state"]:
            raise RuntimeError("Policy initialization is not a matched zero-update control.")
        final = torch.load(arm_report["rounds"][-1]["last_policy"], map_location="cpu", weights_only=False)
        steps = [int(s["step"]) for s in final["optimizer_state_dict"]["state"].values() if "step" in s]
        result["arms"][arm] = {
            "initial_actor_critic_equal": initialization_equal,
            "ppo_updates": arm_report["ppo_updates_completed"],
            "optimizer_step_min": min(steps),
            "optimizer_step_max": max(steps),
            "artificial_resets": sum(r["artificial_resets"] for r in arm_report["rounds"]),
            "pause_integrity_checks_passed": all(
                r["pause_state_and_observations_unchanged"] for r in arm_report["rounds"]
            ),
            "promotions": sum(r.get("decision", {}).get("accepted", False) for r in arm_report["rounds"]),
            "promotions_used_by_ppo": sum(
                r.get("decision", {}).get("accepted", False) for r in arm_report["rounds"][:-1]
            ),
            "final_policy_training_dynamics": arm_report["rounds"][-1]["active_before"],
            "final_active_dynamics": arm_report["active_model"],
            "rounds": [
                {
                    "round": r["round"],
                    "accepted": r.get("decision", {}).get("accepted", False),
                    "decision": r.get("decision"),
                    "active_after_sha256": r["active_after_sha256"],
                }
                for r in arm_report["rounds"]
            ],
            "final_policy": arm_report["rounds"][-1]["last_policy"],
        }
    for name, folder in (
        ("initial", "mj_eval_initial"),
        ("fixed", "mj_eval_fixed_v2"),
        ("online", "mj_eval_online_v2"),
    ):
        evaluation = _read_json(root / folder / "report.json")
        lengths = evaluation["completed_episode_lengths_s"]
        result["policy_evaluation"][name] = {
            "xy_tracking_error_mean_mps": evaluation["xy_tracking_error_mean_mps"],
            "yaw_tracking_error_mean_radps": evaluation["yaw_tracking_error_mean_radps"],
            "completed_episodes": len(lengths),
            "completed_episode_length_mean_s": float(np.mean(lengths)) if lengths else None,
            "right_censored_episode_lengths_s": evaluation["right_censored_episode_lengths_s"],
            "termination_counts": evaluation["termination_counts"],
            "seed": evaluation["seed"],
            "runner_config_recovery": evaluation.get("runner_config_recovery"),
        }
    result["interpretation"] = (
        "No dynamics candidate was accepted: this validates the online collection/fit/rejection path, "
        "not the effectiveness of repeated accepted OPD updates. Policy evaluation differences cannot be attributed "
        "to model adaptation. Fifteen PPO updates and one evaluation seed are not a gait-convergence experiment."
        if result["arms"]["online"]["promotions"] == 0
        else "Pilot only: inspect reference/real-MJ metrics and confirm across seeds before scaling."
    )
    _write_json(root / "result.json", result)
    print(json.dumps(result["policy_evaluation"], indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize", type=Path, required=True)
    summarize_pilot(parser.parse_args().summarize)
