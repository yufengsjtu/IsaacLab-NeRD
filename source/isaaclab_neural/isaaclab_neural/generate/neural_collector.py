# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent frozen-NeRD exploration with bounded, explicitly MJ-labelled sequence rewards."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
from collections import Counter
from pathlib import Path

import torch

from isaaclab_neural.data.opd import file_sha256
from isaaclab_neural.data.opd_sequence import read_sequence_shard, write_sequence_shard
from isaaclab_neural.generate.active_opd import active_coverage_features, active_query_validity
from isaaclab_neural.rl.rsl_rl.online_opd import _pause_fingerprint, _write_json, _write_status
from isaaclab_neural.utils.checkpoint import load_opd_policy_snapshot, save_opd_policy_snapshot
from isaaclab_neural.utils.python_utils import set_random_seed


@torch.no_grad()
def neural_collector_optimism(
    predicted: torch.Tensor,
    oracle: torch.Tensor,
    *,
    height_scale_m: float,
    upright_scale: float,
    cap: float,
) -> dict[str, torch.Tensor]:
    """Score positive root-height/upright optimism without treating it as true survival.

    Args:
        predicted: NeRD endpoint states in raw SI coordinates, shape [N, 87], with XYZW quaternion.
        oracle: Corresponding MJ endpoint states in raw SI coordinates, shape [N, 87].
        height_scale_m: Fixed positive height-error normalization scale [m].
        upright_scale: Fixed positive scale for the difference in vertical body-axis projection.
        cap: Positive cap applied before logarithmic score compression.

    Returns:
        Signed height gaps [m], signed upright gaps, and detached scores in [0, 1].
        Negative gaps are retained in diagnostics but cannot produce positive curiosity rewards.
    """
    if predicted.ndim != 2 or predicted.shape != oracle.shape or predicted.shape[-1] != 87:
        raise ValueError("Neural collector optimism requires matching raw G1 endpoint states.")
    if any(not math.isfinite(value) or value <= 0 for value in (height_scale_m, upright_scale, cap)):
        raise ValueError("Optimism scales and cap must be finite and positive.")
    if not torch.isfinite(predicted).all() or not torch.isfinite(oracle).all():
        raise ValueError("Only finite predictions and MJ labels may produce curiosity rewards.")
    orientations = []
    for state in (predicted, oracle):
        quaternion = state[:, 3:7]
        norm = quaternion.norm(dim=-1, keepdim=True)
        if bool((norm <= 1e-6).any()):
            raise ValueError("Optimism requires nondegenerate root quaternions.")
        quaternion = quaternion / norm
        orientations.append(1 - 2 * quaternion[:, :2].square().sum(dim=-1))
    height = predicted[:, 2] - oracle[:, 2]
    upright = orientations[0] - orientations[1]
    positive = 0.5 * (height.clamp(min=0) / height_scale_m + upright.clamp(min=0) / upright_scale)
    return {
        "height_gap_m": height,
        "upright_gap": upright,
        "score": torch.log1p(positive.clamp(max=cap)) / math.log1p(cap),
    }


def neural_collector_reward_matrix(
    records: list[dict],
    bonuses: torch.Tensor,
    *,
    worlds: int,
    rollout_controls: int,
    first_physics_frame: int,
    decimation: int = 4,
) -> torch.Tensor:
    """Assign delayed sequence bonuses to their final control, before PPO return computation.

    Args:
        records: Control-aligned recorded windows from the current collector rollout.
        bonuses: Finite nonnegative reward increments, shape [N], one per recorded window.
        worlds: Number of collector worlds.
        rollout_controls: Controls in the PPO rollout.
        first_physics_frame: First recorder frame in this rollout.
        decimation: Physical frames per held control.

    Returns:
        Reward increments of shape [rollout_controls, worlds]. Terminated source windows
        receive zero bonus; their oracle labels remain useful data. Existing task rewards
        and timeout bootstrap contributions are not altered by this function.
    """
    if (
        min(worlds, rollout_controls, decimation) < 1
        or first_physics_frame < 0
        or first_physics_frame % decimation
        or bonuses.shape != (len(records),)
        or not torch.isfinite(bonuses).all()
        or bool((bonuses < 0).any())
    ):
        raise ValueError("Invalid collector rollout or sequence reward layout.")
    result = bonuses.new_zeros((rollout_controls, worlds))
    used = set()
    for row, bonus in zip(records, bonuses):
        frames, world = row["frame_ids"], row["env_id"]
        if (
            frames.ndim != 1
            or frames.dtype != torch.int64
            or len(frames) == 0
            or len(frames) % decimation
            or not 0 <= world < worlds
            or int(frames[0]) < first_physics_frame
            or int(frames[0]) % decimation
            or int(frames[-1]) >= first_physics_frame + rollout_controls * decimation
            or not torch.equal(frames, torch.arange(int(frames[0]), int(frames[0]) + len(frames), device=frames.device))
        ):
            raise ValueError("Collector sequence reward crosses a rollout or control boundary.")
        control = (int(frames[-1]) - first_physics_frame) // decimation
        key = (control, world)
        if key in used:
            raise ValueError("Duplicate collector sequence reward destination.")
        used.add(key)
        if not row["terminal"]:
            result[control, world] = bonus
    return result


def run_neural_collector(env, adapter, args) -> None:
    """Train an independent collector while freezing dynamics and retaining typed MJ sequence labels."""
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_neural.generate.opd import simulation_protocol
    from isaaclab_neural.generate.opd_sequence import SequenceRecorder
    from isaaclab_neural.rl.rsl_rl.ppo_diagnostics import PpoDiagnostics

    config = json.loads(Path(args.rl_study_config).read_text())
    base, output = env.unwrapped, args.output.resolve()
    _validate_collector(config, base, args)
    protocol = simulation_protocol(base)
    source = torch.load(args.policy_snapshot, map_location="cpu", weights_only=False)
    if source["infos"]["opd"]["provenance"]["protocol"] != protocol:
        raise ValueError("Independent collector initialization changed the physical/action protocol.")
    cfg = copy.deepcopy(source["infos"]["opd"]["runner_cfg"])
    cfg["device"] = str(base.device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=cfg["clip_actions"])
    runner = OnPolicyRunner(wrapped, copy.deepcopy(cfg), log_dir=None, device=str(base.device))
    load_opd_policy_snapshot(args.policy_snapshot, runner, require_initial=True)
    runner.alg.learning_rate = config["initial_learning_rate"]
    cfg["algorithm"]["learning_rate"] = runner.alg.learning_rate
    for group in runner.alg.optimizer.param_groups:
        group["lr"] = runner.alg.learning_rate
    set_random_seed(config["env_seed"])
    obs, _ = wrapped.reset()
    base.neural_adapter.sync(update_history=False)
    model = base.neural_adapter.solver.neural_model.eval().requires_grad_(False)
    original = {key: value.detach().clone() for key, value in model.state_dict().items()}
    set_random_seed(config["policy_rng_seed"])
    if config["init_at_random_ep_len"]:
        base.episode_length_buf[:] = torch.randint_like(base.episode_length_buf, high=base.max_episode_length)
    provenance = {
        "protocol": protocol,
        "policy_role": "independent_collector",
        "dynamics_sha256": config["dynamics_sha256"],
        "initial_policy_sha256": config["policy_sha256"],
        "collector_config_sha256": file_sha256(args.rl_study_config),
    }
    _write_json(output / "runner_config.json", cfg)
    save_opd_policy_snapshot(output / "policy_0000.pt", runner, cfg, updates_completed=0, provenance=provenance)
    recorder = SequenceRecorder(
        base, adapter, env_ids=list(range(base.num_envs)), sequence_controls=4, verify_frames=32
    )
    oracle = None
    diagnostics = PpoDiagnostics()
    counts = Counter()
    feature_reference = json.loads(Path(config["coverage_reference"]).read_text())
    mean, scale = [torch.tensor(feature_reference[key]) for key in ("mean", "scale")]
    limits = base.scene["robot"].data.joint_pos_limits.torch[0].detach().cpu().clone()
    scales = {key: config[key] for key in ("height_scale_m", "upright_scale", "optimism_cap")}
    iterations, labels, failure, oracle_steps = [], [], None, 0
    completed = 0
    try:
        oracle = _SequenceOracleClient(Path(args.dynamics).resolve(), output, base.num_envs, str(base.device))
        for iteration in range(config["updates"]):
            storage = _collector_storage(output, config)
            actor_hash = _actor_fingerprint(runner.alg._raw_actor)
            first_frame = recorder.frame
            obs, records, rollout = _collector_rollout(runner, obs, recorder, iteration)
            directory = output / f"iteration_{iteration:04d}"
            directory.mkdir()
            metadata = {
                "physics_dt": base.physics_dt,
                "decimation": base.cfg.decimation,
                "history_steps": 10,
                "sequence_controls": 4,
                "iteration_id": iteration,
                "collection_id": str(output),
                "split": "train",
                "source_kind": "nerd",
                "label_origin": "unlabelled",
                "policy_role": "independent_collector",
                "evaluation_only": False,
                "policy_hash": actor_hash,
                "policy_hash_kind": "actor_state_sha256",
                "protocol": protocol,
                "dynamics_hash": config["dynamics_sha256"],
                "initial_policy_sha256": config["policy_sha256"],
                "collector_policy_updates": iteration,
                "policy_rng_seed": config["policy_rng_seed"],
                "initial_solver_state_protocol": "canonical_cold_reset",
                "initial_physical_state_protocol": "native_newton_state_v1",
                "history_origin": "actual_neural_runtime_including_reset_reconstruction",
                "training_semantics": "model-generated history; only subsequent MJ anchor/future states are truth",
            }
            write_sequence_shard(directory / "raw_queries.pt", records, metadata=metadata)
            accepted, screening = _collector_screen(records, limits, config["validity"])
            row = {
                "iteration": iteration,
                "rollout": rollout,
                "screening": screening,
                "actor_before_sha256": actor_hash,
                "storage": storage,
            }
            if accepted:
                query = directory / "queries.pt"
                target = directory / "labels.pt"
                write_sequence_shard(
                    query,
                    accepted,
                    metadata={**metadata, "raw_source_sha256": file_sha256(directory / "raw_queries.pt")},
                )
                expected = _oracle_transitions(accepted, base.num_envs)
                if oracle_steps + expected > config["maximum_oracle_physical_transitions"]:
                    raise RuntimeError("Collector oracle budget exhausted before submitting another request.")
                pause = _pause_fingerprint(base, obs)
                response = oracle.label(query, target, iteration)
                if response["actual_oracle_physics_transitions"] != expected:
                    raise RuntimeError("Oracle transition accounting differs from submitted sequence lengths.")
                oracle_steps += expected
                if pause != _pause_fingerprint(base, obs):
                    raise RuntimeError("Oracle query changed the paused collector state or policy RNG.")
                labelled = read_sequence_shard(target, require_labels=True)
                bonuses, score_report = _collector_bonuses(
                    labelled["records"], protocol, config, scales, counts, mean, scale
                )
                matrix = neural_collector_reward_matrix(
                    labelled["records"],
                    bonuses,
                    worlds=base.num_envs,
                    rollout_controls=cfg["num_steps_per_env"],
                    first_physics_frame=first_frame,
                )
                runner.alg.storage.rewards.add_(matrix.to(runner.device).unsqueeze(-1))
                entry = {
                    "path": str(target),
                    "sha256": file_sha256(target),
                    "iteration": iteration,
                    "records": len(labelled["records"]),
                    "source_kind": "nerd",
                    "split": "train",
                }
                labels.append(entry)
                row.update(oracle=response, scores=score_report, added_reward_sum=float(matrix.sum()), labels=entry)
                if scales.get("calibrated", False) and not (output / "reward_calibration.json").exists():
                    _write_json(
                        output / "reward_calibration.json",
                        {"source_labels": entry, "scales": scales, "evaluation_data_used": False},
                    )
            else:
                row.update(added_reward_sum=0.0, scores={"complete_sequences": 0})
            row["ppo"] = _collector_update(runner, obs, diagnostics)
            completed = iteration + 1
            runner.current_learning_iteration = completed
            row["actor_after_sha256"] = _actor_fingerprint(runner.alg._raw_actor)
            _write_json(directory / "report.json", row)
            iterations.append(row)
            if completed in config["checkpoints"] or completed == config["updates"]:
                _collector_storage(output, config)
                save_opd_policy_snapshot(
                    output / f"policy_{completed:04d}.pt",
                    runner,
                    cfg,
                    updates_completed=completed,
                    provenance=provenance,
                )
            if any(not torch.equal(value, model.state_dict()[key]) for key, value in original.items()):
                raise RuntimeError("Collector updated frozen dynamics parameters or RMS.")
            _write_status(
                output / "status.json",
                {
                    "status": "running",
                    "ppo_updates": completed,
                    "oracle_physical_transitions": oracle_steps,
                    "storage": _collector_storage(output, config),
                },
            )
        if not labels or file_sha256(args.dynamics) != config["dynamics_sha256"]:
            raise RuntimeError("Collector produced no real labels or changed its frozen dynamics source.")
        _write_json(
            output / "training_manifest.json",
            {
                "policy_role": "independent_collector",
                "split": "train",
                "label_origin": "mjwarp",
                "history_origin": "nerd",
                "full_history_truth": False,
                "shards": labels,
                "dynamics_sha256": config["dynamics_sha256"],
            },
        )
    except Exception as error:
        failure = repr(error)
        raise
    finally:
        recorder.close()
        if oracle is not None:
            oracle.close()
        _write_json(
            output / "report.json",
            {
                "config": config,
                "iterations": iterations,
                "failure": failure,
                "ppo_updates": completed,
                "model_optimizer_updates": 0,
                "NeRD_physical_transitions": recorder.frame * base.num_envs,
                "confirmed_oracle_physical_transitions": oracle_steps,
                "oracle_accounting_complete": failure is None,
                "dynamics_and_RMS_unchanged": all(
                    torch.equal(value, model.state_dict()[key]) for key, value in original.items()
                ),
                "evaluation_states_used_for_training": False,
                "training_shards": labels,
                "coverage_bins": len(counts),
                "verified_history_frames": recorder.verified_histories,
            },
        )
        _write_status(
            output / "status.json",
            {
                "status": "failed" if failure else "completed",
                "failure": failure,
                "ppo_updates": completed,
                "oracle_physical_transitions": oracle_steps,
            },
        )


def _validate_collector(config, base, args):
    if (
        args.backend != "nerd"
        or config.get("kind") != "independent_neural_collector_v1"
        or config.get("policy_role") != "independent_collector"
        or config.get("mode") != "train"
        or not 8 <= base.num_envs <= 16
        or config["num_envs"] != base.num_envs
        or not 1 <= config["updates"] <= 64
        or config["env_seed"] not in range(10, 20)
        or not 0 < config["initial_learning_rate"] <= 1e-3
        or not 0 <= config["optimism_weight"] <= 1
        or not 0 <= config["coverage_weight"] <= 0.2
        or file_sha256(args.dynamics) != config["dynamics_sha256"]
        or file_sha256(args.policy_snapshot) != config["policy_sha256"]
        or args.matched_mdp_audit["mdp_sha256"] != config["mdp_sha256"]
        or base.cfg.observations.policy.enable_corruption != config["policy_observation_corruption"]
    ):
        raise ValueError("Independent collector requires a bounded cold-start matched-MDP protocol.")
    if file_sha256(config["coverage_reference"]) != config["coverage_reference_sha256"]:
        raise ValueError("Collector coverage normalization changed.")
    reference = json.loads(Path(config["coverage_reference"]).read_text())
    if reference.get("source_role") != "collector_training_only" or reference.get("evaluation_data_used") is not False:
        raise ValueError("Collector coverage normalization cannot use task-policy evaluation states.")


def _collector_storage(output: Path, config: dict) -> dict:
    root = Path(config["storage_root"])
    if config["minimum_free_bytes"] < 100 * 2**30 or config["maximum_shared_bytes"] > 40 * 2**30:
        raise ValueError("Collector storage limits cannot weaken the shared study protection.")
    used = sum(path.stat().st_blocks * 512 for path in root.rglob("*") if path.is_file() and not path.is_symlink())
    own = sum(path.stat().st_blocks * 512 for path in output.rglob("*") if path.is_file() and not path.is_symlink())
    free = shutil.disk_usage(root).free
    reserve = 64 * 2**20
    if (
        free - reserve < config["minimum_free_bytes"]
        or used + reserve > config["maximum_shared_bytes"]
        or own + reserve > config["maximum_output_bytes"]
    ):
        raise RuntimeError("Collector storage boundary reached; existing files were not deleted.")
    return {"free_bytes": free, "shared_bytes": used, "output_bytes": own, "reservation_bytes": reserve}


def _actor_fingerprint(actor) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(actor.state_dict().items()):
        digest.update(key.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _collector_rollout(runner, obs, recorder, iteration):
    from rsl_rl.utils import check_nan

    recorder.begin_iteration(iteration)
    runner.alg.train_mode()
    reward_total, done_count = 0.0, 0
    with torch.inference_mode():
        for _ in range(runner.cfg["num_steps_per_env"]):
            action = runner.alg.act(obs)
            obs, reward, done, extras = runner.env.step(action.to(runner.env.device))
            check_nan(obs, reward, done)
            obs, reward, done = obs.to(runner.device), reward.to(runner.device), done.to(runner.device)
            recorder.finish_control_step(done)
            runner.alg.process_env_step(obs, reward, done, extras)
            reward_total += float(reward.mean())
            done_count += int(done.sum())
    records = recorder.end_iteration()
    if not records:
        raise RuntimeError("Collector produced no complete-H10 sequence queries.")
    return (
        obs,
        records,
        {"task_reward_mean_per_control": reward_total / runner.cfg["num_steps_per_env"], "done_count": done_count},
    )


def _collector_screen(records, limits, settings):
    states = torch.stack([row["history"][-1]["states"] for row in records])
    gaps = [row["history"][-1]["native"]["separation"] for row in records]
    gaps = torch.tensor([float(value.min()) if value.numel() else 0.0 for value in gaps])
    checks = active_query_validity(states, limits, gaps, **settings)
    valid = torch.stack(list(checks.values())).all(0)
    return [row for index, row in enumerate(records) if valid[index]], {
        "recorded": len(records),
        "accepted": int(valid.sum()),
        "rejected_by_reason_nonexclusive": {key: int((~value).sum()) for key, value in checks.items()},
    }


def _oracle_transitions(records, worlds):
    counts = Counter(len(row["actions"]) for row in records)
    return sum(((count + worlds - 1) // worlds) * worlds * controls * 4 for controls, count in counts.items())


def _collector_bonuses(records, protocol, config, scales, counts, mean, scale):
    complete = [index for index, row in enumerate(records) if len(row["oracle_states"]) == 16]
    bonuses = torch.zeros(len(records))
    if not complete:
        return bonuses, {"complete_sequences": 0}
    predicted = torch.stack([records[index]["student_states"][15] for index in complete])
    truth = torch.stack([records[index]["oracle_states"][15] for index in complete])
    if not scales.get("calibrated", False):
        if config["calibrate_scales_from_first_query"]:
            scales["height_scale_m"] = max(
                scales["height_scale_m"], float(torch.quantile((predicted[:, 2] - truth[:, 2]).clamp(min=0), 0.75))
            )
        scales["calibrated"] = True
    scores = neural_collector_optimism(
        predicted,
        truth,
        height_scale_m=scales["height_scale_m"],
        upright_scale=scales["upright_scale"],
        cap=scales["optimism_cap"],
    )
    foot_ids = [protocol["body_names"].index(name) for name in ("left_ankle_roll_link", "right_ankle_roll_link")]
    contacts, anchors, actions = [], [], []
    for index in complete:
        anchor = records[index]["history"][-1]
        native = anchor["native"]
        active = native["separation"] <= 0
        contacts.append(
            [
                bool((active & ((native["body_slots_0"] == foot) | (native["body_slots_1"] == foot))).any())
                for foot in foot_ids
            ]
        )
        anchors.append(anchor["states"])
        actions.append(anchor["actions"])
    features = active_coverage_features(
        torch.stack(anchors), torch.stack(actions), torch.tensor(contacts), protocol["joint_names"]
    )
    normalized = (features - mean) / scale
    novelty = []
    for index, values in zip(complete, normalized):
        key = tuple(
            torch.floor(values[[0, 1, 2, 4, 5, 6, 9, len(values) - 2, len(values) - 1]].clamp(-10, 10))
            .to(torch.int64)
            .tolist()
        )
        counts[key] += 1
        novelty.append(1 / math.sqrt(counts[key]))
    bonuses[complete] = config["optimism_weight"] * scores["score"] + config["coverage_weight"] * torch.tensor(novelty)
    return bonuses, {
        "complete_sequences": len(complete),
        "height_gap_mean_m": float(scores["height_gap_m"].mean()),
        "height_positive_mean_m": float(scores["height_gap_m"].clamp(min=0).mean()),
        "upright_gap_mean": float(scores["upright_gap"].mean()),
        "optimism_score_mean": float(scores["score"].mean()),
        "novelty_mean": sum(novelty) / len(novelty),
        "normalization": dict(scales),
    }


def _collector_update(runner, obs, diagnostics):
    with torch.inference_mode():
        runner.alg.compute_returns(obs)
    consistency = diagnostics.check_consistency(runner.alg)
    if not consistency["passed"]:
        raise RuntimeError("Collector PPO distribution/log-probability consistency failed before update.")
    with diagnostics.observe_update(runner.alg):
        losses = runner.alg.update()
    if not all(math.isfinite(value) for value in losses.values()):
        raise FloatingPointError("Non-finite collector PPO loss.")
    return {
        "losses": losses,
        "terminal_learning_rate": runner.alg.learning_rate,
        "consistency": consistency,
        "minibatches": diagnostics.minibatches,
    }


class _SequenceOracleClient:
    def __init__(self, dynamics: Path, output: Path, worlds: int, device: str):
        command = [
            sys.executable,
            "-m",
            "isaaclab_neural.generate.opd",
            "--stage",
            "sequence-oracle",
            "--backend",
            "mjwarp",
            "--sequence_geometry_mdp",
            "--sequence_serve",
            "--dynamics",
            str(dynamics),
            "--output",
            str(output / "oracle_worker"),
            "--num-envs",
            str(worlds),
            "--seed",
            "11",
            "--device",
            device,
        ]
        _write_json(output / "oracle.command.json", command)
        self.log = (output / "oracle.log").open("x")
        self.messages = queue.Queue()
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "CUDA_LAUNCH_BLOCKING": "1"},
        )
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        for line in self.process.stdout:
            self.log.write(line)
            self.log.flush()
            if line.startswith("OPD_SEQUENCE_RESPONSE "):
                self.messages.put(line[len("OPD_SEQUENCE_RESPONSE ") :])
        self.messages.put(None)

    def label(self, source: Path, target: Path, request_id: int) -> dict:
        request = {"source": str(source), "target": str(target), "request_id": request_id}
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        response = self.messages.get(timeout=600)
        if response is None:
            raise RuntimeError(
                "Oracle exited without confirming the query; preserve partial accounting and inspect its log."
            )
        result = json.loads(response)
        if result["request_id"] != request_id:
            raise RuntimeError("Persistent oracle returned a different request identity.")
        return result

    def close(self):
        with contextlib.suppress(BrokenPipeError):
            self.process.stdin.close()
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.reader.join(timeout=10)
        self.log.close()
