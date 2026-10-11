# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Portable MJ collection and branched NeRD PPO workers with terrain-aware replay."""

from __future__ import annotations

import copy
import json
import math
import os
import time
from pathlib import Path

import torch
from tensordict import TensorDict

from isaaclab_neural.data.opd import file_sha256
from isaaclab_neural.rl.rsl_rl.mbpo import (
    MODEL_WINDOW,
    BranchBuffer,
    FastHistory,
    capture_manager_state,
    initialize_branch_scene,
    install_branch_reset,
    merge_branch_batches,
)
from isaaclab_neural.utils.checkpoint import load_checkpoint, load_opd_policy_snapshot, save_opd_policy_snapshot
from isaaclab_neural.utils.python_utils import set_random_seed


def atomic_json(value: dict, path: Path) -> None:
    """Publish a mutable protocol message atomically, rejecting non-finite JSON values."""
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, allow_nan=False, default=float))
    os.replace(temporary, path)


def atomic_save(value, path: Path) -> None:
    """Publish a worker-owned tensor message after its write completes."""
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    os.replace(temporary, path)


def wait_for(path: Path, timeout_s: float, stop: Path) -> None:
    """Wait for a peer message with a bounded timeout [s] and explicit peer-failure handling."""
    deadline = time.monotonic() + timeout_s
    while not path.exists():
        if stop.exists():
            raise RuntimeError(f"MBPO peer stopped: {stop.read_text()[:500]}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for MBPO message {path.name}.")
        time.sleep(0.1)


def wait_for_policy(root: Path, iteration: int, timeout_s: float) -> Path | None:
    """Wait for a policy or the producer's actual completion boundary, bounded by timeout [s]."""
    path, done, stop = root / "policies" / f"policy_{iteration:05d}.pt", root / "TRAINING_DONE.json", root / "FAILED"
    deadline = time.monotonic() + timeout_s
    while not path.exists():
        if stop.exists():
            raise RuntimeError(f"MBPO peer stopped: {stop.read_text()[:500]}")
        if done.exists() and iteration >= json.loads(done.read_text())["iterations"]:
            return None
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for MBPO policy {iteration}.")
        time.sleep(0.1)
    return path


def bootstrap_action(config: dict, iteration: int, model_round: int) -> str:
    """Keep collecting during bounded model warm-up without silently consuming the PPO update budget."""
    if iteration < config["warmup_iterations"]:
        return "collect"
    if config["require_model_before_ppo"] and model_round < 0:
        return "collect" if iteration < config["max_warmup_iterations"] else "wait_model"
    return "ppo"


def rollout_length(schedule: list[int], iteration: int) -> int:
    """Return the model-branch horizon in control steps, not physical simulation steps."""
    first, last, low, high = schedule
    if first >= last or min(low, high) < 1:
        raise ValueError("Invalid MBPO branch-length schedule.")
    if iteration <= first:
        return low
    if iteration >= last:
        return high
    return round(low + (high - low) * (iteration - first) / (last - first))


def run_mj_collect(env, adapter, args, config: dict) -> None:
    """Collect actual MJ transitions and labelled physical windows from the current training policy."""
    from isaaclab_neural.generate.opd import simulation_protocol
    from isaaclab_neural.rl.rsl_rl.matched_mj import MatchedGeometryBridge

    root, base = Path(config["run_dir"]), env.unwrapped
    stop = root / "FAILED"
    context = initialize_branch_scene(base)
    bridge = MatchedGeometryBridge(base, adapter)
    recorder = FastHistory(base, adapter)
    wrapped, runner, cfg = _build(env, args)
    atomic_json(
        {
            "protocol": simulation_protocol(base),
            "physics_dt": base.physics_dt,
            "decimation": base.cfg.decimation,
            "history_steps": 10,
            "terrain_context": context,
            "label_origin": "observed_mj_post_step",
            "holdout_env_ids": config["model_holdout_env_ids"],
        },
        root / "model_data" / "meta.json",
    )
    set_random_seed(config["seed"])
    obs, _ = wrapped.reset()
    bridge.refresh()
    base.episode_length_buf[:] = torch.randint_like(base.episode_length_buf, high=base.max_episode_length)
    generator = torch.Generator().manual_seed(config["seed"])
    steps = cfg["num_steps_per_env"]
    lengths = torch.zeros(base.num_envs, device=base.device)
    maximum = config["iterations"] + config["max_warmup_iterations"]
    completed_iterations = 0
    try:
        for iteration in range(maximum):
            if stop.exists():
                raise RuntimeError("MBPO peer stopped during MJ collection.")
            policy_path = wait_for_policy(root, iteration, config["wait_timeout_s"])
            if policy_path is None:
                break
            runner.alg.load(
                torch.load(policy_path, map_location=base.device, weights_only=False),
                {"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False},
                strict=True,
            )
            policy = runner.get_inference_policy(device=str(base.device))
            anchors = set(torch.randperm(steps, generator=generator)[: config["anchor_steps_per_iteration"]].tolist())
            blocks, windows, real, completed = [], [], [], []
            reward_sum, done_count, timeout_count = 0.0, 0, 0
            jumps_before = recorder.external_jump_count
            started = time.monotonic()
            with torch.inference_mode():
                for step in range(steps):
                    actor = runner.alg.actor
                    action = actor(obs, stochastic_output=True)
                    transition = {
                        "observations": obs.clone(),
                        "actions": action.clone(),
                        "actions_log_prob": actor.get_output_log_prob(action).clone(),
                        "distribution_params": tuple(value.clone() for value in actor.output_distribution_params),
                    }
                    obs, reward, done, extras = wrapped.step(action)
                    transition.update(rewards=reward.clone(), dones=done.clone(), time_outs=extras["time_outs"].clone())
                    real.append(transition)
                    policy.reset(done)
                    recorder.finish_control_step(done)
                    reward_sum += float(reward.mean())
                    done_count += int(done.sum())
                    timeout_count += int(extras["time_outs"].sum())
                    lengths += base.step_dt
                    completed.extend(lengths[done.bool()].cpu().tolist())
                    lengths[done.bool()] = 0
                    if step in anchors:
                        ids = recorder.complete()
                        if ids.numel():
                            block = recorder.gather(ids)
                            block["manager"] = capture_manager_state(base, ids)
                            blocks.append(block)
                    if step in config["model_data_steps"]:
                        ids = recorder.complete(MODEL_WINDOW)
                        if ids.numel():
                            windows.append(recorder.model_windows(ids, base.cfg.decimation))
            batch = merge_branch_batches(blocks)
            batch["iteration"] = iteration
            atomic_save(batch, root / "buffers" / f"buffer_{iteration:05d}.pt")
            if windows:
                combined = {key: torch.cat([part[key] for part in windows]) for key in windows[0] if key != "native"}
                combined["native"] = {
                    key: torch.cat([part["native"][key] for part in windows]) for key in windows[0]["native"]
                }
                combined["iteration"] = iteration
                atomic_save(combined, root / "model_data" / f"data_{iteration:05d}.pt")
            atomic_save(
                {"steps": real, "last_observations": obs.clone(), "iteration": iteration},
                root / "real_rollouts" / f"rollout_{iteration:05d}.pt",
            )
            atomic_json(
                {
                    "iteration": iteration + 1,
                    "rows": len(batch["current_state"]),
                    "mj_reward_mean_per_step": reward_sum / steps,
                    "mj_done_count": done_count,
                    "mj_failure_count": done_count - timeout_count,
                    "mj_timeout_count": timeout_count,
                    "external_state_jumps": recorder.external_jump_count - jumps_before,
                    "terrain_level_mean": float(base.scene.terrain.terrain_levels.float().mean()),
                    "completed_episode_duration_s": sum(completed) / len(completed) if completed else None,
                    "completed_episodes": len(completed),
                    "mj_physical_transitions": base.num_envs * steps * base.cfg.decimation,
                    "mj_control_transitions": base.num_envs * steps,
                    "seconds": time.monotonic() - started,
                },
                root / "mj_stats" / f"stats_{iteration:05d}.json",
            )
            completed_iterations = iteration + 1
            if config["delete_consumed_buffers"]:
                policy_path.unlink()
        atomic_json(
            {
                "iterations": completed_iterations,
                "physical_transitions": completed_iterations * base.num_envs * steps * base.cfg.decimation,
            },
            root / "COLLECTION_DONE.json",
        )
    except Exception as error:
        atomic_json({"role": "mj_collect", "error": repr(error)}, stop)
        raise
    finally:
        recorder.close()
        bridge.close()


def run_nerd_train(env, adapter, args, config: dict) -> None:
    """Optimize PPO on terrain-matched NeRD branches and real MJ minibatches without differentiating dynamics."""
    from isaaclab_neural.generate.opd import simulation_protocol

    root, base = Path(config["run_dir"]), env.unwrapped
    stop = root / "FAILED"
    context = initialize_branch_scene(base)
    base.cfg.compute_final_obs = True
    wrapped, runner, cfg = _build(env, args, config["initial_learning_rate"])
    algorithm = runner.alg
    solver = base.neural_adapter.solver
    dynamics = solver.neural_model.eval().requires_grad_(False)
    buffer = BranchBuffer(solver, config["buffer_capacity"], config["seed"], context)
    statistics, mixed = {}, {}
    _install_mixed_generator(algorithm.storage, mixed, config["ppo_real_mix_repeat"])
    provenance = {
        "protocol": simulation_protocol(base),
        "mdp_sha256": args.matched_mdp_audit["mdp_sha256"],
        "training_backend": "nerd_mbpo_branch",
        "dynamics_sha256": file_sha256(args.dynamics),
        "terrain_context": context,
        "mbpo_config": config,
    }
    atomic_save(algorithm.save(), root / "policies" / "policy_00000.pt")
    save_opd_policy_snapshot(
        root / "checkpoints" / "policy_0000.pt", runner, cfg, updates_completed=0, provenance=provenance
    )
    loaded, model_round, actual_updates = 0, -1, 0
    frozen = {key: value.clone() for key, value in dynamics.state_dict().items()}

    def load_buffers(required):
        nonlocal loaded
        if loaded <= required:
            wait_for(root / "buffers" / f"buffer_{required:05d}.pt", config["wait_timeout_s"], stop)
        while (root / "buffers" / f"buffer_{loaded:05d}.pt").exists():
            path = root / "buffers" / f"buffer_{loaded:05d}.pt"
            buffer.add(torch.load(path, map_location="cpu", weights_only=False))
            if config["delete_consumed_buffers"]:
                path.unlink()
            loaded += 1

    def refresh_model():
        nonlocal model_round, frozen
        path = root / "models" / "latest.json"
        if not path.exists():
            return
        latest = json.loads(path.read_text())
        if latest["round"] <= model_round:
            return
        if file_sha256(latest["path"]) != latest["sha256"]:
            raise RuntimeError("Published dynamics checkpoint changed after publication.")
        dynamics.load_state_dict(load_checkpoint(latest["path"], device=base.device)["model_state_dict"], strict=True)
        dynamics.eval().requires_grad_(False)
        solver._mark_history_rows_for_rebuild(torch.arange(base.num_envs, device=solver.torch_device))
        frozen = {key: value.clone() for key, value in dynamics.state_dict().items()}
        model_round = latest["round"]

    set_random_seed(config["seed"])
    try:
        load_buffers(0)
        install_branch_reset(base, buffer, statistics, config["branch_standing_fraction"])
        base.cfg.episode_length_s = rollout_length(config["rollout_schedule"], 0) * base.step_dt
        with torch.inference_mode():
            obs, _ = wrapped.reset()
        iteration, warmup_actual = 0, 0
        while actual_updates < config["iterations"]:
            if stop.exists():
                raise RuntimeError("MBPO peer stopped during PPO training.")
            started = time.monotonic()
            if iteration:
                load_buffers(iteration - 1)
            refresh_model()
            phase = bootstrap_action(config, iteration, model_round)
            if phase == "collect":
                atomic_save(algorithm.save(), root / "policies" / f"policy_{iteration + 1:05d}.pt")
                if iteration:
                    path = root / "real_rollouts" / f"rollout_{iteration - 1:05d}.pt"
                    wait_for(path, config["wait_timeout_s"], stop)
                    if config["delete_consumed_buffers"]:
                        path.unlink()
                _append(
                    root / "nerd_train.jsonl",
                    {"iteration": iteration + 1, "ppo_updates": 0, "warmup": True, "dynamics_round": model_round},
                )
                iteration += 1
                continue
            if phase == "wait_model":
                wait_for(root / "models" / "latest.json", config["wait_timeout_s"], stop)
                refresh_model()
            if actual_updates == 0:
                warmup_actual = iteration
                with torch.inference_mode():
                    obs, _ = wrapped.reset()
            horizon = rollout_length(config["rollout_schedule"], actual_updates)
            base.cfg.episode_length_s = horizon * base.step_dt
            statistics.clear()
            reward_sum, raw_reward_sum, failures, timeouts = 0.0, 0.0, 0, 0
            with torch.inference_mode():
                for _ in range(cfg["num_steps_per_env"]):
                    action = algorithm.act(obs)
                    obs, reward, done, extras = wrapped.step(action.to(base.device))
                    timeout = extras.pop("time_outs").to(algorithm.device)
                    reward = reward.to(algorithm.device)
                    raw_reward_sum += float(reward.mean())
                    if timeout.any():
                        final = TensorDict(extras["final_obs"], batch_size=[base.num_envs]).to(algorithm.device)
                        reward = (
                            reward + algorithm.gamma * algorithm.critic(final).detach().squeeze(-1) * timeout.float()
                        )
                    algorithm.process_env_step(obs.to(algorithm.device), reward, done.to(algorithm.device), extras)
                    reward_sum += float(reward.mean())
                    failures += int((done.bool() & ~timeout.bool()).sum())
                    timeouts += int(timeout.sum())
                algorithm.compute_returns(obs.to(algorithm.device))
            real_iteration = iteration - 1
            path = root / "real_rollouts" / f"rollout_{real_iteration:05d}.pt"
            wait_for(path, config["wait_timeout_s"], stop)
            mixed["storage"] = _real_storage(algorithm, torch.load(path, map_location="cpu", weights_only=False))
            real_rows = mixed["storage"].num_envs * mixed["storage"].num_transitions_per_env
            if config["delete_consumed_buffers"]:
                path.unlink()
            losses, consistency = _update(algorithm, actual_updates < 3)
            mixed["storage"] = None
            actual_updates += 1
            runner.current_learning_iteration = actual_updates
            if actual_updates < config["iterations"]:
                atomic_save(algorithm.save(), root / "policies" / f"policy_{iteration + 1:05d}.pt")
            if actual_updates in config["checkpoints"] or actual_updates == config["iterations"]:
                save_opd_policy_snapshot(
                    root / "checkpoints" / f"policy_{actual_updates:04d}.pt",
                    runner,
                    cfg,
                    updates_completed=actual_updates,
                    provenance={**provenance, "dynamics_round": model_round, "warmup_iterations_actual": warmup_actual},
                )
            if any(not torch.equal(value, dynamics.state_dict()[key]) for key, value in frozen.items()):
                raise RuntimeError("PPO changed dynamics outside the between-iteration model update.")
            ages = statistics.get("source_iterations", [])
            _append(
                root / "nerd_train.jsonl",
                {
                    "iteration": iteration + 1,
                    "ppo_updates": actual_updates,
                    "warmup_iterations_actual": warmup_actual,
                    "terrain_level_mean": float(base.scene.terrain.terrain_levels.float().mean()),
                    "rollout_length_k": horizon,
                    "nerd_reward_mean_per_step_bootstrapped": reward_sum / cfg["num_steps_per_env"],
                    "nerd_reward_mean_per_step_raw": raw_reward_sum / cfg["num_steps_per_env"],
                    "nerd_terminations": failures,
                    "nerd_timeouts": timeouts,
                    "branches": statistics.get("branches", 0),
                    "standing_resets": statistics.get("standing_resets", 0),
                    "branch_source_age_mean": iteration - sum(ages) / len(ages) if ages else None,
                    "ppo_real_rows": real_rows,
                    "buffer_rows": buffer.size,
                    "dynamics_round": model_round,
                    "losses": losses,
                    "learning_rate": algorithm.learning_rate,
                    "consistency": consistency,
                    "seconds": time.monotonic() - started,
                },
            )
            iteration += 1
        atomic_json(
            {
                "ppo_updates": actual_updates,
                "iterations": iteration,
                "dynamics_round": model_round,
                "warmup_iterations_actual": warmup_actual,
            },
            root / "TRAINING_DONE.json",
        )
    except Exception as error:
        atomic_json({"role": "nerd_train", "error": repr(error)}, stop)
        raise


def run_role(env, adapter, args, config: dict) -> None:
    """Execute one declared worker; all workers share only their unique run directory."""
    root = Path(config["run_dir"])
    for name in ("policies", "buffers", "mj_stats", "checkpoints", "model_data", "models", "real_rollouts"):
        (root / name).mkdir(parents=True, exist_ok=True)
    if config["role"] == "mj_collect":
        run_mj_collect(env, adapter, args, config)
    elif config["role"] == "nerd_train":
        run_nerd_train(env, adapter, args, config)
    else:
        raise ValueError("Unknown manager-based MBPO worker role.")


def _build(env, args, learning_rate: float | None = None, *, require_initial: bool = True):
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    source = torch.load(args.policy_snapshot, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(source["infos"]["opd"]["runner_cfg"])
    wrapped = RslRlVecEnvWrapper(env, clip_actions=cfg["clip_actions"])
    runner = OnPolicyRunner(wrapped, copy.deepcopy(cfg), log_dir=None, device=str(env.unwrapped.device))
    load_opd_policy_snapshot(args.policy_snapshot, runner, require_initial=require_initial)
    if learning_rate is not None:
        runner.alg.learning_rate = learning_rate
        for group in runner.alg.optimizer.param_groups:
            group["lr"] = learning_rate
        cfg["algorithm"]["learning_rate"] = learning_rate
    return wrapped, runner, cfg


@torch.no_grad()
def _real_storage(algorithm, rollout: dict):
    from rsl_rl.storage import RolloutStorage

    steps, device = rollout["steps"], algorithm.device
    first = steps[0]["observations"].to(device)
    storage = RolloutStorage("rl", first.batch_size[0], len(steps), first, tuple(steps[0]["actions"].shape[1:]), device)
    for row in steps:
        obs = row["observations"].to(device)
        transition = RolloutStorage.Transition()
        transition.observations, transition.actions = obs, row["actions"].to(device)
        transition.values = algorithm.critic(obs).detach()
        transition.actions_log_prob = row["actions_log_prob"].to(device)
        transition.distribution_params = tuple(value.to(device) for value in row["distribution_params"])
        transition.rewards = (
            row["rewards"].to(device).float()
            + algorithm.gamma * transition.values.squeeze(-1) * row["time_outs"].to(device).float()
        )
        transition.dones = row["dones"].to(device)
        storage.add_transition(transition)
    last = algorithm.critic(rollout["last_observations"].to(device)).detach()
    advantage = 0
    for step in reversed(range(storage.num_transitions_per_env)):
        next_value = last if step == storage.num_transitions_per_env - 1 else storage.values[step + 1]
        continuing = 1.0 - storage.dones[step].float()
        delta = storage.rewards[step] + continuing * algorithm.gamma * next_value - storage.values[step]
        advantage = delta + continuing * algorithm.gamma * algorithm.lam * advantage
        storage.returns[step] = advantage + storage.values[step]
    storage.advantages = storage.returns - storage.values
    storage.advantages = (storage.advantages - storage.advantages.mean()) / (storage.advantages.std() + 1e-8)
    return storage


def _install_mixed_generator(model_storage, holder: dict, repeat: int) -> None:
    original = model_storage.mini_batch_generator

    def combined(num_mini_batches, num_epochs=8):
        real = holder.get("storage")
        if real is None:
            yield from original(num_mini_batches, num_epochs)
            return
        for model_batch, real_batch in zip(
            original(num_mini_batches, num_epochs), real.mini_batch_generator(num_mini_batches, num_epochs)
        ):
            fields = {
                key: torch.cat([getattr(model_batch, key)] + [getattr(real_batch, key)] * repeat)
                for key in ("observations", "actions", "values", "advantages", "returns", "old_actions_log_prob")
            }
            fields["old_distribution_params"] = tuple(
                torch.cat([first] + [second] * repeat)
                for first, second in zip(model_batch.old_distribution_params, real_batch.old_distribution_params)
            )
            yield type(model_batch)(**fields)

    model_storage.mini_batch_generator = combined


def _update(algorithm, audit: bool) -> tuple[dict, dict | None]:
    from isaaclab_neural.rl.rsl_rl.ppo_diagnostics import PpoDiagnostics

    consistency = None
    if audit:
        diagnostics = PpoDiagnostics()
        consistency = diagnostics.check_consistency(algorithm)
        if not consistency["passed"]:
            raise RuntimeError("PPO branch storage/log-probability consistency failed.")
        with diagnostics.observe_update(algorithm):
            losses = algorithm.update()
        consistency["minibatch_kl"] = [row["kl_mean"] for row in diagnostics.minibatches]
        consistency["minibatch_lr"] = [row["effective_learning_rate"] for row in diagnostics.minibatches]
    else:
        losses = algorithm.update()
    if not all(math.isfinite(value) for value in losses.values()):
        raise FloatingPointError("Non-finite MBPO PPO loss.")
    return losses, consistency


def _append(path: Path, value: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False, default=float) + "\n")
    print(json.dumps(value, allow_nan=False, default=float), flush=True)
