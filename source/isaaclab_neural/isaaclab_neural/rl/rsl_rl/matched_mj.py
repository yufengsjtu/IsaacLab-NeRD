# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analytic G1 baseline sharing the NeRD task's force-free contact MDP and timing."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections import defaultdict

import numpy as np
import torch
import warp as wp

from isaaclab.utils.dict import class_to_dict

from isaaclab_neural.utils.checkpoint import load_opd_policy_snapshot, save_opd_policy_snapshot


def mdp_contract(cfg) -> dict:
    """Serialize task semantics, excluding backend implementation and vectorization size."""
    keys = ("actions", "observations", "rewards", "terminations", "commands", "events", "curriculum")
    result = {key: class_to_dict(getattr(cfg, key)) for key in keys}
    scene = class_to_dict(cfg.scene)
    scene.pop("num_envs", None)
    result.update(
        scene=scene,
        physics_dt=cfg.sim.dt,
        decimation=cfg.decimation,
        episode_length_s=cfg.episode_length_s,
        gravity=cfg.sim.gravity,
    )
    return json.loads(json.dumps(result, default=str, sort_keys=True), parse_constant=str)


def configure_matched_backend(cfg, backend: str) -> dict:
    """Replace only the resolved physics configuration of the already-built NeRD task."""
    from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
    from isaaclab_tasks.core.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg
    from isaaclab_tasks.utils.hydra import resolve_presets

    before = mdp_contract(cfg)
    if backend == "mjwarp":
        source = cfg.sim.physics
        reference = G1FlatEnvCfg() if isinstance(cfg, G1FlatEnvCfg) else G1RoughEnvCfg()
        physics = copy.deepcopy(resolve_presets(reference.sim.physics))
        physics.collision_cfg = copy.deepcopy(source.collision_cfg)
        physics.default_shape_cfg = copy.deepcopy(source.default_shape_cfg)
        physics.use_cuda_graph = False
        cfg.sim.physics = physics
    elif backend != "nerd":
        raise ValueError("Matched G1 baseline supports only MJWarp and NeRD.")
    after = mdp_contract(cfg)
    if before != after:
        raise RuntimeError("Replacing the backend changed the task MDP.")
    digest = hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
    return {
        "mdp_unchanged": True,
        "mdp_sha256": digest,
        "mdp": before,
        "backend": backend,
        "physics": json.loads(json.dumps(class_to_dict(cfg.sim.physics), default=str), parse_constant=str),
    }


class MatchedGeometryBridge:
    """Refresh isolated collision geometry before MJ stepping, without overlaying PD effort.

    Reward callbacks consume these pre-step contacts after each complete physical
    frame, matching the one-frame lag of the NeRD task's geometric terms.
    """

    def __init__(self, env, adapter):
        if env.cfg.sim.physics.use_cuda_graph or env.sim.physics_manager._decimation != 1:
            raise ValueError("Matched contact timing requires eager one-frame stepping.")
        if adapter.contact_adapter is None:
            raise ValueError("Matched contact MDP requires a native geometry adapter.")
        self.env, self.adapter = env, adapter
        self.manager = env.sim.physics_manager
        self.calls = 0
        self.verified_readonly_frames = 0
        self._callback = self.update
        env.geometry_contact_adapter = adapter.contact_adapter
        self.manager.register_post_actuator_callback(self._callback)
        self.refresh()

    def refresh(self):
        contacts = self.adapter.backend.collide_native_contacts()
        self.adapter.contact_adapter.update(contacts, self.manager._state_0)

    def update(self):
        guarded = self.calls < 4
        watched = []
        if guarded:
            for obj, names in (
                (self.manager._state_0, ("joint_q", "joint_qd", "body_q", "body_qd", "body_f")),
                (self.manager._control, ("joint_f", "joint_target_q", "joint_target_qd")),
            ):
                for name in names:
                    view = wp.to_torch(getattr(obj, name))
                    watched.append((view, view.clone()))
            rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(self.env.device)
        self.refresh()
        if guarded:
            for view, before in watched:
                torch.testing.assert_close(view, before, rtol=0, atol=0)
            if not torch.equal(rng, torch.get_rng_state()) or not torch.equal(
                cuda_rng, torch.cuda.get_rng_state(self.env.device)
            ):
                raise RuntimeError("Geometry bridge advanced the policy RNG.")
            self.verified_readonly_frames += 1
        if bool(wp.to_torch(self.manager._control.joint_f).any()):
            raise RuntimeError("The analytic baseline must not inject shadow PD as feedforward effort.")
        self.calls += 1

    def close(self):
        self.manager._post_actuator_callbacks.remove(self._callback)
        del self.env.geometry_contact_adapter


class RolloutMetrics:
    """Track completed/censored episodes and nonterminal post-step tracking in SI units."""

    def __init__(self, env):
        self.env = env
        self.policy_update_id = 0
        self.lengths = torch.zeros(env.num_envs, device=env.device)
        self.command = env.command_manager.get_command("base_velocity").clone()
        self.completed = []
        self.errors = []
        self.yaw_errors = []
        self.terminations = defaultdict(int)

    def finish_control_step(self, done):
        env = self.env
        self.lengths += env.step_dt
        done = done.bool()
        self.completed.extend(self.lengths[done].cpu().tolist())
        self.lengths[done] = 0
        command = self.command
        valid = ~done
        velocity = env.scene["robot"].data.root_lin_vel_b.torch
        yaw = env.scene["robot"].data.root_ang_vel_b.torch[:, 2]
        self.errors.extend((velocity[valid, :2] - command[valid, :2]).norm(dim=-1).cpu().tolist())
        self.yaw_errors.extend((yaw[valid] - command[valid, 2]).abs().cpu().tolist())
        self.command.copy_(env.command_manager.get_command("base_velocity"))
        for name in env.termination_manager.active_terms:
            self.terminations[name] += int(env.termination_manager.get_term(name).sum())

    def report(self, *, clear=False):
        result = {
            "completed_episodes": len(self.completed),
            "completed_length_mean_s": float(np.mean(self.completed)) if self.completed else None,
            "completed_lengths_s": self.completed.copy(),
            "right_censored_lengths_s": self.lengths.cpu().tolist(),
            "xy_error_mean_mps": float(np.mean(self.errors)) if self.errors else None,
            "yaw_error_mean_radps": float(np.mean(self.yaw_errors)) if self.yaw_errors else None,
            "terminations": dict(self.terminations),
        }
        if clear:
            self.completed.clear()
            self.errors.clear()
            self.yaw_errors.clear()
            self.terminations.clear()
        return result


def run_matched_baseline(env, adapter, args):
    """Audit, train or evaluate using one canonical NeRD task MDP on the analytic backend."""
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_neural.generate.opd import simulation_protocol
    from isaaclab_neural.rl.rsl_rl.online_opd import _ppo_update, _write_json
    from isaaclab_neural.rl.rsl_rl.ppo_diagnostics import PpoDiagnostics
    from isaaclab_neural.utils.python_utils import set_random_seed

    base = env.unwrapped
    bridge = MatchedGeometryBridge(base, adapter) if args.backend == "mjwarp" else None
    source = torch.load(args.policy_snapshot, map_location="cpu", weights_only=False)
    info = source["infos"]["opd"]
    cfg = copy.deepcopy(info["runner_cfg"])
    mdp = args.matched_mdp_audit
    if "mdp_sha256" in info["provenance"] and info["provenance"]["mdp_sha256"] != mdp["mdp_sha256"]:
        raise ValueError("Checkpoint and evaluation MDP differ.")
    physics = simulation_protocol(base)
    if info["provenance"]["protocol"] != physics:
        raise ValueError("Actuation/physical asset protocol differs from the policy source.")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=cfg["clip_actions"])
    runner = OnPolicyRunner(wrapped, copy.deepcopy(cfg), log_dir=None, device=str(base.device))
    training = args.stage == "matched-mj-train"
    load_opd_policy_snapshot(args.policy_snapshot, runner, require_initial=training)
    if training:
        if not 1 <= args.probe_updates <= 100 or args.ppo_initial_lr != 1e-4:
            raise ValueError("This pilot allows at most 100 PPO updates with initial LR=1e-4.")
        runner.alg.learning_rate = args.ppo_initial_lr
        for group in runner.alg.optimizer.param_groups:
            group["lr"] = args.ppo_initial_lr
        cfg["algorithm"]["learning_rate"] = args.ppo_initial_lr
    set_random_seed(args.seed)
    obs, _ = wrapped.reset()
    if bridge:
        bridge.refresh()
    else:
        base.neural_adapter.sync(update_history=False)
    initial = {
        "observations": {key: value.cpu().clone() for key, value in obs.items()},
        "joint_q": wp.to_torch(base.sim.physics_manager._state_0.joint_q).cpu().clone(),
        "joint_qd": wp.to_torch(base.sim.physics_manager._state_0.joint_qd).cpu().clone(),
        "command": base.command_manager.get_command("base_velocity").cpu().clone(),
    }
    torch.save(initial, args.output / "initial_state.pt")
    _write_json(args.output / "mdp_audit.json", mdp)
    set_random_seed(777 if training else args.seed)
    metrics = RolloutMetrics(base)
    diagnostics = PpoDiagnostics()
    records = []
    failure = None
    try:
        if args.stage == "matched-mdp-audit":
            from isaaclab_neural.contacts.geometry_mdp import body_contacts

            feet = base.scene["robot"].find_bodies(".*_ankle_roll_link")[0]
            contact = body_contacts(feet, env=base).cpu()
            torch.save(contact, args.output / "initial_feet_contacts.pt")
            for _ in range(8):
                with torch.inference_mode():
                    obs, reward, done, extras = wrapped.step(
                        torch.zeros((base.num_envs, wrapped.num_actions), device=base.device)
                    )
                metrics.finish_control_step(done)
            records.append({"audit_steps": 8, "metrics": metrics.report()})
        elif training:
            for iteration in range(args.probe_updates):
                obs, row = _ppo_update(runner, obs, metrics, iteration, diagnostics=diagnostics)
                row["metrics"] = metrics.report(clear=True)
                row["consistency"] = diagnostics.consistency[-1]
                row["minibatches"] = copy.deepcopy(diagnostics.minibatches)
                _write_json(args.output / f"update_{iteration + 1:04d}.json", row)
                records.append({key: value for key, value in row.items() if key != "minibatches"})
                if iteration + 1 in {25, 50, 100, args.probe_updates}:
                    save_opd_policy_snapshot(
                        args.output / f"policy_{iteration + 1:04d}.pt",
                        runner,
                        cfg,
                        updates_completed=iteration + 1,
                        provenance={
                            "protocol": physics,
                            "mdp_sha256": mdp["mdp_sha256"],
                            "training_backend": args.backend,
                            "bounded_baseline": True,
                        },
                    )
                print(
                    json.dumps({key: value for key, value in row.items() if key not in {"minibatches", "metrics"}}),
                    flush=True,
                )
        else:
            policy = runner.get_inference_policy(device=str(base.device))
            for _ in range(args.steps):
                with torch.inference_mode():
                    action = policy(obs, stochastic_output=False)
                    obs, reward, done, extras = wrapped.step(action)
                metrics.finish_control_step(done)
                policy.reset(done)
            records.append({"ppo_updates": info["ppo_updates_completed"], "metrics": metrics.report()})
    except Exception as error:
        failure = repr(error)
        raise
    finally:
        _write_json(
            args.output / "report.json",
            {
                "backend": args.backend,
                "num_envs": base.num_envs,
                "mdp_sha256": mdp["mdp_sha256"],
                "ppo_cfg": cfg,
                "records": records,
                "failure": failure,
                "bridge_frames": bridge.calls if bridge else None,
                "verified_readonly_frames": bridge.verified_readonly_frames if bridge else None,
                "stage": args.stage,
                "env_seed": args.seed,
                "note": "Geometry contact MDP from NeRD, not stock G1 force-sensor rewards. No NeRD training or OPD.",
            },
        )
        if bridge:
            bridge.close()


def validate_policy_study(config: dict, *, num_envs: int, policy_updates: int) -> None:
    """Validate explicit extended-study budgets without changing the legacy pilot contract."""
    if config.get("mode") not in {"train", "eval"} or config.get("physics_scope") != "matched_geometry_nominal":
        raise ValueError("Policy studies require a declared matched-geometry training/evaluation scope.")
    if config.get("num_envs") != num_envs or not 4 <= num_envs <= 4096:
        raise ValueError("Policy study world count must match its explicit4..4096 budget.")
    if config["mode"] == "train":
        if policy_updates != 0 or not 1 <= config.get("updates", 0) <= 1500:
            raise ValueError("From-scratch policy studies require an untrained policy and at most1500 updates.")
        if not 0 < config.get("initial_learning_rate", 0) <= 1e-3:
            raise ValueError("Policy study initial learning rate exceeds its declared bound.")
    elif config.get("updates") != 0 or not 1 <= config.get("eval_steps", 0) <= 2000:
        raise ValueError("Frozen policy evaluation requires zero updates and at most2000 controls.")


class StudyRolloutMetrics(RolloutMetrics):
    """Record nonterminal training tracking in the reward's yaw/world frames."""

    def finish_control_step(self, done):
        from isaaclab_neural.eval.policy_benchmark import task_tracking_state

        self.lengths += self.env.step_dt
        done = done.bool()
        self.completed.extend(self.lengths[done].cpu().tolist())
        self.lengths[done] = 0
        velocity, yaw = task_tracking_state(self.env.scene["robot"].data)
        valid = ~done
        self.errors.extend((velocity[valid] - self.command[valid, :2]).norm(dim=-1).cpu().tolist())
        self.yaw_errors.extend((yaw[valid] - self.command[valid, 2]).abs().cpu().tolist())
        self.command.copy_(self.env.command_manager.get_command("base_velocity"))
        for name in self.env.termination_manager.active_terms:
            self.terminations[name] += int(self.env.termination_manager.get_term(name).sum())


def _evaluate_study_policy(base, runner, obs, steps):
    from isaaclab_neural.eval.policy_benchmark import EpisodeBenchmark, task_tracking_state

    metrics = EpisodeBenchmark(base.num_envs, base.step_dt, base.device)
    policy = runner.get_inference_policy(device=str(base.device))
    controls = 0
    for controls in range(1, steps + 1):
        command = base.command_manager.get_command("base_velocity").clone()
        with torch.inference_mode():
            action = policy(obs, stochastic_output=False)
            obs, reward, done, _ = runner.env.step(action)
            velocity, yaw = task_tracking_state(base.scene["robot"].data)
        if not torch.isfinite(velocity).all() or not torch.isfinite(yaw).all() or not torch.isfinite(reward).all():
            raise FloatingPointError("Nonfinite policy study evaluation.")
        metrics.record(
            velocity,
            yaw,
            command,
            reward,
            base.termination_manager.terminated.bool(),
            base.termination_manager.time_outs.bool(),
        )
        policy.reset(done)
        if not metrics.alive.any():
            break
    return {
        **metrics.report(),
        "control_steps": controls,
        "physical_transitions": controls * base.num_envs * base.cfg.decimation,
    }


def _configure_study_feature_cache(solver, enabled: bool, *, audit_frames: int = 0) -> dict:
    statistics = {
        "enabled": enabled,
        "audited_frames": 0,
        "max_abs_error": 0.0,
        "solver_type": type(solver).__name__,
        "history_capacity": getattr(solver, "num_states_history", None),
    }
    if not enabled:
        return statistics
    if (
        solver.anchor_frame_step != "every"
        or solver.states_frame != "body"
        or solver.use_kv_cache
        or solver.predicted_body_net_contact_forces_b is not None
        or not 0 <= audit_frames <= 128
    ):
        raise ValueError("Feature caching requires per-frame body inputs, no KV/auxiliary cache and bounded auditing.")
    solver.use_feature_cache = True
    if audit_frames:
        original = solver._model_forward

        def audited(inputs):
            reference = {
                key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in inputs.items()
            }
            expected = solver.neural_model(reference, single_step=True)
            actual = original(inputs)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
            statistics["audited_frames"] += 1
            statistics["max_abs_error"] = max(statistics["max_abs_error"], float((actual - expected).abs().max()))
            statistics["last_input_history"] = next(
                value.shape[1] for value in inputs.values() if isinstance(value, torch.Tensor)
            )
            cache = getattr(solver, "feature_cache", None)
            statistics["last_cache_shape"] = list(cache.shape) if cache is not None else None
            if statistics["audited_frames"] == audit_frames:
                solver._model_forward = original
            return actual

        solver._model_forward = audited
    return statistics


def run_policy_study(env, adapter, args) -> None:
    """Run an explicitly budgeted frozen-dynamics RL study without model updates or promotion."""
    from pathlib import Path

    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_neural.data.opd import file_sha256
    from isaaclab_neural.generate.opd import simulation_protocol
    from isaaclab_neural.rl.rsl_rl.online_opd import _ppo_update, _write_json, _write_status
    from isaaclab_neural.rl.rsl_rl.ppo_diagnostics import PpoDiagnostics
    from isaaclab_neural.utils.python_utils import set_random_seed

    config = json.loads(Path(args.rl_study_config).read_text())
    config_hash = file_sha256(args.rl_study_config)
    source = torch.load(args.policy_snapshot, map_location="cpu", weights_only=False)
    info = source["infos"]["opd"]
    base = env.unwrapped
    validate_policy_study(config, num_envs=base.num_envs, policy_updates=info["ppo_updates_completed"])
    if config.get("backend", args.backend) != args.backend or config["env_seed"] != args.seed:
        raise ValueError("Policy study backend or environment seed differs from the launcher.")
    if (
        file_sha256(args.dynamics) != config["dynamics_sha256"]
        or file_sha256(args.policy_snapshot) != config["policy_sha256"]
    ):
        raise ValueError("Policy study changed its locked model or initial policy.")
    physics = simulation_protocol(base)
    if physics != info["provenance"]["protocol"]:
        raise ValueError("Policy study actuator/physics protocol differs from its initialization.")
    mdp = args.matched_mdp_audit
    if config.get("mdp_sha256", mdp["mdp_sha256"]) != mdp["mdp_sha256"]:
        raise ValueError("Policy study task differs from its declared MDP.")
    if base.cfg.observations.policy.enable_corruption != config["policy_observation_corruption"]:
        raise ValueError("Policy observation-noise contract changed.")
    cfg = copy.deepcopy(info["runner_cfg"])
    if cfg["num_steps_per_env"] != 24:
        raise ValueError("Policy study requires the declared24-control rollout.")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=cfg["clip_actions"])
    runner = OnPolicyRunner(wrapped, copy.deepcopy(cfg), log_dir=None, device=str(base.device))
    training = config["mode"] == "train"
    load_opd_policy_snapshot(args.policy_snapshot, runner, require_initial=training)
    if training:
        runner.alg.learning_rate = config["initial_learning_rate"]
        for group in runner.alg.optimizer.param_groups:
            group["lr"] = runner.alg.learning_rate
        cfg["algorithm"]["learning_rate"] = runner.alg.learning_rate
    bridge = MatchedGeometryBridge(base, adapter) if args.backend == "mjwarp" else None
    set_random_seed(config["env_seed"])
    obs, _ = wrapped.reset()
    if bridge:
        bridge.refresh()
    else:
        base.neural_adapter.sync(update_history=False)
    initial_actor = {key: value.detach().clone() for key, value in runner.alg._raw_actor.state_dict().items()}
    _write_json(args.output / "mdp_audit.json", mdp)
    _write_json(args.output / "runner_config.json", cfg)
    torch.save(
        {
            "observations": {key: value.cpu() for key, value in obs.items()},
            "joint_q": wp.to_torch(base.sim.physics_manager._state_0.joint_q).cpu().clone(),
            "joint_qd": wp.to_torch(base.sim.physics_manager._state_0.joint_qd).cpu().clone(),
        },
        args.output / "initial_state.pt",
    )
    set_random_seed(config["policy_rng_seed"])
    if training and config.get("init_at_random_ep_len", True):
        base.episode_length_buf[:] = torch.randint_like(base.episode_length_buf, high=base.max_episode_length)
    provenance = {
        "protocol": physics,
        "mdp_sha256": mdp["mdp_sha256"],
        "training_backend": args.backend,
        "dynamics_sha256": config["dynamics_sha256"],
        "study_config_sha256": config_hash,
    }
    dynamics_model = base.neural_adapter.solver.neural_model if args.backend == "nerd" else None
    dynamics_state, inference_cache = None, None
    if config.get("use_feature_cache", False) and dynamics_model is None:
        raise ValueError("Feature caching is only defined for frozen NeRD inference.")
    if dynamics_model is not None:
        dynamics_model.eval().requires_grad_(False)
        dynamics_state = {key: value.detach().clone() for key, value in dynamics_model.state_dict().items()}
        inference_cache = _configure_study_feature_cache(
            base.neural_adapter.solver,
            config.get("use_feature_cache", False),
            audit_frames=config.get("feature_cache_audit_frames", 0),
        )
    provenance["use_feature_cache"] = bool(inference_cache and inference_cache["enabled"])
    metrics, diagnostic = StudyRolloutMetrics(base), PpoDiagnostics()
    records, failure, evaluation = [], None, None
    started = time.perf_counter()
    try:
        if training:
            save_opd_policy_snapshot(
                args.output / "policy_0000.pt", runner, cfg, updates_completed=0, provenance=provenance
            )
            for iteration in range(config["updates"]):
                checkpoint = iteration + 1 in config["checkpoints"] or iteration + 1 == config["updates"]
                check = diagnostic if iteration < 2 or checkpoint else None
                tick = time.perf_counter()
                obs, row = _ppo_update(runner, obs, metrics, iteration, diagnostics=check)
                torch.cuda.synchronize(base.device)
                row.update(seconds=time.perf_counter() - tick, metrics=metrics.report(clear=True))
                if check is not None:
                    row["consistency"] = diagnostic.consistency[-1]
                    row["minibatches"] = copy.deepcopy(diagnostic.minibatches)
                _write_json(args.output / f"update_{iteration + 1:04d}.json", row)
                records.append({key: value for key, value in row.items() if key != "minibatches"})
                if checkpoint:
                    save_opd_policy_snapshot(
                        args.output / f"policy_{iteration + 1:04d}.pt",
                        runner,
                        cfg,
                        updates_completed=iteration + 1,
                        provenance=provenance,
                    )
                _write_status(
                    args.output / "status.json",
                    {"status": "running", "updates": iteration + 1, "elapsed_seconds": time.perf_counter() - started},
                )
                print(
                    json.dumps({key: value for key, value in row.items() if key not in {"metrics", "minibatches"}}),
                    flush=True,
                )
        else:
            evaluation = _evaluate_study_policy(base, runner, obs, config["eval_steps"])
            assert all(
                torch.equal(value, runner.alg._raw_actor.state_dict()[key]) for key, value in initial_actor.items()
            )
        if dynamics_state is not None and any(
            not torch.equal(value, dynamics_model.state_dict()[key]) for key, value in dynamics_state.items()
        ):
            raise RuntimeError("Policy learning mutated its frozen dynamics model or normalization.")
        if file_sha256(args.dynamics) != config["dynamics_sha256"] or file_sha256(args.rl_study_config) != config_hash:
            raise RuntimeError("Frozen dynamics/study protocol changed during policy learning.")
    except Exception as error:
        failure = repr(error)
        raise
    finally:
        _write_json(
            args.output / "report.json",
            {
                "config": config,
                "config_sha256": config_hash,
                "backend": args.backend,
                "mdp_sha256": mdp["mdp_sha256"],
                "inference_cache": inference_cache,
                "records": records,
                "evaluation": evaluation,
                "failure": failure,
                "model_optimizer_updates": 0,
                "policy_updates": len(records),
                "elapsed_seconds": time.perf_counter() - started,
                "physical_transitions": evaluation["physical_transitions"]
                if evaluation is not None
                else len(records) * base.num_envs * cfg["num_steps_per_env"] * base.cfg.decimation,
                "transition_count_complete": failure is None,
                "note": "Matched geometry nominal MDP; not stock/noisy M0. No dynamics updates or global promotion.",
            },
        )
        _write_status(
            args.output / "status.json",
            {"status": "failed" if failure else "completed", "failure": failure, "updates": len(records)},
        )
        if bridge:
            bridge.close()


def summarize_baseline(root):
    """Audit the matched interface and summarize predeclared checkpoints without selecting a winner."""
    from isaaclab_neural.rl.rsl_rl.online_opd import _write_json

    protocol = json.loads((root / "protocol.json").read_text())
    train = json.loads((root / "train128/report.json").read_text())
    audits = {
        backend: json.loads((root / folder / "mdp_audit.json").read_text())
        for backend, folder in (("nerd", "audit_nerd_v2"), ("mjwarp", "audit_mjwarp_v3"))
    }
    if (
        audits["nerd"]["mdp_sha256"] != audits["mjwarp"]["mdp_sha256"]
        or train["mdp_sha256"] != audits["nerd"]["mdp_sha256"]
    ):
        raise RuntimeError("MDP fingerprint mismatch.")
    initial_states = [
        torch.load(root / folder / "initial_state.pt", map_location="cpu", weights_only=False)
        for folder in ("audit_nerd_v2", "audit_mjwarp_v3")
    ]
    interface_errors = {}
    for key in ("joint_q", "joint_qd", "command"):
        a, b = (state[key] for state in initial_states)
        torch.testing.assert_close(a, b, rtol=0, atol=1e-5)
        interface_errors[key] = float((a - b).abs().max())
    for key in initial_states[0]["observations"]:
        a, b = (state["observations"][key] for state in initial_states)
        torch.testing.assert_close(a, b, rtol=0, atol=1e-5)
        interface_errors[f"observations/{key}"] = float((a - b).abs().max())
    result = {
        "protocol": protocol,
        "mdp_sha256": train["mdp_sha256"],
        "initial_interface_max_errors": interface_errors,
        "ppo_updates": len(train["records"]),
        "all_consistency_checks_passed": all(r["consistency"]["passed"] for r in train["records"]),
        "training_physical_transitions": train["bridge_frames"] * train["num_envs"],
        "verified_readonly_frames": train["verified_readonly_frames"],
        "training_windows": {},
        "evaluation": {},
        "failure": train["failure"],
        "neural_dynamics_updates": 0,
        "opd_updates": 0,
    }
    for start, end in ((0, 10), (40, 50), (90, 100)):
        rows = train["records"][start:end]
        lengths = [value for row in rows for value in row["metrics"]["completed_lengths_s"]]
        result["training_windows"][f"{start + 1}-{end}"] = {
            "reward_mean_per_step": float(np.mean([r["reward_mean_per_step"] for r in rows])),
            "xy_error_mean_mps": float(np.mean([r["metrics"]["xy_error_mean_mps"] for r in rows])),
            "completed_episodes": len(lengths),
            "completed_length_mean_s": float(np.mean(lengths)) if lengths else None,
        }
    for name in ("initial", "0025", "0050", "0100"):
        report = json.loads((root / f"eval_{name}/report.json").read_text())
        if report["mdp_sha256"] != train["mdp_sha256"] or report["failure"] is not None:
            raise RuntimeError("Evaluation failed or changed MDP.")
        metrics = report["records"][0]["metrics"]
        result["evaluation"][name] = {
            key: metrics[key]
            for key in (
                "completed_episodes",
                "completed_length_mean_s",
                "right_censored_lengths_s",
                "xy_error_mean_mps",
                "yaw_error_mean_radps",
                "terminations",
            )
        }
    result["limitations"] = [
        "One training seed and one development evaluation seed; no final test seeds",
        "Geometry-contact MDP, not stock force-contact G1; old stock-MJ evaluation is not a matched comparison",
        "Completed episode lengths exclude right-censored survivors, which are reported separately",
        "This establishes a baseline only; no fixed-NeRD/online-OPD long-training comparison was run",
    ]
    _write_json(root / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize", type=Path, required=True)
    summarize_baseline(parser.parse_args().summarize)
