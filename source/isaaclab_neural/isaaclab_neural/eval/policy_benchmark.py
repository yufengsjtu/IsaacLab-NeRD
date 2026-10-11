# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen-teacher positive controls and common first-episode policy benchmarks."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
import warp as wp
import yaml
from isaaclab_neural.data.opd import file_sha256


class EpisodeBenchmark:
    """Measure each environment's first episode, keeping failure and censoring explicit."""

    def __init__(self, num_envs: int, dt: float, device):
        self.dt = dt
        self.alive = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.failed = torch.zeros_like(self.alive)
        self.timed_out = torch.zeros_like(self.alive)
        self.lengths = torch.zeros(num_envs, device=device)
        self.returns = torch.zeros_like(self.lengths)
        self.xy_sum = torch.zeros_like(self.lengths)
        self.yaw_sum = torch.zeros_like(self.lengths)
        self.counts = torch.zeros(num_envs, dtype=torch.long, device=device)

    def record(self, velocity_xy, yaw_rate, command, reward, terminated, truncated):
        alive = self.alive.clone()
        self.lengths += alive * self.dt
        self.returns += torch.where(alive, reward, 0)
        done = terminated | truncated
        valid = alive & ~done
        self.xy_sum += torch.where(valid, (velocity_xy - command[:, :2]).norm(dim=-1), 0)
        self.yaw_sum += torch.where(valid, (yaw_rate - command[:, 2]).abs(), 0)
        self.counts += valid
        self.failed |= alive & terminated
        self.timed_out |= alive & truncated & ~terminated
        self.alive &= ~done

    def report(self):
        valid = self.counts > 0
        xy = self.xy_sum / self.counts.clamp(min=1)
        yaw = self.yaw_sum / self.counts.clamp(min=1)
        return {
            "failure_rate": float(self.failed.float().mean()),
            "timeout_rate": float(self.timed_out.float().mean()),
            "right_censored_rate": float(self.alive.float().mean()),
            "episode_lengths_s": self.lengths.cpu().tolist(),
            "episode_returns": self.returns.cpu().tolist(),
            "mean_episode_duration_s": float(self.lengths.mean()),
            "mean_episode_return": float(self.returns.mean()),
            "tracking_samples_per_env": self.counts.cpu().tolist(),
            "xy_error_episode_mean_mps": float(xy[valid].mean()) if valid.any() else None,
            "yaw_error_episode_mean_radps": float(yaw[valid].mean()) if valid.any() else None,
            "per_env_xy_error_mps": [float(v) if ok else None for v, ok in zip(xy.cpu(), valid.cpu())],
            "per_env_yaw_error_radps": [float(v) if ok else None for v, ok in zip(yaw.cpu(), valid.cpu())],
            "note": "First episodes only; tracking excludes reset steps and must be read with failures/censoring.",
        }


def load_benchmark_config(checkpoint, path: str) -> tuple[dict, str]:
    """Resolve explicit saved policy configuration without guessing architecture or modifying it."""
    opd = (checkpoint.get("infos") or {}).get("opd")
    if opd is not None:
        cfg = copy.deepcopy(opd["runner_cfg"])
        source = "opd_snapshot"
    else:
        filename = Path(path).parent / "params/agent.yaml"
        if not filename.is_file():
            raise FileNotFoundError(f"Benchmark needs the policy's saved agent config: {filename}")
        cfg = yaml.safe_load(filename.read_text())
        source = str(filename)
    if "class_name" not in cfg["algorithm"]:
        raise ValueError("Use a snapshot with intact/recovered runner metadata for benchmarking.")
    return cfg, source


def task_tracking_state(robot_data):
    """Return yaw-frame linear velocity [m/s] and world yaw rate [rad/s], matching task rewards."""
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat

    velocity = quat_apply_inverse(yaw_quat(robot_data.root_quat_w.torch), robot_data.root_lin_vel_w.torch)
    return velocity[:, :2], robot_data.root_ang_vel_w.torch[:, 2]


@torch.no_grad()
def run_policy_benchmark(env, adapter, args):
    """Evaluate a frozen teacher or OPD candidate with reward-aligned physical tracking metrics."""
    from isaaclab_neural.rl.rsl_rl.matched_mj import MatchedGeometryBridge
    from isaaclab_neural.rl.rsl_rl.online_opd import _write_json
    from isaaclab_neural.utils.python_utils import set_random_seed
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    base = env.unwrapped
    source = torch.load(args.policy_snapshot, map_location="cpu", weights_only=False)
    cfg, config_source = load_benchmark_config(source, args.policy_snapshot)
    bridge = MatchedGeometryBridge(base, adapter) if args.benchmark_mdp == "geometry" else None
    wrapped = RslRlVecEnvWrapper(env, clip_actions=cfg["clip_actions"])
    runner = OnPolicyRunner(wrapped, copy.deepcopy(cfg), log_dir=None, device=str(base.device))
    runner.alg.load(
        source, {"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False}, strict=True
    )
    policy = runner.get_inference_policy(device=str(base.device))
    actual = runner.alg._raw_actor.state_dict()
    if not all(torch.equal(v.cpu(), actual[k].cpu()) for k, v in source["actor_state_dict"].items()):
        raise RuntimeError("Loaded actor differs from the frozen source checkpoint.")
    set_random_seed(args.seed)
    obs, _ = wrapped.reset()
    if bridge:
        bridge.refresh()
    torch.save(
        {
            "joint_q": wp.to_torch(base.sim.physics_manager._state_0.joint_q).cpu().clone(),
            "joint_qd": wp.to_torch(base.sim.physics_manager._state_0.joint_qd).cpu().clone(),
            "command": base.command_manager.get_command("base_velocity").cpu().clone(),
            "observations": {k: v.cpu().clone() for k, v in obs.items()},
        },
        args.output / "initial_state.pt",
    )
    _write_json(args.output / "mdp.json", args.benchmark_mdp_contract)
    metrics = EpisodeBenchmark(base.num_envs, base.step_dt, base.device)
    steps = 0
    try:
        for steps in range(1, args.steps + 1):
            command = base.command_manager.get_command("base_velocity").clone()
            action = policy(obs, stochastic_output=False)
            obs, reward, done, extras = wrapped.step(action)
            robot = base.scene["robot"].data
            velocity, yaw = task_tracking_state(robot)
            if not torch.isfinite(velocity).all() or not torch.isfinite(yaw).all() or not torch.isfinite(reward).all():
                raise FloatingPointError("Non-finite benchmark state or reward.")
            metrics.record(
                velocity[:, :2],
                yaw,
                command,
                reward,
                base.termination_manager.terminated.bool(),
                base.termination_manager.time_outs.bool(),
            )
            policy.reset(done)
            if not metrics.alive.any():
                break
        unchanged = all(torch.equal(v.cpu(), actual[k].cpu()) for k, v in source["actor_state_dict"].items())
        _write_json(
            args.output / "report.json",
            {
                "policy": args.policy_snapshot,
                "policy_sha256": file_sha256(args.policy_snapshot),
                "config_source": config_source,
                "actor_unchanged": unchanged,
                "mdp": args.benchmark_mdp,
                "backend": args.backend,
                "seed": args.seed,
                "num_envs": base.num_envs,
                "max_control_steps": args.steps,
                "executed_control_steps": steps,
                "metrics": metrics.report(),
                "conditions": (
                    "Nominal physics; no observation noise; original commands/heading controller; first episode only"
                ),
                "tracking_frames": {"linear": "gravity-aligned yaw frame", "angular_z": "world frame"},
                "bridge_verified_readonly_frames": bridge.verified_readonly_frames if bridge else None,
            },
        )
    finally:
        if bridge:
            bridge.close()


class ArchiveLoader(yaml.SafeLoader):
    """Read archived config tuples/slices without enabling arbitrary Python object construction."""


ArchiveLoader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: loader.construct_sequence(node))
ArchiveLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:builtins.slice",
    lambda loader, node: str(slice(*loader.construct_sequence(node))),
)


def audit_teacher(run: Path, policy: Path, output: Path):
    """Tie the frozen teacher to saved training configuration and its actual TensorBoard history."""
    from isaaclab_neural.rl.rsl_rl.online_opd import _write_json
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    env = yaml.load((run / "params/env.yaml").read_text(), Loader=ArchiveLoader)
    agent = yaml.safe_load((run / "params/agent.yaml").read_text())
    installed_agent = yaml.safe_load((policy.parent / "params/agent.yaml").read_text())
    if agent != installed_agent or file_sha256(policy) != file_sha256(run / policy.name):
        raise ValueError("Teacher checkpoint/config does not match the located training run.")
    events = EventAccumulator(str(run), size_guidance={"scalars": 0})
    events.Reload()
    names = (
        "Train/mean_reward",
        "Train/mean_episode_length",
        "Loss/learning_rate",
        "Policy/mean_std",
        "Metrics/base_velocity/error_vel_xy",
        "Metrics/base_velocity/error_vel_yaw",
        "Episode_Termination/time_out",
        "Episode_Termination/base_contact",
    )
    curves = {tag: [{"step": value.step, "value": value.value} for value in events.Scalars(tag)] for tag in names}
    dt = env["sim"]["dt"] * env["decimation"]
    selected = {
        str(step): {tag: min(rows, key=lambda x: abs(x["step"] - step)) for tag, rows in curves.items()}
        for step in (0, 25, 100, 500, 1000, 1499)
    }
    for row in selected.values():
        row["mean_episode_length_s"] = row["Train/mean_episode_length"]["value"] * dt
    result = {
        "run": str(run),
        "teacher": str(policy),
        "teacher_sha256": file_sha256(policy),
        "checkpoint_matches_archive": True,
        "agent_config_matches_archive": True,
        "run_metadata": json.loads((run / "run.json").read_text()),
        "num_envs": env["scene"]["num_envs"],
        "configured_iterations": agent["max_iterations"],
        "last_logged_iteration": max(v["step"] for v in curves["Train/mean_reward"]),
        "initial_lr": agent["algorithm"]["learning_rate"],
        "seed": env["seed"],
        "configured_init_at_random_ep_len": agent["init_at_random_ep_len"],
        "control_dt": dt,
        "episode_length_s": env["episode_length_s"],
        "control_transitions": env["scene"]["num_envs"] * agent["num_steps_per_env"] * agent["max_iterations"],
        "selected_training_metrics": selected,
        "config_hashes": {p: file_sha256(run / p) for p in ("params/env.yaml", "params/agent.yaml", "run.json")},
        "warning": "Training log aggregations differ from the new frozen-policy benchmark; do not compare directly.",
    }
    _write_json(output / "report.json", result)
    _write_json(output / "training_curves.json", curves)
    _write_json(output / "archived_env.json", json.loads(json.dumps(env, default=str), parse_constant=str))
    _write_json(output / "archived_agent.json", agent)
    print(json.dumps(result, indent=2))


def summarize_anchor(root: Path, output: Path | None = None):
    """Summarize frozen controls and retain explicit configuration/budget differences."""
    from isaaclab_neural.rl.rsl_rl.online_opd import _write_json

    audit = json.loads((root / "training_audit/report.json").read_text())
    results = {
        "teacher_training": audit,
        "benchmark": {},
        "initial_state_checks": [],
        "claim_status": "Development controls only; final superiority evaluation not conducted",
        "policy_training_this_stage": False,
    }
    for name in ("teacher", "c1000", "opd15", "initial"):
        results["benchmark"][name] = {}
        for mdp in ("stock", "geometry"):
            reports = []
            for seed in (21,) if name == "initial" else (20, 21, 22):
                folder = root / f"{name}_{mdp}_seed{seed}"
                if name == "teacher" and mdp == "geometry" and seed == 21:
                    folder = root / "teacher_geometry_seed21_v2"
                report = json.loads((folder / "report.json").read_text())
                if not report["actor_unchanged"]:
                    raise RuntimeError("A benchmark actor was modified.")
                reports.append(report)
                ref = torch.load(
                    root / f"teacher_stock_seed{seed}" / "initial_state.pt", map_location="cpu", weights_only=False
                )
                state = torch.load(folder / "initial_state.pt", map_location="cpu", weights_only=False)
                for key in ("joint_q", "joint_qd", "command"):
                    torch.testing.assert_close(ref[key], state[key], rtol=0, atol=1e-5)
                for key in ref["observations"]:
                    torch.testing.assert_close(ref["observations"][key], state["observations"][key], rtol=0, atol=1e-5)
                results["initial_state_checks"].append(str(folder))
            metrics = [r["metrics"] for r in reports]
            results["benchmark"][name][mdp] = {
                "seeds": [r["seed"] for r in reports],
                "episodes": sum(r["num_envs"] for r in reports),
                "failure_rate": float(np.mean([m["failure_rate"] for m in metrics])),
                "timeout_rate": float(np.mean([m["timeout_rate"] for m in metrics])),
                "duration_mean_s": float(np.mean([m["mean_episode_duration_s"] for m in metrics])),
                "xy_error_episode_mean_mps": float(np.mean([m["xy_error_episode_mean_mps"] for m in metrics])),
                "yaw_error_episode_mean_radps": float(np.mean([m["yaw_error_episode_mean_radps"] for m in metrics])),
                "seed_results": [{"seed": r["seed"], "metrics": r["metrics"]} for r in reports],
            }
    archive = json.loads((root / "training_audit/archived_env.json").read_text())
    stock = json.loads((root / "teacher_stock_seed21/mdp.json").read_text())
    geometry = json.loads((root / "teacher_geometry_seed21_v2/mdp.json").read_text())

    def differences(a, b, prefix=""):
        if isinstance(a, dict) and isinstance(b, dict):
            rows = []
            for key in sorted(a.keys() | b.keys()):
                if key not in a or key not in b:
                    rows.append({"path": prefix + key, "before": a.get(key), "after": b.get(key)})
                else:
                    rows.extend(differences(a[key], b[key], prefix + key + "."))
            return rows
        return [] if a == b else [{"path": prefix.rstrip("."), "before": a, "after": b}]

    keys = ("actions", "observations", "commands", "events", "rewards", "terminations", "curriculum")
    results["mdp_differences"] = {
        "archived_training_to_nominal_stock": differences({k: archive[k] for k in keys}, {k: stock[k] for k in keys}),
        "nominal_stock_to_geometry": differences({k: stock[k] for k in keys}, {k: geometry[k] for k in keys}),
    }
    results["budget_comparison"] = {
        "teacher_control_transitions": audit["control_transitions"],
        "recent_mj_pilot_control_transitions": 128 * 24 * 100,
        "recent_online_opd_pilot_control_transitions": 8 * 24 * 15,
        "teacher_vs_mj_pilot_ratio": audit["control_transitions"] / (128 * 24 * 100),
        "teacher_vs_opd_pilot_ratio": audit["control_transitions"] / (8 * 24 * 15),
        "warning": "Unmatched budgets: the small pilots cannot establish inability to learn or an OPD effect.",
    }
    teacher = results["benchmark"]["teacher"]["stock"]
    candidate = results["benchmark"]["opd15"]["stock"]
    results["candidate_point_estimate_meets_nominal_target"] = (
        candidate["xy_error_episode_mean_mps"] <= 0.9 * teacher["xy_error_episode_mean_mps"]
        and candidate["yaw_error_episode_mean_radps"] <= 0.9 * teacher["yaw_error_episode_mean_radps"]
        and candidate["failure_rate"] <= teacher["failure_rate"]
    )
    results["limitations"] = [
        "Development evaluation seeds only; final test seeds remain reserved",
        "Nominal physics and no observation noise; not an exact replay of randomized teacher training",
        "Tracking is yaw-aligned/world-frame, not the older body-frame pilot metric",
        "Only first episodes count; failures must not be hidden by survivor-only tracking",
        "One training seed per frozen policy; no new policy training or claim of surpassing teacher",
    ]
    _write_json(output if output is not None else root / "result.json", results)
    print(
        json.dumps(
            {
                name: {mdp: {k: v for k, v in m.items() if k != "seed_results"} for mdp, m in rows.items()}
                for name, rows in results["benchmark"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit_run", "--audit-run", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summarize", type=Path)
    args = parser.parse_args()
    if args.summarize:
        summarize_anchor(args.summarize, args.output)
    elif args.audit_run and args.policy and args.output:
        audit_teacher(args.audit_run, args.policy, args.output)
    else:
        parser.error("Provide --summarize or --audit-run/--policy/--output.")
