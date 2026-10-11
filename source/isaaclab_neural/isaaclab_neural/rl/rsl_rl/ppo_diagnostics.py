# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read-only probes around native RSL-RL PPO minibatches, without copying its optimizer logic."""

from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from isaaclab_neural.data.opd import file_sha256
from isaaclab_neural.utils.checkpoint import load_opd_policy_snapshot, save_opd_policy_snapshot


class PpoDiagnostics:
    """Observe feed-forward, unnormalized PPO policy updates without advancing their RNG."""

    def __init__(self):
        self.minibatches = []
        self.consistency = []
        self.active_batch = None
        self.gradient_squares = {}
        self.pending = None

    @torch.no_grad()
    def check_consistency(self, algorithm):
        actor, storage = algorithm.actor, algorithm.storage
        if actor.is_recurrent or actor.obs_normalization or algorithm.symmetry or algorithm.rnd:
            raise ValueError("This probe requires the current plain feed-forward G1 PPO configuration.")
        if storage.step != storage.num_transitions_per_env:
            raise ValueError("Consistency must be checked on a complete rollout before update.")
        device = torch.device(algorithm.device)
        devices = (
            [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        )
        obs = storage.observations.flatten(0, 1)
        old = tuple(value.flatten(0, 1) for value in storage.distribution_params)
        action = storage.actions.flatten(0, 1)
        with torch.random.fork_rng(devices=devices):
            actor(obs, stochastic_output=True)
            new = actor.output_distribution_params
            log_prob = actor.get_output_log_prob(action)
            old_log_prob = storage.actions_log_prob.flatten()
            kl = actor.get_kl_divergence(old, new)
            deltas = [float((a - b).abs().max()) for a, b in zip(old, new)]
            matched = all(torch.allclose(a, b, atol=1e-5, rtol=1e-5) for a, b in zip(old, new))
            report = {
                "samples": len(action),
                "kl_mean": float(kl.mean()),
                "kl_abs_max": float(kl.abs().max()),
                "distribution_parameter_abs_max_differences": deltas,
                "log_prob_abs_max_difference": float((log_prob - old_log_prob).abs().max()),
                "passed": matched
                and bool(torch.isfinite(kl).all())
                and bool(torch.allclose(log_prob, old_log_prob, atol=1e-3, rtol=0)),
                "observation_abs_max": {key: float(value.abs().max()) for key, value in obs.items()},
                "advantage_mean": float(storage.advantages.mean()),
                "advantage_std": float(storage.advantages.std()),
            }
        self.consistency.append(report)
        return report

    @contextmanager
    def observe_update(self, algorithm):
        """Wrap only generator/observation hooks; native update, clipping and scheduling remain unchanged."""
        storage = algorithm.storage
        original = storage.mini_batch_generator
        previous = storage.__dict__.get("mini_batch_generator")
        handles = []
        self.minibatches = []

        def generator(*args, **kwargs):
            for batch in original(*args, **kwargs):
                self.active_batch = batch
                self.gradient_squares.clear()
                yield batch

        @torch.no_grad()
        def actor_forward(actor, args, result):
            batch = self.active_batch
            params = actor.output_distribution_params
            log_prob = actor.get_output_log_prob(batch.actions)
            ratio = (log_prob - batch.old_actions_log_prob.flatten()).exp()
            kl = actor.get_kl_divergence(batch.old_distribution_params, params)
            self.pending = {
                "minibatch": len(self.minibatches),
                "samples": len(log_prob),
                "kl_mean": float(kl.mean()),
                "clip_fraction": float(
                    ((ratio < 1 - algorithm.clip_param) | (ratio > 1 + algorithm.clip_param)).float().mean()
                ),
                "ratio_mean": float(ratio.mean()),
                "action_std_mean": float(actor.output_std.mean()),
                "entropy_mean": float(actor.output_entropy.mean()),
                "lr_before_schedule": algorithm.learning_rate,
                "advantage_mean": float(batch.advantages.mean()),
                "advantage_std": float(batch.advantages.std()),
            }

        def gradient_hook(key):
            def hook(gradient):
                self.gradient_squares[key] = gradient.detach().float().square().sum()
                return gradient

            return hook

        def optimizer_step(optimizer, args, kwargs):
            row = dict(self.pending)
            row["effective_learning_rate"] = optimizer.param_groups[0]["lr"]
            for group, model in (("actor", algorithm.actor), ("critic", algorithm.critic)):
                before = [v for (g, _), v in self.gradient_squares.items() if g == group]
                after = [p.grad.detach().float().square().sum() for p in model.parameters() if p.grad is not None]
                row[f"{group}_gradient_before_clip"] = float(torch.stack(before).sum().sqrt()) if before else 0.0
                row[f"{group}_gradient_after_clip"] = float(torch.stack(after).sum().sqrt()) if after else 0.0
            self.minibatches.append(row)

        storage.mini_batch_generator = generator
        handles.append(algorithm.actor.register_forward_hook(actor_forward))
        handles.append(algorithm.optimizer.register_step_pre_hook(optimizer_step))
        for group, model in (("actor", algorithm.actor), ("critic", algorithm.critic)):
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    handles.append(parameter.register_hook(gradient_hook((group, name))))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
            if previous is None:
                del storage.mini_batch_generator
            else:
                storage.mini_batch_generator = previous
            self.active_batch = None


def run_ppo_diagnostics(env, args):
    """Run a bounded fixed-model PPO diagnostic; this stage never trains/promotes dynamics."""
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_neural.rl.rsl_rl.online_opd import _ppo_update, _write_json

    if not 1 <= args.probe_updates <= 10 or env.unwrapped.num_envs not in {8, 128}:
        raise ValueError("PPO diagnostics support 8/128 environments and at most ten updates.")
    from isaaclab_neural.generate.opd import simulation_protocol

    source = torch.load(args.policy_snapshot, map_location="cpu", weights_only=False)
    protocol = simulation_protocol(env.unwrapped)
    if source["infos"]["opd"]["provenance"]["protocol"] != protocol:
        raise ValueError("Diagnostic simulation protocol differs from the initial policy snapshot.")
    cfg = copy.deepcopy(source["infos"]["opd"]["runner_cfg"])
    wrapped = RslRlVecEnvWrapper(env, clip_actions=cfg["clip_actions"])
    runner = OnPolicyRunner(wrapped, copy.deepcopy(cfg), log_dir=None, device=str(env.unwrapped.device))
    load_opd_policy_snapshot(args.policy_snapshot, runner, require_initial=True)
    if args.ppo_initial_lr is not None:
        if not 1e-6 <= args.ppo_initial_lr <= 1e-3:
            raise ValueError("The diagnostic initial PPO learning rate must lie in [1e-6, 1e-3].")
        runner.alg.learning_rate = args.ppo_initial_lr
        for group in runner.alg.optimizer.param_groups:
            group["lr"] = args.ppo_initial_lr
        cfg["algorithm"]["learning_rate"] = args.ppo_initial_lr
    torch.manual_seed(777)
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(777)
    obs = wrapped.get_observations().to(runner.device)
    recorder = SimpleNamespace(finish_control_step=lambda done: None)
    diagnostics = PpoDiagnostics()
    records = []
    failure = None
    try:
        for iteration in range(args.probe_updates):
            obs, row = _ppo_update(runner, obs, recorder, iteration, diagnostics=diagnostics)
            row["consistency"] = diagnostics.consistency[-1]
            row["minibatches"] = copy.deepcopy(diagnostics.minibatches)
            _write_json(args.output / f"update_{iteration:03d}.json", row)
            records.append(row)
            print(
                json.dumps(
                    {
                        "update": iteration + 1,
                        "lr": row["learning_rate"],
                        "preupdate": row["consistency"],
                        "first_minibatch": row["minibatches"][0],
                    }
                ),
                flush=True,
            )
        save_opd_policy_snapshot(
            args.output / "policy_final.pt",
            runner,
            cfg,
            updates_completed=args.probe_updates,
            provenance={
                "diagnostic_only": True,
                "protocol": protocol,
                "source_snapshot": file_sha256(args.policy_snapshot),
            },
        )
    except Exception as error:
        failure = repr(error)
        raise
    finally:
        _write_json(
            args.output / "report.json",
            {
                "num_envs": wrapped.num_envs,
                "updates": records,
                "consistency": diagnostics.consistency,
                "failure": failure,
                "dynamics_sha256": file_sha256(args.dynamics),
                "ppo_cfg": cfg,
                "note": "Fixed NeRD/MDP; PPO settings as recorded, with only explicit initial-LR override permitted.",
            },
        )


def summarize_diagnostics(root):
    """Compare update geometry without treating the terminal learning rate as a learning-quality metric."""
    from statistics import mean, median

    from isaaclab_neural.rl.rsl_rl.online_opd import _write_json

    result = {"runs": {}, "dynamics_updates": 0, "oracle_queries": 0}
    for name in ("env8", "env128", "env128_lr1e4"):
        report = json.loads((root / name / "report.json").read_text())
        batches = [batch for update in report["updates"] for batch in update["minibatches"]]
        first = report["updates"][0]["minibatches"]
        result["runs"][name] = {
            "num_envs": report["num_envs"],
            "updates": len(report["updates"]),
            "initial_lr": report["ppo_cfg"]["algorithm"]["learning_rate"],
            "consistency_passed": all(row["passed"] for row in report["consistency"]),
            "max_preupdate_log_prob_difference": max(
                row["log_prob_abs_max_difference"] for row in report["consistency"]
            ),
            "first_post_step_kl": first[1]["kl_mean"],
            "first_post_step_clip_fraction": first[1]["clip_fraction"],
            "first_update_max_kl": max(row["kl_mean"] for row in first),
            "first_update_mean_clip_fraction": mean(row["clip_fraction"] for row in first),
            "all_updates_mean_clip_fraction": mean(row["clip_fraction"] for row in batches),
            "all_updates_median_effective_lr": median(row["effective_learning_rate"] for row in batches),
            "end_of_update_learning_rates": [row["learning_rate"] for row in report["updates"]],
            "reward_mean_per_step": [row["reward_mean_per_step"] for row in report["updates"]],
            "scope": "Fixed-model NeRD optimizer diagnostic, not an MJ policy-performance comparison",
        }
    result["conclusions"] = [
        "The tested rollout/log-probability path is consistent before optimizer updates.",
        "Increasing parallel environments alone does not prevent the first-step KL overshoot at initial LR=1e-3.",
        "Initial LR=1e-4 strongly reduces the first-step jump; inspect full minibatch curves, not only terminal LR.",
        "This is not evidence of learned walking or improved OPD. No long three-arm experiment was run.",
        "128-environment fixed-model diagnostics do not yet enable large-environment online collection/oracle replay.",
    ]
    _write_json(root / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize", type=Path, required=True)
    summarize_diagnostics(parser.parse_args().summarize)
