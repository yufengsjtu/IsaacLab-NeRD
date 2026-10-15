# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnose velocity-limit terminations of control-step SO101 Keyboard NeRD models.

Modes:
    rollout: closed-loop typing (native MJWarp or a NeRD checkpoint) that records the robot joint speeds seen by
        the ``abnormal_robot`` termination term every control step, before resets.
    onestep: one-step prediction errors of NeRD checkpoints on recorded control-step data (true history), binned
        by the true joint speed relative to the velocity limit.
    all: run every rollout and the one-step analysis on the visible GPUs and write a summary.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROBOT_JOINTS = 6
HIST_BINS, HIST_MAX = 200, 10.0
SPEED_BINS = (0.0, 2.0, 3.0, 4.0, 4.5, 4.7, 4.8, 4.9, 5.0, 100.0)


def _rollout(args: argparse.Namespace) -> None:
    import torch

    import isaaclab.envs.mdp.terminations as terminations

    state: dict = {"hist": None, "joint_hist": None, "events": [], "steps": 0, "history": None, "limits": None}
    original_call = terminations.joint_vel_out_of_limit.__call__

    def patched(self, env, asset_cfg=None):
        joint_ids = self._joint_ids if self._joint_ids is not None else slice(None)
        limits = self._asset.data.soft_joint_vel_limits.torch[:, joint_ids]
        speed = self._asset.data.joint_vel.torch[:, joint_ids].abs()
        peak, joint = speed.max(dim=1)
        if state["hist"] is None:
            state["hist"] = torch.zeros(HIST_BINS, device=speed.device)
            state["joint_hist"] = torch.zeros(speed.shape[1], HIST_BINS, device=speed.device)
            state["history"] = torch.zeros(speed.shape[0], 4, device=speed.device)
            state["limits"] = limits[0].tolist()
            state["joint_names"] = list(self._asset.joint_names)
        state["hist"] += torch.histc(peak, bins=HIST_BINS, min=0.0, max=HIST_MAX)
        for column in range(speed.shape[1]):
            state["joint_hist"][column] += torch.histc(speed[:, column], bins=HIST_BINS, min=0.0, max=HIST_MAX)
        terminated = (speed > limits).any(dim=1)
        for world in torch.nonzero(terminated).flatten().tolist():
            state["events"].append(
                {
                    "step": state["steps"],
                    "world": world,
                    "episode_step": int(env.episode_length_buf[world]),
                    "joint": int(joint[world]),
                    "speed": float(peak[world]),
                    "limit": float(limits[world, joint[world]]),
                    "previous_peak_speeds": state["history"][world].tolist(),
                    "joint_speeds": speed[world].tolist(),
                    "joint_pos": self._asset.data.joint_pos.torch[world, joint_ids].tolist(),
                }
            )
        state["history"] = torch.cat((state["history"][:, 1:], peak[:, None]), dim=1)
        state["steps"] += 1
        return terminated

    terminations.joint_vel_out_of_limit.__call__ = patched
    from isaaclab_neural.eval import keyboard_policy

    argv = [
        "keyboard_policy",
        "--checkpoint",
        args.policy,
        "--num-envs",
        str(args.num_envs),
        "--control-steps",
        str(args.control_steps),
        "--seed",
        str(args.seed),
        "--reset-mode",
        "training",
        "--report",
        str(args.out.with_suffix(".policy.json")),
        "--visualizer",
        "none",
    ]
    argv += ["--keyboard-scene", args.keyboard_scene]
    if args.nerd_checkpoint:
        cfg = torch.load(args.nerd_checkpoint, map_location="cpu", weights_only=False)["cfg"]["env"]["neural_solver_cfg"]
        argv += ["--nerd-checkpoint", args.nerd_checkpoint, "--inference-precision", "fp16"]
        # Compiled inference applies to frame-cached models only.
        argv += ["--inference-compile"] if cfg.get("use_frame_cache") else []
        if args.reset_history:
            argv += ["--reset-history", args.reset_history]
    sys.argv = argv
    try:
        keyboard_policy.main()
    finally:
        terminations.joint_vel_out_of_limit.__call__ = original_call
    result = {
        "nerd_checkpoint": args.nerd_checkpoint,
        "seed": args.seed,
        "steps": state["steps"],
        "limits": state["limits"],
        "joint_names": state.get("joint_names"),
        "hist_bins": [HIST_BINS, HIST_MAX],
        "peak_speed_hist": state["hist"].tolist() if state["hist"] is not None else None,
        "joint_speed_hist": state["joint_hist"].tolist() if state["joint_hist"] is not None else None,
        "events": state["events"],
    }
    args.out.write_text(json.dumps(result) + "\n")


def _onestep(args: argparse.Namespace) -> None:
    import torch

    from isaaclab_neural.data.transition_index import SUPERVISION_MASK
    from isaaclab_neural.train import train as train_module
    from isaaclab_neural.train import trainers
    from isaaclab_neural.utils.checkpoint import load_checkpoint, reconstruct_model_from_checkpoint

    checkpoints = args.checkpoints

    def diagnose(self) -> None:
        metadata = self.cfg["keyboard_dataset_contracts"]["train"]
        num_q, keys = metadata["num_q"], set(metadata["key_q_indices"])
        robot_qd = torch.tensor([num_q + i for i in range(num_q) if i not in keys], device=self.device)
        limits = self.neural_env.neural_adapter.isaaclab_env.scene["robot"].data.soft_joint_vel_limits.torch[0]
        limits = limits[: robot_qd.numel()].to(self.device)
        datasets = {"train_seen": self.train_dataset, **{f"valid_{k}": v for k, v in self.valid_datasets.items()}}
        edges = torch.tensor(SPEED_BINS, device=self.device)
        report = {"limits": limits.tolist(), "speed_bins": list(SPEED_BINS), "results": {}}
        for checkpoint in checkpoints:
            model = reconstruct_model_from_checkpoint(load_checkpoint(checkpoint, device=self.device), self.neural_solver, device=self.device)
            model.eval()
            self.neural_model = model
            for name, dataset in datasets.items():
                for precision in ("fp32", "fp16"):
                    self.training_precision = precision
                    loader = self._make_dataloader(dataset=dataset, batch_size=512, shuffle=False, drop_last=False)
                    true_all, pred_all, position_all = [], [], []
                    for index, data in enumerate(loader):
                        if index >= args.max_batches:
                            break
                        data = self.preprocess_data_batch(data)
                        mask = data[SUPERVISION_MASK]
                        with torch.no_grad(), self._training_precision_context():
                            output = self._forward_model(data, False, return_aux=True)
                        prediction = trainers._float_model_output(output).state_prediction
                        next_states = self.neural_solver.convert_prediction_to_next_states(
                            states=data["states"], prediction=prediction, dt=self.neural_env.frame_dt
                        )
                        positions = torch.arange(mask.shape[1], device=self.device).expand_as(mask)
                        true_all.append(data["next_states"].index_select(-1, robot_qd)[mask])
                        pred_all.append(next_states.index_select(-1, robot_qd)[mask])
                        position_all.append(positions[mask])
                    true, pred, position = torch.cat(true_all), torch.cat(pred_all), torch.cat(position_all)
                    error = pred - true
                    along = error * torch.sign(true)
                    speed = true.abs()
                    bins = []
                    for low, high in zip(edges[:-1], edges[1:], strict=True):
                        selected = (speed >= low) & (speed < high)
                        count = int(selected.sum())
                        bins.append(
                            {
                                "true_speed": [float(low), float(high)],
                                "count": count,
                                "rmse": float(error[selected].square().mean().sqrt()) if count else None,
                                "mean_overshoot_along_velocity": float(along[selected].mean()) if count else None,
                                "pred_over_limit_fraction": float((pred.abs() > limits)[selected].float().mean())
                                if count
                                else None,
                            }
                        )
                    true_over = (true.abs() > limits).any(-1)
                    pred_over = (pred.abs() > limits).any(-1)
                    by_position = [
                        float(error[position == p].square().mean().sqrt()) for p in range(int(position.max()) + 1)
                    ]
                    report["results"][f"{Path(checkpoint).stem}/{name}/{precision}"] = {
                        "samples": int(true.shape[0]),
                        "robot_qd_rmse": float(error.square().mean().sqrt()),
                        "per_joint_rmse": error.square().mean(0).sqrt().tolist(),
                        "steps_true_over_limit": float(true_over.float().mean()),
                        "steps_pred_over_limit": float(pred_over.float().mean()),
                        "steps_false_over_limit": float((pred_over & ~true_over).float().mean()),
                        "rmse_by_window_position": by_position,
                        "bins": bins,
                    }
                    print("[onestep]", f"{Path(checkpoint).stem}/{name}/{precision}", json.dumps(report["results"][f"{Path(checkpoint).stem}/{name}/{precision}"])[:600], flush=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n")

    trainers.SequenceModelTrainer.test = diagnose
    overrides = " ".join(
        (
            f"algorithm.dataset.train_datasets.policy.path {args.train_file}",
            f"algorithm.dataset.valid_datasets.policy {args.valid_file}",
            "algorithm.dataset.max_capacity 1000000",
            "algorithm.dataset.valid_max_capacity 1000000",
            "algorithm.dataset.eager_preload_frames 65536",
        )
    )
    sys.argv = [
        "train",
        "--task",
        "IsaacContrib-Keyboard-SO101-NeRD",
        "--cfg",
        args.cfg,
        "--checkpoint",
        checkpoints[0],
        "--test",
        "--num-envs",
        "8",
        "--logdir",
        str(args.out.parent / "onestep_logs"),
        "--no-time-stamp",
        "--cfg-overrides",
        overrides,
        "--visualizer",
        "none",
    ]
    train_module.main()


def _run_all(args: argparse.Namespace) -> None:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    script = str(Path(__file__).resolve())
    # Entries are PATH or PATH@mode[,mode]; bare paths use --reset-history.
    specs = []
    for entry in args.checkpoints:
        path, _, modes = entry.partition("@")
        specs.append((str(Path(path).resolve()), modes.split(",") if modes else args.reset_history))
    checkpoints = [path for path, _ in specs]
    jobs = []
    for seed in args.seeds:
        if not args.skip_native:
            jobs.append(("native", None, None, seed))
        for checkpoint, modes in specs:
            for mode in modes:
                jobs.append((f"{Path(checkpoint).stem}-{mode}", checkpoint, mode, seed))
    commands = [
        (
            f"rollout_{name}_s{seed}",
            [sys.executable, script, "rollout", "--policy", args.policy, "--seed", str(seed), "--out",
             str(out / f"rollout_{name}_s{seed}.json"), "--num-envs", str(args.num_envs),
             "--control-steps", str(args.control_steps), "--keyboard-scene", args.keyboard_scene]
            + (["--nerd-checkpoint", checkpoint, "--reset-history", mode] if checkpoint else []),
        )
        for name, checkpoint, mode, seed in jobs
    ]
    if not args.skip_onestep:
        onestep = [sys.executable, script, "onestep", "--cfg", args.cfg, "--train-file", args.train_file]
        onestep += ["--valid-file", args.valid_file, "--out", str(out / "onestep.json")]
        onestep += ["--max-batches", str(args.max_batches), "--checkpoints", *checkpoints]
        commands.insert(0, ("onestep", onestep))
    slots = {gpu: args.per_gpu for gpu in args.gpus}
    running: list[tuple[subprocess.Popen, str, str]] = []
    failures = []
    while commands or running:
        for gpu in args.gpus:
            while commands and slots[gpu] > 0:
                name, command = commands.pop(0)
                slots[gpu] -= 1
                log = (out / f"{name}.log").open("w")
                print(f"[diag] start {name} on GPU {gpu}", flush=True)
                process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
                )
                running.append((process, name, gpu))
        time.sleep(5)
        for item in list(running):
            process, name, gpu = item
            if process.poll() is not None:
                running.remove(item)
                slots[gpu] += 1
                print(f"[diag] done {name} rc={process.returncode}", flush=True)
                if process.returncode:
                    failures.append(name)
    _summarize(out)
    if failures:
        print(f"[diag] failures: {failures}", flush=True)


def _summarize(out: Path) -> None:
    summary = {}
    for path in sorted(out.glob("rollout_*_s*.json")):
        if path.name.endswith(".policy.json"):
            continue
        data = json.loads(path.read_text())
        policy = json.loads(path.with_suffix(".policy.json").read_text())
        name = path.stem.rsplit("_s", 1)[0].removeprefix("rollout_")
        hist = data["peak_speed_hist"]
        width = HIST_MAX / HIST_BINS
        total = sum(hist)
        tail = {threshold: sum(h for i, h in enumerate(hist) if i * width >= threshold) / total for threshold in (4.0, 4.5, 4.8, 5.0, 6.0, 8.0)}
        events = data["events"]
        entry = summary.setdefault(name, {"episodes": 0, "success": 0, "abnormal": 0, "steps": 0, "tail": {}, "events": []})
        entry["episodes"] += policy["completed_episodes"]
        entry["success"] += policy["clean_successes"]
        entry["abnormal"] += policy["abnormal_robot_episodes"]
        entry["steps"] += data["steps"]
        for threshold, value in tail.items():
            entry["tail"].setdefault(str(threshold), []).append(round(value, 5))
        entry["events"] += events
    lines = []
    for name, entry in summary.items():
        events = entry.pop("events")
        speeds = sorted(event["speed"] - event["limit"] for event in events)
        joints = {}
        for event in events:
            joints[event["joint"]] = joints.get(event["joint"], 0) + 1
        episode_steps = sorted(event["episode_step"] for event in events)
        entry["success_rate"] = entry["success"] / max(1, entry["episodes"])
        entry["abnormal_events"] = len(events)
        entry["excess_over_limit_quantiles"] = (
            [round(speeds[int(q * (len(speeds) - 1))], 4) for q in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)] if speeds else None
        )
        entry["abnormal_joint_counts"] = joints
        entry["abnormal_episode_step_quantiles"] = (
            [episode_steps[int(q * (len(episode_steps) - 1))] for q in (0.0, 0.25, 0.5, 0.75, 1.0)] if episode_steps else None
        )
        lines.append(f"[diag-summary] {name}: {json.dumps(entry)}")
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("\n".join(lines), flush=True)
    onestep = out / "onestep.json"
    if onestep.exists():
        data = json.loads(onestep.read_text())
        for key, value in data["results"].items():
            compact = {k: value[k] for k in ("samples", "robot_qd_rmse", "steps_true_over_limit", "steps_pred_over_limit", "steps_false_over_limit")}
            compact["bins"] = [
                (b["true_speed"], b["count"], None if b["rmse"] is None else round(b["rmse"], 4),
                 None if b["mean_overshoot_along_velocity"] is None else round(b["mean_overshoot_along_velocity"], 4),
                 None if b["pred_over_limit_fraction"] is None else round(b["pred_over_limit_fraction"], 4))
                for b in value["bins"]
            ]
            print(f"[diag-onestep] {key}: {json.dumps(compact)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    rollout = sub.add_parser("rollout")
    rollout.add_argument("--policy", required=True)
    rollout.add_argument("--nerd-checkpoint")
    rollout.add_argument("--seed", type=int, required=True)
    rollout.add_argument("--num-envs", type=int, default=64)
    rollout.add_argument("--control-steps", type=int, default=450)
    rollout.add_argument("--reset-history", choices=("replicate", "episode_start"))
    rollout.add_argument("--keyboard-scene", choices=("fixed", "randomized"), default="fixed")
    rollout.add_argument("--out", type=Path, required=True)
    onestep = sub.add_parser("onestep")
    onestep.add_argument("--cfg", required=True)
    onestep.add_argument("--train-file", required=True)
    onestep.add_argument("--valid-file", required=True)
    onestep.add_argument("--checkpoints", nargs="+", required=True)
    onestep.add_argument("--max-batches", type=int, default=40)
    onestep.add_argument("--out", type=Path, required=True)
    everything = sub.add_parser("all")
    everything.add_argument("--policy", required=True)
    everything.add_argument("--cfg", required=True)
    everything.add_argument("--train-file", required=True)
    everything.add_argument("--valid-file", required=True)
    everything.add_argument("--checkpoints", nargs="*", default=[])
    everything.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    everything.add_argument("--num-envs", type=int, default=64)
    everything.add_argument("--control-steps", type=int, default=450)
    everything.add_argument("--max-batches", type=int, default=40)
    everything.add_argument("--gpus", nargs="+", default=["0", "1"])
    everything.add_argument("--per-gpu", type=int, default=2)
    everything.add_argument("--reset-history", nargs="+", default=["replicate"])
    everything.add_argument("--skip-onestep", action="store_true")
    everything.add_argument("--skip-native", action="store_true")
    everything.add_argument("--keyboard-scene", choices=("fixed", "randomized"), default="fixed")
    everything.add_argument("--out", type=Path, required=True)
    summarize = sub.add_parser("summarize")
    summarize.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "rollout":
        _rollout(args)
    elif args.mode == "onestep":
        _onestep(args)
    elif args.mode == "all":
        _run_all(args)
    else:
        _summarize(args.out)


if __name__ == "__main__":
    main()
