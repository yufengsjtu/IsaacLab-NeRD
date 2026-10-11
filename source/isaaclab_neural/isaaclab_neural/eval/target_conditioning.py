# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnostic equal-capacity target-error ablations, isolated from production rollout loading."""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from isaaclab_neural.data.opd import (
    CONTEXT_FIELDS,
    ContextDataset,
    ContextUpdateSampler,
    file_sha256,
    write_context_shard,
)
from isaaclab_neural.eval.dynamics_consistency import state_errors, summarize_errors
from isaaclab_neural.eval.learnability import shifted_saturated_actions
from isaaclab_neural.train.trainers import masked_dynamics_loss
from isaaclab_neural.utils.checkpoint import get_cfg_from_checkpoint, load_checkpoint, reconstruct_model_from_checkpoint
from isaaclab_neural.utils.running_mean_std import RunningMeanStd
from torch.utils.data import default_collate


class TargetProjection(torch.nn.Module):
    """Block-linear extension that retains the original GEMM shape and bias."""

    def __init__(self, original: torch.nn.Linear, extra_features: int):
        super().__init__()
        self.original = original
        self.target = torch.nn.Linear(
            extra_features,
            original.out_features,
            bias=False,
            device=original.weight.device,
            dtype=original.weight.dtype,
        )
        torch.nn.init.zeros_(self.target.weight)
        self.in_features = original.in_features + extra_features
        self.out_features = original.out_features

    def forward(self, features):
        boundary = self.original.in_features
        return self.original(features[..., :boundary].contiguous()) + self.target(features[..., boundary:].contiguous())


class TargetErrorModel(torch.nn.Module):
    """Append normalized target error [rad] after unchanged baseline encoder features.

    Both arms have the same expanded projection. The control masks the new feature
    after normalization. New columns start at zero; old columns, bias and RMS are copied.
    """

    def __init__(self, baseline, mean: torch.Tensor, variance: torch.Tensor, *, masked: bool):
        super().__init__()
        if not baseline.is_transformer or baseline.is_rnn or baseline.predict_body_net_contact_forces:
            raise ValueError("This diagnostic supports a state-only Transformer baseline.")
        if (
            mean.ndim != 1
            or variance.shape != mean.shape
            or not torch.isfinite(mean).all()
            or not torch.isfinite(variance).all()
        ):
            raise ValueError("Target statistics must be finite, matching feature vectors.")
        if (variance < 0).any():
            raise ValueError("Target variance cannot be negative.")
        self.base = copy.deepcopy(baseline)
        self.masked = masked
        self.target_rms = RunningMeanStd(shape=tuple(mean.shape), device=mean.device)
        self.target_rms.mean.copy_(mean)
        self.target_rms.var.copy_(variance)
        old = self.base.transformer_model.transformer.wte
        self.old_feature_width = old.in_features
        new = TargetProjection(old, mean.numel())
        self.base.transformer_model.transformer.wte = new
        self.base.transformer_model.config.vocab_size = new.in_features

    def forward(self, batch, single_step=False):
        inputs = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        if "joint_target_error" not in inputs:
            targets = getattr(self, "runtime_targets", None)
            if targets is None or targets.shape[:2] != inputs["states"].shape[:2]:
                raise ValueError("Runtime target history must match the actual state context.")
            inputs["joint_target_error"] = targets - inputs["states"][..., 7 : 7 + self.target_rms.mean.numel()]
        target = self.target_rms.normalize(inputs.pop("joint_target_error"))
        if self.masked:
            target = torch.zeros_like(target)
        inputs = self.base._prepare_inputs(inputs, inject_noise=False)
        old_features = self.base.extract_input_features(inputs)
        features = self.base.transformer_model(torch.cat((old_features, target), dim=-1))
        prediction = self.base._decode_state(features)
        return prediction[:, -1:] if single_step else prediction


class TargetContextDataset(ContextDataset):
    """Expose only pre-step position targets in addition to the existing context contract."""

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        file_id, row = self.rows[index]
        with h5py.File(self.paths[file_id], "r") as file:
            sample["action_targets"] = torch.from_numpy(file["data/action_targets"][row].copy())
        return sample


def prepare_target_batch(samples, solver, device):
    """Compute target error [rad] before coordinate conversion and without future-state access."""
    batch = {key: value.to(device) for key, value in default_collate(samples).items()}
    batch["joint_target_error"] = batch.pop("action_targets") - batch["states"][..., 7 : solver.dof_q_per_env]
    solver.process_neural_model_inputs(batch)
    batch["prediction_target"] = solver.convert_next_states_to_prediction(
        batch["states"], batch["next_states"], dt=0.005
    )
    return batch


def _subset_payload(payload, rows):
    length = payload["states"].shape[1]
    selected = {key: payload[key][rows].clone() for key in (*CONTEXT_FIELDS, "actions", "action_targets")}
    native = payload["native_contacts"]
    counts = native["frame_counts"].reshape(-1)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    frame_ids = [row * length + t for row in rows for t in range(length)]
    selected["native_contacts"] = {
        key: torch.cat([value[offsets[i] : offsets[i + 1]] for i in frame_ids])
        for key, value in native.items()
        if key != "frame_counts"
    }
    selected["native_contacts"]["frame_counts"] = native["frame_counts"][rows].clone()
    selected["trajectory_context"] = {key: value[rows].clone() for key, value in payload["trajectory_context"].items()}
    return selected


@torch.no_grad()
def make_pair_contexts(env, adapter, paths, output: Path, *, max_cases: int = 32, seed: int = 100):
    """Save A/B intervention pairs from real saturated states; no synthetic boundary cases."""
    import warp as wp
    from isaaclab_neural.generate.opd import _raw_rollouts, simulation_protocol
    from isaaclab_neural.physics.applied_joint_f import backend_applied_effort

    protocol = simulation_protocol(env)
    kp = torch.tensor(protocol["stiffness"], device=env.device)
    kd = torch.tensor(protocol["damping"], device=env.device)
    limits = torch.tensor(protocol["effort_limits"], device=env.device)
    scale = torch.broadcast_to(
        torch.as_tensor(env.action_manager.get_term("joint_pos")._scale, device=env.device), (env.num_envs, len(kp))
    )
    qsize = adapter.solver.dof_q_per_env
    rng = np.random.default_rng(seed)
    reports = []
    for source_id, filename in enumerate(paths):
        with h5py.File(filename, "r") as file:
            meta = json.loads(file.attrs["opd_metadata"])
            payload = _raw_rollouts(file)
        if meta["protocol"] != protocol or meta["source_kind"] != "mjwarp":
            raise ValueError("Pair probes require matching real MJ histories.")
        states = payload["states"][:, -1].to(env.device)
        targets = payload["action_targets"][:, -1].to(env.device)
        raw_pd = kp * (targets - states[:, 7:qsize]) - kd * states[:, qsize + 6 :]
        saturated = ((raw_pd.abs() >= limits + 1e-3) & (kp > 0)).any(-1)
        selected = torch.nonzero(saturated).flatten().cpu().numpy()
        if len(selected) > max_cases:
            selected = rng.choice(selected, max_cases, replace=False)
        if not len(selected):
            reports.append({"source": str(filename), "pairs": 0})
            continue
        saved_rows, labels, actions, new_targets, efforts, branches, pair_ids = [], [], [], [], [], [], []
        noise = []
        for start in range(0, len(selected), env.num_envs):
            rows = selected[start : start + env.num_envs].tolist()
            valid = len(rows)
            ids = rows + [rows[-1]] * (env.num_envs - valid)
            state = states[ids]
            original = payload["actions"][ids, -1].to(env.device)
            alternate, joints = shifted_saturated_actions(original, raw_pd[ids], limits, scale, delta_target=0.25)
            trial_states, trial_efforts, trial_targets = [], [], []
            for action in (original, original, alternate):
                env.scene["robot"].actuators.reset()
                env.action_manager.reset()
                adapter.backend.state = env.sim.physics_manager._state_0
                adapter.control.joint_f.zero_()
                adapter.reset(state)
                torch.testing.assert_close(adapter.solver.states, state, rtol=0, atol=1e-5)
                adapter.step_action_frame(action, overlay_applied_effort=False)
                if wp.to_torch(adapter.control.joint_f).any():
                    raise RuntimeError("Unexpected analytic feedforward force.")
                trial_states.append(adapter.solver.states.cpu().clone())
                trial_efforts.append(backend_applied_effort(env.scene["robot"]).cpu().clone())
                trial_targets.append(env.action_manager.get_term("joint_pos").processed_actions.cpu().clone())
            torch.testing.assert_close(trial_efforts[0], trial_efforts[2], rtol=0, atol=1e-4)
            torch.testing.assert_close(trial_efforts[0], payload["joint_f"][ids, -1, 6:], rtol=0, atol=1e-3)
            for local, row in enumerate(rows):
                noise.append(float((trial_states[0][local] - trial_states[1][local]).abs().max()))
                for branch, trial in ((0, 0), (1, 2)):
                    saved_rows.append(row)
                    labels.append(trial_states[trial][local])
                    actions.append((original if branch == 0 else alternate)[local].cpu())
                    new_targets.append(trial_targets[trial][local])
                    efforts.append(trial_efforts[0][local])
                    branches.append(branch)
                    pair_ids.append(start + local)
        data = _subset_payload(payload, saved_rows)
        data["actions"][:, -1] = torch.stack(actions)
        data["action_targets"][:, -1] = torch.stack(new_targets)
        data["joint_f"][:, -1, 6:] = torch.stack(efforts)
        meta = {
            **meta,
            "families": [meta["families"][r] for r in saved_rows],
            "pair_intervention": True,
            "source_shard_sha256": file_sha256(filename),
            "selection_seed": seed,
        }
        destination = output / f"pairs_{source_id:02d}.hdf5"
        write_context_shard(
            destination,
            data,
            metadata=meta,
            labels=torch.stack(labels),
            diagnostics={"branch": torch.tensor(branches), "pair_id": torch.tensor(pair_ids)},
        )
        reports.append(
            {
                "source": str(filename),
                "output": str(destination),
                "pairs": len(selected),
                "aa_state_abs_max": max(noise),
                "same_current_pd": True,
            }
        )
    (output / "report.json").write_text(json.dumps(reports, indent=2))
    return reports


def _cache_dataset(dataset):
    return [dataset[i] for i in range(len(dataset))]


def _measure_model(model, samples, solver, weights, device):
    groups = defaultdict(list)
    for sample in samples:
        groups[len(sample["states"])].append(sample)
    errors = defaultdict(list)
    total_loss = 0.0
    with torch.no_grad():
        model.eval()
        for group in groups.values():
            for start in range(0, len(group), 128):
                chunk = group[start : start + 128]
                batch = prepare_target_batch(chunk, solver, device)
                prediction = model(batch)
                loss = masked_dynamics_loss(
                    prediction * weights,
                    batch["prediction_target"] * weights,
                    torch.nn.MSELoss(),
                    batch["supervision_mask"],
                )
                total_loss += float(loss) * len(chunk)
                future = solver.convert_prediction_to_next_states(batch["states"], prediction, dt=0.005)
                for key, value in state_errors(
                    future[:, -1], batch["next_states"][:, -1], solver.dof_q_per_env
                ).items():
                    errors[key].append(value)
    return {
        "loss": total_loss / len(samples),
        "metrics": summarize_errors({k: torch.cat(v) for k, v in errors.items()}),
    }


def run_condition_fit(env, adapter, dynamics: str, config_path: str, output: Path):
    """Train paired diagnostic models with identical initialization, normalization and sampling."""
    import yaml

    config = yaml.safe_load(Path(config_path).read_text())
    updates = int(config["updates"])
    if not 1 <= updates <= 1000:
        raise ValueError("Conditioning diagnostics support 1..1000 updates.")
    solver = adapter.solver
    ckpt = load_checkpoint(dynamics, device=env.device)
    baseline = reconstruct_model_from_checkpoint(ckpt, solver, device=env.device).eval()
    weights = ckpt["loss_weights"].to(env.device)
    options = {
        "physics_dt": 0.005,
        "max_contact_tokens": solver.max_contact_tokens,
        "contact_filter": solver.contact_filter,
    }
    train = TargetContextDataset(config["train"], split="train", **options)
    valid = {
        name: TargetContextDataset(paths, split="validation", **options) for name, paths in config["validation"].items()
    }
    if any(train.families & val.families for val in valid.values()):
        raise ValueError("Pair/episode leakage across training and validation.")
    train_cache = _cache_dataset(train)
    valid_cache = {name: _cache_dataset(data) for name, data in valid.items()}
    errors = torch.cat(
        [sample["action_targets"] - sample["states"][:, 7 : solver.dof_q_per_env] for sample in train_cache]
    )
    mean = errors.mean(0).to(env.device)
    variance = errors.var(0, unbiased=False).to(env.device)
    baseline_rms = {
        k: v.clone() for k, v in baseline.state_dict().items() if k.startswith(("input_rms.", "output_rms."))
    }
    migrated = TargetErrorModel(baseline, mean, variance, masked=False).to(env.device)
    initial_state = copy.deepcopy(migrated.state_dict())
    same_length = [sample for sample in train_cache if len(sample["states"]) == train.lengths[0]]
    sample_batch = prepare_target_batch(same_length[:8], solver, env.device)
    with torch.no_grad():
        original = baseline({k: v.clone() for k, v in sample_batch.items()})
        enabled = migrated.eval()(sample_batch)
        migrated.masked = True
        disabled = migrated(sample_batch)
        torch.testing.assert_close(enabled, original, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(disabled, original, rtol=1e-5, atol=1e-5)
    reports = {}
    for masked in (True, False):
        name = "masked" if masked else "enabled"
        model = TargetErrorModel(baseline, mean, variance, masked=masked).to(env.device)
        model.load_state_dict(initial_state)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
        if config.get("sampling"):
            sampler = ContextUpdateSampler(train, config["batch_size"], seed=config["seed"], **config["sampling"])
            iterator = iter(sampler)
        else:
            sampler = None
        curve = [
            {
                "update": 0,
                "train": _measure_model(model, train_cache, solver, weights, env.device),
                "validation": {
                    k: _measure_model(model, v, solver, weights, env.device) for k, v in valid_cache.items()
                },
            }
        ]
        for update in range(1, updates + 1):
            if sampler is None:
                indices = list(range(len(train_cache)))
            else:
                try:
                    indices = next(iterator)
                except StopIteration:
                    iterator = iter(sampler)
                    indices = next(iterator)
            groups = defaultdict(list)
            for index in indices:
                groups[train.lengths[index]].append(train_cache[index])
            model.train()
            optimizer.zero_grad()
            for group in groups.values():
                batch = prepare_target_batch(group, solver, env.device)
                prediction = model(batch)
                loss = masked_dynamics_loss(
                    prediction * weights,
                    batch["prediction_target"] * weights,
                    torch.nn.MSELoss(),
                    batch["supervision_mask"],
                )
                (loss * (len(group) / len(indices))).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if update in {100, 300, updates}:
                curve.append(
                    {
                        "update": update,
                        "train": _measure_model(model, train_cache, solver, weights, env.device),
                        "validation": {
                            k: _measure_model(model, v, solver, weights, env.device) for k, v in valid_cache.items()
                        },
                    }
                )
                print(name, json.dumps(curve[-1]), flush=True)
        stats_equal = all(torch.equal(v, model.base.state_dict()[k]) for k, v in baseline_rms.items())
        with (output / f"{name}.pt").open("xb") as stream:
            torch.save(
                {
                    "format": "g1_target_conditioning_v1",
                    "masked": masked,
                    "state_dict": model.state_dict(),
                    "mean": mean.cpu(),
                    "variance": variance.cpu(),
                    "baseline_sha256": file_sha256(dynamics),
                    "baseline_cfg": get_cfg_from_checkpoint(ckpt, dynamics),
                    "updates": updates,
                    "config": config,
                    "diagnostic_only": True,
                },
                stream,
            )
        reports[name] = {
            "curve": curve,
            "old_rms_unchanged": stats_equal,
            "target_stats_unchanged": torch.equal(model.target_rms.mean, mean)
            and torch.equal(model.target_rms.var, variance),
            "new_column_norm": float(model.base.transformer_model.transformer.wte.target.weight.norm()),
            "sampling": sampler.audit.summary()
            if sampler is not None
            else {"fixed_rows": len(train), "updates": updates},
            "accepted": False,
        }
    reports["migration"] = {
        "max_abs_difference_enabled": float((enabled - original).abs().max()),
        "max_abs_difference_masked": float((disabled - original).abs().max()),
        "extra_channels": len(mean),
        "new_weights_initially_zero": True,
    }
    reports["data"] = {
        "train": {str(p): file_sha256(p) for p in train.paths},
        "validation": {k: {str(p): file_sha256(p) for p in v.paths} for k, v in valid.items()},
    }
    reports["config"] = config
    (output / "report.json").write_text(json.dumps(reports, indent=2))
    return reports


@torch.no_grad()
def score_condition_pairs(env, adapter, dynamics: str, checkpoint_path: str, paths, output: Path):
    """Measure whether the model resolves paired targets on train or disjoint validation states."""
    checkpoint = torch.load(checkpoint_path, map_location=env.device, weights_only=False)
    if checkpoint["baseline_sha256"] != file_sha256(dynamics):
        raise ValueError("Baseline mismatch.")
    original = load_checkpoint(dynamics, device=env.device)
    baseline = reconstruct_model_from_checkpoint(original, adapter.solver, device=env.device)
    model = TargetErrorModel(baseline, checkpoint["mean"], checkpoint["variance"], masked=checkpoint["masked"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    reports = []
    for path in paths:
        with h5py.File(path, "r") as file:
            meta = json.loads(file.attrs["opd_metadata"])
            branches = file["opd/branch"][:]
        if not np.array_equal(branches, np.tile([0, 1], len(branches) // 2)):
            raise ValueError("Pairs must have adjacent A/B branches.")
        dataset = TargetContextDataset(
            [path],
            split=meta["split"],
            physics_dt=0.005,
            contact_filter=adapter.solver.contact_filter,
            max_contact_tokens=adapter.solver.max_contact_tokens,
        )
        batch = prepare_target_batch([dataset[i] for i in range(len(dataset))], adapter.solver, env.device)
        for key in ("states_embedding", "joint_f", "gravity_dir", "contact_tokens"):
            torch.testing.assert_close(batch[key][0::2], batch[key][1::2], rtol=0, atol=0)
        prediction = model(batch)
        target = batch["prediction_target"]
        weights = original["loss_weights"].to(env.device)
        floor = ((target[1::2, -1] - target[0::2, -1]) * weights).square().mean() / 4
        loss = masked_dynamics_loss(
            prediction * weights, target * weights, torch.nn.MSELoss(), batch["supervision_mask"]
        )
        future = adapter.solver.convert_prediction_to_next_states(batch["states"], prediction, dt=0.005)[:, -1]
        truth = batch["next_states"][:, -1]
        start = adapter.solver.dof_q_per_env + 6
        predicted_delta = future[1::2, start:] - future[0::2, start:]
        true_delta = truth[1::2, start:] - truth[0::2, start:]
        contrast_error = (predicted_delta - true_delta).square().mean(-1).sqrt()
        magnitude = true_delta.square().mean(-1).sqrt()
        active = magnitude > 1e-4
        reports.append(
            {
                "source": str(path),
                "pairs": len(dataset) // 2,
                "masked": model.masked,
                "weighted_loss": float(loss),
                "no_target_pair_loss_lower_bound": float(floor),
                "gt_joint_velocity_contrast_mean_radps": float(magnitude.mean()),
                "contrast_error_mean_radps": float(contrast_error.mean()),
                "active_pairs": int(active.sum()),
                "active_contrast_error_mean_radps": float(contrast_error[active].mean()) if active.any() else None,
            }
        )
        np.savez_compressed(
            output / f"{Path(path).stem}_contrasts.npz",
            predicted=predicted_delta.cpu().numpy(),
            target=true_delta.cpu().numpy(),
        )
    (output / "report.json").write_text(json.dumps(reports, indent=2))
    return reports


@torch.no_grad()
def evaluate_conditioned(env, adapter, dynamics: str, checkpoint_path: str | None, paths, output: Path):
    """Evaluate through the existing NeRD stepping path with a diagnostic target-history hook."""
    from isaaclab_neural.eval.dynamics_consistency import evaluate_contexts

    solver = env.neural_adapter.solver
    if solver.use_feature_cache or solver.use_kv_cache:
        raise ValueError("The target-history diagnostic requires uncached inference.")
    original = solver.neural_model
    if checkpoint_path is None:
        result = evaluate_contexts(env, adapter, paths, output)
    else:
        payload = torch.load(checkpoint_path, map_location=env.device, weights_only=False)
        if payload["format"] != "g1_target_conditioning_v1" or payload["baseline_sha256"] != file_sha256(dynamics):
            raise ValueError("Diagnostic checkpoint/baseline mismatch.")
        model = TargetErrorModel(original, payload["mean"], payload["variance"], masked=payload["masked"])
        model.load_state_dict(payload["state_dict"])
        model.eval()
        solver.set_neural_solver_model(model)

        def start_context(history):
            model.runtime_targets = history["action_targets"][:, :-1].clone()

        def append_target():
            target = env.action_manager.get_term("joint_pos").processed_actions[:, None].clone()
            model.runtime_targets = torch.cat((model.runtime_targets, target), dim=1)[:, -solver.num_states_history :]

        manager = env.sim.physics_manager
        manager.register_post_actuator_callback(append_target)
        try:
            result = evaluate_contexts(
                env, adapter, paths, output, dataset_type=TargetContextDataset, on_context=start_context
            )
        finally:
            manager._post_actuator_callbacks.remove(append_target)
            solver.set_neural_solver_model(original)
            solver.reset_states_history()
    result["diagnostic_checkpoint"] = checkpoint_path
    result["accepted"] = False
    (output / "report.json").write_text(json.dumps(result, indent=2))
    return result
