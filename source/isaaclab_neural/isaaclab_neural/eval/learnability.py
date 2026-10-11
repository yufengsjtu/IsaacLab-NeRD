# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded fitting and input-sufficiency probes; outputs are not deployment candidates."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from isaaclab_neural.data.opd import ContextDataset, file_sha256
from isaaclab_neural.eval.dynamics_consistency import state_errors, summarize_errors
from isaaclab_neural.train.trainers import masked_dynamics_loss, validate_finetune_checkpoint
from isaaclab_neural.utils.checkpoint import (
    get_cfg_from_checkpoint,
    load_checkpoint,
    reconstruct_model_from_checkpoint,
    save_checkpoint,
)
from torch.utils.data import default_collate


def g1_error_groups(joint_names: list[str]) -> dict[str, list[int]]:
    """Partition floating-root state coordinates for physical-error attribution without omitting hands."""
    count = len(joint_names)
    groups = {
        "root_position": list(range(3)),
        "root_orientation": list(range(3, 7)),
        "root_linear_velocity": list(range(7 + count, 10 + count)),
        "root_angular_velocity": list(range(10 + count, 13 + count)),
        "legs": [],
        "torso": [],
        "arms": [],
        "hands": [],
    }
    fingers = tuple(f"_{name}_joint" for name in ("zero", "one", "two", "three", "four", "five", "six"))
    for i, name in enumerate(joint_names):
        group = (
            "hands"
            if name.endswith(fingers)
            else "legs"
            if any(part in name for part in ("hip", "knee", "ankle"))
            else "torso"
            if "torso" in name or "waist" in name
            else "arms"
        )
        groups[group].extend((7 + i, 13 + count + i))
    return {key: sorted(ids) for key, ids in groups.items()}


def sample_context_indices(dataset: ContextDataset, samples: int, *, seed: int) -> list[int]:
    """Select unique anchors with balanced per-shard quotas using a private sampling RNG."""
    if not 1 <= samples <= len(dataset.rows):
        raise ValueError("Requested unique sample count exceeds available contexts.")
    file_ids = sorted({file_id for file_id, _ in dataset.rows})
    groups = [
        np.asarray([i for i, (file_id, _) in enumerate(dataset.rows) if file_id == selected]) for selected in file_ids
    ]
    available = np.asarray([len(group) for group in groups])
    counts = np.minimum(available, samples // len(groups))
    rng = np.random.default_rng(seed)
    while counts.sum() < samples:
        eligible = np.flatnonzero(counts < available)
        chosen = rng.permutation(eligible)[: int(samples - counts.sum())]
        counts[chosen] += 1
    return (
        np.concatenate(
            [np.sort(rng.choice(group, int(count), replace=False)) for group, count in zip(groups, counts) if count]
        )
        .astype(np.int64)
        .tolist()
    )


def pd_input_statistics(path: str | Path) -> dict:
    """Audit actual and unclipped PD effort [N m] and contact-count changes."""
    with h5py.File(path, "r") as file:
        meta = json.loads(file.attrs["opd_metadata"])
        states = file["data/states"][:]
        targets = file["data/action_targets"][:]
        effort = file["data/joint_f"][:, :, 6:]
        protocol = meta["protocol"]
        kp = np.asarray(protocol["stiffness"])
        kd = np.asarray(protocol["damping"])
        limits = np.asarray(protocol["effort_limits"])
        n_joints = len(kp)
        num_q = 7 + n_joints
        raw = kp * (targets - states[..., 7:num_q]) - kd * states[..., num_q + 6 :]
        saturated = np.abs(raw) >= limits - 1e-3
        counts = np.diff(file["contacts/native_pairs/frame_offsets"][:]).reshape(states.shape[:2])
        identity = file["opd/query_identity"][:]
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "source_kind": meta["source_kind"],
        "anchors": len(states),
        "history_length": states.shape[1],
        "anchor_saturated_joint_fraction": float(saturated[:, -1].mean()),
        "anchor_any_saturation_fraction": float(saturated[:, -1].any(axis=-1).mean()),
        "pd_formula_vs_recorded_abs_max_nm": float(np.max(np.abs(np.clip(raw, -limits, limits) - effort))),
        "anchor_abs_effort_p95_nm": float(np.quantile(np.abs(effort[:, -1]), 0.95)),
        "contact_count_change_fraction": float((counts[:, -1] != counts[:, -2]).mean())
        if states.shape[1] > 1
        else None,
        "anchor_contact_count_p95": float(np.quantile(counts[:, -1], 0.95)),
        "per_joint_saturation_fraction": dict(zip(protocol["joint_names"], saturated[:, -1].mean(axis=0).tolist())),
        "control_phase_counts": {str(p): int(((identity[:, 2] % 4) == p).sum()) for p in range(4)},
        "note": (
            "Contact-count change is a proxy, not full contact-identity switching; "
            "overlapping contexts are not independent episodes."
        ),
    }


def _prepared(dataset, ids, solver, device):
    raw = default_collate([dataset[int(i)] for i in ids])
    batch = {key: value.to(device) for key, value in raw.items()}
    solver.process_neural_model_inputs(batch)
    batch["prediction_target"] = solver.convert_next_states_to_prediction(
        states=batch["states"], next_states=batch["next_states"], dt=0.005
    )
    return batch


def _clone_batch(batch):
    return {key: value.clone() for key, value in batch.items()}


@torch.no_grad()
def _context_cache(dataset, indices, solver, device, *, keep_sequence_targets=False):
    if keep_sequence_targets and getattr(dataset, "supervision", None) != "all_positions":
        raise ValueError("Only verified full-history datasets can retain sequence labels.")
    chunks, targets, truths, sequence_targets = {}, [], [], []
    for start in range(0, len(indices), 128):
        batch = _prepared(dataset, indices[start : start + 128], solver, device)
        target = batch.pop("prediction_target")
        targets.append(target[:, -1].cpu())
        truths.append(batch.pop("next_states")[:, -1].cpu())
        mask = batch.pop("supervision_mask")
        if keep_sequence_targets:
            if not bool(mask.all()):
                raise ValueError("Full-history cache cannot contain unsupervised positions.")
            sequence_targets.append(target.cpu())
        for key, value in batch.items():
            chunks.setdefault(key, []).append(value.detach().cpu())
    cache = {
        "inputs": {key: torch.cat(values) for key, values in chunks.items()},
        "target": torch.cat(targets),
        "truth": torch.cat(truths),
        "indices": list(indices),
        "source_hashes": {str(path): file_sha256(path) for path in dataset.paths},
        "families": sorted(dataset.families),
    }
    if keep_sequence_targets:
        cache["sequence_target"] = torch.cat(sequence_targets)
        cache["sequence_label_origin"] = "observed_mj_post_step"
        if not torch.isfinite(cache["sequence_target"]).all():
            raise FloatingPointError("Non-finite full-history target cache.")
    if not all(torch.isfinite(value).all() for value in (*cache["inputs"].values(), cache["target"], cache["truth"])):
        raise FloatingPointError("Non-finite physical-input cache.")
    return cache


def _cache_batch(cache, indices, device):
    indices = torch.as_tensor(indices, dtype=torch.long)
    inputs = {key: value.index_select(0, indices.to(value.device)).to(device) for key, value in cache["inputs"].items()}
    return (
        inputs,
        cache["target"].index_select(0, indices.to(cache["target"].device)).to(device),
        cache["truth"].index_select(0, indices.to(cache["truth"].device)).to(device),
    )


@torch.no_grad()
def _measure_context_cache(cache, model, solver, weights, groups, batch_size=64):
    losses, errors = [], {}
    device = next(model.parameters()).device
    for start in range(0, len(cache["target"]), batch_size):
        ids = list(range(start, min(start + batch_size, len(cache["target"]))))
        inputs, target, truth = _cache_batch(cache, ids, device)
        prediction = model(_clone_batch(inputs), single_step=True)[:, 0]
        components = ((prediction - target) * weights).square()
        if not torch.isfinite(components).all():
            raise FloatingPointError("Non-finite common evaluation loss.")
        losses.append(components.cpu())
        future = solver.convert_prediction_to_next_states(inputs["states"][:, -1:], prediction[:, None], dt=0.005)[:, 0]
        measures = state_errors(future, truth, solver.dof_q_per_env)
        for group in ("legs", "torso", "arms", "hands"):
            positions = [i for i in groups[group] if 7 <= i < solver.dof_q_per_env]
            velocities = [i for i in groups[group] if i >= solver.dof_q_per_env + 6]
            if positions:
                delta = future[:, positions] - truth[:, positions]
                measures[f"{group}_position_rmse_rad"] = torch.atan2(delta.sin(), delta.cos()).square().mean(-1).sqrt()
                measures[f"{group}_velocity_rmse_radps"] = (
                    (future[:, velocities] - truth[:, velocities]).square().mean(-1).sqrt()
                )
        for key, value in measures.items():
            errors.setdefault(key, []).append(value.cpu())
    mean = torch.cat(losses).double().mean(0)
    total = float(mean.mean())
    contributions = {key: float(mean[ids].sum() / len(mean)) for key, ids in groups.items()}
    return {
        "weighted_loss": total,
        "anchors": len(cache["target"]),
        "coordinate_error_mean": mean.tolist(),
        "group_loss_contribution": contributions,
        "group_loss_fraction": {key: value / max(total, 1e-30) for key, value in contributions.items()},
        "metrics": summarize_errors({key: torch.cat(values) for key, values in errors.items()}),
    }


def _selected_pd_statistics(dataset, indices, protocol):
    rows = {}
    for index in indices:
        file_id, row = dataset.rows[index]
        rows.setdefault(file_id, []).append(row)
    states, targets, efforts, contacts = [], [], [], []
    for file_id, selected in rows.items():
        with h5py.File(dataset.paths[file_id], "r") as file:
            meta = json.loads(file.attrs["opd_metadata"])
            if meta["protocol"] != protocol or meta["source_kind"] != "mjwarp":
                raise ValueError("Active fitting requires matching real MJ physics/control inputs.")
            selected = sorted(selected)
            states.append(file["data/states"][selected, -1])
            targets.append(file["data/action_targets"][selected, -1])
            efforts.append(file["data/joint_f"][selected, -1, 6:])
            counts = np.diff(file["contacts/native_pairs/frame_offsets"][:]).reshape(-1, 10)
            contacts.append(counts[selected])
    state, target, effort, counts = map(np.concatenate, (states, targets, efforts, contacts))
    n = len(protocol["joint_names"])
    raw = (
        np.asarray(protocol["stiffness"]) * (target - state[:, 7 : 7 + n])
        - np.asarray(protocol["damping"]) * state[:, 13 + n :]
    )
    limits = np.asarray(protocol["effort_limits"])
    saturated = np.abs(raw) >= limits - 1e-3
    np.testing.assert_allclose(np.clip(raw, -limits, limits), effort, rtol=1e-4, atol=1e-3)
    return {
        "anchors": len(state),
        "any_saturation_fraction": float(saturated.any(-1).mean()),
        "saturated_joint_fraction": float(saturated.mean()),
        "per_joint_saturation": dict(zip(protocol["joint_names"], saturated.mean(0).tolist())),
        "pd_reconstruction_abs_max_nm": float(np.max(np.abs(np.clip(raw, -limits, limits) - effort))),
        "contact_count_p95": float(np.quantile(counts[:, -1], 0.95)),
        "contact_count_changed_fraction": float((counts[:, -1] != counts[:, -2]).mean()),
        "root_height_p05_p95_m": np.quantile(state[:, 2], [0.05, 0.95]).tolist(),
        "joint_speed_abs_p95_radps": float(np.quantile(np.abs(state[:, 13 + n :]), 0.95)),
    }


def normalized_input_statistics(model, batch) -> dict:
    """Report standardized low-dimensional inputs without changing RMS."""
    result = {}
    for key in model.low_dim_input_names:
        normalized = model.input_rms[key].normalize(batch[key]).detach().abs().flatten()
        result[key] = {
            "abs_p95": float(torch.quantile(normalized, 0.95)),
            "abs_p99": float(torch.quantile(normalized, 0.99)),
            "abs_max": float(normalized.max()),
            "fraction_above_5": float((normalized > 5).float().mean()),
        }
    return result


def run_small_fit(
    env,
    dynamics: str,
    train_path: str,
    valid_path: str,
    output: Path,
    *,
    lr: float,
    updates: int,
    samples: int,
    seed: int,
) -> dict:
    """Overfit a fixed full batch from true MJ histories; keep a separate validation batch.

    Architecture, input representation, loss weights and RMS remain unchanged.
    No reference rehearsal, policy optimization, early stopping or promotion occurs.
    """
    solver = env.neural_adapter.solver
    checkpoint = load_checkpoint(dynamics, device=env.device)
    cfg = copy.deepcopy(get_cfg_from_checkpoint(checkpoint, dynamics))
    model = reconstruct_model_from_checkpoint(checkpoint, solver, device=env.device)
    validate_finetune_checkpoint(checkpoint, model, update_statistics=False)
    options = {
        "physics_dt": 0.005,
        "state_dim": solver.state_dim,
        "joint_f_dim": solver.joint_f_dim,
        "contact_filter": solver.contact_filter,
        "max_contact_tokens": solver.max_contact_tokens,
    }
    train_set = ContextDataset([train_path], split="train", **options)
    valid_set = ContextDataset([valid_path], split="validation", **options)
    if train_set.families & valid_set.families:
        raise ValueError("Training/validation family leakage.")
    for path in (train_path, valid_path):
        with h5py.File(path, "r") as file:
            meta = json.loads(file.attrs["opd_metadata"])
            if meta["source_kind"] != "mjwarp":
                raise ValueError("Learnability probes require true MJ histories, not student counterfactual contexts.")
    if len(set(train_set.lengths + valid_set.lengths)) != 1:
        raise ValueError("The fixed-batch probe requires a single exact history length.")
    if min(len(train_set), len(valid_set)) < samples:
        raise ValueError("Insufficient unique query rows for a fixed probe batch.")
    train_ids = np.random.default_rng(seed).choice(len(train_set), samples, replace=False)
    valid_ids = np.random.default_rng(seed + 1).choice(len(valid_set), samples, replace=False)
    with torch.no_grad():
        train = _prepared(train_set, train_ids, solver, env.device)
        valid = _prepared(valid_set, valid_ids, solver, env.device)
    weights = checkpoint["loss_weights"].to(env.device)
    stats_before = {k: v.clone() for k, v in model.state_dict().items() if k.startswith(("input_rms.", "output_rms."))}
    initial_weights = {k: v.clone() for k, v in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()
    curve = []

    @torch.no_grad()
    def measure(batch):
        model.eval()
        predicted = model(_clone_batch(batch))
        loss = masked_dynamics_loss(
            predicted * weights, batch["prediction_target"] * weights, mse, batch["supervision_mask"]
        )
        future = solver.convert_prediction_to_next_states(batch["states"], predicted, dt=0.005)
        errors = state_errors(future[:, -1], batch["next_states"][:, -1], solver.dof_q_per_env)
        return {"weighted_loss": float(loss), "metrics": summarize_errors(errors)}

    metadata = {
        "dynamics_sha256": file_sha256(dynamics),
        "train_sha256": file_sha256(train_path),
        "validation_sha256": file_sha256(valid_path),
        "train_indices": train_ids.tolist(),
        "validation_indices": valid_ids.tolist(),
        "samples": samples,
        "lr": lr,
        "max_updates": updates,
        "history_length": train_set.lengths[0],
        "input_scale_train": normalized_input_statistics(model, train),
        "input_scale_validation": normalized_input_statistics(model, valid),
        "pd_train": pd_input_statistics(train_path),
        "pd_validation": pd_input_statistics(valid_path),
        "protocol": (
            "Fixed full batch, pretrained initialization, no rehearsal, frozen RMS/weights, fresh Adam, grad clip 1.0"
        ),
    }
    (output / "protocol.json").write_text(json.dumps(metadata, indent=2))
    started = time.perf_counter()
    curve.append({"update": 0, "train": measure(train), "validation": measure(valid)})
    failure = None
    for step in range(1, updates + 1):
        model.train()
        optimizer.zero_grad()
        prediction = model(_clone_batch(train))
        loss = masked_dynamics_loss(
            prediction * weights, train["prediction_target"] * weights, mse, train["supervision_mask"]
        )
        if not torch.isfinite(loss):
            failure = f"Non-finite training loss at update {step}"
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step in {10, 100, 300, updates}:
            curve.append({"update": step, "train": measure(train), "validation": measure(valid)})
            print(json.dumps(curve[-1]), flush=True)
    final_state = model.state_dict()
    report = {
        **metadata,
        "curve": curve,
        "failure": failure,
        "elapsed_seconds": time.perf_counter() - started,
        "rms_unchanged": all(torch.equal(value, final_state[key]) for key, value in stats_before.items()),
        "changed_parameter_tensors": sum(
            not torch.equal(value, dict(model.named_parameters())[key]) for key, value in initial_weights.items()
        ),
        "accepted": False,
        "limitation": "Memorization success does not prove input sufficiency or broad generalization.",
    }
    if failure is None:
        cfg["diagnostic_only"] = {"probe": "small_fit", "samples": samples, "lr": lr, "updates": updates}
        save_checkpoint(
            output / "diagnostic_model.pt",
            model,
            "G1-Native-ContactTokens",
            cfg,
            optimizer_state_dict=optimizer.state_dict(),
            epoch=0,
            loss_weights=weights,
        )
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def _prepare_active_study(base, model, checkpoint, config, output):
    from isaaclab_neural.generate.opd import simulation_protocol
    from isaaclab_neural.rl.rsl_rl.online_opd import _write_json

    solver, device = base.neural_adapter.solver, base.device
    protocol = simulation_protocol(base)
    groups = g1_error_groups(protocol["joint_names"])
    comparison_root = Path(config["comparison_root"])
    comparison = json.loads((comparison_root / "result.json").read_text())
    validation = json.loads(Path(config["validation_manifest"]).read_text())
    if comparison["dynamics_updates"] != 0 or not comparison["protected_validation_unchanged"]:
        raise ValueError("Collector comparison did not preserve its frozen model/validation contract.")
    options = {
        "physics_dt": base.physics_dt,
        "state_dim": solver.state_dim,
        "joint_f_dim": solver.joint_f_dim,
        "contact_filter": solver.contact_filter,
        "max_contact_tokens": solver.max_contact_tokens,
    }
    manifest = {
        "schema": "active_context_study_v1",
        "config": config,
        "protocol": protocol,
        "groups": groups,
        "dynamics_sha256": file_sha256(config["dynamics"]),
        "training": {},
        "validation": {},
        "reference": {},
    }
    analysis = {
        "groups": groups,
        "arms": {},
        "scope": "Frozen F_pre, canonical offline batch64; not collector reward or a learned-model claim",
    }
    families = set()
    for arm in ("base", "error", "base_error"):
        report_path = Path(comparison["arms"][arm].get("report", comparison_root / f"train_{arm}/report.json"))
        if file_sha256(report_path) != comparison["arms"][arm]["report_sha256"]:
            raise ValueError("Collector report changed after comparison audit.")
        report = json.loads(report_path.read_text())
        paths = [p for p in report["labelled_training_and_validation_shards"] if Path(p).parent.name == "train"]
        dataset = ContextDataset(paths, split="train", **options)
        if set(dataset.lengths) != {10}:
            raise ValueError("Only real H10 anchors are supported.")
        ids = sample_context_indices(dataset, config["samples_per_arm"], seed=config["sampling_seed"])
        pd = _selected_pd_statistics(dataset, ids, protocol)
        cache = _context_cache(dataset, ids, solver, device)
        cache_path = output / f"train_{arm}.pt"
        torch.save(cache, cache_path)
        measure = _measure_context_cache(cache, model, solver, checkpoint["loss_weights"].to(device), groups)
        inputs, _, _ = _cache_batch(
            cache, np.linspace(0, len(ids) - 1, min(len(ids), 1024), dtype=int).tolist(), device
        )
        analysis["arms"][arm] = {
            "prediction_error": measure,
            "pd": pd,
            "input_scale": normalized_input_statistics(model, inputs),
            "sampled_anchors": len(ids),
        }
        families.update(dataset.families)
        manifest["training"][arm] = {
            "path": str(cache_path.resolve()),
            "sha256": file_sha256(cache_path),
            "anchors": len(ids),
            "source_hashes": cache["source_hashes"],
        }
        _write_json(output / f"analysis_{arm}.json", analysis["arms"][arm])
        print(
            json.dumps(
                {"prepared_arm": arm, "loss": measure["weighted_loss"], "contributions": measure["group_loss_fraction"]}
            ),
            flush=True,
        )
    reference = ContextDataset(config["reference_train"], split="train", **options)
    ref_cache = _context_cache(reference, list(range(len(reference))), solver, device)
    ref_path = output / "reference.pt"
    torch.save(ref_cache, ref_path)
    manifest["reference"] = {
        "path": str(ref_path.resolve()),
        "sha256": file_sha256(ref_path),
        "anchors": len(reference),
        "source_hashes": ref_cache["source_hashes"],
    }
    families.update(reference.families)
    for case in validation["cases"]:
        if not all(file_sha256(path) == digest for path, digest in case["artifacts_sha256"].items()):
            raise ValueError("Protected validation changed.")
        paths = [path for path in case["artifacts_sha256"] if path.endswith(".hdf5")]
        dataset = ContextDataset(paths, split="validation", **options)
        if families & dataset.families:
            raise ValueError("Training/protected-validation family overlap.")
        cache = _context_cache(dataset, list(range(len(dataset))), solver, device)
        name = f"{case['policy']}_seed{case['seed']}"
        path = output / f"validation_{name}.pt"
        torch.save(cache, path)
        sequences = [p for p in case["artifacts_sha256"] if p.endswith("sequences.pt")]
        manifest["validation"][name] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "anchors": len(dataset),
            "source_hashes": cache["source_hashes"],
            "seed": case["seed"],
            "policy": case["policy"],
            "sequence_path": sequences[0],
            "sequence_sha256": case["artifacts_sha256"][sequences[0]],
        }
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "analysis.json", analysis)


def _load_study_cache(entry, device):
    if file_sha256(entry["path"]) != entry["sha256"]:
        raise ValueError("Prepared input cache changed.")
    if not all(file_sha256(path) == digest for path, digest in entry["source_hashes"].items()):
        raise ValueError("A raw source shard changed after preparation.")
    return torch.load(entry["path"], map_location=device, weights_only=True)


@torch.no_grad()
def _measure_study_rollouts(base, model, weights, manifest, config):
    from contextlib import nullcontext

    import newton
    import warp as wp
    from isaaclab_neural.data.opd_sequence import read_sequence_shard
    from isaaclab_neural.train.opd_sequence import G1SequenceStudent

    solver = base.neural_adapter.solver
    fk_state = solver.model.state()
    feet = torch.tensor(
        [
            i
            for i, name in enumerate(solver.model.body_label)
            if str(name).endswith(("left_ankle_roll_link", "right_ankle_roll_link"))
        ],
        device=base.device,
    )
    if len(feet) != 2 * base.num_envs:
        raise ValueError("Sequence diagnostics require two ankle frames per G1 world.")

    def foot_positions(states):
        padded = torch.cat((states, states[-1:].expand(base.num_envs - len(states), -1)))
        stream = (
            wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(base.device)))
            if str(base.device).startswith("cuda")
            else nullcontext()
        )
        with stream:
            solver._assign_states_from_torch(fk_state, padded.contiguous())
            newton.eval_fk(solver.model, fk_state.joint_q, fk_state.joint_qd, fk_state)
            return (
                wp.to_torch(fk_state.body_q)
                .index_select(0, feet)[:, :3]
                .reshape(base.num_envs, 2, 3)[: len(states)]
                .clone()
            )

    result = {}
    for name, entry in manifest["validation"].items():
        if entry.get("seed") != config["heldout_seed"] or "sequence_path" not in entry:
            continue
        if file_sha256(entry["sequence_path"]) != entry["sequence_sha256"]:
            raise ValueError("Protected sequence validation changed.")
        shard = read_sequence_shard(entry["sequence_path"], require_labels=True)
        rng = np.random.default_rng(config["sampling_seed"] + entry["seed"])
        indices = sorted(rng.choice(len(shard["records"]), config["sequence_samples_per_case"], replace=False).tolist())
        records = [shard["records"][i] for i in indices]
        student = G1SequenceStudent(base, solver, model, shard["metadata"], weights)
        legs = torch.tensor(
            [
                i
                for i, joint in enumerate(shard["metadata"]["protocol"]["joint_names"])
                if any(part in joint for part in ("hip", "knee", "ankle"))
            ],
            device=base.device,
        )
        errors = {"4": {}, "16": {}, "short_prefix": {}}
        endpoints = {key: {} for key in errors}
        for length in sorted({len(row["actions"]) for row in records}):
            selected = [row for row in records if len(row["actions"]) == length]
            for start in range(0, len(selected), base.num_envs):
                batch = selected[start : start + base.num_envs]
                predicted, _, _, _ = student.rollout(batch, max_physics_steps=16)
                truth = torch.stack([row["oracle_states"] for row in batch]).to(base.device)
                ankle = torch.stack(
                    [
                        (foot_positions(predicted[:, t]) - foot_positions(truth[:, t])).norm(dim=-1).mean(-1)
                        for t in range(predicted.shape[1])
                    ],
                    dim=1,
                )
                leg_error = (
                    (predicted[..., 50:].index_select(-1, legs) - truth[..., 50:].index_select(-1, legs))
                    .square()
                    .mean(-1)
                    .sqrt()
                )
                for horizon in (4, 16):
                    group = "short_prefix" if horizon == 16 and predicted.shape[1] < 16 else str(horizon)
                    metrics = state_errors(predicted[:, :horizon], truth[:, :horizon], 44)
                    metrics.update(
                        ankle_frame_position_m=ankle[:, :horizon], leg_velocity_rmse_radps=leg_error[:, :horizon]
                    )
                    end = min(horizon, predicted.shape[1]) - 1
                    endpoint = state_errors(predicted[:, end], truth[:, end], 44)
                    endpoint.update(ankle_frame_position_m=ankle[:, end], leg_velocity_rmse_radps=leg_error[:, end])
                    for destination, values in ((errors[group], metrics), (endpoints[group], endpoint)):
                        for key, value in values.items():
                            destination.setdefault(key, []).append(value.cpu().flatten())
        result[name] = {
            "indices": indices,
            "records": len(records),
            "full16_records": sum(len(row["oracle_states"]) == 16 for row in records),
            "terminal_records": sum(row["terminal"] for row in records),
            "short_terminal_prefixes": sum(row["terminal"] and len(row["oracle_states"]) < 16 for row in records),
            "metrics": {
                h: summarize_errors({key: torch.cat(values) for key, values in groups.items()})
                for h, groups in errors.items()
            },
            "endpoint_metrics": {
                h: summarize_errors({key: torch.cat(values) for key, values in groups.items()})
                for h, groups in endpoints.items()
            },
            "scope": "Actual targets; terminal/iteration prefixes use only observed steps, never padding",
        }
    return result


def _fit_active_study(base, model, checkpoint, config, output):
    from dataclasses import asdict

    from isaaclab_neural.data.opd import CurrentEpochSampler
    from isaaclab_neural.rl.rsl_rl.online_opd import _write_json
    from isaaclab_neural.train.opd_sequence import OpdEpochConfig, _training_fingerprint, fit_opd_epochs

    manifest_path = Path(config["prepared_manifest"])
    manifest = json.loads(manifest_path.read_text())
    if manifest["dynamics_sha256"] != file_sha256(config["dynamics"]):
        raise ValueError("Prepared inputs require their original F_pre checkpoint.")
    if (config["learning_rate"], config["epochs"], config["current_batch_size"], config["micro_batch_size"]) != (
        1e-5,
        4,
        256,
        32,
    ):
        raise ValueError("The matched study fixesLR1e-5,4 epochs,current256 and microbatch32.")
    train = _load_study_cache(manifest["training"][config["arm"]], base.device)
    reference = _load_study_cache(manifest["reference"], base.device)
    validation = {name: _load_study_cache(entry, base.device) for name, entry in manifest["validation"].items()}
    solver, weights = base.neural_adapter.solver, checkpoint["loss_weights"].to(base.device).detach().clone()
    groups = manifest["groups"]

    def measure(cache):
        return _measure_context_cache(cache, model, solver, weights, groups)

    selection_protocol = config.get("selection_protocol", "legacy_policy_proxy")
    if selection_protocol == "old_new_retention":
        specification = config["retention_selection"]
        selection_names = specification["old_names"] + specification["new_names"]
        heldout_names = config["heldout_validation_cohorts"]
        if specification["old_kind"] != "original":
            raise ValueError(
                "Fitting selection requires original-data evidence; use read-only proxy evaluation otherwise."
            )
        if not all(
            manifest["validation"][name].get("role") == "old_original_validation" for name in specification["old_names"]
        ):
            raise ValueError("A policy/rehearsal proxy cannot certify original-data retention.")
        if not all(
            manifest["validation"][name].get("role") == "new_visited_distribution"
            for name in specification["new_names"]
        ):
            raise ValueError("New-data validation must use explicit held-out collector distributions.")
    elif selection_protocol == "legacy_policy_proxy":
        selection_names = [
            name for name, entry in manifest["validation"].items() if entry["seed"] == config["selection_seed"]
        ]
        heldout_names = [
            name for name, entry in manifest["validation"].items() if entry["seed"] == config["heldout_seed"]
        ]
        if len(selection_names) != 3 or len(heldout_names) != 3:
            raise ValueError("Legacy evaluation requires all three policy cohorts.")
    else:
        raise ValueError("Unknown model-selection protocol.")
    if not selection_names or not heldout_names or set(selection_names) & set(heldout_names):
        raise ValueError("Selection and final diagnostic cohorts must be nonempty and disjoint.")
    baseline_train = measure(train)
    baseline = {name: measure(validation[name]) for name in selection_names}
    baseline_reference = measure(reference)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    sampler = CurrentEpochSampler(
        len(train["target"]),
        reference_count=len(reference["target"]),
        replay_stage_sizes=[],
        seed=config["optimizer_seed"],
        current_batch_size=256,
    )
    epochs = OpdEpochConfig(
        min_epochs=1,
        num_epochs=4,
        max_epochs=4,
        max_optimizer_steps=4 * len(sampler),
        micro_batch_size=32,
        patience_epochs=5,
        stop_on_validation_gate=False,
    )
    states = model.state_dict()
    frozen = {
        key: value.clone()
        for key, value in states.items()
        if key.startswith(("input_rms.", "output_rms.", "body_net_contact_force_rms."))
    }
    checkpoint_cfg = copy.deepcopy(get_cfg_from_checkpoint(checkpoint, config["dynamics"]))
    checkpoint_cfg["active_model_study"] = {
        "config": config,
        "prepared_manifest_sha256": file_sha256(manifest_path),
        "epochs": asdict(epochs),
        "loss": "0.5 current one-step +0.5 reference one-step; no sequence loss",
    }
    validation_calls = 0

    def loss(source, indices):
        cache = reference if source == "reference" else train
        inputs, target, _ = _cache_batch(cache, indices, base.device)
        prediction = model(_clone_batch(inputs), single_step=True)[:, 0]
        return ((prediction - target) * weights).square().mean(-1)

    def validate():
        nonlocal validation_calls
        metrics = {name: measure(validation[name]) for name in selection_names}
        ratios = {
            name: metrics[name]["weighted_loss"] / max(baseline[name]["weighted_loss"], 1e-12)
            for name in selection_names
        }
        criterion = None
        if selection_protocol == "old_new_retention":
            from isaaclab_neural.eval.coverage import retention_adaptation_score

            criterion = retention_adaptation_score(
                {name: baseline[name]["weighted_loss"] for name in selection_names},
                {name: metrics[name]["weighted_loss"] for name in selection_names},
                **specification,
            )
            score, reference_ok = criterion["new_relative_loss"], criterion["old_preserved"]
            admissible = bool(reference_ok) if validation_calls == 0 else criterion["admissible"]
        else:
            score = float(np.mean(list(ratios.values())))
            reference_ok = all(
                value <= 1 + config["reference_regression_limit"]
                for name, value in ratios.items()
                if name.startswith("reference")
            )
            admissible = reference_ok and (validation_calls == 0 or score <= 1 - config["minimum_relative_gain"])
        state = _training_fingerprint(model, optimizer)
        row = {
            "loss": score,
            "admissible": admissible,
            "cohorts": metrics,
            "relative_losses": ratios,
            "selection_protocol": selection_protocol,
            "retention_adaptation": criterion,
            "reference_gate": reference_ok,
            "train": measure(train),
            "reference_train": measure(reference),
            "state_fingerprint": state,
        }
        save_checkpoint(
            output / f"epoch_{validation_calls:02d}.pt",
            model,
            "G1-Native-ContactTokens",
            checkpoint_cfg,
            optimizer_state_dict=optimizer.state_dict(),
            epoch=validation_calls,
            loss_weights=weights,
        )
        _write_json(output / f"epoch_{validation_calls:02d}.json", row)
        print(
            json.dumps(
                {
                    "epoch": validation_calls,
                    "selection_score": score,
                    "admissible": admissible,
                    "train_loss": row["train"]["weighted_loss"],
                    "ratios": ratios,
                }
            ),
            flush=True,
        )
        validation_calls += 1
        return row

    started = time.perf_counter()
    controller = fit_opd_epochs(model, optimizer, sampler, loss, validate, epochs)
    selected = (
        controller["epochs"][controller["selected_epoch"] - 1]["validation"]
        if controller["selected_epoch"]
        else controller["baseline_validation"]
    )
    if _training_fingerprint(model, optimizer) != selected["state_fingerprint"]:
        raise RuntimeError("Selected model/Adam pair fingerprint mismatch.")
    if not all(torch.equal(value, model.state_dict()[key]) for key, value in frozen.items()):
        raise RuntimeError("Frozen model normalization changed.")
    if not torch.equal(weights, checkpoint["loss_weights"].to(base.device)):
        raise RuntimeError("Frozen loss weights changed.")
    _write_json(output / "controller.json", controller)
    save_checkpoint(
        output / "selected.pt",
        model,
        "G1-Native-ContactTokens",
        checkpoint_cfg,
        optimizer_state_dict=optimizer.state_dict(),
        epoch=controller["selected_epoch"],
        loss_weights=weights,
    )
    selected_heldout = {name: measure(validation[name]) for name in heldout_names}
    selected_rollout = _measure_study_rollouts(base, model, weights, manifest, config)
    initial_model = reconstruct_model_from_checkpoint(checkpoint, solver, device=base.device).eval()
    baseline_heldout = {
        name: _measure_context_cache(validation[name], initial_model, solver, weights, groups) for name in heldout_names
    }
    baseline_rollout = _measure_study_rollouts(base, initial_model, weights, manifest, config)
    _write_json(
        output / "report.json",
        {
            "arm": config["arm"],
            "config": config,
            "epoch_config": asdict(epochs),
            "selection_protocol": selection_protocol,
            "controller": controller,
            "baseline_train": baseline_train,
            "baseline_selection": baseline,
            "baseline_reference_train": baseline_reference,
            "final_train": measure(train),
            "selected_heldout": selected_heldout,
            "baseline_heldout": baseline_heldout,
            "selected_rollout": selected_rollout,
            "baseline_rollout": baseline_rollout,
            "rms_unchanged": True,
            "model_adam_pair_verified": True,
            "policy_updates": 0,
            "selected_state_fingerprint": selected["state_fingerprint"],
            "elapsed_seconds": time.perf_counter() - started,
            "scope": "Single fit seed and matched data/updates; no live promotion or downstream-policy claim",
        },
    )


def run_active_fit_study(env, args) -> None:
    """Prepare or fit equal-size real-data cohorts without updating a policy or live dynamics."""
    from isaaclab_neural.rl.rsl_rl.online_opd import _pause_fingerprint, _write_json

    config = json.loads(Path(args.active_fit_config).read_text())
    if file_sha256(args.dynamics) != file_sha256(config["dynamics"]):
        raise ValueError("Study and launch checkpoints differ.")
    if config["samples_per_arm"] != 8192 or config["sampling_seed"] != 400:
        raise ValueError("This study fixes8192 unique anchors per arm and sampling seed400.")
    base, solver = env.unwrapped, env.unwrapped.neural_adapter.solver
    before = _pause_fingerprint(base, base.obs_buf)
    checkpoint = load_checkpoint(args.dynamics, device=base.device)
    model = reconstruct_model_from_checkpoint(checkpoint, solver, device=base.device)
    validate_finetune_checkpoint(checkpoint, model, update_statistics=False)
    model.eval()
    if config["mode"] == "prepare":
        _prepare_active_study(base, model, checkpoint, config, args.output)
    elif config["mode"] == "fit":
        _fit_active_study(base, model, checkpoint, config, args.output)
    else:
        raise ValueError("Unknown active model-study mode.")
    if _pause_fingerprint(base, base.obs_buf) != before:
        raise RuntimeError("Isolated active-model study changed the live environment.")
    _write_json(
        args.output / "integrity.json",
        {
            "live_environment_unchanged": True,
            "policy_updates": 0,
            "loss_weights_unchanged": True,
            "mode": config["mode"],
        },
    )


def shifted_saturated_actions(actions, raw_pd, limits, scale, *, delta_target: float, joints=None):
    """Change one saturated target [rad] deeper into saturation, preserving initial clipped PD."""
    saturated = raw_pd.abs() >= limits + 1e-3
    if joints is None:
        margin = raw_pd.abs() - limits
        margin = margin.masked_fill(~saturated, float("inf"))
        joints = margin.argmin(dim=-1)
    selected = saturated.gather(-1, joints[:, None]).squeeze(-1)
    if not selected.all():
        raise ValueError("Every selected joint must already be saturated.")
    result = actions.clone()
    rows = torch.arange(len(actions), device=actions.device)
    sign = raw_pd[rows, joints].sign()
    result[rows, joints] += sign * delta_target / scale[rows, joints]
    return result, joints


@torch.no_grad()
def run_target_probe(env, adapter, paths: list[str], output: Path, *, max_cases: int = 32, seed: int = 31) -> dict:
    """Test observed saturated anchors plus explicitly constructed saturation-boundary cases.

    Each A/A/B trial resets the same generalized state. Histories and geometry
    are unchanged; only a position target changes. Only first-step differences
    establish non-uniqueness under identical current model inputs.
    """
    import warp as wp
    from isaaclab_neural.generate.opd import simulation_protocol
    from isaaclab_neural.physics.applied_joint_f import backend_applied_effort

    protocol = simulation_protocol(env)
    kp = torch.tensor(protocol["stiffness"], device=env.device)
    kd = torch.tensor(protocol["damping"], device=env.device)
    limits = torch.tensor(protocol["effort_limits"], device=env.device)
    num_q = adapter.solver.dof_q_per_env
    candidates = []
    audits = []
    constructed_pool = []
    for path in paths:
        audits.append(pd_input_statistics(path))
        with h5py.File(path, "r") as file:
            meta = json.loads(file.attrs["opd_metadata"])
            if meta["protocol"] != protocol or meta["source_kind"] != "mjwarp":
                raise ValueError("Target probes require matching real MJ inputs.")
            state = torch.from_numpy(file["data/states"][:, -1].copy()).to(env.device)
            action = torch.from_numpy(file["data/actions"][:, -1].copy()).to(env.device)
            target = torch.from_numpy(file["data/action_targets"][:, -1].copy()).to(env.device)
        pd = kp * (target - state[:, 7:num_q]) - kd * state[:, num_q + 6 :]
        saturated = (pd.abs() >= limits + 1e-3) & (kp > 0)
        for i in range(len(state)):
            row = {"source": str(path), "index": i, "state": state[i], "action": action[i], "target": target[i]}
            constructed_pool.append(row)
            if saturated[i].any():
                margin = (pd[i].abs() - limits).masked_fill(~saturated[i], float("inf"))
                candidates.append({**row, "joint": int(margin.argmin()), "kind": "observed_saturation"})
    rng = np.random.default_rng(seed)
    if len(candidates) > max_cases:
        candidates = [candidates[int(i)] for i in rng.choice(len(candidates), max_cases, replace=False)]
    leg_joints = [i for i, name in enumerate(protocol["joint_names"]) if "hip_pitch" in name or "knee" in name]
    for i, row_id in enumerate(rng.choice(len(constructed_pool), min(8, len(constructed_pool)), replace=False)):
        candidates.append(
            {
                **constructed_pool[int(row_id)],
                "joint": leg_joints[i % len(leg_joints)],
                "kind": "constructed_boundary_not_observed_target",
            }
        )
    term = env.action_manager.get_term("joint_pos")
    if term.cfg.clip is not None:
        raise ValueError("The target probe requires unclipped affine joint-position actions.")
    scale = torch.broadcast_to(torch.as_tensor(term._scale, device=env.device), (env.num_envs, len(kp)))

    def rollout(state, actions):
        env.scene["robot"].actuators.reset()
        env.action_manager.reset()
        adapter.backend.state = env.sim.physics_manager._state_0
        adapter.control.joint_f.zero_()
        adapter.reset(state)
        torch.testing.assert_close(adapter.solver.states, state, rtol=0, atol=1e-5)
        adapter.step_action_frame(actions, process_action=True, overlay_applied_effort=False)
        if wp.to_torch(adapter.control.joint_f).any():
            raise RuntimeError("Unexpected analytic feedforward effort.")
        return adapter.solver.states.clone(), backend_applied_effort(env.scene["robot"]).clone()

    results = []
    for start in range(0, len(candidates), env.num_envs):
        original = candidates[start : start + env.num_envs]
        chunk = original + [original[-1]] * (env.num_envs - len(original))
        state = torch.stack([row["state"] for row in chunk])
        a = torch.stack([row["action"] for row in chunk])
        targets = torch.stack([row["target"] for row in chunk])
        joints = torch.tensor([row["joint"] for row in chunk], device=env.device)
        for row_id, row in enumerate(chunk):
            if row["kind"].startswith("constructed"):
                j = row["joint"]
                boundary = state[row_id, 7 + j] + (limits[j] + 0.1 + kd[j] * state[row_id, num_q + 6 + j]) / kp[j]
                a[row_id, j] += (boundary - targets[row_id, j]) / scale[row_id, j]
                targets[row_id, j] = boundary
        pd = kp * (targets - state[:, 7:num_q]) - kd * state[:, num_q + 6 :]
        b, _ = shifted_saturated_actions(a, pd, limits, scale, delta_target=0.25, joints=joints)
        first, effort_a = rollout(state, a)
        repeat, effort_repeat = rollout(state, a)
        changed, effort_b = rollout(state, b)
        aa = state_errors(first, repeat, num_q)
        ab = state_errors(first, changed, num_q)
        telemetry = (effort_a - effort_b).abs().amax(dim=-1)
        for i, row in enumerate(original):
            same_input = float(telemetry[i]) <= 1e-4
            metrics = {key: {"aa": float(aa[key][i]), "ab": float(ab[key][i])} for key in aa}
            floors = {
                "root_linear_velocity_mps": 1e-5,
                "joint_velocity_rmse_radps": 1e-4,
                "joint_position_rmse_rad": 1e-6,
            }
            separated = any(metrics[key]["ab"] > max(floor, 10 * metrics[key]["aa"]) for key, floor in floors.items())
            results.append(
                {
                    "source": row["source"],
                    "index": row["index"],
                    "kind": row["kind"],
                    "joint": protocol["joint_names"][row["joint"]],
                    "target_delta_rad": 0.25,
                    "a_target_rad": float(targets[i, row["joint"]]),
                    "same_initial_clipped_pd": same_input,
                    "pd_ab_abs_max_nm": float(telemetry[i]),
                    "pd_aa_abs_max_nm": float((effort_a[i] - effort_repeat[i]).abs().max()),
                    "next_state_separated_above_noise": same_input and separated,
                    "metrics": metrics,
                }
            )
    report = {
        "audits": audits,
        "cases": results,
        "physical_transitions": ((len(candidates) + env.num_envs - 1) // env.num_envs) * env.num_envs * 3,
        "protocol": (
            "Same state/history, same initial clipped PD, changed single target; "
            "first 5 ms A/A/B with cleared solver state"
        ),
        "limitation": (
            "Constructed boundary targets are stress tests, not evidence of their frequency in RL data. "
            "A finite set cannot prove full input sufficiency."
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report
