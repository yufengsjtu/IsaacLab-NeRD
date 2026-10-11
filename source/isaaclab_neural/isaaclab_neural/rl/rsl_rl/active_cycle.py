# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage-boundary active dynamics learning with fresh analytic data and explicit scorer acceptance."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from isaaclab_neural.data.opd import ContextDataset, file_sha256
from isaaclab_neural.rl.rsl_rl.online_opd import _write_json, _write_status
from isaaclab_neural.utils.checkpoint import load_checkpoint, save_checkpoint

_HISTORY_LENGTH = 10


def active_fit_checkpoints(config: dict) -> tuple[int, ...]:
    """Validate fitting budgets and return retained selection/checkpoint updates.

    The default pilot retains its original bounds and checkpoint cadence. An explicit
    fixed-data research study uses a predeclared sparse grid and never opens audit data.

    Args:
        config: Stage fitting configuration, optionally containing ``research_study``.

    Returns:
        Positive updates at which metrics are evaluated and complete checkpoints are retained.
    """
    updates = config.get("updates", 0)
    interval = config.get("check_every", 0)
    research = config.get("research_study")
    limit, maximum_lr = (128, 1e-5) if research is None else (4096, 1e-4)
    if (
        type(updates) is not int
        or not 1 <= updates <= limit
        or type(interval) is not int
        or interval < 1
        or not 0 < config.get("learning_rate", 0) <= maximum_lr
        or not 0 <= config.get("history_fraction", 0) <= 0.5
    ):
        raise ValueError("Invalid bounded stage fitting configuration.")
    if research is None:
        return tuple(sorted({1, updates, *range(interval, updates + 1, interval)}))
    if (
        not isinstance(research, dict)
        or research.get("kind") != "fixed_data_strength_v1"
        or config.get("truth_sampling") != "epoch"
        or not config.get("training_manifest")
        or config.get("defer_audit") is not True
        or config.get("record_step_zero") is not True
    ):
        raise ValueError("Research fitting requires immutable epoch data, step zero and deferred audit.")
    checkpoints = research.get("checkpoint_updates", [])
    if (
        not isinstance(checkpoints, list)
        or not 1 <= len(checkpoints) <= 8
        or any(type(step) is not int or not 1 <= step <= updates for step in checkpoints)
        or checkpoints != sorted(set(checkpoints))
        or checkpoints[-1] != updates
    ):
        raise ValueError("Research checkpoints must be a sparse ordered unique grid ending at the final update.")
    for key in ("minimum_free_bytes", "maximum_output_bytes"):
        if type(research.get(key)) is not int or research[key] <= 0:
            raise ValueError("Research fitting requires explicit positive storage boundaries.")
    return tuple(checkpoints)


def check_active_fit_storage(
    output: Path, *, minimum_free_bytes: int, maximum_output_bytes: int, reserve_bytes: int = 0
) -> dict[str, int]:
    """Check disk boundaries without removing any artifact.

    Args:
        output: Existing study output directory.
        minimum_free_bytes: Required filesystem headroom after the next write [byte].
        maximum_output_bytes: Maximum allocated storage for this stage [byte].
        reserve_bytes: Conservative space reservation for the next checkpoint [byte].

    Returns:
        Free, allocated output and reserved storage [byte].
    """
    if min(minimum_free_bytes, maximum_output_bytes) <= 0 or reserve_bytes < 0:
        raise ValueError("Storage boundaries must be positive and reservations nonnegative.")
    sizes = [path.stat() for path in output.iterdir() if path.is_file() and not path.is_symlink()]
    used = sum(getattr(info, "st_blocks", (info.st_size + 511) // 512) * 512 for info in sizes)
    free = shutil.disk_usage(output).free
    if free - reserve_bytes < minimum_free_bytes or used + reserve_bytes > maximum_output_bytes:
        raise RuntimeError(
            f"Storage boundary reached: free={free}, output={used}, reserve={reserve_bytes}; no files deleted"
        )
    return {"free_bytes": free, "output_bytes": used, "reserve_bytes": reserve_bytes}


def fresh_stage_batches(
    current_count: int, reference_count: int, *, updates: int, batch_size: int, seed: int
) -> list[dict]:
    """Consume distinct current/reference rows within a bounded stage using a private RNG."""
    count = updates * batch_size
    if updates < 1 or batch_size < 1 or min(current_count, reference_count) < count:
        raise ValueError("A refreshed stage needs enough unique anchors for every optimizer exposure.")
    rng = np.random.default_rng(seed)
    current = rng.permutation(current_count)[:count].reshape(updates, batch_size)
    reference = rng.permutation(reference_count)[:count].reshape(updates, batch_size)
    return [{"current": a.tolist(), "reference": b.tolist()} for a, b in zip(current, reference)]


def epoch_stage_batches(
    current_count: int, reference_count: int, *, updates: int, batch_size: int, seed: int
) -> list[dict]:
    """Replay equal-sized truth pools in complete paired epochs with an unchanged first fresh epoch."""
    if updates < 1 or batch_size < 1 or current_count != reference_count or current_count < batch_size:
        raise ValueError("Paired epoch replay requires equal nonempty pools and positive budgets.")
    if current_count % batch_size:
        raise ValueError("Paired epoch pools must divide into complete logical batches.")
    rng = np.random.default_rng(seed)
    result = []
    while len(result) < updates:
        current = rng.permutation(current_count).reshape(-1, batch_size)
        reference = rng.permutation(reference_count).reshape(-1, batch_size)
        result.extend({"current": a.tolist(), "reference": b.tolist()} for a, b in zip(current, reference))
    return result[:updates]


def preservation_batches(
    counts: dict[str, int], proportions: dict[str, float], *, updates: int, batch_size: int, seed: int
) -> list[dict]:
    """Sample input-only teacher domains with exact mixture quotas and explicit per-domain epoch reuse."""
    if updates < 1 or batch_size < 1 or not counts or set(counts) != set(proportions):
        raise ValueError("Preservation sampling requires matching sources and a positive budget.")
    if any(not math.isfinite(value) or value <= 0 for value in proportions.values()):
        raise ValueError("Preservation proportions must be finite and positive.")
    if not math.isclose(sum(proportions.values()), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Preservation proportions must sum to one.")
    result = [{} for _ in range(updates)]
    for offset, name in enumerate(sorted(counts)):
        raw = batch_size * proportions[name]
        quota = round(raw)
        if quota < 1 or not math.isclose(raw, quota, rel_tol=0, abs_tol=1e-10):
            raise ValueError("Preservation mixtures must have integral nonzero row quotas.")
        if counts[name] < quota or counts[name] % quota:
            raise ValueError("Preservation sources must contain complete logical batches without padding.")
        rng = np.random.default_rng(seed + offset)
        rows = []
        while len(rows) < updates * quota:
            rows.extend(rng.permutation(counts[name]).tolist())
        for step in range(updates):
            result[step][name] = rows[step * quota : (step + 1) * quota]
    return result


def _validate_preservation_cache(cache: dict, teacher_sha256: str) -> int:
    if (
        cache.get("kind") != "teacher_inputs_v1"
        or cache.get("split") != "train"
        or cache.get("evaluation_only", False)
        or cache.get("teacher_sha256") != teacher_sha256
        or cache.get("input_origin") not in {"recipe", "observed_sequence", "teacher_rollout"}
        or any(key in cache for key in ("truth", "target", "sequence_target"))
        or not cache.get("source_hashes")
    ):
        raise ValueError(
            "Preservation caches must be teacher-bound training inputs, not oracle truth or evaluation data."
        )
    inputs = cache.get("inputs", {})
    states = inputs.get("states")
    if not isinstance(states, torch.Tensor) or states.ndim != 3 or states.shape[1] != 10 or not len(states):
        raise ValueError("Preservation inputs require nonempty complete H10 state histories.")
    count = len(states)
    identities = cache.get("identities", [])
    if len(identities) != count or len(set(identities)) != count:
        raise ValueError("Preservation inputs need distinct stable row identities.")
    if any(
        not isinstance(value, torch.Tensor) or value.shape[:2] != (count, 10) or not torch.isfinite(value).all()
        for value in inputs.values()
    ):
        raise ValueError("Invalid or nonfinite preservation input histories.")
    return count


def _backward_preservation_inputs(
    model, teacher, caches, indices, weights, device, micro_batch_size, *, beta: float
) -> dict:
    if teacher.training or any(parameter.requires_grad for parameter in teacher.parameters()):
        raise ValueError("Preservation requires a frozen eval-mode teacher.")
    if not math.isfinite(beta) or beta <= 0 or micro_batch_size < 1:
        raise ValueError("Preservation requires positive finite coefficients and a valid microbatch.")
    count = sum(len(rows) for rows in indices.values())
    if not count or any(not rows for rows in indices.values()):
        raise ValueError("Preservation cannot use empty domains.")
    total, sources = 0.0, {}
    for name, rows in indices.items():
        source_loss = 0.0
        for start in range(0, len(rows), micro_batch_size):
            selected = torch.as_tensor(rows[start : start + micro_batch_size], dtype=torch.long)
            inputs = {
                key: value.index_select(0, selected.to(value.device)).to(device)
                for key, value in caches[name]["inputs"].items()
            }
            prediction = model({key: value.clone() for key, value in inputs.items()})
            with torch.no_grad():
                target = teacher({key: value.clone() for key, value in inputs.items()})
            loss = weighted_prediction_mse(prediction, target, weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite input-domain preservation loss.")
            (loss * (beta * len(selected) / count)).backward()
            source_loss += float(loss.detach()) * len(selected) / len(rows)
        proportion = len(rows) / count
        total += proportion * source_loss
        sources[name] = {"loss": source_loss, "inputs": len(rows), "proportion": proportion}
    return {"loss": total, "weighted_loss": beta * total, "sources": sources}


def weighted_prediction_mse(
    prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, *, pretraining: bool = True
) -> torch.Tensor:
    """Average prediction error using the pretrained weighted MSE operator.

    The deprecated ``pretraining`` keyword accepts only True; omit it in new callers.
    """
    if not pretraining:
        raise ValueError("Only the pretraining weighted MSE operator is supported.")
    if prediction.ndim not in (2, 3) or prediction.shape != target.shape or weights.shape != prediction.shape[-1:]:
        raise ValueError("Prediction and real targets must match exactly; anchor labels cannot be broadcast.")
    return torch.nn.functional.mse_loss(prediction * weights, target * weights)


def active_supervision_settings(config: dict) -> dict:
    """Validate the fixed full-history contract while accepting matching deprecated config fields."""
    settings = {
        "current_supervision": "all_positions",
        "reference_supervision": "all_positions",
        "current_validation_supervision": "all_positions",
        "pretraining_loss": True,
    }
    required = {**settings, "retain_current_targets": True, "retain_reference_targets": True}
    incompatible = [key for key, value in required.items() if config.get(key, value) != value]
    if incompatible:
        raise ValueError(f"Active fitting only supports pretraining-aligned full-history supervision: {incompatible}.")
    return settings


def decide_active_stage(baseline: dict, active: dict, candidate: dict, *, minimum_gain: float) -> dict:
    """Require new-stage improvement and cumulative old-recipe/prior-stage retention, without audit tuning."""
    for row in (baseline, active, candidate):
        if any(not math.isfinite(x) or x < 0 for x in (row["old_recipe"], row["current"], *row["history"])):
            raise ValueError("Stage selection requires finite nonnegative losses.")
    if not 0 <= minimum_gain < 1 or len(active["history"]) != len(candidate["history"]):
        raise ValueError("Invalid stage gate or historical cohort layout.")
    old_ok = candidate["old_recipe"] <= baseline["old_recipe"] + 1e-8
    current_ok = candidate["current"] < active["current"] and candidate["current"] <= active["current"] * (
        1 - minimum_gain
    )
    history_ok = all(after <= before + 1e-8 for before, after in zip(active["history"], candidate["history"]))
    return {
        "accepted": bool(old_ok and current_ok and history_ok),
        "old_preserved": bool(old_ok),
        "current_improved": bool(current_ok),
        "history_preserved": bool(history_ok),
        "old_recipe_ratio": candidate["old_recipe"] / max(baseline["old_recipe"], 1e-12),
        "current_ratio": candidate["current"] / max(active["current"], 1e-12),
    }


def next_active_snapshot(active: dict, fit: dict) -> dict:
    """Switch only an accepted child of the exact scorer that generated this stage."""
    if fit["parent_sha256"] != active["sha256"]:
        raise ValueError("A stale model cannot replace the current stage scorer.")
    return dict(fit["candidate"] if fit["accepted"] else active)


def _json(path):
    return json.loads(Path(path).read_text())


def _artifact(path):
    return {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}


def _join_caches(caches):
    for cache in caches:
        _require_full_history_cache("current", cache)
    keys = tuple(caches[0]["inputs"])
    if any(set(row["inputs"]) != set(keys) for row in caches):
        raise ValueError("Context caches use incompatible physical inputs.")
    return {
        "inputs": {key: torch.cat([row["inputs"][key] for row in caches]) for key in keys},
        "target": torch.cat([row["target"] for row in caches]),
        "truth": torch.cat([row["truth"] for row in caches]),
        "sequence_target": torch.cat([row["sequence_target"] for row in caches]),
        "sequence_label_origin": "observed_mj_post_step",
        "source_hashes": {path: digest for row in caches for path, digest in row["source_hashes"].items()},
        "families": sorted({family for row in caches for family in row["families"]}),
    }


def _current_stage_data(config, solver, device, output):
    from isaaclab_neural.eval.coverage import _identity_data, split_new_context_rows
    from isaaclab_neural.eval.learnability import _context_cache, sample_context_indices

    report = _json(config["collector_report"])
    if not report["dynamics_frozen"] or report["provenance"]["dynamics_sha256"] != config["parent_sha256"]:
        raise ValueError("Current data were not collected with the declared frozen scorer.")
    options = {
        "physics_dt": 0.005,
        "state_dim": 87,
        "joint_f_dim": 43,
        "contact_filter": solver.contact_filter,
        "max_contact_tokens": solver.max_contact_tokens,
        "max_history": _HISTORY_LENGTH,
        "supervision": "all_positions",
    }
    paths = report["labelled_training_and_validation_shards"]
    train = ContextDataset([path for path in paths if Path(path).parent.name == "train"], split="train", **options)
    valid = ContextDataset(
        [path for path in paths if Path(path).parent.name == "validation"], split="validation", **options
    )
    previous = config["history_training"]
    history_count = round(config["updates"] * config["batch_size"] * config["history_fraction"]) if previous else 0
    current_count = config["updates"] * config["batch_size"] - history_count
    ids = sample_context_indices(train, current_count, seed=config["seed"])
    if set(train.lengths + valid.lengths) != {_HISTORY_LENGTH}:
        raise ValueError("Active fitting requires ten real history positions.")
    current = _context_cache(train, ids, solver, device, keep_sequence_targets=True)
    history = None
    if history_count:
        history = ContextDataset(previous, split="train", **options)
        old_ids = sample_context_indices(history, history_count, seed=config["seed"] + 1)
        current = _join_caches([current, _context_cache(history, old_ids, solver, device, keep_sequence_targets=True)])
    identities, families = _identity_data(valid)
    count = report["provenance"]["num_envs"]
    start = count - count // 4
    middle = start + (count - start) // 2
    selection_ids, audit_ids = split_new_context_rows(
        identities, selection_worlds=list(range(start, middle)), audit_worlds=list(range(middle, count))
    )
    if (
        not selection_ids
        or not audit_ids
        or train.families & valid.families
        or (history and history.families & valid.families)
    ):
        raise ValueError("Stage validation needs disjoint, nonempty worlds and no training-family overlap.")
    validation = {}
    for name, selected in (("selection", selection_ids), ("audit", audit_ids)):
        cache = _context_cache(valid, selected, solver, device, keep_sequence_targets=True)
        cache["families"] = sorted({families[i] for i in selected})
        path = output / f"current_{name}.pt"
        torch.save(cache, path)
        validation[name] = {
            **_artifact(path),
            "source_hashes": cache["source_hashes"],
            "anchors": len(selected),
            "families": cache["families"],
        }
    if set(validation["selection"]["families"]) & set(validation["audit"]["families"]):
        raise ValueError("Selection and audit share an episode.")
    inventory = {
        "fresh_current_anchors": current_count,
        "history_new_anchors": history_count,
        "available_current_anchors": len(train),
        "train_paths": list(map(str, train.paths)),
        "validation": validation,
        "training_source_hashes": current["source_hashes"],
    }
    return current, validation, inventory


def _reference_stage_data(config, checkpoint, solver, device):
    from isaaclab_neural.data.datasets import LazyTrajectoryDataset
    from isaaclab_neural.eval.coverage import (
        _recipe_cache,
        check_recipe_replay_separation,
        merge_recipe_replay,
        recipe_replay_counts,
    )

    collected = _json(config["reference_collection"])
    protected = _json(config["protected_recipe_collection"])
    if collected["status"] != "completed" or collected["training_updates"] != 0:
        raise ValueError("Fresh reference collection is incomplete or has model updates.")
    entries = collected["phases"]["replay"]
    held = [row for phase in protected["phases"].values() for row in phase] + config["previous_reference_entries"]
    check_recipe_replay_separation(entries, held)
    for path, digest in collected["source_hashes"].items():
        if file_sha256(path) != digest:
            raise ValueError("Reference generation code or teacher changed.")
    files = {row["spec"]["filename"]: row for row in entries}
    paths = {str(Path(row["path"]).resolve()) for row in entries}
    for row in entries:
        if file_sha256(row["path"]) != row["sha256"]:
            raise ValueError("Fresh reference file changed.")
        command = row["command"]
        if (
            "--initial-states-pool" in command
            and str(Path(command[command.index("--initial-states-pool") + 1]).resolve()) not in paths
        ):
            raise ValueError("Reference generation reused another stage's walking pool.")
    sources = checkpoint["cfg"]["algorithm"]["dataset"]["train_datasets"]
    count = config["updates"] * config["batch_size"]
    quotas = recipe_replay_counts({name: row["weight"] for name, row in sources.items()}, count)
    caches = {}
    for index, (name, source) in enumerate(sources.items()):
        entry = files[Path(source["path"]).name]
        if (entry["state_dim"], entry["joint_f_dim"], entry["action_dim"]) != (87, 43, 37):
            raise ValueError("Refreshed reference has an incompatible G1 layout.")
        dataset = LazyTrajectoryDataset(
            entry["path"],
            sample_sequence_length=_HISTORY_LENGTH,
            contact_representation="contact_tokens",
            max_contact_tokens=solver.max_contact_tokens,
            contact_filter=solver.contact_filter,
            contact_packing_policy="body_round_robin_pair_atomic",
        )
        try:
            ids = np.sort(
                np.random.default_rng(config["seed"] + 100 + index).choice(len(dataset), quotas[name], replace=False)
            ).tolist()
            caches[name] = _recipe_cache(dataset, ids, solver, device)
        finally:
            dataset.close()
    reference = merge_recipe_replay(caches, expected_count=count, keep_sequence_targets=True)
    inventory = {
        "unique_reference_anchors": count,
        "source_counts": quotas,
        "sources": entries,
        "source_windows": reference["source_windows"],
        "source_indices": reference["source_indices"],
    }
    return reference, inventory


def _require_full_history_cache(name: str, cache: dict) -> None:
    if name == "current" and cache.get("sequence_label_origin") != "observed_mj_post_step":
        raise ValueError("Current full-history supervision requires verified real-MJ labels.")
    states = cache["inputs"]["states"]
    target = cache.get("sequence_target")
    if states.ndim != 3 or states.shape[1] != _HISTORY_LENGTH or target is None or target.shape != states.shape:
        raise ValueError(f"Full-history {name} fitting requires ten real labels; anchor-only caches are unsupported.")
    if not torch.equal(target[:, -1], cache["target"]):
        raise ValueError("Full-history labels disagree with the cached anchor diagnostic target.")
    if not torch.isfinite(target).all():
        raise FloatingPointError("Non-finite full-history labels.")


def _state_feature_inputs(model, inputs, standard_deviation, generator):
    from isaaclab_neural.eval.learnability import _clone_batch

    result = _clone_batch(inputs)
    if standard_deviation:
        if generator is None or "states_embedding" not in result or model.input_rms is None:
            raise ValueError("State-feature denoising requires frozen RMS and an explicit private generator.")
        feature = result["states_embedding"]
        scale = torch.sqrt(model.input_rms["states_embedding"].var.detach() + 1e-5).to(feature)
        noise = torch.randn(feature.shape, device=feature.device, dtype=feature.dtype, generator=generator)
        result["states_embedding"] = feature + noise * scale * standard_deviation
    return result


def _backward_rollout_loss(student, records, indices, coefficient: float, physics_steps: int) -> float:
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("Rollout coefficient must be finite and nonnegative.")
    if not coefficient:
        return 0.0
    total = 0.0
    for start in range(0, len(indices), student.num_envs):
        ids = indices[start : start + student.num_envs]
        values = student.sequence_loss_terms([records[i] for i in ids], max_physics_steps=physics_steps)["sequence"]
        loss = values.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite rollout objective.")
        (loss * (coefficient * len(ids) / len(indices))).backward()
        total += float(loss.detach()) * len(ids) / len(indices)
    return total


def _backward_history_loss(
    model,
    cache,
    indices,
    weights,
    device,
    micro_batch_size,
    *,
    source="current",
    reference_teacher=None,
    distillation_beta=0.0,
    retention_metrics=None,
    truth_weight=0.5,
    feature_noise_std=0.0,
    noise_generator=None,
) -> float:
    from isaaclab_neural.eval.learnability import _cache_batch, _clone_batch

    if not math.isfinite(truth_weight) or truth_weight < 0 or not 0 <= feature_noise_std <= 0.1:
        raise ValueError("Invalid truth weight or bounded feature-noise scale.")
    if not math.isfinite(distillation_beta) or distillation_beta < 0:
        raise ValueError("Distillation beta must be finite and nonnegative.")
    if distillation_beta:
        if source != "reference" or reference_teacher is None:
            raise ValueError("Output preservation is restricted to reference training data.")
        if reference_teacher.training or any(p.requires_grad for p in reference_teacher.parameters()):
            raise ValueError("The reference teacher must be frozen in eval mode.")
    total, keep_total = 0.0, 0.0
    targets = cache["sequence_target"]
    for start in range(0, len(indices), micro_batch_size):
        ids = indices[start : start + micro_batch_size]
        inputs, _, _ = _cache_batch(cache, ids, device)
        prediction = model(_state_feature_inputs(model, inputs, feature_noise_std, noise_generator))
        target = targets.index_select(0, torch.as_tensor(ids, device=targets.device)).to(device)
        value = weighted_prediction_mse(prediction, target, weights)
        backward_loss = value * (truth_weight * len(ids) / len(indices))
        if distillation_beta:
            clean_prediction = model(_clone_batch(inputs)) if feature_noise_std else prediction
            with torch.no_grad():
                teacher_prediction = reference_teacher(_clone_batch(inputs)).detach()
            keep = weighted_prediction_mse(clean_prediction, teacher_prediction, weights)
            backward_loss = backward_loss + keep * (distillation_beta * len(ids) / len(indices))
            keep_total += float(keep.detach()) * len(ids) / len(indices)
        if not torch.isfinite(backward_loss):
            raise FloatingPointError("Non-finite active-cycle loss.")
        backward_loss.backward()
        total += float(value.detach()) * len(ids) / len(indices)
    if retention_metrics is not None:
        retention_metrics.update(
            loss=keep_total if distillation_beta else None, weighted_loss=distillation_beta * keep_total
        )
    return total


def _frozen_reference_teacher(initial, solver, device):
    from isaaclab_neural.utils.checkpoint import reconstruct_model_from_checkpoint

    target_device = torch.device(device)
    devices = (
        [target_device.index if target_device.index is not None else torch.cuda.current_device()]
        if target_device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=devices):
        return reconstruct_model_from_checkpoint(initial, solver, device=device).eval().requires_grad_(False)


def _make_preservation_inputs(config, reference, truth_batches, *, beta):
    options = config.get("preservation_inputs")
    if options is None:
        return None, None, None
    if beta <= 0:
        raise ValueError("Explicit preservation inputs require a positive distillation coefficient.")
    teacher_hash = file_sha256(config["initial_dynamics"])
    if file_sha256(options["manifest"]) != options["manifest_sha256"]:
        raise ValueError("Preservation input manifest changed.")
    manifest = _json(options["manifest"])
    if manifest["teacher_sha256"] != teacher_hash or "recipe" in manifest["sources"]:
        raise ValueError("Preservation manifest changed its teacher or overrides the recipe input source.")
    if manifest["protocol"] != _json(config["collector_report"])["provenance"]["protocol"]:
        raise ValueError("Preservation inputs and real collector data use different physical protocols.")
    proportions = options["proportions"]
    if "recipe" not in proportions:
        raise ValueError("Input-domain preservation must retain original recipe inputs.")
    caches = {}
    paths = {options["manifest"]: options["manifest_sha256"]}
    for name in proportions:
        if name == "recipe":
            caches[name] = {
                "kind": "teacher_inputs_v1",
                "teacher_sha256": teacher_hash,
                "split": "train",
                "input_origin": "recipe",
                "inputs": reference["inputs"],
                "source_hashes": reference["source_hashes"],
                "identities": [f"recipe:{index}" for index in range(len(reference["target"]))],
            }
        else:
            entry = manifest["sources"][name]
            if file_sha256(entry["path"]) != entry["sha256"]:
                raise ValueError("Preservation input cache changed.")
            caches[name] = torch.load(entry["path"], map_location="cpu", weights_only=True)
            if caches[name]["protocol"] != manifest["protocol"]:
                raise ValueError("Preservation cache physical provenance changed.")
            paths[entry["path"]] = entry["sha256"]
        _validate_preservation_cache(caches[name], teacher_hash)
        for path, digest in caches[name]["source_hashes"].items():
            if file_sha256(path) != digest:
                raise ValueError("Preservation input lineage changed.")
            paths[path] = digest
    counts = {name: len(cache["identities"]) for name, cache in caches.items()}
    batches = preservation_batches(
        counts,
        proportions,
        updates=config["updates"],
        batch_size=config["batch_size"],
        seed=options.get("seed", config["seed"] + 3000),
    )
    recipe_stream = [index for batch in truth_batches for index in batch["reference"]]
    quota = len(batches[0]["recipe"])
    for step, batch in enumerate(batches):
        batch["recipe"] = recipe_stream[step * quota : (step + 1) * quota]
    report = {
        "recipe_schedule": "prefix_of_the_reference_truth_epoch_stream",
        "manifest": _artifact(options["manifest"]),
        "proportions": proportions,
        "unique_inputs_available": counts,
        "inputs_per_update": config["batch_size"],
        "input_origins": {name: cache["input_origin"] for name, cache in caches.items()},
        "exposures": {name: sum(len(batch[name]) for batch in batches) for name in counts},
        "batch_trace_sha256": hashlib.sha256(json.dumps(batches, sort_keys=True).encode()).hexdigest(),
        "source_hashes": paths,
    }
    return caches, batches, report


def _validate_recipe_truth_control(config, prepared, current):
    if not config.get("old_only_truth_control", False):
        return
    if (
        prepared["current"]["sha256"] != prepared["reference"]["sha256"]
        or prepared["current"]["source_hashes"] != prepared["reference"]["source_hashes"]
        or prepared["current_inventory"].get("role") != "old_only_control"
        or current.get("origin") != "resampled_recipe_replay"
    ):
        raise ValueError("Old-only control requires the identical verified recipe cache in both truth roles.")


def _stage_training_data(config, initial, solver, device, output):
    from isaaclab_neural.eval.coverage import _load_cache

    if config.get("old_only_truth_control", False) and not config.get("training_manifest"):
        raise ValueError("Old-only controls require an explicit hashed training manifest.")
    if config.get("training_manifest"):
        if file_sha256(config["training_manifest"]) != config["training_manifest_sha256"]:
            raise ValueError("Prepared training manifest changed.")
        prepared = _json(config["training_manifest"])
        expected_parent = (
            file_sha256(config["initial_dynamics"])
            if config.get("truth_sampling", "fresh") == "epoch"
            else config["parent_sha256"]
        )
        if prepared["parent_sha256"] != expected_parent:
            raise ValueError("Prepared training data belong to another stage scorer.")
        current, reference = [_load_cache(prepared[name], "cpu") for name in ("current", "reference")]
        _validate_recipe_truth_control(config, prepared, current)
        return (
            current,
            reference,
            prepared["validation"],
            prepared["current_inventory"],
            prepared["reference_inventory"],
        )
    current, validation, current_inventory = _current_stage_data(config, solver, device, output)
    if config.get("reference_training_manifest"):
        if file_sha256(config["reference_training_manifest"]) != config["reference_training_manifest_sha256"]:
            raise ValueError("Frozen reference-cache manifest changed.")
        prior = _json(config["reference_training_manifest"])
        if prior["parent_sha256"] != config["parent_sha256"]:
            raise ValueError("Reference cache belongs to another scorer.")
        reference, reference_inventory = _load_cache(prior["reference"], "cpu"), prior["reference_inventory"]
    else:
        reference, reference_inventory = _reference_stage_data(config, initial, solver, device)
    return current, reference, validation, current_inventory, reference_inventory


def _load_rollout_training(config):
    from isaaclab_neural.data.opd_sequence import read_sequence_shard

    if file_sha256(config["rollout_training_manifest"]) != config["rollout_training_manifest_sha256"]:
        raise ValueError("Rollout training manifest changed.")
    manifest = _json(config["rollout_training_manifest"])
    shards = {}
    for name, entry in manifest["sources"].items():
        if file_sha256(entry["path"]) != entry["sha256"]:
            raise ValueError("Rollout training source changed.")
        shard = read_sequence_shard(entry["path"], require_labels=True)
        meta = shard["metadata"]
        if meta["split"] != "train" or meta.get("evaluation_only", False) or meta["source_kind"] != "mjwarp":
            raise ValueError("Rollout fitting is restricted to real-MJ training sequences.")
        if meta["protocol"] != manifest["protocol"] or meta["history_origin"] != "physical_frames":
            raise ValueError("Rollout physical input contract changed.")
        shards[name] = shard
    result = {}
    for source in ("current", "reference"):
        refs = manifest["records"][source]
        identities = [(row["source"], row["index"]) for row in refs]
        if len(identities) != len(set(identities)):
            raise ValueError("A bounded rollout stage requires distinct training anchors.")
        result[source] = [shards[name]["records"][index] for name, index in identities]
        if any(len(row["oracle_states"]) != 16 for row in result[source]):
            raise ValueError("Missing full future labels; do not pad or cross resets.")
    if len(manifest["batches"]) < config["updates"]:
        raise ValueError("Rollout training schedule is shorter than the optimizer budget.")
    return result, next(iter(shards.values()))["metadata"], manifest["batches"]


def _make_rollout_objective(base, model, weights, config, options):
    if not (options["current_rollout_weight"] or options["reference_rollout_weight"]):
        return None, {}, []
    from isaaclab_neural.train.opd_sequence import G1SequenceStudent

    records, metadata, batches = _load_rollout_training(config)
    if metadata["dynamics_hash"] != file_sha256(config["initial_dynamics"]):
        raise ValueError("Rollout histories were collected with another declared scorer.")
    for source in ("current", "reference"):
        ids = [index for batch in batches[: config["updates"]] for index in batch[source]]
        if len(ids) != len(set(ids)) or not ids or min(ids) < 0 or max(ids) >= len(records[source]):
            raise ValueError("Rollout schedule reused or fabricated a training anchor.")
    return G1SequenceStudent(base, base.neural_adapter.solver, model, metadata, weights), records, batches


def _fit_objective_options(config):
    result = {
        "reference_truth_weight": float(config.get("reference_truth_weight", 0.5)),
        "feature_noise_std": float(config.get("feature_noise_std", 0.0)),
        "current_rollout_weight": float(config.get("current_rollout_weight", 0.0)),
        "reference_rollout_weight": float(config.get("reference_rollout_weight", 0.0)),
        "rollout_physics_steps": int(config.get("rollout_physics_steps", 4)),
    }
    if any(not math.isfinite(value) or value < 0 or value > 1e6 for value in result.values()):
        raise ValueError("Objective coefficients must be finite, nonnegative and bounded.")
    if result["feature_noise_std"] > 0.1 or not 1 <= result["rollout_physics_steps"] <= 16:
        raise ValueError("Feature noise/short-rollout budget exceeds the worker contract.")
    return result


def calibrate_active_objective(env, args) -> None:
    """Measure fixed training-only gradient scales without taking an optimizer step."""
    from isaaclab_neural.eval.coverage import _check_candidate, _load_cache
    from isaaclab_neural.eval.learnability import _cache_batch
    from isaaclab_neural.rl.rsl_rl.online_opd import _pause_fingerprint
    from isaaclab_neural.train.opd_sequence import G1SequenceStudent
    from isaaclab_neural.utils.checkpoint import reconstruct_model_from_checkpoint

    config = _json(args.active_fit_config)
    base = env.unwrapped
    paused = _pause_fingerprint(base, base.obs_buf)
    if file_sha256(config["training_manifest"]) != config["training_manifest_sha256"]:
        raise ValueError("Calibration TF data changed.")
    prepared = _json(config["training_manifest"])
    current, reference = [_load_cache(prepared[name], "cpu") for name in ("current", "reference")]
    initial = load_checkpoint(args.dynamics, device=base.device)
    model = reconstruct_model_from_checkpoint(initial, base.neural_adapter.solver, device=base.device)
    teacher = _frozen_reference_teacher(initial, base.neural_adapter.solver, base.device)
    records, metadata, batches = _load_rollout_training(config)
    student = G1SequenceStudent(base, base.neural_adapter.solver, model, metadata, initial["loss_weights"])
    parameters = list(model.parameters())

    def vector():
        return torch.cat(
            [
                (parameter.grad.detach() if parameter.grad is not None else torch.zeros_like(parameter))
                .reshape(-1)
                .cpu()
                .double()
                for parameter in parameters
            ]
        )

    report = {"scope": "Training-only gradient calibration; zero optimizer updates", "probes": []}
    points = config["calibration_points"]
    if len(points) != 3 or config["calibration_batches"] != [0, 7, 15]:
        raise ValueError("Calibration requires the declared three checkpoints and three fixed training batches.")
    for point in points:
        if file_sha256(point["path"]) != point["sha256"]:
            raise ValueError("Calibration checkpoint changed.")
        saved = load_checkpoint(point["path"], device=base.device)
        _check_candidate(saved, initial)
        model.load_state_dict(saved["model_state_dict"], strict=True)
        model.train()
        for index in config["calibration_batches"]:
            batch = prepared["batches"][index]
            model.zero_grad(set_to_none=True)
            new_loss = _backward_history_loss(
                model, current, batch["current"], initial["loss_weights"], base.device, 32
            )
            log = {}
            old_loss = _backward_history_loss(
                model,
                reference,
                batch["reference"],
                initial["loss_weights"],
                base.device,
                32,
                source="reference",
                reference_teacher=teacher,
                distillation_beta=1000,
                retention_metrics=log,
            )
            base_gradient = vector()
            model.zero_grad(set_to_none=True)
            _backward_history_loss(
                model, reference, batch["reference"], initial["loss_weights"], base.device, 32, truth_weight=1
            )
            old_gradient = vector()
            model.zero_grad(set_to_none=True)
            _backward_history_loss(
                model,
                reference,
                batch["reference"],
                initial["loss_weights"],
                base.device,
                32,
                source="reference",
                reference_teacher=teacher,
                distillation_beta=1,
                truth_weight=0,
            )
            keep_gradient = vector()
            row = {
                "point": point["name"],
                "batch_index": index,
                "new_tf": new_loss,
                "old_tf": old_loss,
                "keep_loss": log["loss"],
                "base_norm": float(base_gradient.norm()),
                "old_tf_norm": float(old_gradient.norm()),
                "old_keep_norm": float(keep_gradient.norm()),
                "old_total_norm": float((0.5 * old_gradient + 1000 * keep_gradient).norm()),
                "rollout": {},
            }
            for source in ("current", "reference"):
                model.zero_grad(set_to_none=True)
                value = _backward_rollout_loss(student, records[source], batches[index][source], 1, 4)
                gradient = vector()
                norm = float(gradient.norm())
                if not math.isfinite(norm) or norm <= 0 or row["base_norm"] <= 0:
                    raise FloatingPointError("Invalid calibration gradient norm.")
                row["rollout"][source] = {
                    "loss": value,
                    "norm": norm,
                    "lambda": 0.5 * row["base_norm"] / norm,
                    "cosine_base": float(gradient.dot(base_gradient)) / (norm * row["base_norm"]),
                }
            report["probes"].append(row)
        if not all(torch.equal(value, model.state_dict()[key]) for key, value in saved["model_state_dict"].items()):
            raise RuntimeError("Calibration updated model/RMS state.")
    coefficients = {
        f"{source}_rollout_weight": float(np.median([row["rollout"][source]["lambda"] for row in report["probes"]]))
        for source in ("current", "reference")
    }
    coefficients["reference_truth_weight"] = max(
        0.5,
        float(
            np.median(
                [row["old_total_norm"] / row["old_tf_norm"] for row in report["probes"] if row["point"] != "F_pre"]
            )
        ),
    )
    if any(not math.isfinite(value) or not 0 < value <= 1e6 for value in coefficients.values()):
        raise ValueError("Calibration coefficient exceeds its declared finite budget.")
    model.load_state_dict(initial["model_state_dict"], strict=True)
    inputs, _, _ = _cache_batch(current, prepared["batches"][0]["current"][:32], base.device)
    generator = torch.Generator(device=base.device).manual_seed(5500)
    augmented = _state_feature_inputs(model, inputs, 0.01, generator)
    scale = torch.sqrt(model.input_rms["states_embedding"].var + 1e-5)
    noise = (augmented["states_embedding"] - inputs["states_embedding"]) / scale
    assert all(torch.equal(value, augmented[key]) for key, value in inputs.items() if key != "states_embedding")
    report.update(
        coefficients=coefficients,
        normalized_feature_noise_std=float(noise.std()),
        optimizer_updates=0,
        teacher_unchanged=all(
            torch.equal(value, teacher.state_dict()[key]) for key, value in initial["model_state_dict"].items()
        ),
        live_environment_unchanged=paused == _pause_fingerprint(base, base.obs_buf),
    )
    if not report["teacher_unchanged"] or not report["live_environment_unchanged"]:
        raise RuntimeError("Training-only calibration changed its frozen teacher or live state.")
    _write_json(args.output / "result.json", report)


def _truth_batches(config, current_count, reference_count):
    sampling = config.get("truth_sampling", "fresh")
    if sampling not in {"fresh", "epoch"}:
        raise ValueError("Truth sampling must be fresh or explicitly declared epoch replay.")
    sampler = epoch_stage_batches if sampling == "epoch" else fresh_stage_batches
    return sampling, sampler(
        current_count,
        reference_count,
        updates=config["updates"],
        batch_size=config["batch_size"],
        seed=config["seed"] + 200,
    )


class _CollectorSequenceObjective:
    def __init__(self, env, model, weights, config):
        from isaaclab_neural.data.opd_sequence import read_sequence_shard, validate_collector_training_role
        from isaaclab_neural.train.opd_sequence import G1SequenceStudent

        settings = config["collector_sequences"]
        if config.get("old_only_truth_control", False):
            raise ValueError("An old-only truth control cannot add collector-sequence truth.")
        if file_sha256(settings["manifest"]) != settings["manifest_sha256"]:
            raise ValueError("Independent collector sequence manifest changed.")
        manifest = _json(settings["manifest"])
        if manifest.get("role") != "model_training" or manifest.get("evaluation_data_used") is not False:
            raise ValueError("Collector sequence objective cannot use evaluation data.")
        self.fraction = float(settings["fraction"])
        self.local_weight = 0.5 * self.fraction
        self.rollout_weight = float(settings["root_rollout_weight"])
        self.root_scales = settings["root_correction_scales"]
        if (
            not 0 < self.fraction <= 0.5
            or not math.isfinite(self.rollout_weight)
            or not 0 <= self.rollout_weight <= 1e6
        ):
            raise ValueError("Invalid explicit collector sequence objective weights.")
        self.records, metadata = [], None
        kinds = set()
        identities = set()
        for source in manifest["sources"]:
            if file_sha256(source["path"]) != source["sha256"]:
                raise ValueError("Independent collector sequence source changed.")
            shard = read_sequence_shard(source["path"], require_labels=True, memory_map=True)
            kinds.add(validate_collector_training_role(shard["metadata"]))
            if metadata is not None and metadata["protocol"] != shard["metadata"]["protocol"]:
                raise ValueError("Collector sequences cannot mix physical protocols.")
            metadata = shard["metadata"]
            for index in source["indices"]:
                record = shard["records"][index]
                identity = (source["path"], index)
                if identity in identities or record["env_id"] not in manifest["training_world_ids"]:
                    raise ValueError("Duplicate or held-out collector sequence used for fitting.")
                if len(record["oracle_states"]) not in (4, 8, 12, 16) or len(record["history"]) != 10:
                    raise ValueError(
                        "Current collector objective requires actual control-aligned futures and H10 inputs."
                    )
                identities.add(identity)
                self.records.append(record)
        if not self.records or metadata["physics_dt"] != 0.005 or metadata["decimation"] != 4:
            raise ValueError("Collector sequence timing differs from the frozen G1 contract.")
        batch_size = settings["batch_size"]
        if len(self.records) % batch_size:
            raise ValueError("Collector sequence epochs must have an explicit batch-divisible pool.")
        plans = epoch_stage_batches(
            len(self.records),
            len(self.records),
            updates=config["updates"],
            batch_size=batch_size,
            seed=settings["sampling_seed"],
        )
        self.batches = [row["current"] for row in plans]
        self.student = G1SequenceStudent(env, env.neural_adapter.solver, model, metadata, weights, pd_mode="newton_fma")
        self.report = {
            "manifest": _artifact(settings["manifest"]),
            "source_kinds": sorted(kinds),
            "unique_sequences": len(self.records),
            "local_weight": self.local_weight,
            "common_MJ_current_weight": 0.5 * (1 - self.fraction),
            "root_rollout_weight": self.rollout_weight,
            "root_correction_scales": self.root_scales,
            "local_label_exposures": config["updates"] * batch_size,
            "rollout_label_exposures": (
                sum(len(self.records[index]["oracle_states"]) for batch in self.batches for index in batch)
                if self.rollout_weight
                else 0
            ),
            "full16_sequences": sum(len(row["oracle_states"]) == 16 for row in self.records),
            "valid_prefixes_retained": True,
            "pd_mode": "newton_fma",
            "model_history_relabelled_as_truth": False,
            "sampling_trace_sha256": hashlib.sha256(json.dumps(self.batches).encode()).hexdigest(),
        }

    def backward(self, step: int, *, local_weight: float | None = None, root_weight: float | None = None) -> dict:
        local_weight = self.local_weight if local_weight is None else local_weight
        root_weight = self.rollout_weight if root_weight is None else root_weight
        batch = self.batches[step]
        local_sum, root_sum = 0.0, 0.0
        for horizon in (4, 8, 12, 16):
            group = [index for index in batch if len(self.records[index]["oracle_states"]) == horizon]
            for start in range(0, len(group), self.student.num_envs):
                indices = group[start : start + self.student.num_envs]
                terms = self.student.sequence_loss_terms(
                    [self.records[index] for index in indices],
                    max_physics_steps=16 if root_weight else 1,
                    root_correction_scales=self.root_scales if root_weight else None,
                )
                local = terms["local"].mean()
                root = terms["root_sequence"].mean() if root_weight else local.new_zeros(())
                loss = local_weight * local + root_weight * root
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite independent collector sequence objective.")
                (loss * (len(indices) / len(batch))).backward()
                local_sum += float(local.detach()) * len(indices) / len(batch)
                root_sum += float(root.detach()) * len(indices) / len(batch)
        return {
            "local": local_sum,
            "root_rollout": root_sum,
            "weighted_loss": local_weight * local_sum + root_weight * root_sum,
        }


def calibrate_collector_sequence_objective(env, args) -> None:
    """Calibrate one auxiliary root-rollout coefficient using only independent collector training gradients."""
    from isaaclab_neural.rl.rsl_rl.online_opd import _pause_fingerprint
    from isaaclab_neural.train.trainers import validate_finetune_checkpoint
    from isaaclab_neural.utils.checkpoint import reconstruct_model_from_checkpoint

    base = env.unwrapped
    config = _json(args.active_fit_config)
    if not config.get("defer_audit", False) or config["parent_sha256"] != file_sha256(args.dynamics):
        raise ValueError("Collector objective calibration requires a frozen parent and deferred audit.")
    saved = load_checkpoint(args.dynamics, device=base.device)
    model = reconstruct_model_from_checkpoint(saved, base.neural_adapter.solver, device=base.device)
    validate_finetune_checkpoint(saved, model, update_statistics=False)
    frozen = {key: value.clone() for key, value in model.state_dict().items()}
    paused = _pause_fingerprint(base, base.obs_buf)
    objective = _CollectorSequenceObjective(base, model, saved["loss_weights"].to(base.device), config)
    model.train()
    norms = {"local": [], "root": []}
    for step in (0, 1):
        for name, local_weight, root_weight in (("local", objective.local_weight, 0.0), ("root", 0.0, 1.0)):
            model.zero_grad(set_to_none=True)
            objective.backward(step, local_weight=local_weight, root_weight=root_weight)
            norm = float(
                torch.sqrt(
                    sum(
                        parameter.grad.double().square().sum()
                        for parameter in model.parameters()
                        if parameter.grad is not None
                    )
                )
            )
            if not math.isfinite(norm) or norm <= 0:
                raise ValueError("Collector training calibration produced invalid gradient norms.")
            norms[name].append(norm)
    coefficient = 0.5 * sum(norms["local"]) / sum(norms["root"])
    if not 0 < coefficient <= 1e6:
        raise ValueError("Collector root correction coefficient exceeded the fixed finite bound.")
    model.zero_grad(set_to_none=True)
    assert paused == _pause_fingerprint(base, base.obs_buf)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in frozen.items())
    _write_json(
        args.output / "report.json",
        {
            "root_rollout_weight": coefficient,
            "training_gradient_norms": norms,
            "rule": "root correction gradient target=0.5 times additional local gradient at F_pre",
            "objective": objective.report,
            "model_updates": 0,
            "evaluation_inputs_used": False,
            "frozen_model_and_live_state_unchanged": True,
        },
    )


def _active_fit_optimizer(model, active: dict, config: dict):
    prior = active["cfg"].get("active_cycle")
    if config.get("research_study") and (
        prior is not None or config["parent_sha256"] != file_sha256(config["initial_dynamics"])
    ):
        raise ValueError("Each independent research arm must start from F_pre with fresh Adam.")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    if prior:
        if prior["initial_sha256"] != file_sha256(config["initial_dynamics"]):
            raise ValueError("Cannot continue an optimizer from another initial model.")
        optimizer.load_state_dict(active["optimizer_state_dict"])
        if any(row["lr"] != config["learning_rate"] for row in optimizer.param_groups):
            raise ValueError("Resumed dynamics optimizer learning rate changed.")
    return optimizer, prior


def _active_fit_storage(config: dict, output: Path, *, reserve_bytes: int = 0) -> dict | None:
    research = config.get("research_study")
    if research is None:
        return None
    return check_active_fit_storage(
        output,
        minimum_free_bytes=research["minimum_free_bytes"],
        maximum_output_bytes=research["maximum_output_bytes"],
        reserve_bytes=reserve_bytes,
    )


def _active_fit_research_record(model, initial: dict, config: dict, output: Path) -> dict:
    if not config.get("research_study"):
        return {}
    distance = sum(
        float((value.detach().double() - initial["model_state_dict"][name].double()).square().sum())
        for name, value in model.named_parameters()
    )
    magnitude = sum(
        float(initial["model_state_dict"][name].double().square().sum()) for name, _ in model.named_parameters()
    )
    return {
        "parameter_change_relative_l2": math.sqrt(distance / max(magnitude, 1e-30)),
        "storage": _active_fit_storage(config, output),
    }


def fit_active_stage(env, args) -> None:
    """Fit an isolated candidate on refreshed rows and select it before opening audit data."""
    from isaaclab_neural.eval.coverage import _check_candidate, _load_cache, _measure_recipe_cache, recipe_loss
    from isaaclab_neural.rl.rsl_rl.online_opd import _pause_fingerprint
    from isaaclab_neural.train.opd_sequence import _training_fingerprint
    from isaaclab_neural.train.trainers import validate_finetune_checkpoint
    from isaaclab_neural.utils.checkpoint import reconstruct_model_from_checkpoint

    config = _json(args.active_fit_config)
    supervision_settings = active_supervision_settings(config)
    objective_options = _fit_objective_options(config)
    distillation_beta = float(config.get("reference_distillation_beta", 0.0))
    if not math.isfinite(distillation_beta) or distillation_beta < 0:
        raise ValueError("Reference distillation beta must be finite and nonnegative.")
    if file_sha256(args.dynamics) != config["parent_sha256"]:
        raise ValueError("Invalid scorer parent.")
    checkpoints = active_fit_checkpoints(config)
    research = config.get("research_study")
    base, output = env.unwrapped, args.output
    _active_fit_storage(config, output)
    if base.num_envs != config.get("scratch_worlds", 4):
        raise ValueError("The fixed-data study changed its scratch-world layout.")
    paused = _pause_fingerprint(base, base.obs_buf)
    solver, device = base.neural_adapter.solver, base.device
    initial = load_checkpoint(config["initial_dynamics"], device=device)
    active = load_checkpoint(args.dynamics, device=device)
    _check_candidate(active, initial)
    model = reconstruct_model_from_checkpoint(active, solver, device=device)
    validate_finetune_checkpoint(active, model, update_statistics=False)
    weights = initial["loss_weights"].to(device).detach().clone()
    current, reference, validation, current_inventory, reference_inventory = _stage_training_data(
        config, initial, solver, device, output
    )
    for name, cache in (("current", current), ("reference", reference)):
        _require_full_history_cache("reference" if config.get("old_only_truth_control", False) else name, cache)
    sampling, batches = _truth_batches(config, len(current["target"]), len(reference["target"]))
    trace = hashlib.sha256(json.dumps(batches, sort_keys=True).encode()).hexdigest()
    if config.get("expected_batch_trace_sha256", trace) != trace:
        raise ValueError("A paired fitting run changed its declared batch order.")
    _write_json(
        output / "data_inventory.json",
        {"current": current_inventory, "reference": reference_inventory, "batch_trace_sha256": trace},
    )
    if config.get("prepare_only", False):
        entries = {}
        for name, cache in (("current", current), ("reference", reference)):
            path = output / f"train_{name}.pt"
            torch.save(cache, path)
            entries[name] = {
                **_artifact(path),
                "source_hashes": cache["source_hashes"],
                "anchors": len(cache["target"]),
            }
        _write_json(
            output / "training_manifest.json",
            {
                **entries,
                "parent_sha256": config["parent_sha256"],
                "validation": validation,
                "current_inventory": current_inventory,
                "reference_inventory": reference_inventory,
                "batches": batches,
                "batch_trace_sha256": trace,
            },
        )
        if paused != _pause_fingerprint(base, base.obs_buf):
            raise RuntimeError("Preparing training caches changed the scratch environment.")
        _write_json(
            output / "integrity.json",
            {"model_optimizer_updates": 0, "policy_updates": 0, "live_environment_unchanged": True},
        )
        return
    manifest = _json(config["old_recipe_manifest"])
    if manifest["dynamics_sha256"] != file_sha256(config["initial_dynamics"]):
        raise ValueError("The old retention baseline is not the fixed F_pre.")
    groups = manifest["groups"]
    old_selection = {
        name: _load_cache(entry, device) for name, entry in manifest["phases"]["selection"]["recipe"].items()
    }
    selected_data = _load_cache(validation["selection"], device)
    histories = [_load_cache(entry, device) for entry in config["history_validation"]]

    @torch.no_grad()
    def measure(old_data, new_data, history_data):
        model.eval()
        for cache in (new_data, *history_data):
            _require_full_history_cache("current", cache)
        old = {name: _measure_recipe_cache(cache, model, solver, weights, groups) for name, cache in old_data.items()}
        now = _measure_recipe_cache(new_data, model, solver, weights, groups)
        past = [_measure_recipe_cache(cache, model, solver, weights, groups) for cache in history_data]
        return {
            "old_recipe": recipe_loss(old, manifest["weights"]),
            "current": now["teacher_forced_loss"],
            "current_anchor": now["weighted_loss"],
            "current_supervision": "all_positions",
            "history": [row["teacher_forced_loss"] for row in past],
            "old_metrics": old,
            "current_metrics": now,
            "history_metrics": past,
        }

    model.load_state_dict(initial["model_state_dict"], strict=True)
    baseline = measure(old_selection, selected_data, histories)
    model.load_state_dict(active["model_state_dict"], strict=True)
    before = measure(old_selection, selected_data, histories)
    optimizer, prior = _active_fit_optimizer(model, active, config)
    reference_teacher = _frozen_reference_teacher(initial, solver, device) if distillation_beta else None
    preservation_caches, preservation_plan, preservation_report = _make_preservation_inputs(
        config, reference, batches, beta=distillation_beta
    )
    noise_generator = (
        torch.Generator(device=device).manual_seed(config.get("noise_seed", config["seed"]))
        if objective_options["feature_noise_std"]
        else None
    )
    rollout_student, rollout_records, rollout_batches = _make_rollout_objective(
        base, model, weights, config, objective_options
    )
    collector_sequences = (
        _CollectorSequenceObjective(base, model, weights, config) if config.get("collector_sequences") else None
    )
    current_truth_weight = 0.5 * (1 - collector_sequences.fraction) if collector_sequences else 0.5
    frozen = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if key.startswith(("input_rms.", "output_rms.", "body_net_contact_force_rms."))
    }
    cfg = copy.deepcopy(active["cfg"])
    selected, last, curve = None, None, []
    checkpoint_reservation = (
        4 * sum(value.numel() * value.element_size() for value in model.state_dict().values()) + 2**20
    )
    baseline_snapshot = None
    if config.get("record_step_zero", False):
        _active_fit_storage(config, output, reserve_bytes=checkpoint_reservation)
        path = output / "step_0000.pt"
        save_checkpoint(
            path,
            model,
            "G1-Native-ContactTokens",
            cfg,
            optimizer_state_dict=optimizer.state_dict(),
            epoch=0,
            loss_weights=weights,
        )
        baseline_snapshot = {
            "checkpoint": _artifact(path),
            "update": 0,
            "metrics": before,
            "decision": decide_active_stage(baseline, before, before, minimum_gain=config["minimum_gain"]),
            "model_adam_fingerprint": _training_fingerprint(model, optimizer),
        }
        _write_json(output / "step_0000.json", baseline_snapshot)
    _write_status(output / "training_status.json", {"status": "running", "optimizer_updates": 0})
    for step, batch in enumerate(batches, 1):
        if (step - 1) % 64 == 0:
            _active_fit_storage(config, output)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        retention_metrics = {}
        loss_values = {
            source: _backward_history_loss(
                model,
                cache,
                batch[source],
                weights,
                device,
                config["micro_batch_size"],
                source=source,
                reference_teacher=reference_teacher if source == "reference" else None,
                distillation_beta=distillation_beta if source == "reference" and preservation_plan is None else 0.0,
                retention_metrics=retention_metrics if source == "reference" else None,
                truth_weight=objective_options["reference_truth_weight"]
                if source == "reference"
                else current_truth_weight,
                feature_noise_std=objective_options["feature_noise_std"],
                noise_generator=noise_generator,
            )
            for source, cache in (("current", current), ("reference", reference))
        }
        if preservation_plan is not None:
            retention_metrics = _backward_preservation_inputs(
                model,
                reference_teacher,
                preservation_caches,
                preservation_plan[step - 1],
                weights,
                device,
                config["micro_batch_size"],
                beta=distillation_beta,
            )
        rollout_losses = {}
        if rollout_student is not None:
            rollout_losses = {
                source: _backward_rollout_loss(
                    rollout_student,
                    rollout_records[source],
                    rollout_batches[step - 1][source],
                    objective_options[f"{source}_rollout_weight"],
                    objective_options["rollout_physics_steps"],
                )
                for source in ("current", "reference")
            }
        sequence_metrics = collector_sequences.backward(step - 1) if collector_sequences else {}
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        _write_status(output / "training_status.json", {"status": "running", "optimizer_updates": step})
        row = {"update": step, "loss": loss_values, "gradient_norm_before_clip": float(norm)}
        if sequence_metrics:
            row["collector_sequences"] = sequence_metrics
        if distillation_beta:
            row["reference_distillation"] = retention_metrics
        if (
            distillation_beta
            or objective_options["reference_truth_weight"] != 0.5
            or rollout_losses
            or objective_options["feature_noise_std"]
            or sequence_metrics
        ):
            row["total_loss"] = (
                current_truth_weight * loss_values["current"]
                + objective_options["reference_truth_weight"] * loss_values["reference"]
                + retention_metrics.get("weighted_loss", 0.0)
                + sum(objective_options[f"{source}_rollout_weight"] * value for source, value in rollout_losses.items())
                + sequence_metrics.get("weighted_loss", 0.0)
            )
        if rollout_losses:
            row["rollout_loss"] = rollout_losses
        curve.append(row)
        if step not in checkpoints:
            continue
        _active_fit_storage(config, output, reserve_bytes=checkpoint_reservation)
        measured = measure(old_selection, selected_data, histories)
        decision = decide_active_stage(baseline, before, measured, minimum_gain=config["minimum_gain"])
        cfg["active_cycle"] = {
            "stage": config["stage"],
            **supervision_settings,
            "reference_distillation_beta": distillation_beta,
            "objective_options": objective_options,
            "truth_sampling": sampling,
            "preservation_inputs": preservation_report,
            "initial_sha256": file_sha256(config["initial_dynamics"]),
            "parent_sha256": config["parent_sha256"],
            "optimizer_updates_total": (prior["optimizer_updates_total"] if prior else 0) + step,
        }
        if research:
            cfg["active_cycle"]["research_study"] = research
        if collector_sequences is not None:
            cfg["active_cycle"]["collector_sequences"] = collector_sequences.report
        path = output / f"step_{step:04d}.pt"
        save_checkpoint(
            path,
            model,
            "G1-Native-ContactTokens",
            cfg,
            optimizer_state_dict=optimizer.state_dict(),
            epoch=step,
            loss_weights=weights,
        )
        last = {
            "checkpoint": _artifact(path),
            "update": step,
            "metrics": measured,
            "decision": decision,
            "model_adam_fingerprint": _training_fingerprint(model, optimizer),
        }
        last.update(_active_fit_research_record(model, initial, config, output))
        _write_json(output / f"step_{step:04d}.json", last)
        if decision["accepted"] and (selected is None or measured["current"] < selected["metrics"]["current"]):
            selected = last
        print(json.dumps({"stage": config["stage"], "update": step, **decision}), flush=True)
    if not all(torch.equal(value, model.state_dict()[key]) for key, value in frozen.items()):
        raise RuntimeError("Active-cycle fitting changed frozen normalization.")
    if reference_teacher is not None and (
        any(parameter.grad is not None for parameter in reference_teacher.parameters())
        or not all(
            torch.equal(value, reference_teacher.state_dict()[key])
            for key, value in initial["model_state_dict"].items()
        )
    ):
        raise RuntimeError("Reference distillation modified its frozen F_pre teacher.")
    if preservation_report and any(
        file_sha256(path) != digest for path, digest in preservation_report["source_hashes"].items()
    ):
        raise RuntimeError("Preservation training inputs changed during fitting.")
    chosen = selected or baseline_snapshot or last
    _write_json(
        output / "selection.json",
        {
            "accepted": selected is not None,
            "candidate": chosen["checkpoint"],
            "update": chosen["update"],
            "scope": "TF_PREFILTER_ONLY" if config.get("defer_audit", False) else "teacher_forced_selection",
        },
    )
    audit = audit_baseline = None
    if not config.get("defer_audit", False):
        model.load_state_dict(
            load_checkpoint(chosen["checkpoint"]["path"], device=device)["model_state_dict"], strict=True
        )
        old_audit = {name: _load_cache(entry, device) for name, entry in manifest["phases"]["audit"]["recipe"].items()}
        audit_data = _load_cache(validation["audit"], device)
        audit = measure(old_audit, audit_data, [])
        model.load_state_dict(initial["model_state_dict"], strict=True)
        audit_baseline = measure(old_audit, audit_data, [])
    unchanged = paused == _pause_fingerprint(base, base.obs_buf)
    if not unchanged:
        raise RuntimeError("Fitting changed the scratch environment state.")
    _write_json(
        output / "report.json",
        {
            "parent_sha256": config["parent_sha256"],
            "accepted": selected is not None,
            "candidate": chosen["checkpoint"],
            "selected_update": chosen["update"],
            "selection": chosen["decision"],
            "baseline_selection": baseline,
            "active_selection": before,
            "candidate_selection": chosen["metrics"],
            "audit": audit,
            "audit_F_pre": audit_baseline,
            "audit_deferred": bool(config.get("defer_audit", False)),
            "selection_scope": "TF_PREFILTER_ONLY" if config.get("defer_audit", False) else "teacher_forced_selection",
            "objective_options": objective_options,
            "truth_sampling": sampling,
            **(
                {
                    "research_study": research,
                    "checkpoint_updates": list(checkpoints),
                    "selection_population": "predeclared_retained_checkpoint_grid",
                }
                if research
                else {}
            ),
            "old_only_truth_control": bool(config.get("old_only_truth_control", False)),
            "truth_unique_anchors": {"current": len(current["target"]), "reference": len(reference["target"])},
            "new_truth_label_exposures": (
                (
                    0
                    if config.get("old_only_truth_control", False)
                    else len(batches) * config["batch_size"] * _HISTORY_LENGTH
                )
                + (collector_sequences.report["local_label_exposures"] if collector_sequences else 0)
            ),
            "collector_sequences": collector_sequences.report if collector_sequences else None,
            "preservation_inputs": preservation_report,
            "rollout_manifest": _artifact(config["rollout_training_manifest"])
            if config.get("rollout_training_manifest")
            else None,
            "rollout_label_position_exposures": {
                source: sum(len(batch[source]) for batch in rollout_batches[: config["updates"]])
                * objective_options["rollout_physics_steps"]
                if objective_options[f"{source}_rollout_weight"]
                else 0
                for source in ("current", "reference")
            },
            "curve": curve,
            "optimizer_updates": len(batches),
            **supervision_settings,
            "reference_distillation": {
                "beta": distillation_beta,
                "teacher": _artifact(config["initial_dynamics"]) if distillation_beta else None,
                "teacher_unchanged": True,
                "scope": "Declared input-only preservation mixture; full10 positions; no synthetic truth"
                if preservation_report
                else "Reference training inputs only; full10 positions; beta outside the0.5 truth coefficient",
            },
            "current_label_position_exposures": len(batches) * config["batch_size"] * _HISTORY_LENGTH,
            "reference_label_position_exposures": len(batches) * config["batch_size"] * _HISTORY_LENGTH,
            "optimizer_resumed": prior is not None,
            "rms_unchanged": True,
            "live_environment_unchanged": unchanged,
            "validation": validation,
            "new_training_paths": current_inventory["train_paths"],
            "source_scope": (
                "Real MJ history truth plus independent collector MJ-future labels; never synthetic past truth"
                if collector_sequences
                else (
                    "Real MJ truth with explicitly typed F_pre-preservation inputs"
                    if preservation_report
                    else "MJ collector on-policy active learning, not autoregressive NeRD-rollout distillation"
                )
            ),
            "original_data_retention": "unverified",
        },
    )
    _write_status(output / "training_status.json", {"status": "completed", "optimizer_updates": len(batches)})


def _run(command, log: Path, cwd: str):
    _write_json(log.with_suffix(".command.json"), command)
    with log.open("x") as stream:
        subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=True)


def run_active_cycle(config: dict, output: Path) -> None:
    """Run bounded collector/refresh/fitting stages, retaining rejected scorer and optimizer pairs."""
    rounds = config["rounds"]
    supervision_settings = active_supervision_settings(config)
    if not 1 <= rounds <= 3 or not 1 <= config["collector_updates"] <= 32:
        raise ValueError("Active-cycle pilot is bounded to three stages and32 PPO updates per stage.")
    if not 1 <= config["dynamics_updates"] <= 128 or not 0 < config["learning_rate"] <= 1e-5:
        raise ValueError("Invalid bounded dynamics budget.")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "config.json", config)
    active = _artifact(config["initial_dynamics"])
    policy = _artifact(config["initial_policy"])
    history_train, history_valid, old_entries, records = [], [], [], []
    frozen = {
        str(Path(path).resolve()): file_sha256(path)
        for path in (
            config["initial_dynamics"],
            config["initial_policy"],
            config["old_recipe_manifest"],
            config["protected_recipe_collection"],
            *config["frozen_sources"],
        )
    }
    _write_json(output / "frozen_inputs.json", frozen)
    status = {"status": "running", "stage": 0, "phase": "starting"}
    try:
        for stage in range(rounds):
            if not all(file_sha256(path) == digest for path, digest in frozen.items()):
                raise ValueError("Frozen cycle input changed.")
            directory = output / f"stage_{stage:02d}"
            directory.mkdir()
            status.update(stage=stage, phase="collect")
            (output / "status.json").write_text(json.dumps(status, indent=2))
            common = [sys.executable, "-m", "isaaclab_neural.generate.opd"]
            command = [
                *common,
                "--stage",
                "active-collect",
                "--backend",
                "mjwarp",
                "--active_resume",
                "--active_reward_profile",
                "base_error",
                "--active_error_scope",
                "root_legs",
                "--policy-snapshot",
                policy["path"],
                "--dynamics",
                active["path"],
                "--output",
                str(directory / "collector"),
                "--num-envs",
                str(config["collector_envs"]),
                "--active_record_envs",
                str(config["record_envs"]),
                "--active_capacity",
                str(config["capacity"]),
                "--active_updates",
                str(config["collector_updates"]),
                "--seed",
                str(config["collector_seed"] + stage),
                "--sampler-seed",
                str(config["sampling_seed"] + stage),
                "--active_policy_rng_seed",
                str(config["policy_seed"] + stage),
                "--active_eval_steps",
                "32",
                "--device",
                config["device"],
            ]
            _run(command, directory / "collector.log", config["repository"])
            collected = _json(directory / "collector/report.json")
            if collected["provenance"]["dynamics_sha256"] != active["sha256"] or not collected["dynamics_frozen"]:
                raise ValueError("Stage scorer changed during a PPO block.")
            policy_next = _artifact(collected["last_policy"])
            status["phase"] = "refresh_reference"
            (output / "status.json").write_text(json.dumps(status, indent=2))
            reference_dir = directory / "reference"
            command = [
                part.format(output=str(reference_dir), seed_offset=config["reference_seed_offset"] + 1000 * stage)
                for part in config["reference_command"]
            ]
            _run(command, directory / "reference.log", config["repository"])
            reference = _json(reference_dir / "collection_manifest.json")
            status["phase"] = "fit"
            (output / "status.json").write_text(json.dumps(status, indent=2))
            fitting = {
                **supervision_settings,
                "reference_distillation_beta": config.get("reference_distillation_beta", 0.0),
                "stage": stage,
                "parent_sha256": active["sha256"],
                "initial_dynamics": config["initial_dynamics"],
                "collector_report": str(directory / "collector/report.json"),
                "reference_collection": str(reference_dir / "collection_manifest.json"),
                "protected_recipe_collection": config["protected_recipe_collection"],
                "previous_reference_entries": old_entries,
                "old_recipe_manifest": config["old_recipe_manifest"],
                "history_training": history_train,
                "history_validation": history_valid,
                "history_fraction": config["history_fraction"],
                "updates": config["dynamics_updates"],
                "batch_size": config["batch_size"],
                "micro_batch_size": config["micro_batch_size"],
                "learning_rate": config["learning_rate"],
                "seed": config["sampling_seed"] + 100 * stage,
                "check_every": config["check_every"],
                "minimum_gain": config["minimum_gain"],
            }
            _write_json(directory / "fit_config.json", fitting)
            _run(
                [
                    *common,
                    "--stage",
                    "active-stage-fit",
                    "--backend",
                    "nerd",
                    "--dynamics",
                    active["path"],
                    "--active_fit_config",
                    str(directory / "fit_config.json"),
                    "--output",
                    str(directory / "fit"),
                    "--num-envs",
                    "4",
                    "--seed",
                    "11",
                    "--device",
                    config["device"],
                ],
                directory / "fit.log",
                config["repository"],
            )
            fit = _json(directory / "fit/report.json")
            next_model = next_active_snapshot(active, fit)
            row = {
                "stage": stage,
                "scorer_before": active,
                "scorer_after": next_model,
                "scorer_changed": next_model["sha256"] != active["sha256"],
                "policy_before": policy,
                "policy_after": policy_next,
                "ppo_updates_start": collected["provenance"]["policy_updates_at_start"],
                "ppo_updates_end": collected["total_collector_ppo_updates"],
                "selection": fit["selection"],
                "accepted": fit["accepted"],
                "optimizer_updates": fit["optimizer_updates"],
                "data_inventory": str(directory / "fit/data_inventory.json"),
                "fit_report": str(directory / "fit/report.json"),
            }
            if records and row["ppo_updates_start"] != records[-1]["ppo_updates_end"]:
                raise RuntimeError("Collector PPO progress was reset between stages.")
            _write_json(directory / "report.json", row)
            records.append(row)
            active, policy = next_model, policy_next
            history_train += fit["new_training_paths"]
            history_valid.append(fit["validation"]["selection"])
            old_entries += reference["phases"]["replay"]
        if not all(file_sha256(path) == digest for path, digest in frozen.items()):
            raise ValueError("A protected cycle input changed.")
        _write_json(
            output / "report.json",
            {
                "stages": records,
                "active_model": active,
                "last_policy": policy,
                "accepted_stages": sum(row["accepted"] for row in records),
                "global_default_model_changed": False,
                "scope": (
                    "Bounded stagewise real-MJ active learning with freshly generated reference data; "
                    "original-file retention unverified"
                ),
            },
        )
        status.update(status="completed", phase="completed")
    except Exception as error:
        status.update(status="failed", failure=repr(error))
        raise
    finally:
        (output / "status.json").write_text(json.dumps(status, indent=2))


def main():
    """Launch a local staged active-learning pilot from an explicit immutable configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_active_cycle(_json(args.config), args.output)


if __name__ == "__main__":
    main()
