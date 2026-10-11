# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read-only old/new-distribution evaluation without replacing missing original data with a proxy."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from isaaclab_neural.data.opd import ContextDataset, file_sha256
from isaaclab_neural.utils.checkpoint import get_cfg_from_checkpoint, load_checkpoint, reconstruct_model_from_checkpoint


def retention_adaptation_score(
    baseline: dict[str, float],
    measured: dict[str, float],
    *,
    old_names: list[str],
    new_names: list[str],
    old_kind: str,
    relative_tolerance: float = 0.0,
    absolute_tolerance: float = 1e-8,
) -> dict:
    """Minimize relative new-data error subject to old-data nonregression; proxies never certify originals."""
    if old_kind not in {"original", "proxy", "resampled_recipe", "missing"} or not new_names:
        raise ValueError("Explicit old-data provenance and nonempty new-data cohorts are required.")
    if set(old_names) & set(new_names) or len(set(old_names + new_names)) != len(old_names + new_names):
        raise ValueError("Old and new metric roles must be distinct.")
    if (old_kind == "missing") != (not old_names):
        raise ValueError("Missing original data cannot silently pass a retention constraint.")
    if any(not math.isfinite(x) or x < 0 for x in (relative_tolerance, absolute_tolerance)):
        raise ValueError("Retention tolerances must be finite and nonnegative.")
    for name in old_names + new_names:
        if name not in baseline or name not in measured:
            raise ValueError(f"Missing evaluation cohort: {name}")
        if (
            not math.isfinite(baseline[name])
            or not math.isfinite(measured[name])
            or min(baseline[name], measured[name]) < 0
        ):
            raise ValueError("Evaluation losses must be finite and nonnegative.")
    ratios = {name: 1 + (measured[name] - baseline[name]) / max(baseline[name], 1e-12) for name in new_names}
    score = float(np.mean(list(ratios.values())))
    preserved = (
        all(measured[name] <= baseline[name] * (1 + relative_tolerance) + absolute_tolerance for name in old_names)
        if old_names
        else None
    )
    return {
        "old_kind": old_kind,
        "old_preserved": preserved,
        "original_retention_measured": old_kind == "original" and preserved is not None,
        "original_retention_verified": old_kind == "original" and preserved is True,
        "old_loss_deltas": {name: measured[name] - baseline[name] for name in old_names},
        "new_cohort_ratios": ratios,
        "new_relative_loss": score,
        "new_relative_gain": 1 - score,
        "admissible": old_kind == "original" and preserved is True and score < 1.0,
        "proxy_admissible": old_kind == "proxy" and preserved is True and score < 1.0,
        "resampled_admissible": old_kind == "resampled_recipe" and preserved is True and score < 1.0,
        "resampled_retention_verified": old_kind == "resampled_recipe" and preserved is True,
    }


def sample_recipe_windows(mapping: np.ndarray, *, history: int, count: int, seed: int) -> list[int]:
    """Select unique, nonoverlapping windows from valid trajectory indices with a private RNG."""
    if history < 1 or mapping.ndim != 2 or mapping.shape[1] != 2:
        raise ValueError("Expected positive history length and [trajectory,start] mapping.")
    eligible = np.flatnonzero(mapping[:, 1] % history == 0)
    if not 1 <= count <= len(eligible):
        raise ValueError("Insufficient nonoverlapping trajectory windows.")
    return np.sort(np.random.default_rng(seed).choice(eligible, count, replace=False)).tolist()


def recipe_loss(metrics: dict, weights: dict[str, float], key: str = "teacher_forced_loss") -> float:
    """Aggregate absolute source losses using the original normalized training mixture."""
    if not weights or set(metrics) != set(weights):
        raise ValueError("Every recipe source must be present with its archived weight.")
    if any(not math.isfinite(w) or w <= 0 for w in weights.values()):
        raise ValueError("Recipe weights must be finite and positive.")
    if any(not math.isfinite(row[key]) or row[key] < 0 for row in metrics.values()):
        raise ValueError("Recipe losses must be finite and nonnegative.")
    return sum(weights[name] * row[key] for name, row in metrics.items()) / sum(weights.values())


def recipe_replay_counts(weights: dict[str, float], total: int) -> dict[str, int]:
    """Allocate a fixed replay-row budget with largest-remainder source quotas."""
    if total < 1 or not weights or any(not math.isfinite(w) or w <= 0 for w in weights.values()):
        raise ValueError("Replay size and recipe weights must be positive and finite.")
    raw = np.asarray(list(weights.values()), dtype=np.float64)
    raw = raw / raw.sum() * total
    counts = np.floor(raw).astype(np.int64)
    order = np.argsort(-(raw - counts), kind="stable")
    counts[order[: total - int(counts.sum())]] += 1
    if np.any(counts < 1):
        raise ValueError("Replay budget cannot represent every recipe source.")
    return dict(zip(weights, counts.tolist()))


def check_recipe_replay_separation(replay: list[dict], protected: list[dict]) -> None:
    """Reject reuse of held-out recipe files or their reserved generation seeds."""
    if {row["sha256"] for row in replay} & {row["sha256"] for row in protected}:
        raise ValueError("Replay reused a protected evaluation file.")
    if {row["spec"]["seed"] for row in replay} & {row["spec"]["seed"] for row in protected}:
        raise ValueError("Replay reused an evaluation seed group.")


def merge_recipe_replay(caches: dict[str, dict], *, expected_count: int, keep_sequence_targets: bool = False) -> dict:
    """Join physical-input caches, retaining full real labels only when explicitly requested."""
    if not caches or sum(len(row["target"]) for row in caches.values()) != expected_count:
        raise ValueError("Replacement replay must preserve the reference-row budget.")
    keys = tuple(next(iter(caches.values()))["inputs"])
    for row in caches.values():
        if set(row["inputs"]) != set(keys) or not torch.equal(row["target"], row["sequence_target"][:, -1]):
            raise ValueError("Replay must share input fields and supervise only its last anchor.")
    result = {
        "inputs": {key: torch.cat([row["inputs"][key] for row in caches.values()]) for key in keys},
        "target": torch.cat([row["target"] for row in caches.values()]),
        "truth": torch.cat([row["truth"] for row in caches.values()]),
        "indices": list(range(expected_count)),
        "source_hashes": {path: digest for row in caches.values() for path, digest in row["source_hashes"].items()},
        "families": sorted({family for row in caches.values() for family in row["families"]}),
        "source_names": [name for name, row in caches.items() for _ in range(len(row["target"]))],
        "source_indices": {name: row["indices"] for name, row in caches.items()},
        "source_windows": {name: row["window_mapping"] for name, row in caches.items()},
        "origin": "resampled_recipe_replay",
    }
    if keep_sequence_targets:
        result["sequence_target"] = torch.cat([row["sequence_target"] for row in caches.values()])
    return result


def split_new_context_rows(
    identities: np.ndarray, *, selection_worlds: list[int], audit_worlds: list[int]
) -> tuple[list[int], list[int]]:
    """Partition complete worlds, so overlapping histories and episodes cannot straddle evaluation roles."""
    if identities.ndim != 2 or identities.shape[1] != 4 or not np.issubdtype(identities.dtype, np.integer):
        raise ValueError("Expected integer [world,episode,frame,policy-update] identities.")
    if not selection_worlds or not audit_worlds or set(selection_worlds) & set(audit_worlds):
        raise ValueError("Selection and audit worlds must be nonempty and disjoint.")
    if len(np.unique(identities, axis=0)) != len(identities):
        raise ValueError("Repeated query identities would silently reweight evaluation.")
    selected = np.isin(identities[:, 0], selection_worlds)
    audited = np.isin(identities[:, 0], audit_worlds)
    if not (selected | audited).all():
        raise ValueError("An input world is outside the declared held-out roles.")
    return np.flatnonzero(selected).tolist(), np.flatnonzero(audited).tolist()


def _write(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def _original_inventory(checkpoint, config):
    cfg = get_cfg_from_checkpoint(checkpoint, config["dynamics"])
    dataset = cfg["algorithm"]["dataset"]
    rows = []
    for role in ("train_datasets", "valid_datasets"):
        for name, entry in dataset[role].items():
            path = Path(entry["path"] if isinstance(entry, dict) else entry).resolve()
            row = {"name": name, "role": role, "configured_path": str(path), "exists": path.exists()}
            if path.exists():
                with h5py.File(path, "r") as file:
                    row["state_dim"] = int(file["data/states"].shape[-1])
                    row["joint_f_dim"] = int(file["data/joint_f"].shape[-1])
                    row["compatible"] = (row["state_dim"], row["joint_f_dim"]) == (87, 43)
                row["sha256"] = file_sha256(path)
            rows.append(row)
    return {
        "expected_state_dim": 87,
        "expected_joint_f_dim": 43,
        "configured_sources": rows,
        "status": "unverified_original_data",
        "note": (
            "Matching paths/shapes do not establish original training provenance. "
            "Missing/incompatible data are not padded or replaced by a proxy."
        ),
    }


def _identity_data(dataset):
    identities, families = [], []
    for path in dataset.paths:
        with h5py.File(path, "r") as file:
            identities.append(file["opd/query_identity"][:])
            families.extend(json.loads(file.attrs["opd_metadata"])["families"])
    return np.concatenate(identities), families


def _verify_entry(entry):
    if file_sha256(entry["path"]) != entry["sha256"]:
        raise ValueError("A fixed evaluation cache changed.")
    if not all(file_sha256(path) == digest for path, digest in entry["source_hashes"].items()):
        raise ValueError("A raw evaluation source changed.")


def _prepare_coverage(base, checkpoint, config, output):
    from isaaclab_neural.eval.learnability import _context_cache, g1_error_groups
    from isaaclab_neural.generate.opd import simulation_protocol

    solver = base.neural_adapter.solver
    options = {
        "physics_dt": base.physics_dt,
        "state_dim": 87,
        "joint_f_dim": 43,
        "contact_filter": solver.contact_filter,
        "max_contact_tokens": solver.max_contact_tokens,
    }
    protocol = simulation_protocol(base)
    manifest = {
        "schema": "old_new_coverage_v1",
        "config": config,
        "physics": protocol,
        "dynamics_sha256": file_sha256(config["dynamics"]),
        "groups": g1_error_groups(protocol["joint_names"]),
        "original_data": _original_inventory(checkpoint, config),
        "cohorts": {},
        "new_region_claim": "Visited distributions, not certified novel support; original coverage data unavailable",
    }
    all_train_families = set()
    reports = {}
    for name, entry in config["cohorts"].items():
        if file_sha256(entry["report"]) != entry["report_sha256"]:
            raise ValueError("Collector report changed.")
        report = json.loads(Path(entry["report"]).read_text())
        if report["provenance"]["protocol"] != protocol or not report["dynamics_frozen"]:
            raise ValueError("Collector protocol or frozen model contract differs.")
        reports[name] = report
        paths = [p for p in report["labelled_training_and_validation_shards"] if Path(p).parent.name == "train"]
        dataset = ContextDataset(paths, split="train", **options)
        train_ids, _ = _identity_data(dataset)
        if not np.all((train_ids[:, 0] >= 0) & (train_ids[:, 0] < config["training_world_stop"])):
            raise ValueError("Training data invaded the declared held-out worlds.")
        all_train_families.update(dataset.families)
    for name, report in reports.items():
        paths = [p for p in report["labelled_training_and_validation_shards"] if Path(p).parent.name == "validation"]
        dataset = ContextDataset(paths, split="validation", **options)
        if set(dataset.lengths) != {10} or dataset.families & all_train_families:
            raise ValueError("New-cohort validation leaked training episodes or mixed histories.")
        identities, families = _identity_data(dataset)
        selected, audited = split_new_context_rows(
            identities, selection_worlds=config["selection_worlds"], audit_worlds=config["audit_worlds"]
        )
        manifest["cohorts"][name] = {}
        for phase, indices in (("selection", selected), ("audit", audited)):
            cache = _context_cache(dataset, indices, solver, base.device)
            cache["families"] = sorted({families[i] for i in indices})
            path = output / f"{name}_{phase}.pt"
            torch.save(cache, path)
            manifest["cohorts"][name][phase] = {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "anchors": len(indices),
                "worlds": sorted(set(identities[indices, 0].tolist())),
                "families": cache["families"],
                "source_hashes": cache["source_hashes"],
                "role": "new_visited_distribution",
            }
        if set(manifest["cohorts"][name]["selection"]["families"]) & set(
            manifest["cohorts"][name]["audit"]["families"]
        ):
            raise ValueError("Selection/audit episode leakage.")
        print(json.dumps({"prepared_cohort": name, "selection": len(selected), "audit": len(audited)}), flush=True)
    prior = json.loads(Path(config["prior_prepared_manifest"]).read_text())
    if prior["dynamics_sha256"] != manifest["dynamics_sha256"]:
        raise ValueError("Reference proxy was prepared for another checkpoint.")
    manifest["old_proxy"] = {**prior["reference"], "role": "rehearsal_training_proxy_not_original_holdout"}
    _verify_entry(manifest["old_proxy"])
    _write(output / "manifest.json", manifest)
    _write(output / "original_data_inventory.json", manifest["original_data"])


def _load_cache(entry, device):
    _verify_entry(entry)
    return torch.load(entry["path"], map_location=device, weights_only=True)


def _check_candidate(checkpoint, baseline):
    if not torch.equal(checkpoint["loss_weights"].cpu(), baseline["loss_weights"].cpu()):
        raise ValueError("Candidate changed the evaluation loss weights.")
    for key, value in baseline["model_state_dict"].items():
        if key.startswith(("input_rms.", "output_rms.", "body_net_contact_force_rms.")):
            if not torch.equal(value.cpu(), checkpoint["model_state_dict"][key].cpu()):
                raise ValueError("Candidate changed frozen normalization statistics.")


@torch.no_grad()
def _evaluate_coverage(base, model, checkpoint, config, output):
    from isaaclab_neural.eval.learnability import _measure_context_cache

    manifest_path = Path(config["prepared_manifest"])
    if file_sha256(manifest_path) != config["prepared_manifest_sha256"]:
        raise ValueError("Evaluation split manifest changed.")
    manifest = json.loads(manifest_path.read_text())
    if manifest["dynamics_sha256"] != file_sha256(config["dynamics"]):
        raise ValueError("Evaluation checkpoint changed.")
    selection = {name: _load_cache(entry["selection"], base.device) for name, entry in manifest["cohorts"].items()}
    proxy = _load_cache(manifest["old_proxy"], base.device)
    groups, weights, solver = manifest["groups"], checkpoint["loss_weights"].to(base.device), base.neural_adapter.solver
    old_name = "rehearsal_training_proxy"

    def measure(cache):
        return _measure_context_cache(cache, model, solver, weights, groups)

    baseline_metrics = {name: measure(cache) for name, cache in selection.items()}
    baseline_metrics[old_name] = measure(proxy)
    baseline = {name: row["weighted_loss"] for name, row in baseline_metrics.items()}
    matrix = {"F_pre": {"checkpoint": config["dynamics"], "metrics": baseline_metrics, "epoch": 0}}
    chosen = {family: "F_pre" for family in config["method_cohorts"]}
    candidate_paths = {"F_pre": config["dynamics"]}
    _write(output / "selection_F_pre.json", matrix["F_pre"])
    for candidate in config["candidates"]:
        path = candidate["path"]
        if file_sha256(path) != candidate["sha256"]:
            raise ValueError("An existing epoch checkpoint changed.")
        state = load_checkpoint(path, device=base.device)
        _check_candidate(state, checkpoint)
        model.load_state_dict(state["model_state_dict"], strict=True)
        model.eval()
        metrics = {name: measure(cache) for name, cache in selection.items()}
        metrics[old_name] = measure(proxy)
        values = {name: row["weighted_loss"] for name, row in metrics.items()}
        kwargs = {
            "old_names": [old_name],
            "old_kind": "proxy",
            "absolute_tolerance": config["old_absolute_tolerance"],
            "relative_tolerance": config["old_relative_tolerance"],
        }
        family, cohort = candidate["method"], config["method_cohorts"][candidate["method"]]
        own = retention_adaptation_score(baseline, values, new_names=[cohort], **kwargs)
        common = retention_adaptation_score(baseline, values, new_names=config["common_new_cohorts"], **kwargs)
        row = {
            **candidate,
            "metrics": metrics,
            "own_new_rule": own,
            "common_new_rule": common,
            "original_data_retention": "unverified",
            "selection_scope": "provisional_proxy_only",
        }
        matrix[candidate["name"]] = row
        candidate_paths[candidate["name"]] = path
        best = chosen[family]
        previous = matrix[best]["metrics"][cohort]["weighted_loss"]
        if own["proxy_admissible"] and metrics[cohort]["weighted_loss"] < previous:
            chosen[family] = candidate["name"]
        _write(output / f"selection_{candidate['name']}.json", row)
        print(
            json.dumps(
                {
                    "candidate": candidate["name"],
                    "own_new_ratio": own["new_relative_loss"],
                    "common_new_ratio": common["new_relative_loss"],
                    "old_proxy_delta": own["old_loss_deltas"],
                    "proxy_admissible": own["proxy_admissible"],
                }
            ),
            flush=True,
        )
    _write(output / "selection_matrix.json", matrix)
    _write(
        output / "provisional_selection.json",
        {
            "chosen": chosen,
            "formal_acceptance": None,
            "reason": "Original training/validation data unverified; replay-proxy constraint only",
        },
    )
    del selection
    audit = {name: _load_cache(entry["audit"], base.device) for name, entry in manifest["cohorts"].items()}
    audited = {}
    for name in ["F_pre", *sorted(set(chosen.values()) - {"F_pre"})]:
        state = load_checkpoint(candidate_paths[name], device=base.device)
        model.load_state_dict(state["model_state_dict"], strict=True)
        model.eval()
        metrics = {key: measure(cache) for key, cache in audit.items()}
        audited[name] = metrics
        _write(output / f"audit_{name}.json", metrics)
    ratios = {
        name: {
            cohort: stats["weighted_loss"] / audited["F_pre"][cohort]["weighted_loss"] for cohort, stats in row.items()
        }
        for name, row in audited.items()
    }
    _write(
        output / "result.json",
        {
            "status": "Read-only old/new evaluation completed",
            "provisional_selected": chosen,
            "formal_acceptance": None,
            "original_data_retention": "unverified_missing_or_incompatible_sources",
            "original_data_inventory": manifest["original_data"],
            "proxy_constraint": "Fixed768-sample rehearsal TRAINING pool, not the original corpus or an old holdout",
            "old_absolute_tolerance": config["old_absolute_tolerance"],
            "old_relative_tolerance": config["old_relative_tolerance"],
            "audit_loss_ratios": ratios,
            "audit_anchor_counts": {name: len(cache["target"]) for name, cache in audit.items()},
            "common_new_score": {
                name: float(np.mean([row[c] for c in config["common_new_cohorts"]])) for name, row in ratios.items()
            },
            "new_region_status": "Held-out visited distributions; original-policy support expansion not certified",
            "model_optimizer_updates": 0,
            "policy_updates": 0,
            "live_promotion": False,
            "multistep_metrics": "Historical4/16-step reports remain separate, unchanged deployment-risk diagnostics",
            "scope": "Post-hoc development protocol, fixed selection/audit worlds; not a preregistered blind test",
        },
    )


@torch.no_grad()
def _recipe_cache(dataset, indices, solver, device):
    from torch.utils.data import default_collate

    fields = {"states", "next_states", "root_body_q", "root_body_qd", "gravity_dir", "joint_f"}
    fields.update(dataset.native_projection.output_keys)
    inputs, targets, truths = {}, [], []
    for start in range(0, len(indices), 128):
        raw = default_collate(
            [
                {key: value for key, value in dataset[index].items() if key in fields}
                for index in indices[start : start + 128]
            ]
        )
        batch = {key: value.to(device) for key, value in raw.items()}
        solver.process_neural_model_inputs(batch)
        target = solver.convert_next_states_to_prediction(
            states=batch["states"], next_states=batch["next_states"], dt=0.005
        )
        targets.append(target.cpu())
        truths.append(batch.pop("next_states")[:, -1].cpu())
        for key, value in batch.items():
            inputs.setdefault(key, []).append(value.cpu())
    target = torch.cat(targets)
    digest = file_sha256(dataset.dataset_path)
    mapping = dataset.mapping_index2traj[indices]
    cache = {
        "inputs": {key: torch.cat(value) for key, value in inputs.items()},
        "target": target[:, -1],
        "sequence_target": target,
        "truth": torch.cat(truths),
        "indices": indices,
        "window_mapping": mapping.tolist(),
        "trajectory_ids": torch.as_tensor(mapping[:, 0]),
        "source_hashes": {str(dataset.dataset_path): digest},
        "families": sorted({f"{digest}:{trajectory}" for trajectory in mapping[:, 0]}),
    }
    if not all(torch.isfinite(value).all() for value in (*cache["inputs"].values(), target, cache["truth"])):
        raise FloatingPointError("Non-finite resampled-recipe cache.")
    return cache


def _prepare_recipe(base, checkpoint, config, output):
    from isaaclab_neural.data.datasets import LazyTrajectoryDataset
    from isaaclab_neural.eval.learnability import g1_error_groups
    from isaaclab_neural.generate.opd import simulation_protocol

    collection_path = Path(config["collection_manifest"])
    if file_sha256(collection_path) != config["collection_manifest_sha256"]:
        raise ValueError("Collection provenance changed.")
    collected = json.loads(collection_path.read_text())
    if collected["status"] != "completed" or collected["training_updates"] != 0:
        raise ValueError("Recipe collection is incomplete or not read-only.")
    for path, digest in collected["source_hashes"].items():
        if file_sha256(path) != digest:
            raise ValueError("Recipe generator or policy source changed.")
    archived = get_cfg_from_checkpoint(checkpoint, config["dynamics"])["algorithm"]["dataset"]
    weights = {name: entry["weight"] for name, entry in archived["train_datasets"].items()}
    counts = {
        name: round(config["recipe_samples"] * weight / sum(weights.values())) for name, weight in weights.items()
    }
    if sum(counts.values()) != config["recipe_samples"]:
        raise ValueError("Choose a sample count exactly compatible with all recipe proportions.")
    protocol, solver = simulation_protocol(base), base.neural_adapter.solver
    manifest = {
        "schema": "resampled_old_recipe_v1",
        "config": config,
        "weights": weights,
        "counts": counts,
        "phases": {},
        "dynamics_sha256": file_sha256(config["dynamics"]),
        "groups": g1_error_groups(protocol["joint_names"]),
        "original_data_retention": "unverified",
        "origin": "resampled_recipe",
        "physics_dt": base.physics_dt,
    }
    sources = [
        (name, Path(entry["path"]).name, counts[name], "recipe") for name, entry in archived["train_datasets"].items()
    ]
    sources += [
        (name, Path(path).name, config["diagnostic_samples"], "diagnostic")
        for name, path in archived["valid_datasets"].items()
    ]
    for phase_index, phase in enumerate(("selection", "audit")):
        files = {entry["spec"]["filename"]: entry for entry in collected["phases"][phase]}
        manifest["phases"][phase] = {"recipe": {}, "diagnostic": {}}
        for source_index, (name, filename, count, role) in enumerate(sources):
            entry = files[filename]
            if file_sha256(entry["path"]) != entry["sha256"]:
                raise ValueError("A generated recipe dataset changed.")
            if (entry["state_dim"], entry["joint_f_dim"], entry["action_dim"]) != (87, 43, 37):
                raise ValueError("Recipe data have the wrong robot layout.")
            dataset = LazyTrajectoryDataset(
                entry["path"],
                sample_sequence_length=10,
                contact_representation="contact_tokens",
                max_contact_tokens=solver.max_contact_tokens,
                contact_filter=solver.contact_filter,
                contact_packing_policy="body_round_robin_pair_atomic",
            )
            try:
                indices = sample_recipe_windows(
                    dataset.mapping_index2traj,
                    history=10,
                    count=count,
                    seed=config["sampling_seed"] + 100 * phase_index + source_index,
                )
                cache = _recipe_cache(dataset, indices, solver, base.device)
            finally:
                dataset.close()
            path = output / f"{phase}_{role}_{name}.pt"
            torch.save(cache, path)
            manifest["phases"][phase][role][name] = {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "anchors": len(indices),
                "families": cache["families"],
                "source_hashes": cache["source_hashes"],
                "role": "resampled_recipe",
                "generation_seed": entry["spec"]["seed"],
                "source_file": entry["path"],
                "source_trajectory_count": entry["trajectories"],
                "evaluated_trajectories": len(cache["families"]),
            }
            print(json.dumps({"prepared": phase, "role": role, "source": name, "anchors": len(indices)}), flush=True)
    families = {
        phase: {
            family
            for entries in manifest["phases"][phase].values()
            for entry in entries.values()
            for family in entry["families"]
        }
        for phase in manifest["phases"]
    }
    if families["selection"] & families["audit"]:
        raise ValueError("Resampled-old selection and audit share generated trajectories.")
    _write(output / "manifest.json", manifest)


def _prepare_recipe_replay(base, checkpoint, config, output):
    from isaaclab_neural.data.datasets import LazyTrajectoryDataset

    for key in ("collection_manifest", "protected_collection", "parent_manifest"):
        if file_sha256(config[key]) != config[f"{key}_sha256"]:
            raise ValueError("Replay preparation provenance changed.")
    collected = json.loads(Path(config["collection_manifest"]).read_text())
    protected = json.loads(Path(config["protected_collection"]).read_text())
    parent = json.loads(Path(config["parent_manifest"]).read_text())
    if collected["status"] != "completed" or collected["training_updates"] != 0:
        raise ValueError("Replay requires completed analytic collection without fitting.")
    if parent["dynamics_sha256"] != file_sha256(config["dynamics"]):
        raise ValueError("The current-data cache and replacement replay require the same F_pre.")
    for path, digest in collected["source_hashes"].items():
        if file_sha256(path) != digest:
            raise ValueError("Replay collection source changed.")
    entries = collected["phases"]["replay"]
    check_recipe_replay_separation(entries, [row for phase in protected["phases"].values() for row in phase])
    paths = {str(Path(row["path"]).resolve()) for row in entries}
    for row in entries:
        command = row["command"]
        if "--initial-states-pool" in command:
            dependency = Path(command[command.index("--initial-states-pool") + 1]).resolve()
            if str(dependency) not in paths:
                raise ValueError("Replay must use its independently collected walking pool.")
    original_reference = _load_cache(parent["reference"], "cpu")
    count = len(original_reference["target"])
    if count != config["reference_count"] or parent["reference"]["anchors"] != count:
        raise ValueError("Replay replacement changed the fixed reference-row budget.")
    sources = get_cfg_from_checkpoint(checkpoint, config["dynamics"])["algorithm"]["dataset"]["train_datasets"]
    weights = {name: entry["weight"] for name, entry in sources.items()}
    quotas = recipe_replay_counts(weights, count)
    files = {Path(row["path"]).name: row for row in entries}
    caches = {}
    for index, (name, source) in enumerate(sources.items()):
        entry = files[Path(source["path"]).name]
        if file_sha256(entry["path"]) != entry["sha256"] or (
            entry["state_dim"],
            entry["joint_f_dim"],
            entry["action_dim"],
        ) != (87, 43, 37):
            raise ValueError("Replay data changed or have an incompatible robot layout.")
        solver = base.neural_adapter.solver
        dataset = LazyTrajectoryDataset(
            entry["path"],
            sample_sequence_length=10,
            contact_representation="contact_tokens",
            max_contact_tokens=solver.max_contact_tokens,
            contact_filter=solver.contact_filter,
            contact_packing_policy="body_round_robin_pair_atomic",
        )
        try:
            if quotas[name] > len(dataset):
                raise ValueError("Insufficient unique replay anchors.")
            indices = np.sort(
                np.random.default_rng(config["replay_sampling_seed"] + index).choice(
                    len(dataset), quotas[name], replace=False
                )
            ).tolist()
            caches[name] = _recipe_cache(dataset, indices, solver, base.device)
        finally:
            dataset.close()
    replay = merge_recipe_replay(caches, expected_count=count)
    if set(replay["inputs"]) != set(original_reference["inputs"]):
        raise ValueError("Replacement replay changed the input contract.")
    for key, value in replay["inputs"].items():
        previous = original_reference["inputs"][key]
        if value.shape != previous.shape or value.dtype != previous.dtype:
            raise ValueError(f"Replacement replay changed the batch layout: {key}")
    replay["source_hashes"].update({row["path"]: row["sha256"] for row in entries})
    path = output / "reference_recipe.pt"
    torch.save(replay, path)
    manifest = copy.deepcopy(parent)
    manifest["reference"] = {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "anchors": count,
        "source_hashes": replay["source_hashes"],
        "role": "resampled_recipe_replay",
    }
    manifest["reference_replacement"] = {
        "parent_manifest_sha256": config["parent_manifest_sha256"],
        "collection_manifest_sha256": config["collection_manifest_sha256"],
        "counts": quotas,
        "realized_fractions": {name: n / count for name, n in quotas.items()},
        "supervision": "last anchor only",
        "generation_seeds": {row["spec"]["filename"]: row["spec"]["seed"] for row in entries},
        "sampling_seed": config["replay_sampling_seed"],
        "reference_count_unchanged": True,
        "validation_unchanged": True,
    }
    manifest["config"]["reference_train"] = [row["path"] for row in entries if row["spec"]["split"] == "train"]
    for key in ("training", "validation", "groups", "protocol", "dynamics_sha256"):
        if manifest[key] != parent[key]:
            raise RuntimeError("Replay replacement changed a protected current/evaluation input.")
    _write(output / "manifest.json", manifest)
    _write(output / "replay_audit.json", manifest["reference_replacement"])


@torch.no_grad()
def _measure_recipe_cache(cache, model, solver, weights, groups):
    from isaaclab_neural.eval.learnability import _cache_batch, _clone_batch, _measure_context_cache

    result = _measure_context_cache(cache, model, solver, weights, groups)
    losses, positions = [], []
    device = next(model.parameters()).device
    for start in range(0, len(cache["target"]), 64):
        ids = torch.arange(start, min(start + 64, len(cache["target"])), dtype=torch.long)
        inputs, _, _ = _cache_batch(cache, ids, device)
        target = cache["sequence_target"].index_select(0, ids.to(cache["sequence_target"].device)).to(device)
        prediction = model(_clone_batch(inputs))
        if prediction.shape != target.shape:
            raise ValueError("Original-style sequence loss requires all teacher-forced positions.")
        components = (prediction * weights - target * weights).square()
        if not torch.isfinite(components).all():
            raise FloatingPointError("Non-finite old-recipe teacher-forced loss.")
        losses.append(components.double().mean((1, 2)).cpu())
        positions.append(components.double().mean(2).cpu())
    sample_loss = torch.cat(losses)
    result["teacher_forced_position_loss"] = torch.cat(positions).mean(0).tolist()
    result["teacher_forced_loss"] = float(sample_loss.mean())
    result["teacher_forced_per_window_loss"] = sample_loss.tolist()
    return result


@torch.no_grad()
def _evaluate_recipe(base, model, checkpoint, config, output):
    from isaaclab_neural.eval.learnability import _measure_context_cache

    reuse_new = bool(config.get("new_selection_matrix"))
    keys = ["prepared_manifest", "new_manifest"] + (["new_selection_matrix"] if reuse_new else [])
    for key in keys:
        if file_sha256(config[key]) != config[f"{key}_sha256"]:
            raise ValueError(f"Frozen evaluation input changed: {key}")
    manifest = json.loads(Path(config["prepared_manifest"]).read_text())
    previous = json.loads(Path(config["new_selection_matrix"]).read_text()) if reuse_new else {}
    new_manifest = json.loads(Path(config["new_manifest"]).read_text())
    if any(item["dynamics_sha256"] != file_sha256(config["dynamics"]) for item in (manifest, new_manifest)):
        raise ValueError("Old and new evaluation baselines differ.")
    names = {row["name"] for row in config["candidates"]}
    if len(names) != len(config["candidates"]) or "F_pre" in names:
        raise ValueError("Candidate names must be unique and distinct from the baseline.")
    if set(config.get("fixed_audit_candidates", [])) - names:
        raise ValueError("Fixed audit endpoints must be declared candidates.")
    if reuse_new and set(previous) - {"F_pre"} != names:
        raise ValueError("Existing candidates and new-data measurements must match exactly.")
    for candidate in config["candidates"]:
        if file_sha256(candidate["path"]) != candidate["sha256"]:
            raise ValueError("A candidate checkpoint changed.")
        if reuse_new and candidate["sha256"] != previous[candidate["name"]]["sha256"]:
            raise ValueError("Cannot reuse new-selection metrics after candidate changes.")
    source_weights, groups, solver = manifest["weights"], manifest["groups"], base.neural_adapter.solver
    weights = checkpoint["loss_weights"].to(base.device)

    def caches(phase):
        return {
            role: {name: _load_cache(entry, base.device) for name, entry in sources.items()}
            for role, sources in manifest["phases"][phase].items()
        }

    def measure(loaded):
        return {
            role: {
                name: _measure_recipe_cache(cache, model, solver, weights, groups) for name, cache in sources.items()
            }
            for role, sources in loaded.items()
        }

    new_selection = (
        {name: _load_cache(entry["selection"], base.device) for name, entry in new_manifest["cohorts"].items()}
        if not reuse_new
        else {}
    )

    def measure_new():
        return {
            name: _measure_context_cache(cache, model, solver, weights, groups) for name, cache in new_selection.items()
        }

    if not reuse_new:
        previous["F_pre"] = {"checkpoint": config["dynamics"], "metrics": measure_new(), "epoch": 0}
    selected_data = caches("selection")
    measured = measure(selected_data)
    baseline_loss = recipe_loss(measured["recipe"], source_weights)
    baseline = {name: entry["weighted_loss"] for name, entry in previous["F_pre"]["metrics"].items()}
    baseline["resampled_old_recipe"] = baseline_loss
    matrix = {"F_pre": {"old_metrics": measured, "recipe_loss": baseline_loss, "checkpoint": config["dynamics"]}}
    chosen = {name: "F_pre" for name in config["method_cohorts"]}
    paths = {"F_pre": config["dynamics"]}
    _write(output / "selection_F_pre.json", matrix["F_pre"])
    for candidate in config["candidates"]:
        state = load_checkpoint(candidate["path"], device=base.device)
        _check_candidate(state, checkpoint)
        model.load_state_dict(state["model_state_dict"], strict=True)
        model.eval()
        measured = measure(selected_data)
        old_loss = recipe_loss(measured["recipe"], source_weights)
        name, method = candidate["name"], candidate["method"]
        cohort = config["method_cohorts"][method]
        if not reuse_new:
            previous[name] = {**candidate, "metrics": measure_new()}
        current = {key: value["weighted_loss"] for key, value in previous[name]["metrics"].items()}
        current["resampled_old_recipe"] = old_loss
        rule = retention_adaptation_score(
            baseline,
            current,
            old_names=["resampled_old_recipe"],
            new_names=[cohort],
            old_kind="resampled_recipe",
            absolute_tolerance=config["old_absolute_tolerance"],
            relative_tolerance=config["old_relative_tolerance"],
        )
        matrix[name] = {
            **candidate,
            "old_metrics": measured,
            "recipe_loss": old_loss,
            "retention_adaptation": rule,
            "old_loss_ratio": old_loss / baseline_loss,
            "old_source_ratios": {
                key: row["teacher_forced_loss"] / matrix["F_pre"]["old_metrics"]["recipe"][key]["teacher_forced_loss"]
                for key, row in measured["recipe"].items()
            },
        }
        best = chosen[method]
        if rule["resampled_admissible"] and current[cohort] < previous[best]["metrics"][cohort]["weighted_loss"]:
            chosen[method] = name
        paths[name] = candidate["path"]
        _write(output / f"selection_{name}.json", matrix[name])
        print(
            json.dumps(
                {
                    "candidate": name,
                    "old_recipe_ratio": old_loss / baseline_loss,
                    "own_new_ratio": rule["new_relative_loss"],
                    "resampled_admissible": rule["resampled_admissible"],
                }
            ),
            flush=True,
        )
    _write(output / "selection_matrix.json", matrix)
    _write(
        output / "selection.json",
        {
            "chosen": chosen,
            "basis": "Fresh old recipe selection plus preserved new-selection data",
            "original_data_retention": "unverified",
        },
    )
    if not reuse_new:
        _write(output / "new_selection_matrix.json", previous)
    del selected_data
    new_selection.clear()
    old_audit = caches("audit")
    new_audit = {name: _load_cache(entry["audit"], base.device) for name, entry in new_manifest["cohorts"].items()}
    audits = {}
    audit_names = (set(chosen.values()) | set(config.get("fixed_audit_candidates", []))) - {"F_pre"}
    for name in ["F_pre", *sorted(audit_names)]:
        state = load_checkpoint(paths[name], device=base.device)
        model.load_state_dict(state["model_state_dict"], strict=True)
        model.eval()
        old_metrics = measure(old_audit)
        new_metrics = {
            key: _measure_context_cache(cache, model, solver, weights, groups) for key, cache in new_audit.items()
        }
        audits[name] = {
            "old_metrics": old_metrics,
            "new_metrics": new_metrics,
            "recipe_loss": recipe_loss(old_metrics["recipe"], source_weights),
            "old_anchor_recipe_loss": recipe_loss(old_metrics["recipe"], source_weights, key="weighted_loss"),
        }
        _write(output / f"audit_{name}.json", audits[name])
    summary = {}
    for name, row in audits.items():
        old_ratios = {
            key: value["teacher_forced_loss"] / audits["F_pre"]["old_metrics"]["recipe"][key]["teacher_forced_loss"]
            for key, value in row["old_metrics"]["recipe"].items()
        }
        new_ratios = {
            key: value["weighted_loss"] / audits["F_pre"]["new_metrics"][key]["weighted_loss"]
            for key, value in row["new_metrics"].items()
        }
        summary[name] = {
            "old_recipe_loss": row["recipe_loss"],
            "old_recipe_ratio": row["recipe_loss"] / audits["F_pre"]["recipe_loss"],
            "old_anchor_recipe_ratio": row["old_anchor_recipe_loss"] / audits["F_pre"]["old_anchor_recipe_loss"],
            "old_source_ratios": old_ratios,
            "new_ratios": new_ratios,
            "old_recipe_audit_preserved": row["recipe_loss"]
            <= audits["F_pre"]["recipe_loss"] * (1 + config["old_relative_tolerance"])
            + config["old_absolute_tolerance"],
            "common_new_relative_loss": float(np.mean([new_ratios[key] for key in config["common_new_cohorts"]])),
        }
    _write(
        output / "result.json",
        {
            "selected": chosen,
            "fixed_audit_candidates": config.get("fixed_audit_candidates", []),
            "new_selection_metrics_reused": reuse_new,
            "audit": summary,
            "original_data_retention": "unverified",
            "evidence_kind": "Independent reseeded current pretraining-recipe samples, not original files",
            "old_metric": "Full teacher-forced10-position weighted MSE; recipe source weights preserved",
            "new_metric": "H10 last-anchor error on reused development audit corpus",
            "old_selection_samples": sum(row["anchors"] for row in manifest["phases"]["selection"]["recipe"].values()),
            "old_audit_samples": sum(row["anchors"] for row in manifest["phases"]["audit"]["recipe"].values()),
            "optimizer_updates": 0,
            "policy_updates": 0,
            "live_promotion": False,
            "scope": "Small resampled-recipe retention test, not an exact historical-data or population guarantee",
        },
    )


@torch.no_grad()
def _audit_prior_recipe_candidates(base, model, checkpoint, config, output):
    for key in ("prepared_manifest", "prior_selection", "current_selection"):
        if file_sha256(config[key]) != config[f"{key}_sha256"]:
            raise ValueError("Fixed-candidate diagnostic provenance changed.")
    manifest = json.loads(Path(config["prepared_manifest"]).read_text())
    if manifest["dynamics_sha256"] != file_sha256(config["dynamics"]):
        raise ValueError("Diagnostic baseline differs from prepared recipe data.")
    prior = json.loads(Path(config["prior_selection"]).read_text())["chosen"]
    names = sorted(set(prior.values()) - {"F_pre"})
    candidates = {row["name"]: row for row in config["candidates"]}
    caches = {
        role: {name: _load_cache(entry, base.device) for name, entry in sources.items()}
        for role, sources in manifest["phases"]["audit"].items()
    }
    groups, weights = manifest["groups"], checkpoint["loss_weights"].to(base.device)
    results = {}
    for name in ["F_pre", *names]:
        path = config["dynamics"] if name == "F_pre" else candidates[name]["path"]
        if name != "F_pre" and file_sha256(path) != candidates[name]["sha256"]:
            raise ValueError("A pre-existing diagnostic candidate changed.")
        state = load_checkpoint(path, device=base.device)
        _check_candidate(state, checkpoint)
        model.load_state_dict(state["model_state_dict"], strict=True)
        model.eval()
        measured = {
            role: {
                source: _measure_recipe_cache(cache, model, base.neural_adapter.solver, weights, groups)
                for source, cache in entries.items()
            }
            for role, entries in caches.items()
        }
        row = {
            "model": name,
            "path": path,
            "metrics": measured,
            "recipe_loss": recipe_loss(measured["recipe"], manifest["weights"]),
            "anchor_recipe_loss": recipe_loss(measured["recipe"], manifest["weights"], key="weighted_loss"),
        }
        results[name] = row
        _write(output / f"{name}.json", row)
    baseline = results["F_pre"]
    summary = {
        name: {
            "recipe_loss": row["recipe_loss"],
            "recipe_ratio": row["recipe_loss"] / baseline["recipe_loss"],
            "anchor_ratio": row["anchor_recipe_loss"] / baseline["anchor_recipe_loss"],
            "source_ratios": {
                source: data["teacher_forced_loss"] / baseline["metrics"]["recipe"][source]["teacher_forced_loss"]
                for source, data in row["metrics"]["recipe"].items()
            },
        }
        for name, row in results.items()
    }
    if file_sha256(config["current_selection"]) != config["current_selection_sha256"]:
        raise RuntimeError("Diagnostic evaluation changed the locked selection.")
    _write(
        output / "result.json",
        {
            "scope": "Diagnostic-only audit of candidates fixed in the preceding study",
            "selection_changed": False,
            "original_data_retention": "unverified",
            "audit": summary,
            "optimizer_updates": 0,
            "policy_updates": 0,
            "live_promotion": False,
        },
    )


@torch.no_grad()
def _evaluate_recorded_sequences(base, model, checkpoint, config, output):
    from isaaclab_neural.data.opd_sequence import read_sequence_shard
    from isaaclab_neural.eval.learnability import _measure_study_rollouts

    if file_sha256(config["sequence_manifest"]) != config["sequence_manifest_sha256"]:
        raise ValueError("Recorded-sequence manifest changed.")
    manifest = json.loads(Path(config["sequence_manifest"]).read_text())
    for entry in manifest["validation"].values():
        shard = read_sequence_shard(entry["sequence_path"], require_labels=True)
        if (
            shard["metadata"].get("source_kind") != "mjwarp"
            or shard["metadata"].get("history_origin") != "physical_frames"
        ):
            raise ValueError("This audit requires real MJ histories and recorded future controls.")
    result = {}
    for candidate in config["models"]:
        if file_sha256(candidate["path"]) != candidate["sha256"]:
            raise ValueError("Recorded-sequence candidate changed.")
        state = load_checkpoint(candidate["path"], device=base.device)
        _check_candidate(state, checkpoint)
        model.load_state_dict(state["model_state_dict"], strict=True)
        model.eval()
        result[candidate["name"]] = _measure_study_rollouts(
            base, model, checkpoint["loss_weights"].to(base.device), manifest, config
        )
        _write(output / f"{candidate['name']}.json", result[candidate["name"]])
    _write(
        output / "result.json",
        {
            "models": result,
            "model_optimizer_updates": 0,
            "policy_updates": 0,
            "scope": "Recorded future actions; true MJ labels; full and terminal-prefix horizons separated",
        },
    )


@torch.no_grad()
def run_joint_f_retention(env, args) -> None:
    """Replay held-out recorded forces with the pretrained evaluator in isolated scratch worlds."""
    from isaaclab_neural.eval.training_evaluator import TrainingRolloutEvaluator
    from isaaclab_neural.train.trainers import adapt_training_env

    config = json.loads(Path(args.active_fit_config).read_text())
    base = env.unwrapped
    entry = config["dataset"]
    count = config["num_rollouts"]
    if not 1 <= count <= 1024 or count % base.num_envs:
        raise ValueError("Joint-force retention needs complete scratch batches and at most1024 rollouts.")
    if file_sha256(entry["path"]) != entry["sha256"] or file_sha256(args.dynamics) != config["initial_sha256"]:
        raise ValueError("Joint-force evaluation data or fixed F_pre changed.")
    if file_sha256(config["training_manifest"]) != config["training_manifest_sha256"]:
        raise ValueError("Training manifest changed before the held-out rollout audit.")
    training = json.loads(Path(config["training_manifest"]).read_text())
    if any(entry["sha256"] in training[name]["source_hashes"].values() for name in ("current", "reference")):
        raise ValueError("Joint-force evaluation cannot reuse an optimization source.")
    initial = load_checkpoint(args.dynamics, device=base.device)
    solver = base.neural_adapter.solver
    model = reconstruct_model_from_checkpoint(initial, solver, device=base.device).eval().requires_grad_(False)
    evaluator = TrainingRolloutEvaluator(
        adapt_training_env(env),
        hdf5_dataset_path=entry["path"],
        eval_horizon=10,
        device=base.device,
        contact_representation=solver.contact_representation,
        num_contacts_per_env=solver.num_contacts_per_env,
        max_contact_tokens=solver.max_contact_tokens,
        contact_filter=solver.contact_filter,
        contact_packing_policy=initial["cfg"]["env"]["neural_solver_cfg"]["contact_packing_policy"],
    )
    random_state = np.random.get_state()
    try:
        np.random.seed(config["sampling_seed"])
        trajectories = evaluator._sample_dataset_trajectories(count)
    finally:
        np.random.set_state(random_state)
    if evaluator.history_length != 10 or evaluator._history_offset(trajectories) != 9:
        raise ValueError("Joint-force audit requires nine real warm-start frames and a real anchor.")
    report = {
        "dataset": entry,
        "num_rollouts": count,
        "num_envs": base.num_envs,
        "physics_dt": base.physics_dt,
        "horizon": 10,
        "control": "recorded joint_f, no PD recomputation",
        "loss": "unweighted state MSE over all10 future steps",
        "trajectory_indices": trajectories["_dataset_trajectory_index"].tolist(),
        "window_starts": trajectories["_dataset_window_start"].tolist(),
        "models": {},
        "model_optimizer_updates": 0,
        "policy_updates": 0,
        "scratch_worlds_advanced": True,
        "original_data_retention": "unverified; held-out resampled policy trajectories, not historical originals",
    }
    original_model = solver.neural_model
    solver.set_neural_solver_model(model)
    try:
        for candidate in config["models"]:
            if file_sha256(candidate["path"]) != candidate["sha256"]:
                raise ValueError("Joint-force candidate changed after selection.")
            checkpoint = load_checkpoint(candidate["path"], device=base.device)
            _check_candidate(checkpoint, initial)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            delta, _, metrics = evaluator._evaluate_control_rollout(
                trajectories,
                control_key="joint_f",
                step_fn=evaluator._step_joint_f,
                eval_mode="rollout",
                render=False,
            )
            if not torch.isfinite(delta).all():
                raise FloatingPointError("Non-finite joint-force rollout predictions.")
            row = {
                section: {key: value.detach().cpu().tolist() for key, value in values.items()}
                for section, values in metrics.items()
            }
            row["nonfinite_entries"] = 0
            report["models"][candidate["name"]] = row
            _write(args.output / f"{candidate['name']}.json", row)
            if not all(
                torch.equal(value, model.state_dict()[key]) for key, value in checkpoint["model_state_dict"].items()
            ):
                raise RuntimeError("Joint-force evaluation modified model weights or normalization.")
    finally:
        solver.set_neural_solver_model(original_model)
    if file_sha256(args.dynamics) != config["initial_sha256"] or file_sha256(entry["path"]) != entry["sha256"]:
        raise RuntimeError("Joint-force evaluation modified its frozen inputs.")
    _write(args.output / "result.json", report)


@torch.no_grad()
def _evaluate_rl_coverage(base, model, checkpoint, config, output):
    from isaaclab_neural.eval.learnability import (
        _context_cache,
        _measure_study_rollouts,
        g1_error_groups,
        sample_context_indices,
    )

    if file_sha256(config["benchmark_manifest"]) != config["benchmark_manifest_sha256"]:
        raise ValueError("RL coverage benchmark changed.")
    manifest = json.loads(Path(config["benchmark_manifest"]).read_text())
    caches = {}
    sequences = {"validation": {}}
    for index, (name, entry) in enumerate(manifest["cohorts"].items()):
        if file_sha256(entry["report"]["path"]) != entry["report"]["sha256"]:
            raise ValueError("Benchmark collection report changed.")
        collected = json.loads(Path(entry["report"]["path"]).read_text())
        if collected["provenance"]["collection_mode"] != "collect_benchmark" or not collected["policy_unchanged"]:
            raise ValueError("RL coverage requires evaluation-only frozen-policy data.")
        for path, digest in entry["source_hashes"].items():
            if file_sha256(path) != digest:
                raise ValueError("Benchmark raw data changed.")
        solver = base.neural_adapter.solver
        dataset = ContextDataset(
            collected["context_shards"],
            split="validation",
            physics_dt=0.005,
            state_dim=87,
            joint_f_dim=43,
            contact_filter=solver.contact_filter,
            max_contact_tokens=solver.max_contact_tokens,
            supervision="all_positions",
        )
        ids = sample_context_indices(
            dataset, min(config["context_samples"], len(dataset)), seed=config["sampling_seed"] + index
        )
        caches[name] = _context_cache(dataset, ids, solver, base.device, keep_sequence_targets=True)
        sequences["validation"][name] = {
            "seed": config["heldout_seed"],
            "sequence_path": collected["sequence_shard"],
            "sequence_sha256": entry["source_hashes"][collected["sequence_shard"]],
        }
    groups = g1_error_groups(manifest["protocol"]["joint_names"])
    result = {}
    for candidate in config["models"]:
        if file_sha256(candidate["path"]) != candidate["sha256"]:
            raise ValueError("Coverage candidate changed.")
        state = load_checkpoint(candidate["path"], device=base.device)
        _check_candidate(state, checkpoint)
        model.load_state_dict(state["model_state_dict"], strict=True)
        model.eval()
        result[candidate["name"]] = {
            "teacher_forced": {
                name: _measure_recipe_cache(cache, model, solver, checkpoint["loss_weights"].to(base.device), groups)
                for name, cache in caches.items()
            },
            "action_sequences": _measure_study_rollouts(
                base, model, checkpoint["loss_weights"].to(base.device), sequences, config
            ),
        }
        _write(output / f"{candidate['name']}.json", result[candidate["name"]])
    _write(
        output / "result.json",
        {
            "models": result,
            "context_indices": {name: cache["indices"] for name, cache in caches.items()},
            "optimizer_updates": 0,
            "evaluation_only": True,
        },
    )


@torch.no_grad()
def _audit_fixed_training_models(base, model, checkpoint, config, output):
    for key in ("training_manifest", "old_recipe_manifest", "selection_lock"):
        if file_sha256(config[key]) != config[f"{key}_sha256"]:
            raise ValueError("Locked post-selection audit input changed.")
    locked = json.loads(Path(config["selection_lock"]).read_text())
    allowed = {row["sha256"] for row in locked["audit_models"]}
    prepared = json.loads(Path(config["training_manifest"]).read_text())
    old = json.loads(Path(config["old_recipe_manifest"]).read_text())
    current = _load_cache(prepared["validation"]["audit"], base.device)
    references = {name: _load_cache(entry, base.device) for name, entry in old["phases"]["audit"]["recipe"].items()}
    result = {}
    for candidate in config["models"]:
        if candidate["sha256"] not in allowed or file_sha256(candidate["path"]) != candidate["sha256"]:
            raise ValueError("Audit model was not locked before opening audit data.")
        saved = load_checkpoint(candidate["path"], device=base.device)
        _check_candidate(saved, checkpoint)
        model.load_state_dict(saved["model_state_dict"], strict=True)
        model.eval()
        metrics = {
            name: _measure_recipe_cache(
                cache, model, base.neural_adapter.solver, checkpoint["loss_weights"].to(base.device), old["groups"]
            )
            for name, cache in references.items()
        }
        now = _measure_recipe_cache(
            current, model, base.neural_adapter.solver, checkpoint["loss_weights"].to(base.device), old["groups"]
        )
        result[candidate["name"]] = {
            "old_recipe": recipe_loss(metrics, old["weights"]),
            "current": now["teacher_forced_loss"],
            "current_anchor": now["weighted_loss"],
            "old_metrics": metrics,
            "current_metrics": now,
        }
    _write(
        output / "result.json",
        {"models": result, "optimizer_updates": 0, "selection_lock_sha256": config["selection_lock_sha256"]},
    )


def run_coverage_study(env, args) -> None:
    """Evaluate preserved checkpoints on explicit old/new roles, without optimization or live promotion."""
    from isaaclab_neural.rl.rsl_rl.online_opd import _pause_fingerprint

    config = json.loads(Path(args.active_fit_config).read_text())
    base = env.unwrapped
    if base.num_envs != config.get("scratch_worlds", 4):
        raise ValueError("Evaluation scratch-world layout differs from its declared numerical profile.")
    before = _pause_fingerprint(base, base.obs_buf)
    dynamics_hash = file_sha256(args.dynamics)
    if dynamics_hash != file_sha256(config["dynamics"]):
        raise ValueError("Configured baseline and launch checkpoint differ.")
    checkpoint = load_checkpoint(args.dynamics, device=base.device)
    if config["mode"] == "prepare":
        _prepare_coverage(base, checkpoint, config, args.output)
    elif config["mode"] == "recipe_prepare":
        _prepare_recipe(base, checkpoint, config, args.output)
    elif config["mode"] == "replay_prepare":
        _prepare_recipe_replay(base, checkpoint, config, args.output)
    elif config["mode"] in {
        "evaluate",
        "recipe_evaluate",
        "recipe_audit_prior",
        "sequence_retention",
        "rl_coverage",
        "fixed_training_audit",
    }:
        model = (
            reconstruct_model_from_checkpoint(checkpoint, base.neural_adapter.solver, device=base.device)
            .eval()
            .requires_grad_(False)
        )
        if config["mode"] == "fixed_training_audit":
            _audit_fixed_training_models(base, model, checkpoint, config, args.output)
        elif config["mode"] == "rl_coverage":
            _evaluate_rl_coverage(base, model, checkpoint, config, args.output)
        elif config["mode"] == "sequence_retention":
            _evaluate_recorded_sequences(base, model, checkpoint, config, args.output)
        elif config["mode"] == "recipe_evaluate":
            _evaluate_recipe(base, model, checkpoint, config, args.output)
        elif config["mode"] == "recipe_audit_prior":
            _audit_prior_recipe_candidates(base, model, checkpoint, config, args.output)
        else:
            _evaluate_coverage(base, model, checkpoint, config, args.output)
    else:
        raise ValueError("Unknown coverage evaluation mode.")
    if _pause_fingerprint(base, base.obs_buf) != before or file_sha256(args.dynamics) != dynamics_hash:
        raise RuntimeError("Read-only evaluation changed the baseline or live environment.")
    _write(
        args.output / "integrity.json",
        {
            "live_environment_unchanged": True,
            "baseline_unchanged": True,
            "model_optimizer_updates": 0,
            "policy_updates": 0,
        },
    )
