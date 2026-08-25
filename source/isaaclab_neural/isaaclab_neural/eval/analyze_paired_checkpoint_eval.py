# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare standalone fixed-checkpoint evaluations on paired transitions."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from isaaclab_neural.eval.paired_checkpoint_eval import summarize_paired_trajectory_metrics

DEFAULT_METRICS = (
    "state_MSE",
    "q_MSE",
    "qd_MSE",
    "state_L2",
    "q_error_norm",
    "qd_error_norm",
)
DEFAULT_COMPARISONS = (
    "c-active-vs-a-active=c_active:a_active",
    "a-mean-count-vs-a-active=a_mean_count:a_active",
    "c-no-other-vs-c-active=c_no_other:c_active",
    "c-no-other-vs-a-mean-count=c_no_other:a_mean_count",
)
_COORDINATE_KEYS = (
    "window_index",
    "trajectory_index",
    "start_step",
    "trajectory_terrain_level",
    "trajectory_terrain_type",
    "trajectory_source_env_id",
)
_PAIRING_KEYS = (
    "shared_data_sha256",
    "shared_context_sha256",
    "num_trajectories",
    "steps_per_trajectory",
)


def parse_comparison(value: str) -> tuple[str, str, str]:
    """Parse ``LABEL=CANDIDATE:REFERENCE`` comparison syntax."""
    try:
        label, designs = value.split("=", 1)
        candidate, reference = designs.split(":", 1)
    except ValueError as exc:
        raise ValueError("Comparison must use LABEL=CANDIDATE:REFERENCE syntax.") from exc
    if not label or not candidate or not reference or candidate == reference:
        raise ValueError("Comparison must use LABEL=CANDIDATE:REFERENCE syntax.")
    return label, candidate, reference


def _resolve_result_paths(path: str | Path) -> tuple[Path, Path]:
    result_path = Path(path).expanduser().resolve()
    metadata_path = result_path / "metrics.json" if result_path.is_dir() else result_path
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Evaluation metadata does not exist: {metadata_path}.")
    metadata = json.loads(metadata_path.read_text())
    payload_path = metadata_path.parent / metadata["per_window_metrics"]
    if not payload_path.is_file():
        raise FileNotFoundError(f"Evaluation payload does not exist: {payload_path}.")
    return metadata_path, payload_path


def _checkpoint_seed_ids(metadata: Mapping[str, Any]) -> list[int]:
    seed_pattern = re.compile(r"seed([0-9]+)")
    seeds = []
    for key in metadata["checkpoints"]:
        match = seed_pattern.fullmatch(key)
        if match is None:
            raise ValueError(f"Unexpected checkpoint label: {key!r}.")
        seeds.append(int(match.group(1)))
    return sorted(seeds)


def load_evaluation_result(
    design: str,
    path: str | Path,
    *,
    metric_names: Sequence[str] = DEFAULT_METRICS,
    expected_seed_ids: Sequence[int] = (0, 1, 2),
) -> dict[str, Any]:
    """Load and validate one design's metadata and per-window metrics."""
    metadata_path, payload_path = _resolve_result_paths(path)
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema_version") != 1:
        raise ValueError(f"Unsupported evaluation schema in {metadata_path}.")
    if metadata.get("design") != design:
        raise ValueError(f"Evaluation design mismatch: requested {design!r}, metadata has {metadata.get('design')!r}.")
    seed_ids = _checkpoint_seed_ids(metadata)
    expected = sorted(int(seed) for seed in expected_seed_ids)
    if seed_ids != expected:
        raise ValueError(f"Checkpoint seeds are {seed_ids}, expected {expected}.")

    with np.load(payload_path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    missing_coordinates = sorted(set(_COORDINATE_KEYS) - set(payload))
    if missing_coordinates:
        raise ValueError(f"Evaluation payload is missing coordinate arrays: {missing_coordinates}.")
    num_windows = int(metadata["num_windows"])
    if payload["window_index"].shape != (num_windows,):
        raise ValueError("window_index count does not match evaluation metadata.")
    if not np.array_equal(payload["window_index"], np.arange(num_windows, dtype=np.int64)):
        raise ValueError("window_index must be the complete deterministic zero-based sequence.")

    metrics: dict[str, np.ndarray] = {}
    for metric_name in metric_names:
        arrays = []
        for seed in seed_ids:
            key = f"seed{seed}_{metric_name}"
            if key not in payload:
                raise ValueError(f"Evaluation payload is missing {key!r}.")
            values = np.asarray(payload[key], dtype=np.float64)
            if values.shape != (num_windows,) or not np.isfinite(values).all():
                raise ValueError(f"Evaluation metric {key!r} must contain {num_windows} finite values.")
            arrays.append(values)
        metrics[metric_name] = np.stack(arrays)

    return {
        "design": design,
        "metadata_path": str(metadata_path),
        "payload_path": str(payload_path),
        "metadata": metadata,
        "payload": payload,
        "seed_ids": seed_ids,
        "metrics": metrics,
    }


def _pairing_identity(result: Mapping[str, Any]) -> dict[str, Any]:
    pairing = result["metadata"]["dataset"]["pairing"]
    missing = sorted(set(_PAIRING_KEYS) - set(pairing))
    if missing:
        raise ValueError(f"Paired dataset manifest is missing identity fields: {missing}.")
    return {key: pairing[key] for key in _PAIRING_KEYS}


def validate_aligned_evaluation_results(
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Require every design to cover the same paired targets and windows."""
    if len(results) < 2:
        raise ValueError("At least two evaluation results are required.")
    iterator = iter(results.items())
    reference_name, reference = next(iterator)
    reference_pairing = _pairing_identity(reference)
    reference_payload = reference["payload"]
    reference_seeds = reference["seed_ids"]
    for design, result in iterator:
        if _pairing_identity(result) != reference_pairing:
            raise ValueError(f"Paired dataset pairing identity differs for {reference_name!r} and {design!r}.")
        if result["seed_ids"] != reference_seeds:
            raise ValueError(f"Checkpoint seed sets differ for {reference_name!r} and {design!r}.")
        for key in _COORDINATE_KEYS:
            if not np.array_equal(reference_payload[key], result["payload"][key]):
                raise ValueError(f"Paired evaluation coordinate {key!r} differs for design {design!r}.")

    trajectory_index = np.asarray(reference_payload["trajectory_index"])
    num_trajectories = int(trajectory_index.max()) + 1
    if num_trajectories != int(reference_pairing["num_trajectories"]):
        raise ValueError("Window mapping trajectory count does not match the paired dataset manifest.")
    return {
        **reference_pairing,
        "num_windows": int(reference_payload["window_index"].size),
        "num_trajectories": num_trajectories,
        "seed_ids": list(reference_seeds),
        "designs": sorted(results),
    }


def _parse_result(value: str) -> tuple[str, Path]:
    try:
        design, path = value.split("=", 1)
    except ValueError as exc:
        raise ValueError("Result must use DESIGN=PATH syntax.") from exc
    if not design or not path:
        raise ValueError("Result must use DESIGN=PATH syntax.")
    return design, Path(path)


def _stratified_summaries(
    candidate: np.ndarray,
    reference: np.ndarray,
    trajectory_index: np.ndarray,
    trajectory_labels: np.ndarray,
    *,
    seed_ids: Sequence[int],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    summaries = []
    window_labels = trajectory_labels[trajectory_index]
    for group_index, label in enumerate(np.unique(trajectory_labels)):
        mask = window_labels == label
        original_indices = trajectory_index[mask]
        _unique_indices, remapped_indices = np.unique(original_indices, return_inverse=True)
        summary = summarize_paired_trajectory_metrics(
            candidate[:, mask],
            reference[:, mask],
            remapped_indices,
            seed_ids=seed_ids,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + group_index,
        )
        summaries.append({"value": int(label), **summary})
    return summaries


def _design_summary(result: Mapping[str, Any], metric_names: Sequence[str]) -> dict[str, Any]:
    checkpoints = result["metadata"]["checkpoints"]
    dimension_mse = np.stack(
        [np.asarray(checkpoints[f"seed{seed}"]["state_dimension_MSE"], dtype=np.float64) for seed in result["seed_ids"]]
    )
    return {
        "dataset_manifest_key": result["metadata"]["dataset"]["manifest_key"],
        "checkpoint_run_ids": {
            f"seed{seed}": checkpoints[f"seed{seed}"]["wandb_run_id"] for seed in result["seed_ids"]
        },
        "metric_means": {
            metric_name: {
                "ensemble": float(result["metrics"][metric_name].mean(dtype=np.float64)),
                "per_seed": [float(value) for value in result["metrics"][metric_name].mean(axis=1)],
            }
            for metric_name in metric_names
        },
        "state_dimension_MSE": {
            "ensemble": dimension_mse.mean(axis=0).tolist(),
            "per_seed": dimension_mse.tolist(),
        },
    }


def _contact_count_alignment(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_counts = np.asarray(candidate["payload"]["contact_count_mean"], dtype=np.float64)
    reference_counts = np.asarray(reference["payload"]["contact_count_mean"], dtype=np.float64)
    difference = candidate_counts - reference_counts
    return {
        "candidate_mean": float(candidate_counts.mean()),
        "reference_mean": float(reference_counts.mean()),
        "exact_match_fraction": float(np.mean(candidate_counts == reference_counts)),
        "mean_delta": float(difference.mean()),
        "mean_absolute_delta": float(np.abs(difference).mean()),
        "nonzero_delta_fraction": float(np.mean(difference != 0.0)),
    }


def main() -> None:
    """Load four design outputs and write deterministic paired comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", required=True, help="DESIGN=DIR_OR_METRICS_JSON")
    parser.add_argument("--comparison", action="append", help="LABEL=CANDIDATE:REFERENCE")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    metric_names = tuple(name.strip() for name in args.metrics.split(",") if name.strip())
    if not metric_names:
        raise ValueError("--metrics must contain at least one metric name.")
    result_specs = [_parse_result(value) for value in args.result]
    if len({design for design, _path in result_specs}) != len(result_specs):
        raise ValueError("--result design names must be unique.")
    results = {design: load_evaluation_result(design, path, metric_names=metric_names) for design, path in result_specs}
    alignment = validate_aligned_evaluation_results(results)
    comparisons = [parse_comparison(value) for value in (args.comparison or DEFAULT_COMPARISONS)]
    trajectory_index = np.asarray(next(iter(results.values()))["payload"]["trajectory_index"], dtype=np.int64)
    seed_ids = alignment["seed_ids"]
    comparison_output = {}
    for comparison_index, (label, candidate_name, reference_name) in enumerate(comparisons):
        if candidate_name not in results or reference_name not in results:
            raise ValueError(f"Comparison {label!r} refers to an unloaded design.")
        candidate = results[candidate_name]
        reference = results[reference_name]
        metric_output = {
            metric_name: summarize_paired_trajectory_metrics(
                candidate["metrics"][metric_name],
                reference["metrics"][metric_name],
                trajectory_index,
                seed_ids=seed_ids,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed + comparison_index,
            )
            for metric_name in metric_names
        }
        primary_candidate = candidate["metrics"]["state_MSE"]
        primary_reference = reference["metrics"]["state_MSE"]
        comparison_output[label] = {
            "candidate": candidate_name,
            "reference": reference_name,
            "metrics": metric_output,
            "by_terrain_level": _stratified_summaries(
                primary_candidate,
                primary_reference,
                trajectory_index,
                np.asarray(candidate["payload"]["trajectory_terrain_level"]),
                seed_ids=seed_ids,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed + 1000 + comparison_index * 100,
            ),
            "by_terrain_type": _stratified_summaries(
                primary_candidate,
                primary_reference,
                trajectory_index,
                np.asarray(candidate["payload"]["trajectory_terrain_type"]),
                seed_ids=seed_ids,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed + 2000 + comparison_index * 100,
            ),
            "post_filter_contact_count_alignment": _contact_count_alignment(candidate, reference),
        }

    output = {
        "schema_version": 1,
        "alignment": alignment,
        "designs": {design: _design_summary(result, metric_names) for design, result in results.items()},
        "comparisons": comparison_output,
        "interpretation_scope": {
            "primary_metric": "state_MSE on the paired policy-validation suite",
            "trajectory_bootstrap": "dataset uncertainty conditional on the fixed Epoch99 checkpoints",
            "training_seed_uncertainty": "reported per seed; three checkpoints do not justify a population CI",
            "causal_boundary": (
                "Paired evaluation removes evaluation-transition and target differences. It does not make A and C "
                "training datasets or encoder architectures identical."
            ),
        },
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(output_path)
    print(f"[paired-analysis] wrote {output_path}")


if __name__ == "__main__":
    main()
