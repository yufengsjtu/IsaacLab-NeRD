# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analyze old/new E199 checkpoints on frozen old/new validation suites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

METRICS = ("state_MSE", "q_MSE", "qd_MSE", "state_L2", "q_L2", "qd_L2")
REGIMES = (
    "exp_trajectory",
    "zero_action_trajectory",
    "lstm_actuator_zero_action_trajectory",
    "lstm_actuator_policy_trajectory",
)


def _load_result(path: str | Path) -> dict[str, Any]:
    result_dir = Path(path).expanduser().resolve()
    metadata_path = result_dir / "metrics.json" if result_dir.is_dir() else result_dir
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema_version") != 1:
        raise ValueError(f"Unsupported result schema: {metadata_path}.")
    payload_path = metadata_path.parent / metadata["per_sample_metrics"]
    with np.load(payload_path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    return {"metadata": metadata, "payload": payload, "path": str(metadata_path)}


def _group_metric(payload: dict[str, np.ndarray], group: str, metric: str, prefix: str = "") -> np.ndarray:
    arrays = []
    for seed in (0, 1, 2):
        key = f"{prefix}{group}_seed{seed}_{metric}"
        if key not in payload:
            raise ValueError(f"Result payload is missing {key!r}.")
        values = np.asarray(payload[key], dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError(f"Result payload {key!r} must be a finite vector.")
        arrays.append(values)
    return np.stack(arrays)


def _trajectory_means(values: np.ndarray, trajectory_index: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(trajectory_index, return_inverse=True)
    counts = np.bincount(inverse, minlength=unique.size)
    rows = []
    for values_for_seed in values:
        rows.append(np.bincount(inverse, weights=values_for_seed, minlength=unique.size) / counts)
    return np.stack(rows)


def _paired_summary(
    new_values: np.ndarray,
    old_values: np.ndarray,
    trajectory_index: np.ndarray,
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    if new_values.shape != old_values.shape:
        raise ValueError("Old/new metric arrays must have the same shape.")
    new_mean = float(new_values.mean())
    old_mean = float(old_values.mean())
    new_trajectory = _trajectory_means(new_values, trajectory_index)
    old_trajectory = _trajectory_means(old_values, trajectory_index)
    trajectory_delta = (new_trajectory - old_trajectory).mean(axis=0)
    rng = np.random.default_rng(bootstrap_seed)
    samples = rng.integers(
        0,
        trajectory_delta.size,
        size=(bootstrap_samples, trajectory_delta.size),
    )
    bootstrap = trajectory_delta[samples].mean(axis=1)
    return {
        "new_mean": new_mean,
        "old_mean": old_mean,
        "mean_delta": new_mean - old_mean,
        "ratio": new_mean / old_mean,
        "relative_change_percent": (new_mean / old_mean - 1.0) * 100.0,
        "per_seed_relative_change_percent": [
            float((new_values[index].mean() / old_values[index].mean() - 1.0) * 100.0)
            for index in range(new_values.shape[0])
        ],
        "trajectory_bootstrap_delta_ci95": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "num_trajectories": int(trajectory_delta.size),
    }


def _stratify_primary(
    payload: dict[str, np.ndarray],
    new_values: np.ndarray,
    old_values: np.ndarray,
    label_key: str,
) -> list[dict[str, Any]]:
    labels = np.asarray(payload[label_key])
    rows = []
    for label in np.unique(labels):
        mask = labels == label
        if int(mask.sum()) < 50:
            continue
        new_mean = float(new_values[:, mask].mean())
        old_mean = float(old_values[:, mask].mean())
        rows.append(
            {
                "value": int(label),
                "num_windows": int(mask.sum()),
                "new_mean": new_mean,
                "old_mean": old_mean,
                "relative_change_percent": (new_mean / old_mean - 1.0) * 100.0,
            }
        )
    return rows


def analyze_results(
    paths: list[str | Path],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20_260_826,
) -> dict[str, Any]:
    """Return within-suite fixed-E199 comparisons for all eight suites."""
    results = [_load_result(path) for path in paths]
    encoders = {result["metadata"]["encoder"] for result in results}
    if len(encoders) != 1:
        raise ValueError(f"Results contain multiple encoders: {sorted(encoders)}.")
    by_key = {}
    for result in results:
        suite = result["metadata"]["suite"]
        key = (suite["sampling_strategy"], suite["regime"])
        if key in by_key:
            raise ValueError(f"Duplicate result for {key}.")
        by_key[key] = result
    expected = {(sampling, regime) for sampling in ("old", "new") for regime in REGIMES}
    if set(by_key) != expected:
        raise ValueError(f"Result suites are {sorted(by_key)}, expected {sorted(expected)}.")

    suite_summaries = {}
    for suite_index, (key, result) in enumerate(sorted(by_key.items())):
        sampling, regime = key
        payload = result["payload"]
        trajectory_index = np.asarray(payload["trajectory_index"], dtype=np.int64)
        metrics = {}
        for metric_index, metric in enumerate(METRICS):
            new_values = _group_metric(payload, "new", metric)
            old_values = _group_metric(payload, "old", metric)
            metrics[metric] = _paired_summary(
                new_values,
                old_values,
                trajectory_index,
                bootstrap_seed=bootstrap_seed + suite_index * 100 + metric_index,
                bootstrap_samples=bootstrap_samples,
            )
        primary_new = _group_metric(payload, "new", "state_MSE")
        primary_old = _group_metric(payload, "old", "state_MSE")
        summary: dict[str, Any] = {
            "suite_id": result["metadata"]["suite"]["suite_id"],
            "sampling_strategy": sampling,
            "regime": regime,
            "dataset_sha256": result["metadata"]["suite"]["sha256"],
            "metrics": metrics,
            "state_MSE_by_terrain_level": _stratify_primary(
                payload,
                primary_new,
                primary_old,
                "trajectory_terrain_level",
            ),
            "state_MSE_by_terrain_type": _stratify_primary(
                payload,
                primary_new,
                primary_old,
                "trajectory_terrain_type",
            ),
        }
        if result["metadata"]["rollout"] is not None:
            rollout_trajectory = np.asarray(payload["rollout_trajectory_index"], dtype=np.int64)
            summary["rollout"] = {
                metric: _paired_summary(
                    _group_metric(payload, "new", metric, prefix="rollout_"),
                    _group_metric(payload, "old", metric, prefix="rollout_"),
                    rollout_trajectory,
                    bootstrap_seed=bootstrap_seed + suite_index * 100 + 50 + metric_index,
                    bootstrap_samples=bootstrap_samples,
                )
                for metric_index, metric in enumerate(METRICS)
            }
        suite_summaries[result["metadata"]["suite"]["suite_id"]] = summary

    return {
        "schema_version": 1,
        "encoder": encoders.pop(),
        "checkpoint_epoch": 199,
        "comparison": "new-sampling checkpoints versus old-sampling checkpoints on the same frozen suite",
        "lower_is_better": True,
        "suites": suite_summaries,
    }


def _markdown_report(analysis: dict[str, Any]) -> str:
    lines = [
        f"# Encoder {analysis['encoder'].upper()} sampling-strategy evaluation",
        "",
        "All comparisons use fixed `model_epoch199.pt`. Negative percentages favor new sampling.",
        "",
        "| Evaluation distribution | Regime | state_MSE change | state_L2 change |",
        "|---|---|---:|---:|",
    ]
    for suite in analysis["suites"].values():
        mse = suite["metrics"]["state_MSE"]["relative_change_percent"]
        l2 = suite["metrics"]["state_L2"]["relative_change_percent"]
        lines.append(f"| {suite['sampling_strategy']} | {suite['regime']} | {mse:+.2f}% | {l2:+.2f}% |")
    policy_suites = [
        suite
        for suite in analysis["suites"].values()
        if suite["regime"] == "lstm_actuator_policy_trajectory" and "rollout" in suite
    ]
    if policy_suites:
        lines.extend(
            (
                "",
                "## Deterministic 10-step policy-suite rollouts",
                "",
                "| Evaluation distribution | rollout state_MSE change | rollout state_L2 change |",
                "|---|---:|---:|",
            )
        )
        for suite in policy_suites:
            mse = suite["rollout"]["state_MSE"]["relative_change_percent"]
            l2 = suite["rollout"]["state_L2"]["relative_change_percent"]
            lines.append(f"| {suite['sampling_strategy']} | {mse:+.2f}% | {l2:+.2f}% |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_826)
    args = parser.parse_args()
    analysis = analyze_results(
        args.result,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    output_json = Path(args.output_json).expanduser().resolve()
    output_markdown = Path(args.output_markdown).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    output_markdown.write_text(_markdown_report(analysis))
    print(f"[sampling-eval] wrote {output_json}")
    print(f"[sampling-eval] wrote {output_markdown}")


if __name__ == "__main__":
    main()
