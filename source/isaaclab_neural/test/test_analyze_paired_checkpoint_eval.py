# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the standalone paired-checkpoint comparison CLI."""

from __future__ import annotations

import json

import numpy as np
import pytest
from isaaclab_neural.eval.analyze_paired_checkpoint_eval import (
    load_evaluation_result,
    parse_comparison,
    validate_aligned_evaluation_results,
)


def _write_result(tmp_path, design: str, *, manifest_key: str, offset: float = 0.0):
    result_dir = tmp_path / design
    result_dir.mkdir()
    checkpoints = {
        f"seed{seed}": {
            "wandb_run_id": f"run-{seed}",
            "checkpoint_sha256": str(seed) * 64,
            "metrics": {},
            "state_dimension_MSE": [0.1, 0.2],
        }
        for seed in range(3)
    }
    metadata = {
        "schema_version": 1,
        "design": design,
        "num_windows": 6,
        "dataset": {
            "manifest_key": manifest_key,
            "pairing": {
                "shared_data_sha256": "data-hash",
                "shared_context_sha256": "context-hash",
                "num_trajectories": 3,
                "steps_per_trajectory": 4,
            },
        },
        "checkpoints": checkpoints,
        "per_window_metrics": "per_window_metrics.npz",
    }
    (result_dir / "metrics.json").write_text(json.dumps(metadata))
    payload = {
        "window_index": np.arange(6, dtype=np.int64),
        "trajectory_index": np.repeat(np.arange(3, dtype=np.int64), 2),
        "start_step": np.tile(np.arange(2, dtype=np.int64), 3),
        "trajectory_terrain_level": np.array([0, 1, 0], dtype=np.int64),
        "trajectory_terrain_type": np.array([2, 2, 3], dtype=np.int64),
        "trajectory_source_env_id": np.array([10, 11, 12], dtype=np.int64),
        "contact_count_mean": np.full(6, 2.0),
    }
    for seed in range(3):
        payload[f"seed{seed}_state_MSE"] = np.arange(6, dtype=np.float32) + seed + offset
    np.savez_compressed(result_dir / "per_window_metrics.npz", **payload)
    return result_dir


def test_load_and_validate_results_allow_different_paired_contact_files(tmp_path) -> None:
    raw = load_evaluation_result(
        "a_active",
        _write_result(tmp_path, "a_active", manifest_key="raw15"),
        metric_names=["state_MSE"],
    )
    contact = load_evaluation_result(
        "c_active",
        _write_result(tmp_path, "c_active", manifest_key="contact_tokens"),
        metric_names=["state_MSE"],
    )

    alignment = validate_aligned_evaluation_results({"a_active": raw, "c_active": contact})

    assert alignment["num_windows"] == 6
    assert alignment["num_trajectories"] == 3
    assert alignment["shared_data_sha256"] == "data-hash"
    assert raw["seed_ids"] == [0, 1, 2]
    assert raw["metrics"]["state_MSE"].shape == (3, 6)


def test_alignment_rejects_coordinate_or_pairing_mismatch(tmp_path) -> None:
    first = load_evaluation_result(
        "first",
        _write_result(tmp_path, "first", manifest_key="raw15"),
        metric_names=["state_MSE"],
    )
    second_path = _write_result(tmp_path, "second", manifest_key="contact_tokens")
    second = load_evaluation_result("second", second_path, metric_names=["state_MSE"])
    second["payload"]["start_step"][0] = 99
    with pytest.raises(ValueError, match="start_step"):
        validate_aligned_evaluation_results({"first": first, "second": second})

    second = load_evaluation_result("second", second_path, metric_names=["state_MSE"])
    second["metadata"]["dataset"]["pairing"]["shared_data_sha256"] = "different"
    with pytest.raises(ValueError, match="pairing identity"):
        validate_aligned_evaluation_results({"first": first, "second": second})


def test_load_rejects_incomplete_seed_set(tmp_path) -> None:
    result_path = _write_result(tmp_path, "incomplete", manifest_key="raw15")
    metrics_path = result_path / "metrics.json"
    metadata = json.loads(metrics_path.read_text())
    metadata["checkpoints"].pop("seed2")
    metrics_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match=r"expected \[0, 1, 2\]"):
        load_evaluation_result("incomplete", result_path, metric_names=["state_MSE"])


def test_parse_comparison_requires_label_candidate_and_reference() -> None:
    assert parse_comparison("c-vs-a=c_active:a_active") == ("c-vs-a", "c_active", "a_active")
    with pytest.raises(ValueError, match="LABEL=CANDIDATE:REFERENCE"):
        parse_comparison("c_active:a_active")
