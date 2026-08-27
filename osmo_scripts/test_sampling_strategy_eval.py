# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the standalone fixed-E199 sampling evaluator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from osmo_scripts.sampling_strategy_eval.analyze import _paired_summary
from osmo_scripts.sampling_strategy_eval.contract import (
    deterministic_indices,
    load_verified_checkpoints,
    prepare_eval_cfg,
)

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "osmo_scripts" / "sampling_strategy_eval"


def _a_cfg() -> dict:
    return {
        "env": {
            "neural_solver_cfg": {
                "contact_representation": "active15_tokens",
                "states_frame": "body",
            }
        },
        "algorithm": {
            "num_epochs": 1000,
            "num_iters_per_epoch": 5000,
            "sample_sequence_length": 10,
            "batch_size": 512,
            "num_valid_batches": 100,
            "update_dataset_statistics": True,
            "optimizer": {"type": "Adam", "lr_start": "1e-4", "lr_end": "1e-5"},
            "dataset": {
                "max_capacity": 20_000_000,
                "train_dataset_path": "train.hdf5",
                "valid_datasets": {"valid": "valid.hdf5"},
            },
        },
        "inputs": {
            "low_dim": ["states_embedding", "joint_f", "gravity_dir"],
            "contact_set": {
                "dim": 17,
                "encoder_type": "body_routed_active15",
                "body_latent_dim": 64,
                "hidden_dim": 32,
            },
        },
        "network": {"transformer": {"n_embd": 384}},
    }


def _write_checkpoint(path: Path, cfg: dict) -> tuple[int, str]:
    checkpoint = {
        "version": 2,
        "epoch": 199,
        "cfg": cfg,
        "model_state_dict": {"weight": torch.arange(4, dtype=torch.float32).reshape(2, 2)},
    }
    torch.save(checkpoint, path)
    return path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()


def test_checkpoint_manifest_requires_exact_old_new_three_seed_e199(tmp_path: Path) -> None:
    entries = []
    for group in ("old", "new"):
        for seed in range(3):
            checkpoint_path = tmp_path / f"{group}_seed{seed}.pt"
            size, digest = _write_checkpoint(checkpoint_path, _a_cfg())
            entries.append(
                {
                    "group": group,
                    "seed": seed,
                    "wandb_run_id": f"{group}{seed}",
                    "path": checkpoint_path.name,
                    "size_bytes": size,
                    "sha256": digest,
                }
            )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": 1, "encoder": "a", "checkpoints": entries}))

    loaded = load_verified_checkpoints(manifest_path, "a")

    assert [(entry["group"], entry["seed"]) for entry, _checkpoint in loaded] == [
        ("new", 0),
        ("new", 1),
        ("new", 2),
        ("old", 0),
        ("old", 1),
        ("old", 2),
    ]
    assert all(checkpoint["epoch"] == 199 for _entry, checkpoint in loaded)


def test_prepare_eval_cfg_decouples_model_semantics_from_suite_storage() -> None:
    raw_cfg = prepare_eval_cfg(
        _a_cfg(),
        encoder="a",
        dataset_path="/tmp/raw.hdf5",
        dataset_representation="raw15_tokens",
        batch_size=512,
        num_workers=8,
        num_envs=64,
        seed=7,
    )
    active_cfg = prepare_eval_cfg(
        _a_cfg(),
        encoder="a",
        dataset_path="/tmp/active.hdf5",
        dataset_representation="active15_tokens",
        batch_size=512,
        num_workers=8,
        num_envs=64,
        seed=7,
    )

    assert raw_cfg["env"]["neural_solver_cfg"]["contact_representation"] == "active15_tokens"
    assert raw_cfg["env"]["neural_solver_cfg"]["contact_filter"] == "solver_active"
    assert raw_cfg["algorithm"]["dataset"]["contact_representation"] == "raw15_tokens"
    assert "contact_filter" not in active_cfg["env"]["neural_solver_cfg"]
    assert active_cfg["algorithm"]["dataset"]["contact_representation"] == "active15_tokens"
    assert raw_cfg["cli"]["eval_interval"] == 0
    assert raw_cfg["algorithm"]["update_dataset_statistics"] is False


def test_deterministic_window_selection_is_sorted_unique_and_reproducible() -> None:
    first = deterministic_indices(10_000, 512, 20260826)
    second = deterministic_indices(10_000, 512, 20260826)

    assert np.array_equal(first, second)
    assert np.all(first[1:] > first[:-1])
    assert first.shape == (512,)
    assert deterministic_indices(4, 10, 0).tolist() == [0, 1, 2, 3]


def test_paired_summary_reports_new_minus_old_with_trajectory_bootstrap() -> None:
    old = np.array([[2.0, 2.0, 4.0, 4.0], [2.0, 2.0, 4.0, 4.0]])
    new = old * 0.75
    summary = _paired_summary(
        new,
        old,
        np.array([0, 0, 1, 1]),
        bootstrap_seed=0,
        bootstrap_samples=100,
    )

    assert summary["relative_change_percent"] == -25.0
    assert summary["per_seed_relative_change_percent"] == [-25.0, -25.0]
    assert summary["trajectory_bootstrap_delta_ci95"][1] < 0.0


def test_osmo_workflows_and_entrypoint_freeze_the_sampling_contract() -> None:
    a_workflow = (EVAL_DIR / "eval_a_workflow.yaml").read_text()
    d_workflow = (EVAL_DIR / "eval_d_workflow.yaml").read_text()
    entrypoint = (EVAL_DIR / "entrypoint.sh").read_text()

    for workflow in (a_workflow, d_workflow):
        assert "gpu: 1" in workflow
        assert "cpu: 12" in workflow
        assert "memory: 96Gi" in workflow
        assert "platform: ovx-l40" in workflow
        assert "/tmp/sampling-eval-entrypoint.sh" in workflow
        assert "osmo_scripts/start.sh" not in workflow
        assert "run_experiment.py" not in workflow
        assert "WANDB_API_KEY: wandb_api_key" in workflow
        assert "/tmp/sampling-eval-code-input/IsaacLab-NeRD.tar.gz.base64" in workflow
        assert "SAMPLING_EVAL_CODE_SHA256" in workflow
    assert a_workflow.count("    - url:") == 0
    assert d_workflow.count("    - url:") == 1
    assert "model_epoch199.pt" not in entrypoint
    assert "--max-windows 51200" in entrypoint
    assert '--rollout-count "$rollout_count"' in entrypoint
    assert "--rollout-horizon 10" in entrypoint
    assert "--window-seed 20260826" in entrypoint
    assert "--rollout-seed 20260826" in entrypoint
    assert "sampling_strategy_eval.run_eval" in entrypoint
    assert "isaaclab_neural.train.train" not in entrypoint
    assert "sampling_strategy_eval.download_checkpoints" in entrypoint
    assert "SAMPLING_EVAL_CHECKPOINT_INPUT_ROOT" in entrypoint
    assert "verified {len(loaded)} preloaded checkpoints" in entrypoint
    assert entrypoint.index('cd "$PROJECT_ROOT"') < entrypoint.index("\nprepare_checkpoints\n")
    assert "base64 --decode" in entrypoint
    assert "sha256sum --check" in entrypoint
    assert "SAMPLING_EVAL_ANALYSIS_JSON_BEGIN" in entrypoint
    assert "RESULTS_ACKNOWLEDGED" in entrypoint
    assert "osmo data upload" not in entrypoint
    assert "RESULTS_DOWNLOADED" not in entrypoint


def test_osmo_data_workflows_mount_both_sampling_generations() -> None:
    a_workflow = (EVAL_DIR / "eval_a_osmo_data_workflow.yaml").read_text()
    d_workflow = (EVAL_DIR / "eval_d_osmo_data_workflow.yaml").read_text()

    for workflow in (a_workflow, d_workflow):
        assert workflow.count("    - url:") == 2
        assert "gpu: 1" in workflow
        assert "storage: 512Gi" in workflow
        assert "/mnt/amlfs" not in workflow
        assert "/osmo/data/input/0/" in workflow
        assert "/osmo/data/input/1/" in workflow
        assert 'SAMPLING_EVAL_CHECKPOINT_ARTIFACT: "{{ checkpoint_artifact }}"' in workflow
        assert 'old_dataset_url: ""' in workflow
        assert 'new_dataset_url: ""' in workflow
        assert 'checkpoint_artifact: ""' in workflow
    assert "Anymal-C-Rough-Native-Active15" in a_workflow
    assert "Anymal-C-Rough-Native-Raw15" in a_workflow
    assert d_workflow.count("Anymal-C-Rough-Native-ContactTokens") == 2


def test_dataset_manifests_freeze_eight_suites_per_encoder() -> None:
    for encoder in ("a", "d"):
        manifest = json.loads((EVAL_DIR / f"dataset_manifest_{encoder}.json").read_text())
        assert manifest["encoder"] == encoder
        assert len(manifest["suites"]) == 8
        assert {(entry["sampling_strategy"], entry["regime"]) for entry in manifest["suites"]} == {
            (sampling, regime)
            for sampling in ("old", "new")
            for regime in (
                "exp_trajectory",
                "zero_action_trajectory",
                "lstm_actuator_zero_action_trajectory",
                "lstm_actuator_policy_trajectory",
            )
        }
        if encoder == "a":
            assert all(entry["sha256"] for entry in manifest["suites"])
        else:
            assert all(entry["sha256"] for entry in manifest["suites"] if entry["sampling_strategy"] == "new")
