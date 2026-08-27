# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for verified W&B checkpoint ingress in standalone sampling eval."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch

from osmo_scripts.sampling_strategy_eval.contract import load_verified_checkpoints
from osmo_scripts.sampling_strategy_eval.download_checkpoints import (
    download_checkpoint_artifact_set,
    download_checkpoint_set,
)


def _cfg() -> dict:
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
            "optimizer": {"type": "Adam", "lr_start": "1e-4", "lr_end": "1e-5"},
            "dataset": {"max_capacity": 20_000_000},
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


def test_download_checkpoint_set_rebuilds_and_verifies_manifest(tmp_path: Path) -> None:
    entries = []
    remote_paths = {}
    for group in ("old", "new"):
        for seed in range(3):
            run_id = f"{group}{seed}"
            remote_path = tmp_path / "remote" / run_id / "model_epoch199.pt"
            remote_path.parent.mkdir(parents=True)
            torch.save(
                {
                    "version": 2,
                    "epoch": 199,
                    "cfg": _cfg(),
                    "model_state_dict": {"weight": torch.arange(4).reshape(2, 2)},
                },
                remote_path,
            )
            entries.append(
                {
                    "group": group,
                    "seed": seed,
                    "wandb_run_id": run_id,
                    "path": f"{group}_a/seed{seed}/model_epoch199.pt",
                    "size_bytes": remote_path.stat().st_size,
                    "sha256": hashlib.sha256(remote_path.read_bytes()).hexdigest(),
                }
            )
            remote_paths[f"entity/project/{run_id}"] = remote_path

    class FakeFile:
        def __init__(self, source: Path):
            self.source = source

        def download(self, *, root: str, replace: bool) -> SimpleNamespace:
            assert replace is True
            target = Path(root) / self.source.name
            shutil.copy2(self.source, target)
            return SimpleNamespace(name=str(target))

    class FakeRun:
        def __init__(self, source: Path):
            self.source = source

        def file(self, name: str) -> FakeFile:
            assert name == "model_epoch199.pt"
            return FakeFile(self.source)

    class FakeApi:
        def run(self, path: str) -> FakeRun:
            return FakeRun(remote_paths[path])

    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "encoder": "a",
                "checkpoint_epoch": 199,
                "checkpoints": entries,
            }
        )
    )
    output_manifest = download_checkpoint_set(
        source_manifest,
        tmp_path / "downloaded",
        entity="entity",
        project="project",
        api=FakeApi(),
    )

    assert len(load_verified_checkpoints(output_manifest, "a")) == 6


def test_download_checkpoint_artifact_set_verifies_downloaded_bundle(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifact"
    entries = []
    for group in ("old", "new"):
        for seed in range(3):
            checkpoint_path = artifact_root / f"{group}_a" / f"seed{seed}" / "model_epoch199.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "version": 2,
                    "epoch": 199,
                    "cfg": _cfg(),
                    "model_state_dict": {"weight": torch.arange(4).reshape(2, 2)},
                },
                checkpoint_path,
            )
            entries.append(
                {
                    "group": group,
                    "seed": seed,
                    "wandb_run_id": f"{group}{seed}",
                    "path": str(checkpoint_path.relative_to(artifact_root)),
                    "size_bytes": checkpoint_path.stat().st_size,
                    "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                }
            )
    (artifact_root / "checkpoint_manifest_a.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "encoder": "a",
                "checkpoint_epoch": 199,
                "checkpoints": entries,
            }
        )
    )

    class FakeArtifact:
        def download(self, *, root: str) -> str:
            shutil.copytree(artifact_root, root, dirs_exist_ok=True)
            return root

    class FakeApi:
        def artifact(self, name: str) -> FakeArtifact:
            assert name == "entity/project/checkpoints:v3"
            return FakeArtifact()

    output_manifest = download_checkpoint_artifact_set(
        "entity/project/checkpoints:v3",
        tmp_path / "downloaded",
        encoder="a",
        api=FakeApi(),
    )

    assert len(load_verified_checkpoints(output_manifest, "a")) == 6
