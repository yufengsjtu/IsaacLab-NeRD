# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure contract helpers for fixed-checkpoint sampling evaluation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

CHECKPOINT_EPOCH = 199
EXPECTED_SEEDS = (0, 1, 2)
SUPPORTED_ENCODERS = ("a", "d")
SUPPORTED_REPRESENTATIONS = ("active15_tokens", "raw15_tokens", "contact_tokens")


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_indices(length: int, count: int, seed: int) -> np.ndarray:
    """Select a sorted, deterministic subset without replacement."""
    if length <= 0:
        raise ValueError("Dataset length must be positive.")
    if count <= 0:
        raise ValueError("Sample count must be positive.")
    if count >= length:
        return np.arange(length, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(length, size=count, replace=False)).astype(np.int64, copy=False)


def _checkpoint_training_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    algorithm = cfg["algorithm"]
    solver_cfg = copy.deepcopy(cfg["env"]["neural_solver_cfg"])
    solver_cfg.pop("contact_filter", None)
    solver_cfg.pop("exclude_robot_self_collisions", None)
    return {
        "solver": solver_cfg,
        "inputs": cfg["inputs"],
        "network": cfg["network"],
        "num_epochs": int(algorithm["num_epochs"]),
        "num_iters_per_epoch": int(algorithm["num_iters_per_epoch"]),
        "sample_sequence_length": int(algorithm["sample_sequence_length"]),
        "batch_size": int(algorithm["batch_size"]),
        "optimizer": algorithm["optimizer"],
        "max_capacity": int(algorithm["dataset"]["max_capacity"]),
    }


def _validate_encoder_cfg(encoder: str, cfg: dict[str, Any]) -> None:
    solver_cfg = cfg["env"]["neural_solver_cfg"]
    contact_cfg = cfg["inputs"]["contact_set"]
    if encoder == "a":
        expected = {
            "contact_representation": "active15_tokens",
            "encoder_type": "body_routed_active15",
            "dim": 17,
            "body_latent_dim": 64,
            "hidden_dim": 32,
        }
        actual = {
            "contact_representation": solver_cfg["contact_representation"],
            "encoder_type": contact_cfg["encoder_type"],
            "dim": int(contact_cfg["dim"]),
            "body_latent_dim": int(contact_cfg["body_latent_dim"]),
            "hidden_dim": int(contact_cfg["hidden_dim"]),
        }
    elif encoder == "d":
        expected = {
            "contact_representation": "contact_tokens",
            "dim": 17,
            "encoder_layers": 2,
            "encoder_heads": 4,
            "num_latent_queries": 8,
            "hidden_size": 384,
        }
        actual = {
            "contact_representation": solver_cfg["contact_representation"],
            "dim": int(contact_cfg["dim"]),
            "encoder_layers": int(contact_cfg["encoder_layers"]),
            "encoder_heads": int(contact_cfg["encoder_heads"]),
            "num_latent_queries": int(contact_cfg["num_latent_queries"]),
            "hidden_size": int(contact_cfg["hidden_size"]),
        }
    else:
        raise ValueError(f"Unsupported encoder: {encoder!r}.")
    if actual != expected:
        raise ValueError(f"Checkpoint does not match encoder {encoder!r}: expected {expected}, got {actual}.")


def _state_signature(checkpoint: dict[str, Any]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple((name, tuple(value.shape)) for name, value in checkpoint["model_state_dict"].items())


def load_verified_checkpoints(
    manifest_path: str | Path,
    encoder: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Load the exact old/new three-seed E199 checkpoint set."""
    if encoder not in SUPPORTED_ENCODERS:
        raise ValueError(f"Unsupported encoder: {encoder!r}.")
    path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("encoder") != encoder:
        raise ValueError("Checkpoint manifest schema or encoder does not match the request.")
    entries = list(manifest.get("checkpoints", []))
    expected_pairs = {(group, seed) for group in ("old", "new") for seed in EXPECTED_SEEDS}
    actual_pairs = {(str(entry.get("group")), int(entry.get("seed", -1))) for entry in entries}
    if actual_pairs != expected_pairs or len(entries) != len(expected_pairs):
        raise ValueError(f"Checkpoint manifest pairs are {sorted(actual_pairs)}, expected {sorted(expected_pairs)}.")

    loaded = []
    reference_training_contract = None
    reference_signature = None
    for entry in sorted(entries, key=lambda item: (str(item["group"]), int(item["seed"]))):
        checkpoint_path = (path.parent / entry["path"]).resolve()
        if checkpoint_path.stat().st_size != int(entry["size_bytes"]):
            raise ValueError(f"Checkpoint size mismatch: {checkpoint_path}.")
        actual_sha256 = sha256_file(checkpoint_path)
        if actual_sha256 != entry["sha256"]:
            raise ValueError(f"Checkpoint SHA-256 mismatch: {checkpoint_path}.")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("version") != 2 or checkpoint.get("epoch") != CHECKPOINT_EPOCH:
            raise ValueError(f"Expected a v2 Epoch{CHECKPOINT_EPOCH} checkpoint: {checkpoint_path}.")
        if "model_state_dict" not in checkpoint or "cfg" not in checkpoint:
            raise ValueError(f"Checkpoint is missing reconstruction state: {checkpoint_path}.")
        _validate_encoder_cfg(encoder, checkpoint["cfg"])
        training_contract = _checkpoint_training_contract(checkpoint["cfg"])
        signature = _state_signature(checkpoint)
        if reference_training_contract is None:
            reference_training_contract = training_contract
            reference_signature = signature
        elif training_contract != reference_training_contract:
            raise ValueError("Old/new checkpoints do not share the same model and training contract.")
        elif signature != reference_signature:
            raise ValueError("Old/new checkpoints do not share the same state-dict structure.")
        resolved_entry = dict(entry)
        resolved_entry["resolved_path"] = str(checkpoint_path)
        loaded.append((resolved_entry, checkpoint))
    return loaded


def load_verified_suite(
    manifest_path: str | Path,
    suite_id: str,
    dataset_path: str | Path,
) -> dict[str, Any]:
    """Validate one immutable validation-suite file against its manifest."""
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_file.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported dataset manifest schema.")
    matches = [entry for entry in manifest.get("suites", []) if entry.get("suite_id") == suite_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one dataset-manifest entry for {suite_id!r}.")
    entry = dict(matches[0])
    if entry.get("contact_representation") not in SUPPORTED_REPRESENTATIONS:
        raise ValueError(f"Unsupported suite contact representation: {entry.get('contact_representation')!r}.")
    path = Path(dataset_path).expanduser().resolve()
    if path.name != entry["filename"]:
        raise ValueError(f"Dataset filename mismatch for suite {suite_id!r}.")
    expected_size = entry.get("size_bytes")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise ValueError(f"Dataset size mismatch for suite {suite_id!r}.")
    actual_sha256 = sha256_file(path)
    expected_sha256 = entry.get("sha256")
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(f"Dataset SHA-256 mismatch for suite {suite_id!r}.")
    entry["resolved_path"] = str(path)
    entry["size_bytes"] = path.stat().st_size
    entry["sha256"] = actual_sha256
    return entry


def prepare_eval_cfg(
    checkpoint_cfg: dict[str, Any],
    *,
    encoder: str,
    dataset_path: str | Path,
    dataset_representation: str,
    batch_size: int,
    num_workers: int,
    num_envs: int,
    seed: int,
) -> dict[str, Any]:
    """Adapt a checkpoint config to one read-only validation suite."""
    if encoder == "a" and dataset_representation not in ("active15_tokens", "raw15_tokens"):
        raise ValueError("Encoder A requires Active15 or Raw15 validation data.")
    if encoder == "d" and dataset_representation != "contact_tokens":
        raise ValueError("Encoder D requires ContactTokens validation data.")
    cfg = copy.deepcopy(checkpoint_cfg)
    cfg["env"]["num_envs"] = num_envs
    solver_cfg = cfg["env"]["neural_solver_cfg"]
    if encoder == "a":
        solver_cfg["contact_representation"] = "active15_tokens"
        if dataset_representation == "raw15_tokens":
            solver_cfg["contact_filter"] = "solver_active"
        else:
            solver_cfg.pop("contact_filter", None)
    else:
        solver_cfg["contact_representation"] = "contact_tokens"
        solver_cfg.pop("contact_filter", None)

    algorithm = cfg["algorithm"]
    algorithm["seed"] = seed
    algorithm["batch_size"] = batch_size
    algorithm["num_valid_batches"] = 0
    algorithm["update_dataset_statistics"] = False
    dataset_cfg = algorithm["dataset"]
    dataset_cfg["contact_representation"] = dataset_representation
    dataset_cfg["train_dataset_path"] = str(Path(dataset_path).resolve())
    dataset_cfg["valid_datasets"] = {}
    dataset_cfg["max_capacity"] = 1_000_000_000
    dataset_cfg["load_mode"] = "lazy"
    dataset_cfg["num_data_workers"] = num_workers
    dataset_cfg["pin_memory"] = True
    dataset_cfg["non_blocking"] = True
    dataset_cfg["persistent_workers"] = num_workers > 0
    cfg["cli"] = {
        "train": False,
        "distributed": False,
        "rank": 0,
        "local_rank": 0,
        "world_size": 1,
        "render": False,
        "eval_interval": 0,
        "enable_wandb": False,
    }
    return cfg
