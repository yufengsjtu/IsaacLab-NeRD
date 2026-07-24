# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain provenance helpers for reproducible NeRD datasets."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

TERRAIN_CONTEXT_SCHEMA_VERSION = 1


def _canonicalize(value: Any) -> Any:
    """Convert a config value to a stable JSON-compatible representation."""
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _hash_config(config: Any) -> str:
    payload = json.dumps(_canonicalize(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def terrain_generator_cfg(env_cfg: Any) -> Any | None:
    """Return the terrain generator config from an IsaacLab environment config."""
    scene_cfg = getattr(env_cfg, "scene", None)
    terrain_cfg = getattr(scene_cfg, "terrain", None)
    return getattr(terrain_cfg, "terrain_generator", None)


def set_terrain_seed(env_cfg: Any, seed: int) -> bool:
    """Set the environment and terrain-generator seeds when available."""
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = seed
    generator_cfg = terrain_generator_cfg(env_cfg)
    if generator_cfg is None:
        return False
    generator_cfg.seed = seed
    return True


def build_terrain_context(env_cfg: Any, seed: int) -> dict[str, Any] | None:
    """Build deterministic terrain provenance without advancing caller RNG state."""
    generator_cfg = terrain_generator_cfg(env_cfg)
    if generator_cfg is None:
        return None

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        cfg = copy.deepcopy(generator_cfg)
        cfg.seed = seed
        generator = cfg.class_type(cfg=cfg, device="cpu")
        mesh = generator.terrain_mesh
        origins = np.asarray(generator.terrain_origins)
        return {
            "schema_version": TERRAIN_CONTEXT_SCHEMA_VERSION,
            "seed": int(seed),
            "config_sha256": _hash_config(cfg.to_dict()),
            "mesh_sha256": _hash_arrays(np.asarray(mesh.vertices), np.asarray(mesh.faces)),
            "origins_sha256": _hash_arrays(origins),
            "num_rows": int(cfg.num_rows),
            "num_cols": int(cfg.num_cols),
        }
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def read_terrain_context(dataset_path: str | Path) -> dict[str, Any] | None:
    """Read terrain provenance from a NeRD HDF5 dataset."""
    with h5py.File(Path(dataset_path).expanduser(), "r", swmr=True, libver="latest") as dataset_file:
        if "context" not in dataset_file or "terrain" not in dataset_file["context"]:
            return None
        terrain_group = dataset_file["context"]["terrain"]
        return {
            key: (
                value.decode("utf-8")
                if isinstance(value, bytes)
                else value.item()
                if isinstance(value, np.generic)
                else value
            )
            for key, value in terrain_group.attrs.items()
        }


def validate_terrain_context(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    """Raise when reconstructed terrain provenance differs from a dataset."""
    keys = ("schema_version", "seed", "config_sha256", "mesh_sha256", "origins_sha256", "num_rows", "num_cols")
    mismatches = {key: (expected.get(key), actual.get(key)) for key in keys if expected.get(key) != actual.get(key)}
    if mismatches:
        details = ", ".join(f"{key}: dataset={old!r}, reconstructed={new!r}" for key, (old, new) in mismatches.items())
        raise ValueError(
            f"Terrain context mismatch: {details}. Regenerate the evaluation dataset for this task config."
        )
