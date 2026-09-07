# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load trajectory initial-state pools for dataset generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_initial_states_pool(path: str | Path, device: str | torch.device) -> torch.Tensor:
    """Load a ``[num_states, state_dim]`` pool from HDF5 or a torch file.

    HDF5 files must contain ``data/states`` with shape
    ``[num_trajectories, horizon, state_dim]``. Torch tensors may be either that
    rank-3 layout or an already flattened rank-2 pool.

    Args:
        path: HDF5 dataset or serialized torch tensor.
        device: Device that receives the returned pool.

    Returns:
        Flattened generalized-state pool, shape ``[num_states, state_dim]``.
    """
    pool_path = Path(path)
    if not pool_path.is_file():
        raise FileNotFoundError(pool_path)

    if pool_path.suffix.lower() in {".hdf5", ".h5"}:
        import h5py

        with h5py.File(pool_path, "r") as handle:
            if "data" not in handle or "states" not in handle["data"]:
                raise KeyError(f"{pool_path} does not contain data/states.")
            states = np.asarray(handle["data"]["states"])
        if states.ndim != 3:
            raise ValueError(
                "HDF5 initial-state pools must have shape "
                f"[num_trajectories, horizon, state_dim], got {states.shape}."
            )
        return torch.as_tensor(states, dtype=torch.float32, device=device).reshape(-1, states.shape[-1])

    loaded = torch.load(pool_path, map_location=device)
    if not isinstance(loaded, torch.Tensor):
        raise TypeError(f"Torch initial-state pool must be a tensor, got {type(loaded).__name__}.")
    if loaded.ndim == 3:
        loaded = loaded.reshape(-1, loaded.shape[-1])
    if loaded.ndim != 2:
        raise ValueError(f"initial_states_pool must be rank 2 or 3, got shape {tuple(loaded.shape)}.")
    return loaded.to(device=device, dtype=torch.float32)
