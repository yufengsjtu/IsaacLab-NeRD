# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for dataset-generation initial-state pools."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from isaaclab_neural.generate.initial_states import load_initial_states_pool


def test_load_initial_states_pool_flattens_hdf5_trajectories(tmp_path: Path):
    path = tmp_path / "pool.hdf5"
    states = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    with h5py.File(path, "w") as handle:
        handle.create_group("data").create_dataset("states", data=states)

    pool = load_initial_states_pool(path, device="cpu")

    assert tuple(pool.shape) == (6, 4)
    assert torch.equal(pool[0], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.equal(pool[-1], torch.tensor([20.0, 21.0, 22.0, 23.0]))


def test_load_initial_states_pool_accepts_torch_rank3(tmp_path: Path):
    path = tmp_path / "pool.pt"
    torch.save(torch.arange(12, dtype=torch.float32).reshape(2, 2, 3), path)

    pool = load_initial_states_pool(path, device="cpu")

    assert tuple(pool.shape) == (4, 3)


def test_load_initial_states_pool_rejects_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_initial_states_pool(tmp_path / "missing.hdf5", device="cpu")
