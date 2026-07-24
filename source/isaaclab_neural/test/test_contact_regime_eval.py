# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from isaaclab_neural.eval.contact_regime_eval import classify_regimes, evaluate_dataset_regimes


def test_classify_regimes_marks_touchdown_and_impact():
    depths = np.array([[[0.1, 0.2], [0.1, -0.01], [-0.02, -0.03], [-0.01, 0.0]]])
    regimes = classify_regimes(depths, impact_frames=2, settled_frames=1)
    assert regimes["swing"][0, 0]
    assert regimes["touchdown"][0, 1]
    assert regimes["impact"][0, 2]
    assert regimes["settled"][0, -1]


def test_evaluate_dataset_regimes_on_synthetic_arrays(tmp_path):
    path = tmp_path / "regime.hdf5"
    import h5py

    states = np.random.randn(2, 4, 8).astype(np.float32)
    next_states = states + 0.01
    depths = np.array(
        [
            [[0.2, 0.3], [0.1, 0.0], [-0.01, 0.0], [-0.02, 0.0]],
            [[0.2, 0.3], [0.1, 0.0], [-0.01, 0.0], [-0.02, 0.0]],
        ],
        dtype=np.float32,
    )
    with h5py.File(path, "w") as handle:
        group = handle.create_group("data")
        group.create_dataset("states", data=states)
        group.create_dataset("next_states", data=next_states)
        group.create_dataset("joint_f", data=np.zeros((2, 4, 3), dtype=np.float32))
        group.create_dataset("contact_depths", data=depths)
        group.create_dataset("contact_masks", data=depths <= 0.0)
        group.attrs["mode"] = "trajectory"

    metrics = evaluate_dataset_regimes(path)
    assert np.isfinite(metrics["swing_rmse"])
    assert metrics["finite_fraction"] == 1.0
