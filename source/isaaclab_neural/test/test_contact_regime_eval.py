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


def test_evaluate_dataset_regimes_ignores_invalid_token_padding(tmp_path):
    """Padding tokens have gap=0; they must not be treated as touchdown."""
    path = tmp_path / "regime_tokens.hdf5"
    import h5py

    from isaaclab_neural.contacts.contact_set_schema import CONTACT_TOKEN_DIM, CONTACT_TOKEN_GAP_INDEX
    from isaaclab_neural.eval.contact_regime_eval import classify_regimes

    states = np.random.randn(1, 4, 8).astype(np.float32)
    next_states = states + 0.01
    tokens = np.zeros((1, 4, 3, CONTACT_TOKEN_DIM), dtype=np.float32)
    # Step 0-1: only padding (valid=0, gap=0). Step 2: first valid contact with gap<=0.
    tokens[0, 2, 0, 0] = 1.0
    tokens[0, 2, 0, CONTACT_TOKEN_GAP_INDEX] = -0.01
    tokens[0, 3, 0, 0] = 1.0
    tokens[0, 3, 0, CONTACT_TOKEN_GAP_INDEX] = -0.02

    with h5py.File(path, "w") as handle:
        group = handle.create_group("data")
        group.create_dataset("states", data=states)
        group.create_dataset("next_states", data=next_states)
        group.create_dataset("joint_f", data=np.zeros((1, 4, 3), dtype=np.float32))
        group.create_dataset("contact_tokens", data=tokens)
        group.create_dataset("contact_token_overflow", data=np.zeros((1, 4), dtype=np.int64))
        group.attrs["mode"] = "trajectory"
        group.attrs["max_contact_tokens"] = 3
        group.attrs["contact_representation"] = "contact_tokens"

    # Invalid padding must not create a false early touchdown at step 0.
    valid = tokens[..., 0] > 0.5
    gaps = np.where(valid, tokens[..., CONTACT_TOKEN_GAP_INDEX], np.inf)
    regimes = classify_regimes(gaps, impact_frames=1, settled_frames=1)
    assert regimes["swing"][0, 0] and regimes["swing"][0, 1]
    assert regimes["touchdown"][0, 2]

    metrics = evaluate_dataset_regimes(path)
    assert np.isfinite(metrics["swing_rmse"])
    assert metrics.get("token_overflow_frames", 0) == 0
    assert metrics.get("capacity_overflow_frames", 0) == 0
