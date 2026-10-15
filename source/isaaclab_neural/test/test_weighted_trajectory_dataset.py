# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for weighted mixed trajectory datasets."""

from __future__ import annotations

import pytest
import torch
from isaaclab_neural.data.datasets import WeightedMixedTrajectoryDataset


class _FakeTrajectoryDataset:
    def __init__(self, values: list[int]):
        self._values = values

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"value": torch.tensor(self._values[index])}


def test_weighted_mix_allocates_by_normalized_weights_and_cycles_short_sources():
    mixed = WeightedMixedTrajectoryDataset(
        {
            "policy": _FakeTrajectoryDataset([10, 11]),
            "zero": _FakeTrajectoryDataset([20, 21]),
        },
        {"policy": 3.0, "zero": 1.0},
    )

    assert len(mixed) == 4
    assert mixed.weights.tolist() == pytest.approx([0.75, 0.25])
    assert mixed.source_counts.tolist() == [3, 1]
    assert mixed.source_for_index(0) == ("policy", 0)
    assert mixed.source_for_index(1) == ("policy", 1)
    assert mixed.source_for_index(2) == ("policy", 0)
    assert mixed.source_for_index(3) == ("zero", 0)
    assert int(mixed[2]["value"]) == 10
    assert int(mixed[3]["value"]) == 20


def test_weighted_mix_rejects_empty_or_mismatched_sources():
    with pytest.raises(ValueError, match="at least one dataset"):
        WeightedMixedTrajectoryDataset({}, {})
    with pytest.raises(ValueError, match="names must match"):
        WeightedMixedTrajectoryDataset({"a": _FakeTrajectoryDataset([1])}, {"b": 1.0})
    with pytest.raises(ValueError, match="non-empty"):
        WeightedMixedTrajectoryDataset({"a": _FakeTrajectoryDataset([])}, {"a": 1.0})


def test_weighted_mix_exposes_a_shared_body_force_order_only():
    def source(names):
        dataset = _FakeTrajectoryDataset([1])
        dataset.body_net_contact_force_body_names = names
        return dataset

    order = ("base", "LF_FOOT")
    shared = WeightedMixedTrajectoryDataset({"a": source(order), "b": source(order)}, {"a": 1.0, "b": 1.0})
    assert shared.body_net_contact_force_body_names == order
    # Differing orders are reported as missing so trainer validation rejects the mixture.
    mixed = WeightedMixedTrajectoryDataset({"a": source(order), "b": source(order[::-1])}, {"a": 1.0, "b": 1.0})
    assert mixed.body_net_contact_force_body_names == ()
