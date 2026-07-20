# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for contact reconstruction diagnostic edge cases."""

import math

import torch
from isaaclab_neural.eval.contact_reconstruction_diagnostic import contact_metrics, input_difference_metrics


def _inputs(mask: torch.Tensor) -> dict[str, torch.Tensor]:
    batch_size, num_contacts = mask.shape
    return {
        "states": torch.zeros(batch_size, 1, 4),
        "root_body_q": torch.zeros(batch_size, 1, 7),
        "joint_f": torch.zeros(batch_size, 1, 2),
        "gravity_dir": torch.zeros(batch_size, 1, 3),
        "contact_masks": mask.unsqueeze(1),
        "contact_points_0": torch.zeros(batch_size, 1, num_contacts * 3),
        "contact_points_1": torch.zeros(batch_size, 1, num_contacts * 3),
        "contact_normals": torch.zeros(batch_size, 1, num_contacts * 3),
        "contact_depths": torch.zeros(batch_size, 1, num_contacts),
        "contact_thicknesses_0": torch.zeros(batch_size, 1, num_contacts),
        "contact_thicknesses_1": torch.zeros(batch_size, 1, num_contacts),
    }


def test_input_difference_metrics_handles_no_shared_contacts():
    dataset_inputs = _inputs(torch.tensor([[True, False]]))
    runtime_inputs = _inputs(torch.tensor([[False, True]]))

    metrics = input_difference_metrics(dataset_inputs, runtime_inputs, 2)

    assert math.isnan(metrics["contact_points_0_l2_mean_active"])
    assert math.isnan(metrics["contact_points_0_l2_max_active"])
    assert math.isnan(metrics["contact_depths_abs_mean_active"])
    assert math.isnan(metrics["contact_depths_abs_max_active"])


def test_contact_metrics_reports_one_sided_empty_contacts():
    dataset_inputs = _inputs(torch.tensor([[True, False], [True, False]]))
    runtime_inputs = _inputs(torch.tensor([[False, False], [True, False]]))

    metrics = contact_metrics(dataset_inputs, runtime_inputs, 2)

    assert math.isinf(metrics["set_chamfer_distance_mean"])
    assert metrics["one_sided_empty_count"] == 1.0
    assert metrics["one_sided_empty_fraction"] == 0.5
