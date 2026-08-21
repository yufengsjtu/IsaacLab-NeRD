# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Anymal contact-force validation play helpers."""

import argparse
import math

import pytest
import torch
from isaaclab_neural.eval.play_anymal_rough_contact_forces import (
    requested_newton_visualizer,
    rotate_body_forces_to_world,
    save_contact_force_plot,
    save_foot_contact_force_plots,
)


def test_requested_newton_visualizer_handles_single_and_combined_backends():
    assert requested_newton_visualizer(argparse.Namespace(visualizer=["newton"]))
    assert requested_newton_visualizer(argparse.Namespace(visualizer="kit,newton"))
    assert not requested_newton_visualizer(argparse.Namespace(visualizer=["kit"]))
    assert not requested_newton_visualizer(argparse.Namespace(visualizer=None))


def test_rotate_body_forces_to_world_uses_root_orientation():
    half_angle = math.pi / 4.0
    root_body_q = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]])
    forces_b = torch.tensor([[[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]])

    forces_w = rotate_body_forces_to_world(forces_b, root_body_q)

    expected = torch.tensor([[[0.0, 2.0, 0.0], [-3.0, 0.0, 0.0]]])
    torch.testing.assert_close(forces_w, expected, atol=1.0e-6, rtol=0.0)


def test_rotate_body_forces_to_world_rejects_mismatched_root_shape():
    with pytest.raises(ValueError, match="root_body_q"):
        rotate_body_forces_to_world(torch.zeros(2, 17, 3), torch.zeros(1, 7))


def test_save_contact_force_plot_writes_all_body_curves(tmp_path):
    pytest.importorskip("matplotlib")
    body_names = [f"body_{index:02d}" for index in range(17)]
    sensor_history = torch.arange(5 * 17, dtype=torch.float32).view(5, 17)
    predicted_history = sensor_history + 1.0
    output_path = tmp_path / "contact_sensor_forces.png"

    result = save_contact_force_plot(sensor_history, predicted_history, body_names, 0.01, output_path)

    assert result == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_save_contact_force_plot_rejects_mismatched_histories(tmp_path):
    with pytest.raises(ValueError, match="same shape"):
        save_contact_force_plot(
            torch.zeros(5, 17),
            torch.zeros(4, 17),
            [f"body_{index:02d}" for index in range(17)],
            0.01,
            tmp_path / "forces.png",
        )


def test_save_foot_contact_force_plots_writes_four_files(tmp_path):
    pytest.importorskip("matplotlib")
    body_names = [
        "base",
        "LF_HIP",
        "LF_THIGH",
        "LF_SHANK",
        "LF_FOOT",
        "LH_HIP",
        "LH_THIGH",
        "LH_SHANK",
        "LH_FOOT",
        "RF_HIP",
        "RF_THIGH",
        "RF_SHANK",
        "RF_FOOT",
        "RH_HIP",
        "RH_THIGH",
        "RH_SHANK",
        "RH_FOOT",
    ]
    sensor_history = torch.arange(5 * 17, dtype=torch.float32).view(5, 17)
    predicted_history = sensor_history + 1.0

    paths = save_foot_contact_force_plots(
        sensor_history,
        predicted_history,
        body_names,
        0.01,
        tmp_path / "contact_sensor_forces.png",
    )

    assert [path.name for path in paths] == [
        "contact_sensor_forces_04_LF_FOOT.png",
        "contact_sensor_forces_08_LH_FOOT.png",
        "contact_sensor_forces_12_RF_FOOT.png",
        "contact_sensor_forces_16_RH_FOOT.png",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
