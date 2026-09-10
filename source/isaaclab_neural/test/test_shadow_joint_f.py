# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for implicit shadow-PD probe helpers."""

from __future__ import annotations

import torch
from isaaclab_neural.eval.shadow_joint_f import (
    evaluate_shadow_joint_f_probe,
    inverse_dynamics_inertial_gravity,
    pearson_corrcoef,
    reconstruct_shadow_pd,
)


def test_reconstruct_shadow_pd_matches_clipped_ideal_pd() -> None:
    joint_pos = torch.tensor([[0.1, -0.2]])
    joint_vel = torch.tensor([[0.3, -0.4]])
    joint_pos_target = torch.tensor([[0.5, 0.0]])
    joint_vel_target = torch.zeros(1, 2)
    stiffness = torch.tensor([[80.0, 80.0]])
    damping = torch.tensor([[4.0, 4.0]])
    effort_limit = torch.tensor([[87.0, 12.0]])

    effort = reconstruct_shadow_pd(
        joint_pos, joint_vel, joint_pos_target, joint_vel_target, stiffness, damping, effort_limit
    )

    expected = torch.tensor([[80.0 * 0.4 + 4.0 * (-0.3), 12.0]])
    torch.testing.assert_close(effort, expected)


def test_inverse_dynamics_inertial_gravity_is_mass_times_acc_plus_g() -> None:
    mass = torch.tensor([[[2.0, 0.0], [0.0, 3.0]]])
    acc = torch.tensor([[1.0, -2.0]])
    gravity = torch.tensor([[0.5, 0.25]])

    force = inverse_dynamics_inertial_gravity(mass, acc, gravity)

    torch.testing.assert_close(force, torch.tensor([[2.5, -5.75]]))


def test_pearson_corrcoef_is_one_for_scaled_shift() -> None:
    lhs = torch.tensor([0.0, 1.0, 2.0, 3.0])
    rhs = 2.0 * lhs + 5.0
    torch.testing.assert_close(pearson_corrcoef(lhs, rhs), torch.tensor(1.0))


def test_evaluate_shadow_joint_f_probe_passes_on_consistent_pre_step_pd() -> None:
    applied = torch.tensor([8.0, -3.0, 0.2]).view(1, 1, 3).expand(6, 2, 3).clone()
    applied = applied + 0.05 * torch.arange(6, dtype=applied.dtype).view(6, 1, 1)
    reconstructed = applied.clone()
    reconstructed_post = applied + 0.4
    recorded = torch.zeros_like(applied)
    id_force = 0.85 * applied
    arm_mask = torch.tensor([True, True, False])

    report = evaluate_shadow_joint_f_probe(
        applied=applied,
        reconstructed=reconstructed,
        reconstructed_post=reconstructed_post,
        recorded_joint_f=recorded,
        id_force=id_force,
        abs_qd_baseline=0.4,
        abs_qd_inject=1.1,
        inject_nan_frac=0.0,
        arm_joint_mask=arm_mask,
    )

    assert report["verdict"]["identity_ok"]
    assert report["verdict"]["recorded_joint_f_empty_ok"]
    assert report["verdict"]["arm_shadow_signal_ok"]
    assert report["verdict"]["pre_step_timing_ok"]
    assert report["verdict"]["writeback_dangerous_ok"]
    assert report["verdict"]["passed"]


def test_evaluate_shadow_joint_f_probe_fails_when_hdf5_channel_is_already_filled() -> None:
    applied = torch.ones(4, 1, 2)
    report = evaluate_shadow_joint_f_probe(
        applied=applied,
        reconstructed=applied,
        reconstructed_post=applied + 1.0,
        recorded_joint_f=applied,
        id_force=applied,
        arm_joint_mask=torch.tensor([True, True]),
    )

    assert not report["verdict"]["recorded_joint_f_empty_ok"]
    assert not report["verdict"]["passed"]
