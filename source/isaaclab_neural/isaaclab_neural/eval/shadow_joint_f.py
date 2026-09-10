# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for probing implicit-actuator shadow PD as a NeRD ``joint_f`` channel.

Isaac Lab's implicit actuators apply PD inside the solver and leave Newton
``control.joint_f`` empty. They still compute clipped PD telemetry
(``applied_effort`` / ``applied_torque``) with the IdealPD formula. These
helpers reconstruct that telemetry, build an inertial-plus-gravity inverse-
dynamics residual, and turn stacked rollout tensors into pass/fail gates.
"""

from __future__ import annotations

from typing import Any

import torch

IDENTITY_MAX_ABS_NM = 1.0e-3
"""Maximum allowed |applied − reconstructed PD| [N·m]."""

RECORDED_JOINT_F_MAX_ABS_NM = 1.0e-4
"""Maximum allowed |control.joint_f| [N·m] on the implicit recording path."""

ARM_SHADOW_RMS_MIN_NM = 0.5
"""Minimum arm-joint RMS of shadow PD [N·m] for a usable control feature."""

TIMING_ERROR_RATIO_MIN = 5.0
"""Post-step PD reconstruction error must exceed the pre-step error by this factor."""

WRITEBACK_ABS_QD_RATIO_MIN = 1.25
"""Injecting shadow PD into ``control.joint_f`` must raise mean |qd| by this factor."""


def reconstruct_shadow_pd(
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    joint_pos_target: torch.Tensor,
    joint_vel_target: torch.Tensor,
    stiffness: torch.Tensor,
    damping: torch.Tensor,
    effort_limit: torch.Tensor,
) -> torch.Tensor:
    """Return clipped PD effort matching Isaac Lab implicit telemetry.

    Args:
        joint_pos: Joint positions [m or rad, depending on joint type].
        joint_vel: Joint velocities [m/s or rad/s, depending on joint type].
        joint_pos_target: Position targets [m or rad, depending on joint type].
        joint_vel_target: Velocity targets [m/s or rad/s, depending on joint type].
        stiffness: Joint stiffness [N/m or N·m/rad, depending on joint type].
        damping: Joint damping [N·s/m or N·m·s/rad, depending on joint type].
        effort_limit: Joint effort limits [N or N·m, depending on joint type].

    Returns:
        Clipped PD effort [N or N·m, depending on joint type], same shape as
        ``joint_pos``.
    """
    effort = stiffness * (joint_pos_target - joint_pos) + damping * (joint_vel_target - joint_vel)
    return torch.clamp(effort, min=-effort_limit, max=effort_limit)


def inverse_dynamics_inertial_gravity(
    mass_matrix: torch.Tensor,
    joint_acc: torch.Tensor,
    gravity_compensation: torch.Tensor,
) -> torch.Tensor:
    """Return ``M(q) qdd + g(q)`` [N or N·m, depending on joint type].

    This omits Coriolis/centrifugal terms. Use it as an approximate residual
    against shadow PD, not as a claim of exact inverse dynamics.

    Args:
        mass_matrix: Generalized mass matrix [kg or kg·m²], shape ``(..., J, J)``.
        joint_acc: Generalized acceleration [m/s² or rad/s²], shape ``(..., J)``.
        gravity_compensation: Gravity load ``g(q)`` [N or N·m], shape ``(..., J)``.

    Returns:
        Inertial plus gravity generalized force, shape ``(..., J)``.
    """
    inertial = torch.matmul(mass_matrix, joint_acc.unsqueeze(-1)).squeeze(-1)
    return inertial + gravity_compensation


def pearson_corrcoef(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Return Pearson correlation of two 1-D tensors, or 0 when either is constant."""
    lhs = lhs.reshape(-1).float()
    rhs = rhs.reshape(-1).float()
    if lhs.numel() < 2:
        return torch.zeros((), dtype=torch.float32, device=lhs.device)
    lhs = lhs - lhs.mean()
    rhs = rhs - rhs.mean()
    lhs_std = torch.linalg.vector_norm(lhs)
    rhs_std = torch.linalg.vector_norm(rhs)
    if float(lhs_std) < 1.0e-12 or float(rhs_std) < 1.0e-12:
        return torch.zeros((), dtype=torch.float32, device=lhs.device)
    return torch.dot(lhs, rhs) / (lhs_std * rhs_std)


def _finite_mask(tensor: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(tensor).all(dim=-1)


def _masked_rms(tensor: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return float("nan")
    values = tensor[mask]
    return float(torch.sqrt(torch.mean(values * values)))


def _masked_max_abs(tensor: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return float("nan")
    return float(tensor[mask].abs().max())


def _masked_mean_abs(tensor: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return float("nan")
    return float(tensor[mask].abs().mean())


def _clip_fraction(computed: torch.Tensor, applied: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return float("nan")
    clipped = (computed[mask] - applied[mask]).abs() > 1.0e-4
    return float(clipped.float().mean())


def summarize_channel(
    *,
    applied: torch.Tensor,
    reconstructed: torch.Tensor,
    reconstructed_post: torch.Tensor,
    recorded_joint_f: torch.Tensor,
    id_force: torch.Tensor,
    joint_mask: torch.Tensor,
) -> dict[str, float]:
    """Summarize one joint subset from stacked ``(T, N, J)`` tensors."""
    sample_mask = _finite_mask(applied) & _finite_mask(reconstructed)
    applied_sel = applied[:, :, joint_mask]
    recon_sel = reconstructed[:, :, joint_mask]
    post_sel = reconstructed_post[:, :, joint_mask]
    recorded_sel = recorded_joint_f[:, :, joint_mask]
    id_sel = id_force[:, :, joint_mask]
    residual = applied_sel - recon_sel
    post_residual = applied_sel - post_sel
    id_residual = applied_sel - id_sel
    return {
        "applied_rms_nm": _masked_rms(applied_sel, sample_mask),
        "applied_mean_abs_nm": _masked_mean_abs(applied_sel, sample_mask),
        "applied_max_abs_nm": _masked_max_abs(applied_sel, sample_mask),
        "identity_max_abs_nm": _masked_max_abs(residual, sample_mask),
        "identity_rms_nm": _masked_rms(residual, sample_mask),
        "identity_post_max_abs_nm": _masked_max_abs(post_residual, sample_mask),
        "recorded_joint_f_max_abs_nm": _masked_max_abs(recorded_sel, sample_mask),
        "id_rmse_nm": _masked_rms(id_residual, sample_mask),
        "id_pearson": float(pearson_corrcoef(applied_sel[sample_mask], id_sel[sample_mask])),
        "nan_frac": float((~sample_mask).float().mean()),
    }


def evaluate_shadow_joint_f_probe(
    *,
    applied: torch.Tensor,
    reconstructed: torch.Tensor,
    reconstructed_post: torch.Tensor,
    recorded_joint_f: torch.Tensor,
    id_force: torch.Tensor,
    computed_unclipped: torch.Tensor | None = None,
    abs_qd_baseline: float | None = None,
    abs_qd_inject: float | None = None,
    inject_nan_frac: float | None = None,
    arm_joint_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Return metrics and pass/fail gates for a shadow-PD probe.

    Args:
        applied: Shadow PD telemetry [N·m], shape ``(T, N, J)``.
        reconstructed: PD reconstructed from pre-step state [N·m].
        reconstructed_post: PD reconstructed from post-step state [N·m].
        recorded_joint_f: Newton ``control.joint_f`` after the step [N·m].
        id_force: ``M qdd + g`` [N·m] from the pre-step configuration.
        computed_unclipped: Unclipped PD before effort limits, optional.
        abs_qd_baseline: Mean |qd| without writing shadow PD into ``joint_f``.
        abs_qd_inject: Mean |qd| when shadow PD is injected as feed-forward.
        inject_nan_frac: Fraction of inject-rollout samples that were non-finite.
        arm_joint_mask: Boolean mask over joints. Defaults to every joint.

    Returns:
        JSON-serializable dict with ``metrics`` and ``verdict``.
    """
    num_joints = int(applied.shape[-1])
    if arm_joint_mask is None:
        arm_joint_mask = torch.ones(num_joints, dtype=torch.bool, device=applied.device)
    arm_joint_mask = arm_joint_mask.to(device=applied.device, dtype=torch.bool)
    finger_joint_mask = ~arm_joint_mask
    sample_mask = _finite_mask(applied)
    metrics: dict[str, Any] = {
        "all": summarize_channel(
            applied=applied,
            reconstructed=reconstructed,
            reconstructed_post=reconstructed_post,
            recorded_joint_f=recorded_joint_f,
            id_force=id_force,
            joint_mask=torch.ones(num_joints, dtype=torch.bool, device=applied.device),
        ),
        "arm": summarize_channel(
            applied=applied,
            reconstructed=reconstructed,
            reconstructed_post=reconstructed_post,
            recorded_joint_f=recorded_joint_f,
            id_force=id_force,
            joint_mask=arm_joint_mask,
        ),
    }
    if bool(finger_joint_mask.any()):
        metrics["finger"] = summarize_channel(
            applied=applied,
            reconstructed=reconstructed,
            reconstructed_post=reconstructed_post,
            recorded_joint_f=recorded_joint_f,
            id_force=id_force,
            joint_mask=finger_joint_mask,
        )
    if computed_unclipped is not None:
        metrics["arm"]["clip_frac"] = _clip_fraction(
            computed_unclipped[:, :, arm_joint_mask], applied[:, :, arm_joint_mask], sample_mask
        )
        if bool(finger_joint_mask.any()):
            metrics["finger"]["clip_frac"] = _clip_fraction(
                computed_unclipped[:, :, finger_joint_mask], applied[:, :, finger_joint_mask], sample_mask
            )

    identity_ok = metrics["all"]["identity_max_abs_nm"] <= IDENTITY_MAX_ABS_NM
    recorded_empty_ok = metrics["all"]["recorded_joint_f_max_abs_nm"] <= RECORDED_JOINT_F_MAX_ABS_NM
    signal_ok = metrics["arm"]["applied_rms_nm"] >= ARM_SHADOW_RMS_MIN_NM
    pre_err = max(metrics["all"]["identity_max_abs_nm"], 1.0e-12)
    post_err = metrics["all"]["identity_post_max_abs_nm"]
    timing_ok = post_err >= TIMING_ERROR_RATIO_MIN * pre_err or (identity_ok and post_err >= 0.05)
    writeback: dict[str, float | bool | None] = {
        "abs_qd_baseline": abs_qd_baseline,
        "abs_qd_inject": abs_qd_inject,
        "inject_nan_frac": inject_nan_frac,
        "checked": abs_qd_inject is not None,
    }
    writeback_ok = True
    if abs_qd_inject is not None:
        inject_is_nan = inject_nan_frac is not None and inject_nan_frac > 0.0
        qd_ratio = float("nan")
        if abs_qd_baseline is not None and abs_qd_baseline > 1.0e-8:
            qd_ratio = abs_qd_inject / abs_qd_baseline
        writeback["abs_qd_ratio"] = qd_ratio
        writeback_ok = inject_is_nan or (
            abs_qd_baseline is not None and abs_qd_inject >= WRITEBACK_ABS_QD_RATIO_MIN * abs_qd_baseline
        )
    writeback["ok"] = writeback_ok
    metrics["writeback"] = writeback

    verdict = {
        "identity_ok": bool(identity_ok),
        "recorded_joint_f_empty_ok": bool(recorded_empty_ok),
        "arm_shadow_signal_ok": bool(signal_ok),
        "pre_step_timing_ok": bool(timing_ok),
        "writeback_dangerous_ok": bool(writeback_ok),
        "id_is_approximate": True,
    }
    required = [
        "identity_ok",
        "recorded_joint_f_empty_ok",
        "arm_shadow_signal_ok",
        "pre_step_timing_ok",
    ]
    if metrics["writeback"]["checked"]:
        required.append("writeback_dangerous_ok")
    verdict["passed"] = all(bool(verdict[key]) for key in required)
    return {"metrics": metrics, "verdict": verdict}
