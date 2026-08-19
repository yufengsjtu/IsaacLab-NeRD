# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regime-split offline evaluation for NeRD contact datasets and checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from isaaclab_neural.eval.contact_distribution_stats import analyze_contact_distribution


def _touchdown_mask(contact_depths: np.ndarray) -> np.ndarray:
    """Return per-trajectory first touchdown step from signed contact depths.

    Args:
        contact_depths: Signed depths with shape ``(num_traj, num_steps, num_contacts)``.
    """
    # Collapse contact slots first so we detect any foot/body contact per step.
    touching = (contact_depths <= 0.0).any(axis=-1)
    has_touch = touching.any(axis=-1)
    first_touch = touching.argmax(axis=-1)
    return np.where(has_touch, first_touch, contact_depths.shape[1])


def classify_regimes(
    contact_depths: np.ndarray, *, impact_frames: int = 15, settled_frames: int = 15
) -> dict[str, np.ndarray]:
    """Classify trajectory steps into swing, touchdown, impact, and settled regimes."""
    num_traj, steps = contact_depths.shape[:2]
    touchdown = _touchdown_mask(contact_depths)
    step_ids = np.tile(np.arange(steps), (num_traj, 1))
    swing = step_ids < touchdown[:, None]
    impact_end = np.minimum(touchdown + impact_frames, steps)
    settled_start = max(steps - settled_frames, 0)
    impact = (step_ids >= touchdown[:, None]) & (step_ids < impact_end[:, None])
    settled = step_ids >= settled_start
    return {
        "swing": swing,
        "touchdown": step_ids == touchdown[:, None],
        "impact": impact,
        "settled": settled,
    }


def _state_rmse(states: np.ndarray, next_states: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    delta = next_states - states
    return float(np.sqrt(np.mean(delta[mask] ** 2)))


def evaluate_dataset_regimes(dataset_path: str | Path) -> dict[str, float]:
    """Compute regime-split one-step state RMSE directly from an HDF5 dataset."""
    dataset_path = Path(dataset_path).expanduser()
    with h5py.File(dataset_path, "r") as handle:
        data = handle["data"]
        states = np.asarray(data["states"])
        next_states = np.asarray(data["next_states"])
        if "contact_depths" in data:
            contact_depths = np.asarray(data["contact_depths"])
        else:
            from isaaclab_neural.contacts.contact_set_schema import (
                ACTIVE15_GAP_INDEX,
                CONTACT_TOKEN_GAP_INDEX,
                CONTACT_TOKEN_VALID_INDEX,
                is_native15_contact_representation,
            )

            tokens = np.asarray(data["contact_tokens"])
            # Padding rows have gap=0; only valid tokens may count as contact.
            valid = tokens[..., CONTACT_TOKEN_VALID_INDEX] > 0.5
            representation = str(data.attrs.get("contact_representation", ""))
            gap_index = (
                ACTIVE15_GAP_INDEX if is_native15_contact_representation(representation) else CONTACT_TOKEN_GAP_INDEX
            )
            gaps = tokens[..., gap_index]
            contact_depths = np.where(valid, gaps, np.inf)

    regimes = classify_regimes(contact_depths)
    metrics = {}
    for name, mask in regimes.items():
        metrics[f"{name}_rmse"] = _state_rmse(states, next_states, mask)
    metrics["finite_fraction"] = float(np.isfinite(states).all(axis=-1).mean())
    metrics.update(
        {k: v for k, v in analyze_contact_distribution(dataset_path).items() if "overflow" in k or "capacity" in k}
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="NeRD HDF5 dataset path.")
    parser.add_argument(
        "--overflow-gate",
        action="store_true",
        help="Exit with code 1 when token overflow or capacity truncation is detected.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_dataset_regimes(args.dataset)
    print(f"Regime metrics for {args.dataset}")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    if args.overflow_gate:
        overflow_frames = int(metrics.get("token_overflow_frames", 0)) + int(metrics.get("capacity_overflow_frames", 0))
        if overflow_frames > 0:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
