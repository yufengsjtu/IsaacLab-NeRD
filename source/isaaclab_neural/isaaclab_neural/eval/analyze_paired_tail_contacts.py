# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analyze solver-active self-collisions in paired-evaluation error tails."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from isaaclab_neural.contacts.contact_set_schema import (
    CONTACT_TOKEN_BODY_SLOT_INDEX,
    CONTACT_TOKEN_GAP_INDEX,
    CONTACT_TOKEN_OTHER_BODY_SLOT_INDEX,
    CONTACT_TOKEN_OTHER_DYNAMIC_INDEX,
    CONTACT_TOKEN_SOLVER_ACTIVE_FIELD,
    CONTACT_TOKEN_VALID_INDEX,
)


def _exact_top_mask(values: np.ndarray, fraction: float) -> np.ndarray:
    count = int(np.ceil(values.size * fraction))
    indices = np.argpartition(values, values.size - count)[-count:]
    mask = np.zeros(values.size, dtype=bool)
    mask[indices] = True
    return mask


def _window_group(
    mask: np.ndarray,
    tail_mask: np.ndarray,
    a_mse: np.ndarray,
    c_mse: np.ndarray,
    a_l2: np.ndarray,
    c_l2: np.ndarray,
) -> dict[str, int | float]:
    count = int(mask.sum())
    if count == 0:
        return {"num_windows": 0}
    return {
        "num_windows": count,
        "window_fraction": count / mask.size,
        "tail_windows": int((mask & tail_mask).sum()),
        "tail_rate": float(tail_mask[mask].mean()),
        "a_state_MSE": float(a_mse[mask].mean()),
        "c_state_MSE": float(c_mse[mask].mean()),
        "c_over_a_state_MSE": float(c_mse[mask].mean() / a_mse[mask].mean()),
        "a_state_L2": float(a_l2[mask].mean()),
        "c_state_L2": float(c_l2[mask].mean()),
        "c_over_a_state_L2": float(c_l2[mask].mean() / a_l2[mask].mean()),
        "mean_MSE_improvement": float((a_mse[mask] - c_mse[mask]).mean()),
    }


def _distribution(values: list[np.ndarray]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    merged = np.concatenate(values).astype(np.float64, copy=False)
    if merged.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    return {
        "count": int(merged.size),
        "mean": float(merged.mean()),
        "p50": float(np.quantile(merged, 0.50)),
        "p95": float(np.quantile(merged, 0.95)),
        "p99": float(np.quantile(merged, 0.99)),
    }


def _load_metric_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as handle:
        return {name: np.asarray(handle[name]) for name in handle.files}


def analyze_tail_contacts(
    dataset_path: str | Path,
    a_metrics_path: str | Path,
    c_metrics_path: str | Path,
    *,
    tail_fraction: float = 0.001,
    chunk_trajectories: int = 16,
) -> dict:
    """Return self-collision exposure and tail-attribution statistics."""
    if not 0.0 < tail_fraction < 1.0:
        raise ValueError("tail_fraction must be between zero and one.")
    a_metrics = _load_metric_arrays(Path(a_metrics_path))
    c_metrics = _load_metric_arrays(Path(c_metrics_path))
    for key in ("trajectory_index", "start_step"):
        if not np.array_equal(a_metrics[key], c_metrics[key]):
            raise ValueError(f"A/C metrics are not aligned for {key}.")

    trajectory_index = a_metrics["trajectory_index"].astype(np.int64, copy=False)
    start_step = a_metrics["start_step"].astype(np.int64, copy=False)
    if np.any(np.diff(trajectory_index) < 0):
        raise ValueError("trajectory_index must be grouped in ascending order.")
    num_windows = trajectory_index.size
    a_seed_mse = np.stack([a_metrics[f"seed{seed}_state_MSE"] for seed in range(3)])
    c_seed_mse = np.stack([c_metrics[f"seed{seed}_state_MSE"] for seed in range(3)])
    a_seed_l2 = np.stack([a_metrics[f"seed{seed}_state_L2"] for seed in range(3)])
    c_seed_l2 = np.stack([c_metrics[f"seed{seed}_state_L2"] for seed in range(3)])
    a_mse = a_seed_mse.mean(axis=0, dtype=np.float64)
    c_mse = c_seed_mse.mean(axis=0, dtype=np.float64)
    a_l2 = a_seed_l2.mean(axis=0, dtype=np.float64)
    c_l2 = c_seed_l2.mean(axis=0, dtype=np.float64)
    tail_mask = _exact_top_mask(a_mse, tail_fraction)

    with h5py.File(dataset_path, "r", swmr=True, libver="latest") as handle:
        data = handle["data"]
        tokens_dataset = data["contact_tokens"]
        solver_active_dataset = data[CONTACT_TOKEN_SOLVER_ACTIVE_FIELD]
        num_trajectories, num_steps, _, _ = tokens_dataset.shape
        sequence_length = num_steps - int(start_step.max())
        if sequence_length <= 0:
            raise ValueError("Window start steps are incompatible with dataset length.")
        if int(trajectory_index.max()) >= num_trajectories:
            raise ValueError("Metrics reference a trajectory outside the HDF5 dataset.")
        offsets = np.searchsorted(
            trajectory_index,
            np.arange(num_trajectories + 1),
        )
        window_active_mean = np.zeros(num_windows, dtype=np.float32)
        window_self_mean = np.zeros(num_windows, dtype=np.float32)
        window_self_max = np.zeros(num_windows, dtype=np.int16)
        frame_tail = np.zeros((num_trajectories, num_steps), dtype=bool)
        for index in np.flatnonzero(tail_mask):
            trajectory = int(trajectory_index[index])
            start = int(start_step[index])
            frame_tail[trajectory, start : start + sequence_length] = True

        self_gap_all: list[np.ndarray] = []
        self_speed_all: list[np.ndarray] = []
        self_gap_tail: list[np.ndarray] = []
        self_speed_tail: list[np.ndarray] = []
        external_gap_tail: list[np.ndarray] = []
        external_speed_tail: list[np.ndarray] = []
        pair_counts_all: Counter[tuple[int, int]] = Counter()
        pair_counts_tail: Counter[tuple[int, int]] = Counter()
        self_active_tokens = 0
        self_active_tokens_tail_frames = 0
        active_tokens = 0
        active_tokens_tail_frames = 0
        self_frames = 0
        odd_self_frames = 0
        overflow_frames = 0

        for lower in range(0, num_trajectories, chunk_trajectories):
            upper = min(lower + chunk_trajectories, num_trajectories)
            tokens = np.asarray(tokens_dataset[lower:upper])
            solver_active = np.asarray(solver_active_dataset[lower:upper])
            valid = tokens[..., CONTACT_TOKEN_VALID_INDEX] > 0.5
            active = valid & solver_active
            self_collision = (
                active
                & (tokens[..., CONTACT_TOKEN_OTHER_DYNAMIC_INDEX] > 0.5)
                & (tokens[..., CONTACT_TOKEN_OTHER_BODY_SLOT_INDEX] >= 0)
            )
            external = active & ~self_collision
            active_count = active.sum(axis=-1)
            self_count = self_collision.sum(axis=-1)
            frame_tail_chunk = frame_tail[lower:upper]

            active_tokens += int(active.sum())
            self_active_tokens += int(self_collision.sum())
            active_tokens_tail_frames += int(active[frame_tail_chunk].sum())
            self_active_tokens_tail_frames += int(self_collision[frame_tail_chunk].sum())
            self_frames += int((self_count > 0).sum())
            odd_self_frames += int(((self_count % 2) == 1).sum())
            if "contact_token_overflow" in data:
                overflow_frames += int(np.count_nonzero(np.asarray(data["contact_token_overflow"][lower:upper])))

            self_gap_all.append(tokens[..., CONTACT_TOKEN_GAP_INDEX][self_collision])
            self_speed_all.append(np.linalg.norm(tokens[..., 14:17][self_collision], axis=-1))
            tail_token_mask = frame_tail_chunk[..., None]
            self_tail = self_collision & tail_token_mask
            external_tail = external & tail_token_mask
            self_gap_tail.append(tokens[..., CONTACT_TOKEN_GAP_INDEX][self_tail])
            self_speed_tail.append(np.linalg.norm(tokens[..., 14:17][self_tail], axis=-1))
            external_gap_tail.append(tokens[..., CONTACT_TOKEN_GAP_INDEX][external_tail])
            external_speed_tail.append(np.linalg.norm(tokens[..., 14:17][external_tail], axis=-1))

            owner = tokens[..., CONTACT_TOKEN_BODY_SLOT_INDEX].astype(np.int64)
            other = tokens[..., CONTACT_TOKEN_OTHER_BODY_SLOT_INDEX].astype(np.int64)
            for first, second in zip(owner[self_collision], other[self_collision], strict=True):
                pair_counts_all[tuple(sorted((int(first), int(second))))] += 1
            for first, second in zip(owner[self_tail], other[self_tail], strict=True):
                pair_counts_tail[tuple(sorted((int(first), int(second))))] += 1

            for trajectory in range(lower, upper):
                window_slice = slice(offsets[trajectory], offsets[trajectory + 1])
                starts = start_step[window_slice]
                expected = np.arange(num_steps - sequence_length + 1)
                if not np.array_equal(starts, expected):
                    raise ValueError(f"Trajectory {trajectory} does not contain every ordered window.")
                local = trajectory - lower
                active_windows = np.lib.stride_tricks.sliding_window_view(
                    active_count[local],
                    sequence_length,
                )
                self_windows = np.lib.stride_tricks.sliding_window_view(
                    self_count[local],
                    sequence_length,
                )
                window_active_mean[window_slice] = active_windows.mean(axis=-1)
                window_self_mean[window_slice] = self_windows.mean(axis=-1)
                window_self_max[window_slice] = self_windows.max(axis=-1)

    c_count_mean = c_metrics["contact_count_mean"].astype(np.float64, copy=False)
    a_count_mean = a_metrics["contact_count_mean"].astype(np.float64, copy=False)
    c_count_max_error = float(np.max(np.abs(window_active_mean - c_count_mean)))
    if c_count_max_error > 1e-5:
        raise RuntimeError(f"Derived C active counts do not match paired-eval metrics: {c_count_max_error}.")
    count_delta = c_count_mean - a_count_mean
    count_residual = count_delta - window_self_mean
    self_present = window_self_mean > 0
    no_self = ~self_present
    tail_self = tail_mask & self_present
    tail_no_self = tail_mask & no_self
    improvement = a_mse - c_mse

    return {
        "scope": {
            "num_trajectories": num_trajectories,
            "num_steps": num_steps,
            "sequence_length": sequence_length,
            "num_windows": num_windows,
            "tail_fraction": tail_fraction,
            "tail_windows": int(tail_mask.sum()),
            "tail_threshold_a_state_MSE": float(a_mse[tail_mask].min()),
            "contact_token_overflow_frames": overflow_frames,
        },
        "count_reconstruction": {
            "max_abs_derived_c_count_error": c_count_max_error,
            "all_mean_abs_c_minus_a_minus_self": float(np.abs(count_residual).mean()),
            "tail_mean_abs_c_minus_a_minus_self": float(np.abs(count_residual[tail_mask]).mean()),
            "all_exact_within_1e_5_fraction": float((np.abs(count_residual) <= 1e-5).mean()),
            "tail_exact_within_1e_5_fraction": float((np.abs(count_residual[tail_mask]) <= 1e-5).mean()),
        },
        "self_collision_exposure": {
            "active_tokens": active_tokens,
            "self_active_directed_tokens": self_active_tokens,
            "self_token_fraction": self_active_tokens / active_tokens,
            "active_tokens_in_tail_frames": active_tokens_tail_frames,
            "self_active_tokens_in_tail_frames": self_active_tokens_tail_frames,
            "tail_frame_self_token_fraction": (self_active_tokens_tail_frames / active_tokens_tail_frames),
            "frames_with_self_collision": self_frames,
            "odd_directed_self_count_frames": odd_self_frames,
            "all_windows_with_self_collision": int(self_present.sum()),
            "all_window_self_fraction": float(self_present.mean()),
            "tail_windows_with_self_collision": int(tail_self.sum()),
            "tail_window_self_fraction": float(self_present[tail_mask].mean()),
            "tail_window_self_enrichment": float(self_present[tail_mask].mean() / self_present.mean()),
            "all_window_mean_self_tokens_per_frame": float(window_self_mean.mean()),
            "tail_window_mean_self_tokens_per_frame": float(window_self_mean[tail_mask].mean()),
            "tail_improvement_fraction_from_self_windows": float(
                improvement[tail_self].sum() / improvement[tail_mask].sum()
            ),
        },
        "error_by_self_collision": {
            "self_present": _window_group(
                self_present,
                tail_mask,
                a_mse,
                c_mse,
                a_l2,
                c_l2,
            ),
            "self_absent": _window_group(
                no_self,
                tail_mask,
                a_mse,
                c_mse,
                a_l2,
                c_l2,
            ),
            "tail_self_present": _window_group(
                tail_self,
                tail_mask,
                a_mse,
                c_mse,
                a_l2,
                c_l2,
            ),
            "tail_self_absent": _window_group(
                tail_no_self,
                tail_mask,
                a_mse,
                c_mse,
                a_l2,
                c_l2,
            ),
        },
        "geometry": {
            "self_gap_all": _distribution(self_gap_all),
            "self_relative_speed_all": _distribution(self_speed_all),
            "self_gap_tail_frames": _distribution(self_gap_tail),
            "self_relative_speed_tail_frames": _distribution(self_speed_tail),
            "external_gap_tail_frames": _distribution(external_gap_tail),
            "external_relative_speed_tail_frames": _distribution(external_speed_tail),
        },
        "self_body_pairs": {
            "all": {f"{first}:{second}": count for (first, second), count in pair_counts_all.most_common()},
            "tail_frames": {f"{first}:{second}": count for (first, second), count in pair_counts_tail.most_common()},
        },
        "top_tail_windows": [
            {
                "window_index": int(index),
                "trajectory_index": int(trajectory_index[index]),
                "start_step": int(start_step[index]),
                "a_state_MSE": float(a_mse[index]),
                "c_state_MSE": float(c_mse[index]),
                "a_state_L2": float(a_l2[index]),
                "c_state_L2": float(c_l2[index]),
                "a_contact_count_mean": float(a_count_mean[index]),
                "c_contact_count_mean": float(c_count_mean[index]),
                "self_tokens_per_frame": float(window_self_mean[index]),
                "self_tokens_max_frame": int(window_self_max[index]),
            }
            for index in np.flatnonzero(tail_mask)[np.argsort(a_mse[tail_mask])[::-1]][:100]
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--a-metrics", required=True)
    parser.add_argument("--c-metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tail-fraction", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze_tail_contacts(
        args.dataset,
        args.a_metrics,
        args.c_metrics,
        tail_fraction=args.tail_fraction,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["self_collision_exposure"], indent=2))


if __name__ == "__main__":
    main()
