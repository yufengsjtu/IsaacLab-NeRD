# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline analysis of contact counts and packing overflow in NeRD HDF5 datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def analyze_contact_distribution(dataset_path: str | Path) -> dict[str, float | int | np.ndarray]:
    """Summarize per-frame valid contact counts and optional token overflow."""
    dataset_path = Path(dataset_path).expanduser()
    with h5py.File(dataset_path, "r") as handle:
        data = handle["data"]
        if "contact_masks" in data:
            valid_counts = np.asarray(data["contact_masks"]).sum(axis=-1).astype(np.int64)
        elif "contact_tokens" in data:
            tokens = np.asarray(data["contact_tokens"])
            valid_counts = (tokens[..., 0] > 0.5).sum(axis=-1).astype(np.int64)
        else:
            raise ValueError("Dataset must contain contact_masks or contact_tokens.")

        overflow = None
        if "contact_token_overflow" in data:
            overflow = np.asarray(data["contact_token_overflow"])

        num_contacts_per_env = int(data.attrs.get("num_contacts_per_env", valid_counts.max(initial=0)))
        max_contact_tokens = int(data.attrs.get("max_contact_tokens", num_contacts_per_env))
        uses_tokens = "contact_tokens" in data

    flat_counts = valid_counts.reshape(-1)
    # Packed tokens never exceed K; real drops are recorded in contact_token_overflow.
    if uses_tokens:
        capacity_overflow_frames = 0
    else:
        capacity_overflow_frames = int((flat_counts > num_contacts_per_env).sum())
    summary: dict[str, float | int | np.ndarray] = {
        "frames": int(flat_counts.size),
        "min_valid_contacts": int(flat_counts.min(initial=0)),
        "max_valid_contacts": int(flat_counts.max(initial=0)),
        "mean_valid_contacts": float(flat_counts.mean()),
        "p95_valid_contacts": float(np.percentile(flat_counts, 95)),
        "p99_valid_contacts": float(np.percentile(flat_counts, 99)),
        "num_contacts_per_env": num_contacts_per_env,
        "max_contact_tokens": max_contact_tokens,
        "capacity_overflow_frames": capacity_overflow_frames,
    }
    if overflow is not None:
        summary["token_overflow_frames"] = int((overflow > 0).sum())
        summary["token_overflow_total"] = int(overflow.sum())
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Path to NeRD HDF5 dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze_contact_distribution(args.dataset)
    print(f"Contact distribution for {args.dataset}")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
