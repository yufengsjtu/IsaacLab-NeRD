# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark eager versus lazy NeRD HDF5 dataset loading."""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from isaaclab_neural.data import (
    create_batch_transition_dataset,
    create_trajectory_dataset,
    write_rollouts_to_hdf5,
)


def _rss_gib() -> float:
    """Return current peak resident set size in GiB."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024**3)
    return rss / (1024**2)


def _current_rss_gib() -> float:
    """Return current resident set size in GiB."""
    with Path("/proc/self/status").open("r", encoding="utf-8") as status_file:
        for line in status_file:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / (1024**2)
    return _rss_gib()


def _make_synthetic_rollouts(
    *,
    num_trajectories: int,
    trajectory_length: int,
    state_dim: int,
    joint_f_dim: int,
    num_contacts_per_env: int,
    include_actions: bool,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    """Create rollout tensors with the same schema as NeRD trajectory datasets."""
    shape = (num_trajectories, trajectory_length)
    rollouts: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
        "states": torch.randn(*shape, state_dim, dtype=torch.float32),
        "next_states": torch.randn(*shape, state_dim, dtype=torch.float32),
        "joint_f": torch.randn(*shape, joint_f_dim, dtype=torch.float32),
        "root_body_q": torch.randn(*shape, 7, dtype=torch.float32),
        "gravity_dir": torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32).expand(*shape, 3).clone(),
        "contacts": {
            "contact_masks": torch.rand(*shape, num_contacts_per_env) > 0.5,
            "contact_normals": torch.randn(*shape, num_contacts_per_env * 3, dtype=torch.float32),
            "contact_depths": torch.rand(*shape, num_contacts_per_env, dtype=torch.float32),
            "contact_points_0": torch.randn(*shape, num_contacts_per_env * 3, dtype=torch.float32),
            "contact_points_1": torch.randn(*shape, num_contacts_per_env * 3, dtype=torch.float32),
            "contact_thicknesses_0": torch.rand(*shape, num_contacts_per_env, dtype=torch.float32),
            "contact_thicknesses_1": torch.rand(*shape, num_contacts_per_env, dtype=torch.float32),
        },
    }
    if include_actions:
        rollouts["actions"] = torch.randn(*shape, 12, dtype=torch.float32)
    return rollouts


def _write_benchmark_dataset(
    output_path: Path,
    *,
    num_trajectories: int,
    trajectory_length: int,
    state_dim: int,
    joint_f_dim: int,
    num_contacts_per_env: int,
) -> None:
    rollouts = _make_synthetic_rollouts(
        num_trajectories=num_trajectories,
        trajectory_length=trajectory_length,
        state_dim=state_dim,
        joint_f_dim=joint_f_dim,
        num_contacts_per_env=num_contacts_per_env,
        include_actions=True,
    )
    write_rollouts_to_hdf5(output_path, rollouts, env_name="Benchmark")


def _time_dataloader_epoch(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    collate_fn,
    shuffle: bool,
) -> tuple[float, int]:
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    start = time.perf_counter()
    batch_count = 0
    for batch in loader:
        batch_count += 1
        _ = batch["states"].shape
    return time.perf_counter() - start, batch_count


def _benchmark_trajectory_dataset(
    dataset_path: Path,
    *,
    load_mode: str,
    sample_sequence_length: int,
    max_capacity: int,
    batch_size: int,
    num_workers: int,
    warmup_epochs: int,
    measure_epochs: int,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, float]:
    gc.collect()
    rss_before = _rss_gib()

    init_start = time.perf_counter()
    dataset = create_trajectory_dataset(
        load_mode=load_mode,
        hdf5_dataset_path=dataset_path,
        sample_sequence_length=sample_sequence_length,
        max_capacity=max_capacity,
        rank=rank,
        world_size=world_size,
    )
    init_seconds = time.perf_counter() - init_start
    rss_after_init = _current_rss_gib()

    for _ in range(warmup_epochs):
        dataset.shuffle()
        _time_dataloader_epoch(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=None,
            shuffle=False,
        )

    epoch_seconds = []
    batches_per_epoch = 0
    for _ in range(measure_epochs):
        dataset.shuffle()
        seconds, batches_per_epoch = _time_dataloader_epoch(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=None,
            shuffle=False,
        )
        epoch_seconds.append(seconds)

    if hasattr(dataset, "close"):
        dataset.close()
    del dataset
    gc.collect()
    rss_after_epoch = _current_rss_gib()

    return {
        "init_seconds": init_seconds,
        "epoch_seconds_mean": float(np.mean(epoch_seconds)),
        "epoch_seconds_std": float(np.std(epoch_seconds)),
        "batches_per_epoch": float(batches_per_epoch),
        "samples_per_second": float((batches_per_epoch * batch_size) / np.mean(epoch_seconds)),
        "rss_before_gib": rss_before,
        "rss_after_init_gib": rss_after_init,
        "rss_after_epoch_gib": rss_after_epoch,
    }


def _benchmark_batch_transition_dataset(
    dataset_path: Path,
    *,
    load_mode: str,
    batch_size: int,
    max_capacity: int,
    num_workers: int,
    warmup_epochs: int,
    measure_epochs: int,
) -> dict[str, float]:
    from isaaclab_neural.data import collate_fn_BatchTransitionDataset

    gc.collect()
    rss_before = _rss_gib()

    init_start = time.perf_counter()
    dataset = create_batch_transition_dataset(
        load_mode=load_mode,
        batch_size=batch_size,
        hdf5_dataset_path=dataset_path,
        max_capacity=max_capacity,
        device="cpu",
    )
    init_seconds = time.perf_counter() - init_start
    rss_after_init = _current_rss_gib()

    for _ in range(warmup_epochs):
        dataset.shuffle()
        _time_dataloader_epoch(
            dataset,
            batch_size=1,
            num_workers=num_workers,
            collate_fn=collate_fn_BatchTransitionDataset,
            shuffle=False,
        )

    epoch_seconds = []
    batches_per_epoch = 0
    for _ in range(measure_epochs):
        dataset.shuffle()
        seconds, batches_per_epoch = _time_dataloader_epoch(
            dataset,
            batch_size=1,
            num_workers=num_workers,
            collate_fn=collate_fn_BatchTransitionDataset,
            shuffle=False,
        )
        epoch_seconds.append(seconds)

    if hasattr(dataset, "close"):
        dataset.close()
    del dataset
    gc.collect()
    rss_after_epoch = _current_rss_gib()

    return {
        "init_seconds": init_seconds,
        "epoch_seconds_mean": float(np.mean(epoch_seconds)),
        "epoch_seconds_std": float(np.std(epoch_seconds)),
        "batches_per_epoch": float(batches_per_epoch),
        "samples_per_second": float((batches_per_epoch * batch_size) / np.mean(epoch_seconds)),
        "rss_before_gib": rss_before,
        "rss_after_init_gib": rss_after_init,
        "rss_after_epoch_gib": rss_after_epoch,
    }


def _print_result(label: str, result: dict[str, float]) -> None:
    print(f"\n[{label}]")
    print(f"  init:            {result['init_seconds']:.3f}s")
    print(f"  epoch:           {result['epoch_seconds_mean']:.3f}s +/- {result['epoch_seconds_std']:.3f}s")
    print(f"  batches/epoch:   {int(result['batches_per_epoch'])}")
    print(f"  samples/s:       {result['samples_per_second']:.1f}")
    print(f"  rss after init:  {result['rss_after_init_gib']:.2f} GiB")
    print(f"  rss after epoch: {result['rss_after_epoch_gib']:.2f} GiB")


def _run_worker(payload: dict[str, object]) -> dict[str, float]:
    """Run one benchmark mode in an isolated process."""
    dataset_path = Path(cast(str, payload["dataset_path"]))
    load_mode = cast(str, payload["load_mode"])
    dataset_kind = cast(str, payload["dataset_kind"])
    if dataset_kind == "trajectory":
        return _benchmark_trajectory_dataset(
            dataset_path,
            load_mode=load_mode,
            sample_sequence_length=int(payload["sample_sequence_length"]),
            max_capacity=int(payload["max_capacity"]),
            batch_size=int(payload["batch_size"]),
            num_workers=int(payload["num_workers"]),
            warmup_epochs=int(payload["warmup_epochs"]),
            measure_epochs=int(payload["measure_epochs"]),
            rank=0,
            world_size=int(payload["eager_world_size"]) if load_mode == "eager" else 1,
        )
    return _benchmark_batch_transition_dataset(
        dataset_path,
        load_mode=load_mode,
        batch_size=int(payload["batch_size"]),
        max_capacity=int(payload["max_capacity"]),
        num_workers=int(payload["num_workers"]),
        warmup_epochs=int(payload["warmup_epochs"]),
        measure_epochs=int(payload["measure_epochs"]),
    )


def _benchmark_in_subprocess(payload: dict[str, object]) -> dict[str, float]:
    """Spawn a clean process so memory numbers are not polluted by the other mode."""
    command = [
        sys.executable,
        "-c",
        "import json, sys; from isaaclab_neural.data.benchmark_dataset_loading import _run_worker; "
        "print(json.dumps(_run_worker(json.loads(sys.argv[1]))))",
        json.dumps(payload),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, env=os.environ.copy())
    return json.loads(completed.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=str, default=None, help="Existing NeRD HDF5 dataset path.")
    parser.add_argument("--dataset-kind", choices=("trajectory", "batch_transition"), default="trajectory")
    parser.add_argument("--num-trajectories", type=int, default=256)
    parser.add_argument("--trajectory-length", type=int, default=64)
    parser.add_argument("--state-dim", type=int, default=48)
    parser.add_argument("--joint-f-dim", type=int, default=12)
    parser.add_argument("--num-contacts-per-env", type=int, default=4)
    parser.add_argument("--sample-sequence-length", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-capacity", type=int, default=1_000_000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--measure-epochs", type=int, default=3)
    parser.add_argument(
        "--eager-world-size",
        type=int,
        default=1,
        help="Simulate one rank of eager DDP trajectory sharding.",
    )
    args = parser.parse_args()

    if args.dataset_path is None:
        temp_dir = tempfile.TemporaryDirectory()
        dataset_path = Path(temp_dir.name) / "benchmark_dataset.hdf5"
        _write_benchmark_dataset(
            dataset_path,
            num_trajectories=args.num_trajectories,
            trajectory_length=args.trajectory_length,
            state_dim=args.state_dim,
            joint_f_dim=args.joint_f_dim,
            num_contacts_per_env=args.num_contacts_per_env,
        )
        print(f"[benchmark] wrote synthetic dataset to {dataset_path}")
    else:
        temp_dir = None
        dataset_path = Path(args.dataset_path).expanduser()

    try:
        payload = {
            "dataset_path": str(dataset_path),
            "dataset_kind": args.dataset_kind,
            "sample_sequence_length": args.sample_sequence_length,
            "batch_size": args.batch_size,
            "max_capacity": args.max_capacity,
            "num_workers": args.num_workers,
            "warmup_epochs": args.warmup_epochs,
            "measure_epochs": args.measure_epochs,
            "eager_world_size": args.eager_world_size,
        }
        eager = _benchmark_in_subprocess({**payload, "load_mode": "eager"})
        lazy = _benchmark_in_subprocess({**payload, "load_mode": "lazy"})

        _print_result("eager", eager)
        _print_result("lazy", lazy)

        init_ratio = lazy["init_seconds"] / max(eager["init_seconds"], 1e-6)
        epoch_ratio = lazy["epoch_seconds_mean"] / max(eager["epoch_seconds_mean"], 1e-6)
        rss_ratio = lazy["rss_after_init_gib"] / max(eager["rss_after_init_gib"], 1e-6)
        throughput_ratio = lazy["samples_per_second"] / max(eager["samples_per_second"], 1e-6)
        print("\n[summary]")
        print(f"  lazy/eager init speed ratio: {init_ratio:.2f}x")
        print(f"  lazy/eager epoch speed ratio: {epoch_ratio:.2f}x")
        print(f"  lazy/eager throughput ratio: {throughput_ratio:.2f}x")
        print(f"  lazy/eager rss-after-init ratio: {rss_ratio:.2f}x")
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
