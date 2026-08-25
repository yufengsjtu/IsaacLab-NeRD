# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate one design's E99 checkpoints on one immutable paired HDF5 file."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import isaaclab_neural.envs  # noqa: F401 - registers NeRD tasks
import numpy as np
import torch
from isaaclab_neural.eval.paired_checkpoint_eval import (
    validate_paired_eval_design_contract,
    window_contact_count_metrics,
    window_state_dimension_mse,
    window_state_error_metrics,
)
from isaaclab_neural.eval.paired_contact_dataset import sha256_file, validate_paired_policy_suite_manifest
from isaaclab_neural.physics import NerdNewtonCfg, NerdSolverCfg
from isaaclab_neural.train import SequenceModelTrainer
from isaaclab_neural.utils.checkpoint import load_checkpoint, reconstruct_model_from_checkpoint
from isaaclab_neural.utils.python_utils import set_random_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from isaaclab_tasks.utils.hydra import hydra_task_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    from isaaclab_tasks.utils import add_launcher_args

    add_launcher_args(parser)
    parser.add_argument("--task", default="Isaac-Velocity-Rough-Anymal-C-NeRD-v0")
    parser.add_argument("--dataset", required=True, help="Paired Raw15 or ContactTokens HDF5 file.")
    parser.add_argument("--paired-dataset-manifest", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--expected-seeds",
        default="0,1,2",
        help="Comma-separated checkpoint seeds required for this design.",
    )
    parser.add_argument("--force-overwrite", action="store_true")
    return parser


args_cli, hydra_args = _parser().parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]


def _expected_seed_ids(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--expected-seeds must contain unique integer seeds.")
    return seeds


def _checkpoint_entries(
    manifest_path: Path,
    design: str,
    expected_seeds: list[int],
) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    entries = [entry for entry in manifest["checkpoints"] if entry["design"] == design]
    entries.sort(key=lambda entry: int(entry["seed"]))
    if not entries:
        raise ValueError(f"Checkpoint manifest has no entries for design {design!r}.")
    seeds = [int(entry["seed"]) for entry in entries]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Checkpoint manifest has duplicate seeds for design {design!r}: {seeds}.")
    if seeds != sorted(expected_seeds):
        raise ValueError(f"Checkpoint seeds for design {design!r} are {seeds}, expected {sorted(expected_seeds)}.")
    return entries


def _load_verified_checkpoints(
    manifest_path: Path,
    design: str,
    expected_seeds: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entries = _checkpoint_entries(manifest_path, design, expected_seeds)
    checkpoints = []
    for entry in entries:
        checkpoint_path = (manifest_path.parent / entry["path"]).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}.")
        if checkpoint_path.stat().st_size != int(entry["size_bytes"]):
            raise ValueError(f"Checkpoint size mismatch: {checkpoint_path}.")
        actual_sha256 = sha256_file(checkpoint_path)
        if actual_sha256 != entry["sha256"]:
            raise ValueError(f"Checkpoint SHA-256 mismatch: {checkpoint_path}.")
        checkpoint = load_checkpoint(checkpoint_path, device="cpu")
        if checkpoint.get("version") != 2 or checkpoint.get("epoch") != 99:
            raise ValueError(f"Expected a v2 Epoch99 checkpoint: {checkpoint_path}.")
        entry = dict(entry)
        entry["resolved_path"] = str(checkpoint_path)
        checkpoints.append(checkpoint)
    return entries, checkpoints


def _comparison_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "neural_solver_cfg": cfg["env"]["neural_solver_cfg"],
        "inputs": cfg["inputs"],
        "network": cfg["network"],
        "sample_sequence_length": cfg["algorithm"]["sample_sequence_length"],
        "dataset_contact_representation": cfg["algorithm"]["dataset"].get("contact_representation"),
    }


def _prepare_eval_cfg(
    checkpoint: dict[str, Any],
    dataset_path: Path,
) -> dict[str, Any]:
    cfg = copy.deepcopy(checkpoint["cfg"])
    cfg["env"]["num_envs"] = args_cli.num_envs
    algorithm = cfg["algorithm"]
    algorithm["seed"] = args_cli.seed
    algorithm["batch_size"] = args_cli.batch_size
    algorithm["num_valid_batches"] = 0
    algorithm["update_dataset_statistics"] = False
    dataset_cfg = algorithm["dataset"]
    dataset_cfg["train_dataset_path"] = str(dataset_path)
    dataset_cfg["valid_datasets"] = {}
    dataset_cfg["max_capacity"] = 1_000_000_000
    dataset_cfg["load_mode"] = "lazy"
    dataset_cfg["num_data_workers"] = args_cli.num_workers
    dataset_cfg["pin_memory"] = True
    dataset_cfg["non_blocking"] = True
    dataset_cfg["persistent_workers"] = args_cli.num_workers > 0
    cfg["cli"] = {
        "train": False,
        "distributed": False,
        "rank": 0,
        "local_rank": 0,
        "world_size": 1,
        "render": False,
    }
    return cfg


def _configure_env(env_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device


def _build_launch_cfg(env_cfg):
    physics_cfg = env_cfg.sim.physics
    if not isinstance(physics_cfg, NerdNewtonCfg):
        return env_cfg

    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

    launch_cfg = copy.deepcopy(env_cfg)
    launch_cfg.sim.physics = NewtonCfg(
        num_substeps=physics_cfg.num_substeps,
        debug_mode=physics_cfg.debug_mode,
        use_cuda_graph=physics_cfg.use_cuda_graph,
    )
    return launch_cfg


def _verify_dataset(
    dataset_path: Path,
    manifest_path: Path,
    dataset_representation: str,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    policy_suite = validate_paired_policy_suite_manifest(manifest)
    representation_keys = {
        "raw15_tokens": "raw15",
        "contact_tokens": "contact_tokens",
    }
    if dataset_representation not in representation_keys:
        raise ValueError(f"Unsupported paired-eval dataset representation: {dataset_representation!r}.")
    manifest_key = representation_keys[dataset_representation]
    expected = manifest[manifest_key]
    if dataset_path.stat().st_size != int(expected["size_bytes"]):
        raise ValueError(f"Dataset size mismatch for {dataset_path}.")
    actual_sha256 = sha256_file(dataset_path)
    if actual_sha256 != expected["sha256"]:
        raise ValueError(f"Dataset SHA-256 mismatch for {dataset_path}.")
    return {
        "manifest_key": manifest_key,
        "size_bytes": dataset_path.stat().st_size,
        "sha256": actual_sha256,
        "pairing": manifest["pairing"],
        "policy_suite": policy_suite,
    }


def _summarize_metrics(metrics: dict[str, np.ndarray]) -> dict[str, dict[str, float | int]]:
    return {
        name: {
            "count": int(values.size),
            "mean": float(np.mean(values, dtype=np.float64)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
        }
        for name, values in metrics.items()
    }


@hydra_task_config(args_cli.task, "")
def main(env_cfg, _agent_cfg=None) -> None:
    """Run deterministic full-window inference for all seeds in one design."""
    if args_cli.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args_cli.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")

    dataset_path = Path(args_cli.dataset).expanduser().resolve()
    paired_manifest_path = Path(args_cli.paired_dataset_manifest).expanduser().resolve()
    checkpoint_manifest_path = Path(args_cli.checkpoint_manifest).expanduser().resolve()
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    result_json = output_dir / "metrics.json"
    result_npz = output_dir / "per_window_metrics.npz"
    if (result_json.exists() or result_npz.exists()) and not args_cli.force_overwrite:
        raise FileExistsError(f"Evaluation output already exists under {output_dir}.")
    output_dir.mkdir(parents=True, exist_ok=True)

    entries, checkpoints = _load_verified_checkpoints(
        checkpoint_manifest_path,
        args_cli.design,
        _expected_seed_ids(args_cli.expected_seeds),
    )
    for checkpoint in checkpoints:
        validate_paired_eval_design_contract(args_cli.design, checkpoint["cfg"])
    reference_contract = _comparison_contract(checkpoints[0]["cfg"])
    for checkpoint in checkpoints[1:]:
        if _comparison_contract(checkpoint["cfg"]) != reference_contract:
            raise ValueError("Checkpoints within one design do not share the same inference contract.")
    dataset_identity = _verify_dataset(
        dataset_path,
        paired_manifest_path,
        reference_contract["dataset_contact_representation"],
    )
    cfg = _prepare_eval_cfg(checkpoints[0], dataset_path)
    _configure_env(env_cfg)
    neural_solver_cfg = dict(cfg["env"]["neural_solver_cfg"])
    neural_solver_cfg.pop("use_cuda_graph", None)
    solver_cfg = NerdSolverCfg(**neural_solver_cfg)

    from isaaclab_neural.utils.usd_utils import newton_material_binding_api_autofix

    from isaaclab_tasks.utils import launch_simulation

    with launch_simulation(_build_launch_cfg(env_cfg), args_cli):
        import gymnasium as gym

        with newton_material_binding_api_autofix():
            env = gym.make(
                args_cli.task,
                cfg=env_cfg,
                device=args_cli.device,
                solver_cfg=solver_cfg,
            ).unwrapped
            env.reset()
        trainer = SequenceModelTrainer(
            neural_env=env,
            checkpoint=checkpoints[0],
            cfg=cfg,
            device=args_cli.device,
        )
        models = [trainer.neural_model_unwrapped]
        models.extend(
            reconstruct_model_from_checkpoint(
                checkpoint,
                trainer.neural_solver,
                device=args_cli.device,
            )
            for checkpoint in checkpoints[1:]
        )
        for model in models:
            model.eval()

        loader = DataLoader(
            trainer.train_dataset,
            batch_size=args_cli.batch_size,
            shuffle=False,
            num_workers=args_cli.num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=args_cli.num_workers > 0,
        )
        metric_chunks: list[dict[str, list[np.ndarray]]] = [{} for _entry in entries]
        dimension_mse_sums: list[np.ndarray | None] = [None for _entry in entries]
        dimension_mse_counts = [0 for _entry in entries]
        contact_metric_chunks: dict[str, list[np.ndarray]] = {}
        with torch.inference_mode():
            for batch in tqdm(loader, desc=f"evaluate {args_cli.design}"):
                data = trainer.preprocess_data_batch(batch)
                contact_metrics = window_contact_count_metrics(data["contact_tokens"])
                for name, values in contact_metrics.items():
                    contact_metric_chunks.setdefault(name, []).append(
                        values.detach().cpu().numpy().astype(np.float32, copy=False)
                    )
                for model_index, model in enumerate(models):
                    prediction = model(dict(data))
                    predicted_next_states = trainer.neural_solver.convert_prediction_to_next_states(
                        states=data["states"],
                        prediction=prediction,
                        dt=trainer.neural_env.frame_dt,
                    )
                    trainer.neural_solver.wrap2PI(predicted_next_states)
                    batch_metrics = window_state_error_metrics(
                        predicted_next_states,
                        data["next_states"],
                        dof_q=trainer.neural_solver.dof_q_per_env,
                    )
                    dimension_mse = (
                        window_state_dimension_mse(
                            predicted_next_states,
                            data["next_states"],
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    dimension_sum = dimension_mse.sum(axis=0, dtype=np.float64)
                    if dimension_mse_sums[model_index] is None:
                        dimension_mse_sums[model_index] = dimension_sum
                    else:
                        dimension_mse_sums[model_index] += dimension_sum
                    dimension_mse_counts[model_index] += int(dimension_mse.shape[0])
                    for name, values in batch_metrics.items():
                        metric_chunks[model_index].setdefault(name, []).append(
                            values.detach().cpu().numpy().astype(np.float32, copy=False)
                        )

        dataset = trainer.train_dataset
        mapping = np.asarray(dataset.mapping_index2traj, dtype=np.int64)
        payload: dict[str, np.ndarray] = {
            "window_index": np.arange(mapping.shape[0], dtype=np.int64),
            "trajectory_index": mapping[:, 0],
            "start_step": mapping[:, 1],
        }
        for name, chunks in contact_metric_chunks.items():
            payload[name] = np.concatenate(chunks)
        with h5py.File(dataset_path, "r", swmr=True, libver="latest") as dataset_file:
            trajectory_context = dataset_file["context"]["trajectories"]
            for name in ("terrain_level", "terrain_type", "source_env_id"):
                payload[f"trajectory_{name}"] = np.asarray(trajectory_context[name])
        summary: dict[str, Any] = {}
        for model_index, (entry, chunks) in enumerate(zip(entries, metric_chunks, strict=True)):
            label = f"seed{entry['seed']}"
            merged = {name: np.concatenate(values) for name, values in chunks.items()}
            if any(values.shape != (mapping.shape[0],) for values in merged.values()):
                raise RuntimeError(f"Per-window metric count mismatch for {label}.")
            dimension_sum = dimension_mse_sums[model_index]
            dimension_count = dimension_mse_counts[model_index]
            if dimension_sum is None or dimension_count != mapping.shape[0]:
                raise RuntimeError(f"Per-dimension metric count mismatch for {label}.")
            summary[label] = {
                "wandb_run_id": entry["wandb_run_id"],
                "checkpoint_sha256": entry["sha256"],
                "metrics": _summarize_metrics(merged),
                "state_dimension_MSE": (dimension_sum / dimension_count).tolist(),
            }
            for name, values in merged.items():
                payload[f"{label}_{name}"] = values

        temporary_npz = result_npz.with_suffix(".npz.tmp")
        with temporary_npz.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        temporary_npz.replace(result_npz)
        result = {
            "schema_version": 1,
            "design": args_cli.design,
            "dataset": dataset_identity,
            "num_windows": int(mapping.shape[0]),
            "sample_sequence_length": int(cfg["algorithm"]["sample_sequence_length"]),
            "checkpoints": summary,
            "per_window_metrics": result_npz.name,
        }
        temporary_json = result_json.with_suffix(".json.tmp")
        temporary_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary_json.replace(result_json)
        if hasattr(trainer.train_dataset, "close"):
            trainer.train_dataset.close()
        env.close()
        print(f"[paired-eval] wrote {result_json}")
        print(f"[paired-eval] wrote {result_npz}")


if __name__ == "__main__":
    set_random_seed(args_cli.seed)
    main()  # type: ignore[call-arg]
