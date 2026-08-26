# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate old/new E199 checkpoints on one frozen validation suite."""

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
from isaaclab_neural.eval.paired_checkpoint_eval import window_state_error_metrics
from isaaclab_neural.eval.training_evaluator import TrainingRolloutEvaluator
from isaaclab_neural.physics import NerdNewtonCfg, NerdSolverCfg
from isaaclab_neural.train import SequenceModelTrainer
from isaaclab_neural.utils.checkpoint import reconstruct_model_from_checkpoint
from isaaclab_neural.utils.python_utils import set_random_seed
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from isaaclab_tasks.utils.hydra import hydra_task_config

from osmo_scripts.sampling_strategy_eval.contract import (
    deterministic_indices,
    load_verified_checkpoints,
    load_verified_suite,
    prepare_eval_cfg,
)

METRIC_NAMES = ("state_MSE", "q_MSE", "qd_MSE", "state_L2", "q_L2", "qd_L2")
SOURCE_METRIC_NAMES = {
    "state_MSE": "state_MSE",
    "q_MSE": "q_MSE",
    "qd_MSE": "qd_MSE",
    "state_L2": "state_L2",
    "q_error_norm": "q_L2",
    "qd_error_norm": "qd_L2",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    from isaaclab_tasks.utils import add_launcher_args

    add_launcher_args(parser)
    parser.add_argument("--task", default="Isaac-Velocity-Rough-Anymal-C-NeRD-v0")
    parser.add_argument("--encoder", choices=("a", "d"), required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--max-windows", type=int, default=51_200)
    parser.add_argument("--window-seed", type=int, default=20_260_826)
    parser.add_argument("--rollout-count", type=int, default=0)
    parser.add_argument("--rollout-horizon", type=int, default=10)
    parser.add_argument("--rollout-seed", type=int, default=20_260_826)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-overwrite", action="store_true")
    return parser


args_cli, hydra_args = _parser().parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]


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


def _model_label(entry: dict[str, Any]) -> str:
    return f"{entry['group']}_seed{entry['seed']}"


def _summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def _context_payload(dataset_path: Path, trajectory_indices: np.ndarray) -> dict[str, np.ndarray]:
    payload = {}
    with h5py.File(dataset_path, "r", swmr=True, libver="latest") as dataset_file:
        context = dataset_file.get("context/trajectories")
        for name in ("terrain_level", "terrain_type", "source_env_id"):
            if context is None or name not in context:
                payload[f"trajectory_{name}"] = np.full(trajectory_indices.shape, -1, dtype=np.int64)
            else:
                values = np.asarray(context[name])
                payload[f"trajectory_{name}"] = values[trajectory_indices]
    return payload


def _one_step_metrics(
    trainer: SequenceModelTrainer,
    models: list[torch.nn.Module],
    entries: list[dict[str, Any]],
    selected_indices: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    loader = DataLoader(
        Subset(trainer.train_dataset, selected_indices.tolist()),
        batch_size=args_cli.batch_size,
        shuffle=False,
        num_workers=args_cli.num_workers,
        drop_last=False,
        pin_memory=True,
        persistent_workers=args_cli.num_workers > 0,
    )
    chunks: list[dict[str, list[np.ndarray]]] = [{} for _model in models]
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"one-step {args_cli.suite_id}"):
            data = trainer.preprocess_data_batch(batch)
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
                for source_name, values in batch_metrics.items():
                    output_name = SOURCE_METRIC_NAMES[source_name]
                    chunks[model_index].setdefault(output_name, []).append(
                        values.detach().cpu().numpy().astype(np.float32, copy=False)
                    )

    payload = {}
    summary = {}
    for entry, model_chunks in zip(entries, chunks, strict=True):
        label = _model_label(entry)
        summary[label] = {}
        for metric_name in METRIC_NAMES:
            values = np.concatenate(model_chunks[metric_name])
            if values.shape != selected_indices.shape or not np.isfinite(values).all():
                raise RuntimeError(f"Invalid one-step {metric_name} payload for {label}.")
            payload[f"{label}_{metric_name}"] = values
            summary[label][metric_name] = _summary(values)
    return payload, summary


def _rollout_metric_arrays(diff: torch.Tensor, dof_q: int) -> dict[str, np.ndarray]:
    state_error = diff.float()
    q_error = state_error[..., :dof_q]
    qd_error = state_error[..., dof_q:]
    arrays = {
        "state_MSE": state_error.square().mean(dim=(1, 2)),
        "q_MSE": q_error.square().mean(dim=(1, 2)),
        "qd_MSE": qd_error.square().mean(dim=(1, 2)),
        "state_L2": state_error.norm(dim=-1).mean(dim=1),
        "q_L2": q_error.norm(dim=-1).mean(dim=1),
        "qd_L2": qd_error.norm(dim=-1).mean(dim=1),
        "final_state_MSE": state_error[:, -1].square().mean(dim=1),
        "final_state_L2": state_error[:, -1].norm(dim=-1),
    }
    return {name: values.detach().cpu().numpy().astype(np.float32, copy=False) for name, values in arrays.items()}


def _rollout_metrics(
    trainer: SequenceModelTrainer,
    models: list[torch.nn.Module],
    entries: list[dict[str, Any]],
    dataset_path: Path,
    dataset_representation: str,
    require_solver_active: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    evaluator = TrainingRolloutEvaluator(
        trainer.neural_env,
        hdf5_dataset_path=str(dataset_path),
        eval_horizon=args_cli.rollout_horizon,
        dataset_contact_representation=dataset_representation,
        require_solver_active=require_solver_active,
        device=args_cli.device,
        require_terrain_context=True,
        contact_context_validation="warn",
    )
    dataset = evaluator.trajectory_dataset
    if dataset is None:
        raise RuntimeError("Rollout evaluator did not create its trajectory dataset.")
    if args_cli.rollout_count < args_cli.num_envs or args_cli.rollout_count % args_cli.num_envs != 0:
        raise ValueError("--rollout-count must be a positive multiple of --num-envs.")
    rng_state = np.random.get_state()
    np.random.seed(args_cli.rollout_seed)
    rollout_indices = np.random.randint(0, len(dataset), size=args_cli.rollout_count)
    np.random.set_state(rng_state)
    mapping = np.asarray(dataset.mapping_index2traj, dtype=np.int64)[rollout_indices]
    payload: dict[str, np.ndarray] = {
        "rollout_trajectory_index": mapping[:, 0],
        "rollout_start_step": mapping[:, 1],
    }
    summary = {}
    for entry, model in zip(entries, models, strict=True):
        label = _model_label(entry)
        trainer.neural_solver.set_neural_solver_model(model)
        model.eval()
        np.random.seed(args_cli.rollout_seed)
        diff, _trajectories, error_stats = evaluator.evaluate_joint_f_mode(
            num_traj=args_cli.rollout_count,
            eval_mode="rollout",
            trajectory_source="dataset",
            passive=False,
            render=False,
        )
        arrays = _rollout_metric_arrays(diff, trainer.neural_solver.dof_q_per_env)
        summary[label] = {name: _summary(values) for name, values in arrays.items()}
        summary[label]["step_wise"] = {
            name: [float(value) for value in values.detach().cpu()] for name, values in error_stats["step-wise"].items()
        }
        for name, values in arrays.items():
            payload[f"rollout_{label}_{name}"] = values
    if hasattr(dataset, "close"):
        dataset.close()
    return payload, summary


@hydra_task_config(args_cli.task, "")
def main(env_cfg, _agent_cfg=None) -> None:
    """Run one immutable-suite evaluation."""
    if args_cli.batch_size <= 0 or args_cli.num_workers < 0 or args_cli.num_envs <= 0:
        raise ValueError("Batch size and environment count must be positive; workers must be non-negative.")
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    result_json = output_dir / "metrics.json"
    result_npz = output_dir / "per_sample_metrics.npz"
    if (result_json.exists() or result_npz.exists()) and not args_cli.force_overwrite:
        raise FileExistsError(f"Evaluation output already exists under {output_dir}.")
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_verified_checkpoints(args_cli.checkpoint_manifest, args_cli.encoder)
    entries = [entry for entry, _checkpoint in loaded]
    checkpoints = [checkpoint for _entry, checkpoint in loaded]
    dataset_path = Path(args_cli.dataset).expanduser().resolve()
    suite = load_verified_suite(args_cli.dataset_manifest, args_cli.suite_id, dataset_path)
    cfg = prepare_eval_cfg(
        checkpoints[0]["cfg"],
        encoder=args_cli.encoder,
        dataset_path=dataset_path,
        dataset_representation=suite["contact_representation"],
        batch_size=args_cli.batch_size,
        num_workers=args_cli.num_workers,
        num_envs=args_cli.num_envs,
        seed=args_cli.seed,
    )
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
            reconstruct_model_from_checkpoint(checkpoint, trainer.neural_solver, device=args_cli.device)
            for checkpoint in checkpoints[1:]
        )
        for model in models:
            model.eval()

        selected_indices = deterministic_indices(
            len(trainer.train_dataset),
            args_cli.max_windows,
            args_cli.window_seed,
        )
        mapping = np.asarray(trainer.train_dataset.mapping_index2traj, dtype=np.int64)[selected_indices]
        payload: dict[str, np.ndarray] = {
            "window_index": selected_indices,
            "trajectory_index": mapping[:, 0],
            "start_step": mapping[:, 1],
        }
        payload.update(_context_payload(dataset_path, mapping[:, 0]))
        one_step_payload, one_step_summary = _one_step_metrics(
            trainer,
            models,
            entries,
            selected_indices,
        )
        payload.update(one_step_payload)
        rollout_summary = None
        if args_cli.rollout_count:
            rollout_payload, rollout_summary = _rollout_metrics(
                trainer,
                models,
                entries,
                dataset_path,
                suite["contact_representation"],
                suite["contact_representation"] == "raw15_tokens",
            )
            payload.update(rollout_payload)

        with result_npz.with_suffix(".npz.tmp").open("wb") as stream:
            np.savez_compressed(stream, **payload)
        result_npz.with_suffix(".npz.tmp").replace(result_npz)
        result = {
            "schema_version": 1,
            "encoder": args_cli.encoder,
            "suite": suite,
            "checkpoint_epoch": 199,
            "checkpoints": {
                _model_label(entry): {
                    "group": entry["group"],
                    "seed": int(entry["seed"]),
                    "wandb_run_id": entry["wandb_run_id"],
                    "sha256": entry["sha256"],
                }
                for entry in entries
            },
            "selection": {
                "num_windows": int(selected_indices.size),
                "total_windows": int(len(trainer.train_dataset)),
                "window_seed": args_cli.window_seed,
                "rollout_count": args_cli.rollout_count,
                "rollout_horizon": args_cli.rollout_horizon if args_cli.rollout_count else 0,
                "rollout_seed": args_cli.rollout_seed if args_cli.rollout_count else None,
            },
            "one_step": one_step_summary,
            "rollout": rollout_summary,
            "per_sample_metrics": result_npz.name,
        }
        temporary_json = result_json.with_suffix(".json.tmp")
        temporary_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary_json.replace(result_json)
        if hasattr(trainer.train_dataset, "close"):
            trainer.train_dataset.close()
        env.close()
        print(f"[sampling-eval] wrote {result_json}")
        print(f"[sampling-eval] wrote {result_npz}")


if __name__ == "__main__":
    set_random_seed(args_cli.seed)
    main()  # type: ignore[call-arg]
