# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kinematically replay recorded Franka HDF5 trajectories in Newton.

The script writes ``data/states`` back into the ground-truth Newton solver and
renders them. It does **not** re-simulate with a policy or actions, so random,
policy, and zero-action splits are all visible as collected. Recorded clips do
not store the EE pose command, so command markers are not the logged goals.

Example::

    PYTHONPATH="$(pwd)/source/isaaclab_neural:${PYTHONPATH}" ./isaaclab.sh -p \\
      -m isaaclab_neural.eval.replay_franka_dataset \\
      --dataset ./data/datasets/Franka-Reach/dataset_valid.hdf5 \\
      --num-envs 1 --loop --playback-hz 30 --viz newton
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import h5py
import isaaclab_neural.envs  # noqa: F401 - registers Franka dataset-generation tasks
import numpy as np
import torch

REACH_DATASET_GEN_TASK = "Isaac-Reach-Franka-Dataset-Gen-v0"
LIFT_DATASET_GEN_TASK = "Isaac-Lift-Cube-Franka-Dataset-Gen-v0"


@dataclass(frozen=True)
class ReplayDatasetInfo:
    """Metadata loaded from a NeRD trajectory HDF5 file."""

    path: Path
    env_name: str
    num_trajectories: int
    horizon: int
    state_dim: int
    traj_lengths: np.ndarray | None


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse replay and Isaac Lab launcher arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="HDF5 trajectory dataset to replay.")
    parser.add_argument(
        "--task",
        default=None,
        help=(
            "Registered ground-truth IsaacLab task. Defaults to Franka Dataset-Gen "
            "Reach or Lift based on the HDF5 env name."
        ),
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of trajectories shown at once.")
    parser.add_argument(
        "--trajectory-index",
        type=int,
        nargs="+",
        default=None,
        help="Explicit trajectory indices. A single index is broadcast to --num-envs.",
    )
    parser.add_argument("--start-step", type=int, default=0, help="First recorded frame to play.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of frames to play from each trajectory.",
    )
    parser.add_argument(
        "--playback-hz",
        type=float,
        default=30.0,
        help="Visualizer playback rate [Hz]. Franka Reach datasets record 60 Hz physics frames (dt=1/60 s).",
    )
    parser.add_argument("--loop", action="store_true", help="Keep sampling and replaying until interrupted.")
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.25,
        help="Pause between replay cycles when --loop is set.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for random trajectory sampling.")

    from isaaclab.app import add_launcher_args

    add_launcher_args(parser)
    return parser.parse_known_args()


def infer_replay_task(env_name: str | None = None, task: str | None = None) -> str:
    """Return the IsaacLab task used to host kinematic replay.

    Args:
        env_name: HDF5 ``data.attrs['env']`` value, if present.
        task: Optional explicit gym task id.

    Returns:
        A Dataset-Gen gym id, or ``task`` when it is provided.
    """
    if task:
        return task
    name = (env_name or "").lower()
    if "lift" in name:
        return LIFT_DATASET_GEN_TASK
    return REACH_DATASET_GEN_TASK


def sample_trajectory_indices(
    num_trajectories: int,
    num_envs: int,
    rng: np.random.Generator,
    trajectory_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """Sample or validate trajectory indices for a replay batch.

    Args:
        num_trajectories: Number of trajectories stored in the dataset.
        num_envs: Number of parallel environments / robots to show.
        rng: NumPy generator used for random sampling.
        trajectory_indices: Optional explicit indices. A single index is
            broadcast to ``num_envs``.

    Returns:
        Integer index array with shape ``[num_envs]``.
    """
    if num_trajectories <= 0:
        raise ValueError("Dataset contains no trajectories.")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive.")
    if trajectory_indices:
        indices = np.asarray(trajectory_indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= num_trajectories):
            raise ValueError(f"Trajectory indices {indices.tolist()} are outside [0, {num_trajectories}).")
        if indices.size == 1 and num_envs > 1:
            return np.full(num_envs, int(indices[0]), dtype=np.int64)
        if indices.size != num_envs:
            raise ValueError(f"Expected {num_envs} trajectory indices, got {indices.size}.")
        return indices
    replace = num_envs > num_trajectories
    return rng.choice(num_trajectories, size=num_envs, replace=replace).astype(np.int64)


def _decode_attr(value: Any) -> str:
    """Decode an HDF5 attribute to ``str``."""
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def read_replay_dataset_info(dataset_path: str | Path) -> ReplayDatasetInfo:
    """Read trajectory counts and state size from a NeRD HDF5 dataset.

    Args:
        dataset_path: Path to ``dataset_*.hdf5``.

    Returns:
        Dataset metadata used to sample and validate replay batches.
    """
    path = Path(dataset_path).expanduser()
    with h5py.File(path, "r", swmr=True, libver="latest") as dataset_file:
        if "data" not in dataset_file or "states" not in dataset_file["data"]:
            raise ValueError(f"Dataset {path} is missing data/states.")
        data_group = cast(h5py.Group, dataset_file["data"])
        states = cast(h5py.Dataset, data_group["states"])
        if states.ndim != 3:
            raise ValueError(f"data/states must have shape [N, T, D], got {tuple(states.shape)}.")
        traj_lengths = None
        if "traj_lengths" in data_group:
            traj_lengths = np.asarray(cast(h5py.Dataset, data_group["traj_lengths"])[()], dtype=np.int32)
        return ReplayDatasetInfo(
            path=path,
            env_name=_decode_attr(data_group.attrs.get("env", "")),
            num_trajectories=int(states.shape[0]),
            horizon=int(states.shape[1]),
            state_dim=int(states.shape[-1]),
            traj_lengths=traj_lengths,
        )


def _gather_rows(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    """Load selected trajectory rows, preserving duplicate indices."""
    indices = np.asarray(indices, dtype=np.int64)
    unique, inverse = np.unique(indices, return_inverse=True)
    rows = np.stack([np.asarray(dataset[int(index)]) for index in unique], axis=0)
    return rows[inverse]


def load_replay_batch(
    dataset_path: str | Path,
    indices: Sequence[int] | np.ndarray,
) -> dict[str, torch.Tensor]:
    """Load recorded states and optional trajectory context.

    Args:
        dataset_path: Path to ``dataset_*.hdf5``.
        indices: Trajectory indices, shape ``[num_envs]``.

    Returns:
        Dictionary with ``states`` shaped ``[num_envs, T, D]`` and optional
        ``traj_lengths``, ``env_origin``, ``terrain_level``, and ``terrain_type``.
    """
    indices_np = np.asarray(indices, dtype=np.int64)
    path = Path(dataset_path).expanduser()
    with h5py.File(path, "r", swmr=True, libver="latest") as dataset_file:
        data_group = cast(h5py.Group, dataset_file["data"])
        states = _gather_rows(cast(h5py.Dataset, data_group["states"]), indices_np)
        batch: dict[str, torch.Tensor] = {
            "states": torch.as_tensor(states, dtype=torch.float32),
        }
        if "traj_lengths" in data_group:
            lengths = _gather_rows(cast(h5py.Dataset, data_group["traj_lengths"]), indices_np)
            batch["traj_lengths"] = torch.as_tensor(lengths, dtype=torch.int32)
        if "context" in dataset_file:
            context_group = cast(h5py.Group, dataset_file["context"])
            if "trajectories" in context_group:
                trajectory_group = cast(h5py.Group, context_group["trajectories"])
                for key in ("env_origin", "terrain_level", "terrain_type"):
                    if key not in trajectory_group:
                        continue
                    array = _gather_rows(cast(h5py.Dataset, trajectory_group[key]), indices_np)
                    dtype = torch.float32 if key == "env_origin" else torch.int64
                    batch[key] = torch.as_tensor(array, dtype=dtype)
    return batch


def clip_and_pad_states(
    states: torch.Tensor,
    traj_lengths: np.ndarray | torch.Tensor | None,
    *,
    start_step: int = 0,
    max_steps: int | None = None,
) -> torch.Tensor:
    """Slice trajectories and hold the last valid frame when lengths differ.

    Args:
        states: Recorded generalized states [num_envs, horizon, state_dim].
        traj_lengths: Optional per-trajectory valid lengths.
        start_step: First frame to include.
        max_steps: Optional cap on the number of played frames.

    Returns:
        States shaped ``[num_envs, play_horizon, state_dim]``.
    """
    if states.ndim != 3:
        raise ValueError(f"states must have shape [N, T, D], got {tuple(states.shape)}.")
    if start_step < 0:
        raise ValueError("start_step must be non-negative.")
    num_envs, horizon, _ = states.shape
    if start_step >= horizon:
        raise ValueError(f"start_step {start_step} is outside trajectory horizon {horizon}.")
    if traj_lengths is None:
        lengths = np.full(num_envs, horizon, dtype=np.int32)
    else:
        if isinstance(traj_lengths, torch.Tensor):
            length_array = traj_lengths.detach().cpu().numpy()
        else:
            length_array = np.asarray(traj_lengths)
        lengths = length_array.astype(np.int32, copy=False).reshape(-1)
        if lengths.shape[0] != num_envs:
            raise ValueError(f"traj_lengths must have {num_envs} entries, got {lengths.shape[0]}.")
    play_horizon = horizon - start_step
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        play_horizon = min(play_horizon, max_steps)
    if play_horizon <= 0:
        raise ValueError(f"No frames to play: start_step={start_step}, horizon={horizon}.")
    play_end = start_step + play_horizon
    clipped = states[:, start_step:play_end].clone()
    for env_id, length in enumerate(lengths):
        valid = int(min(int(length), play_end) - start_step)
        if valid <= 0:
            raise ValueError(f"Trajectory {env_id} has no frames after start_step {start_step} (length={length}).")
        if valid < clipped.shape[1]:
            clipped[env_id, valid:] = clipped[env_id, valid - 1]
    return clipped


def apply_terrain_context(terrain: Any, context: Mapping[str, torch.Tensor]) -> None:
    """Write recorded terrain patch metadata onto a TerrainImporter.

    Flat/grid terrains only restore ``env_origins``. Curriculum terrains also
    restore ``terrain_levels`` / ``terrain_types`` when those values are not
    the Dataset-Gen sentinel ``-1``.

    Args:
        terrain: IsaacLab ``TerrainImporter``, or ``None``.
        context: Per-env tensors from ``context/trajectories``.
    """
    if terrain is None:
        return
    device = getattr(terrain, "device", "cpu")
    env_origin = context.get("env_origin")
    if env_origin is not None and getattr(terrain, "env_origins", None) is not None:
        terrain.env_origins[: env_origin.shape[0]].copy_(env_origin.to(device=device, dtype=terrain.env_origins.dtype))
    terrain_level = context.get("terrain_level")
    terrain_type = context.get("terrain_type")
    has_curriculum = (
        getattr(terrain, "terrain_levels", None) is not None and getattr(terrain, "terrain_types", None) is not None
    )
    if not has_curriculum or terrain_level is None or terrain_type is None:
        return
    if torch.all(terrain_level < 0):
        return
    count = terrain_level.shape[0]
    terrain.terrain_levels[:count].copy_(terrain_level.to(device=device, dtype=terrain.terrain_levels.dtype))
    terrain.terrain_types[:count].copy_(terrain_type.to(device=device, dtype=terrain.terrain_types.dtype))


def requested_newton_visualizer(args: argparse.Namespace) -> bool:
    """Return whether the launcher requested a Newton visualizer."""
    visualizers = getattr(args, "visualizer", None)
    if isinstance(visualizers, str):
        visualizers = visualizers.split(",")
    return bool(visualizers) and "newton" in {str(value).strip().lower() for value in visualizers}


def configure_env(env_cfg, args: argparse.Namespace) -> None:
    """Apply replay CLI overrides and require stock Newton MJWarp physics."""
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"):
        env_cfg.scene.num_envs = args.num_envs
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = args.seed
    if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "device") and getattr(args, "device", None) is not None:
        env_cfg.sim.device = args.device

    from isaaclab_neural.data import read_terrain_context, set_terrain_seed
    from isaaclab_neural.physics import NerdNewtonCfg

    terrain_context = read_terrain_context(args.dataset)
    if terrain_context is not None:
        set_terrain_seed(env_cfg, int(terrain_context["seed"]))

    from isaaclab_newton.physics import MJWarpSolverCfg
    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

    physics_cfg = getattr(getattr(env_cfg, "sim", None), "physics", None)
    if isinstance(physics_cfg, NerdNewtonCfg):
        raise ValueError(
            "Franka dataset replay writes recorded states into stock Newton. "
            "Use Isaac-Reach-Franka-Dataset-Gen-v0 or "
            "Isaac-Lift-Cube-Franka-Dataset-Gen-v0 with presets=newton_mjwarp, "
            "not a NeRD task."
        )
    if not isinstance(physics_cfg, NewtonCfg):
        raise ValueError("Franka dataset replay requires a Newton physics preset (presets=newton_mjwarp).")
    if not isinstance(physics_cfg.solver_cfg, MJWarpSolverCfg):
        raise ValueError("Franka dataset replay requires the MJWarp solver.")
    # Replay writes joint coordinates without ``sim.step``. A captured CUDA
    # graph would replay the reset physics buffers over those writes.
    physics_cfg.use_cuda_graph = False


def _build_replay_solver_cfg():
    """Return a contact-free recording solver used only to write Newton state."""
    from isaaclab_neural.physics import NerdSolverCfg

    return NerdSolverCfg(
        name="NeuralSolver",
        states_frame="world",
        num_contacts_per_env=0,
        contact_mode="newton_native",
    )


def _write_replay_frame(adapter, states: torch.Tensor) -> None:
    """Write one generalized-state snapshot into Newton and refresh the scene."""
    adapter.backend.assign_solver_states(adapter.solver, states)
    if hasattr(adapter.env, "scene"):
        adapter.env.scene.update(dt=float(getattr(adapter.env, "physics_dt", 0.0)))


def _pace_playback(args: argparse.Namespace, last_time: float) -> float:
    """Sleep so visualizer playback matches ``--playback-hz``."""
    if args.playback_hz <= 0.0 or not requested_newton_visualizer(args):
        return time.perf_counter()
    period = 1.0 / float(args.playback_hz)
    remaining = period - (time.perf_counter() - last_time)
    if remaining > 0.0:
        time.sleep(remaining)
    return time.perf_counter()


def run_replay(env_cfg, args: argparse.Namespace, info: ReplayDatasetInfo) -> None:
    """Launch the Dataset-Gen env and kinematically play selected trajectories."""
    import gymnasium as gym
    from isaaclab_neural.generate.adapter import DataGenerationAdapter
    from isaaclab_neural.utils.usd_utils import newton_material_binding_api_autofix

    from isaaclab.app import launch_simulation

    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive.")
    if args.playback_hz <= 0.0:
        raise ValueError("--playback-hz must be positive.")
    if args.pause_seconds < 0.0:
        raise ValueError("--pause-seconds must be non-negative.")

    configure_env(env_cfg, args)
    solver_cfg = _build_replay_solver_cfg()
    rng = np.random.default_rng(args.seed)
    print(
        f"[replay] dataset={info.path} env={info.env_name!r} task={args.task} "
        f"trajectories={info.num_trajectories} horizon={info.horizon} state_dim={info.state_dim}"
    )

    with launch_simulation(env_cfg, args):
        with newton_material_binding_api_autofix():
            env = gym.make(args.task, cfg=env_cfg, device=args.device).unwrapped
            env.reset()
        adapter = DataGenerationAdapter(env, solver_cfg)
        if adapter.state_dim != info.state_dim:
            raise ValueError(
                f"Dataset state_dim={info.state_dim} does not match env state_dim={adapter.state_dim}. "
                "Replay needs the same Dataset-Gen scene used during collection."
            )
        cycle = 0
        try:
            while True:
                indices = sample_trajectory_indices(
                    info.num_trajectories,
                    args.num_envs,
                    rng,
                    args.trajectory_index,
                )
                batch = load_replay_batch(info.path, indices)
                states = clip_and_pad_states(
                    batch["states"],
                    batch.get("traj_lengths"),
                    start_step=args.start_step,
                    max_steps=args.max_steps,
                ).to(device=adapter.device, dtype=torch.float32)
                apply_terrain_context(
                    getattr(getattr(adapter.env, "scene", None), "terrain", None),
                    {key: value for key, value in batch.items() if key != "states"},
                )
                print(f"[replay] cycle={cycle} indices={indices.tolist()} frames={states.shape[1]}")
                last_time = time.perf_counter()
                for step in range(states.shape[1]):
                    _write_replay_frame(adapter, states[:, step])
                    env.sim.render()
                    last_time = _pace_playback(args, last_time)
                cycle += 1
                if not args.loop:
                    break
                if args.pause_seconds > 0.0 and requested_newton_visualizer(args):
                    time.sleep(args.pause_seconds)
        except KeyboardInterrupt:
            print("[replay] interrupted")
        env.close()


def main() -> None:
    """Parse Hydra configuration and replay recorded Franka trajectories."""
    args, hydra_args = parse_args()
    info = read_replay_dataset_info(args.dataset)
    args.task = infer_replay_task(info.env_name, args.task)
    if not any(str(value).startswith("presets=") for value in hydra_args):
        hydra_args = ["presets=newton_mjwarp", *hydra_args]
    sys.argv = [sys.argv[0], *hydra_args]

    from isaaclab_tasks.utils.hydra import hydra_task_config

    @hydra_task_config(args.task, "")
    def hydra_main(env_cfg, agent_cfg=None) -> None:
        run_replay(env_cfg, args, info)

    hydra_main()


if __name__ == "__main__":
    main()
