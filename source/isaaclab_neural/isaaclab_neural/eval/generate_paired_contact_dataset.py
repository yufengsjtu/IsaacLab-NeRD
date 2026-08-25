# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate transition-paired Raw15 and ContactTokens policy-validation data."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import isaaclab_neural.envs  # noqa: F401 - registers NeRD tasks
from isaaclab_neural.contacts.contact_set_schema import (
    CONTACT_REPRESENTATION_RAW15,
    CONTACT_REPRESENTATION_TOKENS,
)
from isaaclab_neural.eval.paired_contact_dataset import (
    PairedActionTrajectorySampler,
    PairedDataGenerationAdapter,
    sha256_file,
    split_paired_rollouts,
    truncation_summary_delta,
    validate_paired_hdf5_files,
)
from isaaclab_neural.generate.arguments import get_parser
from isaaclab_neural.utils.python_utils import set_random_seed

from isaaclab_tasks.utils.hydra import hydra_task_config


def _configure_env(env_cfg, args) -> None:
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    terrain_cfg = getattr(env_cfg.scene.terrain, "terrain_generator", None)
    if terrain_cfg is not None:
        terrain_cfg.seed = args.seed
    env_cfg.sim.device = args.device

    from isaaclab_newton.physics import MJWarpSolverCfg
    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

    if not isinstance(env_cfg.sim.physics, NewtonCfg):
        raise ValueError("Paired dataset generation requires a Newton physics preset.")
    if not isinstance(env_cfg.sim.physics.solver_cfg, MJWarpSolverCfg):
        raise ValueError("Paired dataset generation requires the MJWarp solver.")
    env_cfg.sim.physics.solver_cfg.use_mujoco_contacts = False


def _build_solver_cfg(args):
    from isaaclab_neural.contacts import resolve_contact_packing_policy
    from isaaclab_neural.physics import NerdSolverCfg

    packing_policy = resolve_contact_packing_policy("newton_native", args.contact_packing_policy)
    return NerdSolverCfg(
        name="NeuralSolver",
        states_frame=args.states_frame,
        anchor_frame_step=args.anchor_frame_step,
        states_embedding_type=args.states_embedding_type,
        prediction_type=args.prediction_type,
        orientation_prediction_parameterization=args.orientation_prediction_parameterization,
        min_contact_event_threshold=args.min_contact_event_threshold,
        num_contacts_per_env=args.num_contacts_per_env,
        contact_mode="newton_native",
        contact_packing_policy=packing_policy,
        contact_representation=CONTACT_REPRESENTATION_RAW15,
        max_contact_tokens=args.max_contact_tokens,
    )


def _output_paths(args) -> tuple[Path, Path, Path]:
    output_dir = Path(args.dataset_dir).expanduser()
    return (
        output_dir / args.raw15_env_name / args.dataset_name,
        output_dir / args.contact_tokens_env_name / args.paired_dataset_name,
        output_dir / args.manifest_name,
    )


def _prepare_outputs(paths: tuple[Path, ...], *, force_overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force_overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Paired dataset outputs already exist: {joined}.")
    if force_overwrite:
        for path in existing:
            path.unlink()
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


parser = get_parser()
parser.description = __doc__
parser.set_defaults(
    task="Isaac-Velocity-Rough-Anymal-C-v0",
    env_name=None,
    robot_name="Anymal-C",
    sample_mode="policy",
    initial_states_source="env",
    contact_mode="newton_native",
    contact_packing_policy="body_round_robin_pair_atomic",
    contact_representation=CONTACT_REPRESENTATION_RAW15,
    max_contact_tokens=64,
    num_contacts_per_env=64,
    num_envs=1024,
    num_transitions=1_000_000,
    trajectory_length=400,
    write_chunk_transitions=409_600,
    seed=40,
    data_device="cpu",
)
parser.add_argument(
    "--paired-dataset-name",
    default="dataset_lstm_actuator_policy_valid.hdf5",
    help="Filename under the ContactTokens output directory.",
)
parser.add_argument(
    "--raw15-env-name",
    default="Anymal-C-Rough-Native-Raw15-Paired-Eval",
    help="Raw15 output directory and HDF5 env metadata.",
)
parser.add_argument(
    "--contact-tokens-env-name",
    default="Anymal-C-Rough-Native-ContactTokens-Paired-Eval",
    help="ContactTokens output directory and HDF5 env metadata.",
)
parser.add_argument(
    "--manifest-name",
    default="paired_contact_policy_eval_manifest.json",
    help="Pairing/provenance manifest relative to dataset-dir.",
)
parser.set_defaults(dataset_name="dataset_lstm_actuator_policy_valid.hdf5")
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], "presets=newton_mjwarp", *hydra_args]


@hydra_task_config(args_cli.task, args_cli.policy_agent)
def main(env_cfg, agent_cfg=None) -> None:
    """Generate and validate the two paired HDF5 files."""
    if args_cli.sample_mode != "policy":
        raise ValueError("This standalone generator only supports --sample-mode policy.")
    if args_cli.policy_checkpoint is None:
        raise ValueError("--policy-checkpoint is required.")
    if args_cli.contact_representation != CONTACT_REPRESENTATION_RAW15:
        raise ValueError("The primary representation must remain raw15_tokens.")
    if args_cli.randomize_pd_gains:
        raise ValueError("The deployment policy-validation suite does not use extra PD gain randomization.")
    if args_cli.write_chunk_transitions <= 0:
        raise ValueError("--write-chunk-transitions must be positive.")

    import gymnasium as gym
    import torch
    from isaaclab_neural.data import append_rollouts_to_hdf5, build_terrain_context
    from isaaclab_neural.eval.eval import build_policy_env, load_saved_agent_cfg_for_policy
    from isaaclab_neural.utils.commons import JOINT_F_LIM, JOINT_Q_MAX, JOINT_Q_MIN, JOINT_QD_LIM
    from isaaclab_neural.utils.usd_utils import newton_material_binding_api_autofix

    from isaaclab_tasks.utils import launch_simulation

    raw15_path, contact_tokens_path, manifest_path = _output_paths(args_cli)
    _prepare_outputs(
        (raw15_path, contact_tokens_path, manifest_path),
        force_overwrite=args_cli.force_overwrite,
    )
    policy_path = Path(args_cli.policy_checkpoint).expanduser()
    if not policy_path.is_file():
        raise FileNotFoundError(f"Policy checkpoint does not exist: {policy_path}.")
    policy_agent_config_path = policy_path.parent / "params" / "agent.yaml"
    if not policy_agent_config_path.is_file():
        raise FileNotFoundError(f"Policy agent configuration does not exist: {policy_agent_config_path}.")

    _configure_env(env_cfg, args_cli)
    solver_cfg = _build_solver_cfg(args_cli)
    with launch_simulation(env_cfg, args_cli):
        with newton_material_binding_api_autofix():
            env = gym.make(args_cli.task, cfg=env_cfg, device=args_cli.device).unwrapped
            env.reset()
        terrain_context = build_terrain_context(env_cfg, args_cli.seed)
        adapter = PairedDataGenerationAdapter(
            env,
            solver_cfg,
            paired_contact_representation=CONTACT_REPRESENTATION_TOKENS,
        )
        sampler = PairedActionTrajectorySampler(
            adapter,
            trajectory_length=args_cli.trajectory_length,
            data_device=args_cli.data_device,
            action_low=args_cli.action_low,
            action_high=args_cli.action_high,
            joint_q_min=JOINT_Q_MIN["Anymal-C"],
            joint_q_max=JOINT_Q_MAX["Anymal-C"],
            joint_qd_lim=JOINT_QD_LIM["Anymal-C"],
            joint_f_lim=JOINT_F_LIM["Anymal-C"],
        )
        policy_agent_cfg = load_saved_agent_cfg_for_policy(
            str(policy_path),
            agent_cfg,
            device=args_cli.device,
            seed=args_cli.seed,
        )
        env_wrapped, policy, runner = build_policy_env(
            env,
            policy_agent_cfg,
            str(policy_path),
            args_cli.device,
        )
        raw15_truncation_baseline = adapter.contact_truncation_summary()
        contact_tokens_truncation_baseline = adapter.paired_contact_truncation_summary()
        if raw15_truncation_baseline is None:
            raise RuntimeError("Raw15 contact truncation counters are unavailable.")

        total_transitions = 0
        while total_transitions < args_cli.num_transitions:
            requested_chunk = min(
                args_cli.write_chunk_transitions,
                args_cli.num_transitions - total_transitions,
            )
            combined = sampler.sample_trajectories_policy_mode(
                requested_chunk,
                policy=policy,
                env_wrapped=env_wrapped,
                initial_states_source="env",
                render=args_cli.render,
            )
            raw15_rollouts, contact_token_rollouts = split_paired_rollouts(combined)
            append_rollouts_to_hdf5(
                raw15_path,
                raw15_rollouts,
                args_cli.raw15_env_name,
                terrain_context=terrain_context,
                contact_representation=CONTACT_REPRESENTATION_RAW15,
            )
            append_rollouts_to_hdf5(
                contact_tokens_path,
                contact_token_rollouts,
                args_cli.contact_tokens_env_name,
                terrain_context=terrain_context,
                contact_representation=CONTACT_REPRESENTATION_TOKENS,
            )
            written = int(combined["states"].shape[0] * combined["states"].shape[1])
            total_transitions += written
            print(f"[paired-dataset] appended {written} transitions ({total_transitions} total).")
            del combined, raw15_rollouts, contact_token_rollouts
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        pairing = validate_paired_hdf5_files(raw15_path, contact_tokens_path)
        raw15_truncation = truncation_summary_delta(
            adapter.contact_truncation_summary() or {},
            raw15_truncation_baseline,
        )
        contact_tokens_truncation = truncation_summary_delta(
            adapter.paired_contact_truncation_summary(),
            contact_tokens_truncation_baseline,
        )
        for counter in ("frames", "raw_contacts"):
            if raw15_truncation[counter] != contact_tokens_truncation[counter]:
                raise RuntimeError(
                    f"Paired contact encoders observed different {counter}: "
                    f"{raw15_truncation[counter]} != {contact_tokens_truncation[counter]}."
                )
        manifest = {
            "schema_version": 1,
            "requested_transitions": int(args_cli.num_transitions),
            "actual_transitions": total_transitions,
            "num_envs": int(args_cli.num_envs),
            "trajectory_length": int(args_cli.trajectory_length),
            "seed": int(args_cli.seed),
            "randomize_pd_gains": False,
            "task": args_cli.task,
            "contact_packing_policy": args_cli.contact_packing_policy,
            "max_contact_tokens": int(args_cli.max_contact_tokens),
            "policy_checkpoint": {
                "path": str(policy_path),
                "size_bytes": policy_path.stat().st_size,
                "sha256": sha256_file(policy_path),
            },
            "policy_agent_config": {
                "path": str(policy_agent_config_path),
                "size_bytes": policy_agent_config_path.stat().st_size,
                "sha256": sha256_file(policy_agent_config_path),
            },
            "raw15": {
                "path": str(raw15_path),
                "size_bytes": raw15_path.stat().st_size,
                "sha256": sha256_file(raw15_path),
            },
            "contact_tokens": {
                "path": str(contact_tokens_path),
                "size_bytes": contact_tokens_path.stat().st_size,
                "sha256": sha256_file(contact_tokens_path),
            },
            "pairing": pairing,
            "truncation": {
                "raw15": raw15_truncation,
                "contact_tokens": contact_tokens_truncation,
            },
        }
        temporary_manifest = manifest_path.with_suffix(f"{manifest_path.suffix}.tmp")
        temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary_manifest.replace(manifest_path)
        print(f"[paired-dataset] wrote {raw15_path}")
        print(f"[paired-dataset] wrote {contact_tokens_path}")
        print(f"[paired-dataset] wrote {manifest_path}")
        del runner
        env.close()


if __name__ == "__main__":
    set_random_seed(args_cli.seed)
    main()  # type: ignore[call-arg]
