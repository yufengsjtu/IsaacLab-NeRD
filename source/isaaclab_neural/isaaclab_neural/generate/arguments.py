# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line arguments for NeRD dataset generation."""

from __future__ import annotations

import argparse


def get_parser() -> argparse.ArgumentParser:
    """Create the NeRD dataset-generation argument parser."""
    parser = argparse.ArgumentParser(description="Generate NeRD HDF5 trajectory datasets from IsaacLab/Newton.")
    parser.add_argument(
        "--task", type=str, default="Isaac-Cartpole-v0", help="Registered ground-truth IsaacLab task id."
    )
    parser.add_argument(
        "--dataset-dir", type=str, default="./data/datasets", help="Directory to store generated datasets."
    )
    parser.add_argument("--dataset-name", type=str, default="dataset_train.hdf5", help="Generated HDF5 filename.")
    parser.add_argument("--env-name", type=str, default=None, help="Metadata env name stored in the HDF5 file.")
    parser.add_argument("--robot-name", type=str, default=None, help="Robot key used for default sampling ranges.")
    parser.add_argument("--sample-mode", choices=["action", "joint_f", "policy"], default="action")
    parser.add_argument(
        "--initial-states-source",
        choices=["sample", "input_states_pool", "env"],
        default="env",
        help="Source of trajectory initial generalized states.",
    )
    parser.add_argument(
        "--initial-states-pool", type=str, default=None, help="Optional torch file with initial states."
    )
    parser.add_argument(
        "--policy-checkpoint", type=str, default=None, help="Optional RSL-RL policy checkpoint for policy mode."
    )
    parser.add_argument("--policy-agent", type=str, default="rsl_rl_cfg_entry_point")
    parser.add_argument("--step-granularity", choices=["frame", "env"], default="frame")
    parser.add_argument("--num-transitions", type=int, default=100_000, help="Number of transitions to collect.")
    parser.add_argument("--trajectory-length", type=int, default=100, help="Number of env steps per trajectory.")
    parser.add_argument("--num-envs", type=int, default=1024, help="Number of vectorized environments.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--data-device", type=str, default=None, help="Device used to stage generated tensors before HDF5."
    )
    parser.add_argument(
        "--write-chunk-transitions",
        type=int,
        default=10000000,
        help="Append generated rollouts to HDF5 every N transitions. Disabled when 0.",
    )
    parser.add_argument("--zero-actions", action="store_true", help="Generate passive trajectories with zero actions.")
    parser.add_argument("--action-low", type=float, default=-1.0, help="Fallback lower action sampling bound.")
    parser.add_argument("--action-high", type=float, default=1.0, help="Fallback upper action sampling bound.")
    parser.add_argument(
        "--randomize-pd-gains", action="store_true", help="Randomize Anymal-C actuator gains per frame."
    )
    parser.add_argument("--kp-min", type=float, default=20.0, help="Minimum randomized actuator stiffness.")
    parser.add_argument("--kp-max", type=float, default=80.0, help="Maximum randomized actuator stiffness.")
    parser.add_argument("--kd-min", type=float, default=0.5, help="Minimum randomized actuator damping.")
    parser.add_argument("--kd-max", type=float, default=4.0, help="Maximum randomized actuator damping.")
    parser.add_argument("--render", action="store_true", help="Render while generating trajectories.")
    parser.add_argument("--contact-mode", choices=["fixed_ground", "newton_native"], default="fixed_ground")
    parser.add_argument("--num-contacts-per-env", type=int, default=0)
    parser.add_argument(
        "--contact-packing-policy",
        choices=["stable_index", "penetration_priority", "random", "force_priority"],
        default=None,
        help="Contact slot ordering. Defaults to penetration_priority for newton_native and stable_index otherwise.",
    )
    parser.add_argument("--states-frame", choices=["world", "body", "body_translation_only"], default="body")
    parser.add_argument("--anchor-frame-step", choices=["first", "last", "every"], default="every")
    parser.add_argument("--states-embedding-type", choices=["identical", "sinusoidal"], default="identical")
    parser.add_argument("--prediction-type", choices=["absolute", "relative", "acceleration"], default="relative")
    parser.add_argument(
        "--orientation-prediction-parameterization",
        choices=["quaternion", "exponential", "naive"],
        default="quaternion",
    )
    parser.add_argument("--min-contact-event-threshold", type=float, default=0.12)
    parser.add_argument(
        "--force-overwrite", action="store_true", help="Overwrite an existing dataset without prompting."
    )

    from isaaclab_tasks.utils import add_launcher_args

    add_launcher_args(parser)
    return parser
