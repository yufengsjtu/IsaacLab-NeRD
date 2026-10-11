# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Expand a compact G1 mix recipe into per-source dataset specifications."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

MIX_SOURCE_KEYS = (
    "policy",
    "random",
    "zero",
    "policy_noise",
    "rl_ckpts",
    "policy_pool_perturbed",
)
DEFAULT_MIX_RATIOS = (2.0, 0.0, 0.0, 2.0, 2.0, 4.0)
DEFAULT_POLICY_CHECKPOINT_ITERS = (500, 1000, 1499)
DEFAULT_POLICY_CHECKPOINT_DIR = "pretrained/control_policy/g1_flat_terrain/checkpoint_mix"
# Lab-policy OSMO records every physics frame (dt=0.005, decimation=4).
# Clean/noise policy uses 250 control ticks → 1000 frames. Pool-perturb uses
# 100 physics frames with a new random action every frame, matching jiex
# ``--perturbed-trajectory-length 100``. Env-reset random/zero match
# generate_dataset_g1.py ``--trajectory-length 500``. The teacher state pool
# has the same transition count as the perturb train split, but each pool
# clip uses ``experiment["trajectory_length"]`` (1000 walking frames), not
# ``perturbed_trajectory_length``.
DEFAULT_PERTURBED_TRAJECTORY_LENGTH = 100
DEFAULT_RESET_TRAJECTORY_LENGTH = 500
DEFAULT_PERTURBED_RANDOM_FRACTION = 1.0
DEFAULT_VALID_TRANSITIONS = 250_000
DEFAULT_VALID_ALWAYS = ("policy", "random", "zero", "deployment")
POLICY_ACTION_NOISE_STD = 0.2
CLIP_POLICY_ACTION_NOISE = False

_TRAIN_SOURCE_FILENAMES = {
    "policy": "dataset_policy_train.hdf5",
    "random": "dataset_random_action_train.hdf5",
    "zero": "dataset_zero_action_train.hdf5",
    "policy_noise": "dataset_policy_noisy_train.hdf5",
}
_CORE_VALID_FILES = {
    "policy": ("exp_trajectory", "dataset_valid.hdf5"),
    "random": ("random_action_trajectory", "dataset_random_action_valid.hdf5"),
    "zero": ("zero_action_trajectory", "dataset_zero_action_valid.hdf5"),
    "deployment": ("deployment_policy_trajectory", "dataset_deployment_policy_valid.hdf5"),
}


def allocate_counts(total: int, weights: Sequence[float]) -> list[int]:
    """Split ``total`` by non-negative weights using the largest-remainder method."""
    if total < 0:
        raise ValueError("Transition counts must be non-negative.")
    raw_weights = np.asarray([float(weight) for weight in weights], dtype=np.float64)
    if raw_weights.ndim != 1:
        raise ValueError("Mix weights must be one-dimensional.")
    if not np.all(np.isfinite(raw_weights)) or np.any(raw_weights < 0.0):
        raise ValueError("Mix weights must be finite and non-negative.")
    if raw_weights.size == 0:
        return []
    if total == 0 or float(raw_weights.sum()) <= 0.0:
        return [0] * int(raw_weights.size)

    raw_counts = raw_weights / raw_weights.sum() * float(total)
    counts = np.floor(raw_counts).astype(np.int64)
    remainder = int(total) - int(counts.sum())
    if remainder:
        fractional_order = np.argsort(-(raw_counts - counts), kind="stable")
        counts[fractional_order[:remainder]] += 1
    return [int(count) for count in counts]


def parse_mix_ratios(dataset_mix: Mapping) -> dict[str, float]:
    """Return named mix ratios from a list or mapping."""
    if "ratios" in dataset_mix:
        values = list(dataset_mix["ratios"])
        if len(values) != len(MIX_SOURCE_KEYS):
            raise ValueError(
                f"dataset_mix.ratios must have {len(MIX_SOURCE_KEYS)} entries {MIX_SOURCE_KEYS}, got {len(values)}."
            )
        ratios = {key: float(value) for key, value in zip(MIX_SOURCE_KEYS, values)}
    else:
        ratios = {key: float(dataset_mix.get(key, 0.0)) for key in MIX_SOURCE_KEYS}

    if any(value < 0.0 or not np.isfinite(value) for value in ratios.values()):
        raise ValueError("dataset_mix ratios must be finite and non-negative.")
    if sum(ratios.values()) <= 0.0:
        raise ValueError("dataset_mix ratios must sum to a positive value.")
    return ratios


def parse_valid_config(dataset_mix: Mapping) -> tuple[int, tuple[str, ...]]:
    """Return per-file valid count and always-on core valid source names."""
    valid_cfg = dataset_mix.get("valid")
    if isinstance(valid_cfg, Mapping):
        transitions = int(valid_cfg.get("transitions", DEFAULT_VALID_TRANSITIONS))
        always_values = valid_cfg.get("always", DEFAULT_VALID_ALWAYS)
    else:
        transitions = int(dataset_mix.get("valid_transitions", DEFAULT_VALID_TRANSITIONS))
        always_values = DEFAULT_VALID_ALWAYS
    if transitions <= 0:
        raise ValueError("dataset_mix valid transitions must be positive.")
    always = tuple(str(value) for value in always_values)
    unknown = [key for key in always if key not in _CORE_VALID_FILES]
    if unknown:
        raise ValueError(f"dataset_mix.valid.always entries must be in {DEFAULT_VALID_ALWAYS}, got {unknown}.")
    ordered = tuple(key for key in DEFAULT_VALID_ALWAYS if key in set(always))
    if not ordered:
        raise ValueError("dataset_mix.valid.always must include at least one core valid source.")
    return transitions, ordered


def _checkpoint_filename(iteration: int) -> str:
    return f"dataset_policy_train_ckpt_{iteration:04d}.hdf5"


def _checkpoint_path(checkpoint_dir: str, iteration: int) -> str:
    return str(Path(checkpoint_dir) / f"model_{iteration}.pt")


def _spec(
    *,
    filename: str,
    seed: int,
    split: str,
    task_kind: str,
    zero_actions: bool,
    use_policy: bool,
    num_transitions: int,
    policy_checkpoint: str | None = None,
    policy_action_noise_std: float = 0.0,
    clip_policy_action_noise: bool | None = None,
    trajectory_length: int | None = None,
    initial_states_source: str | None = None,
    initial_states_pool: str | None = None,
    resample_actions_every_frame: bool = False,
) -> dict:
    item = {
        "filename": filename,
        "seed": seed,
        "split": split,
        "task_kind": task_kind,
        "zero_actions": zero_actions,
        "use_policy": use_policy,
        "num_transitions": int(num_transitions),
    }
    if policy_checkpoint is not None:
        item["policy_checkpoint"] = policy_checkpoint
    if policy_action_noise_std:
        item["policy_action_noise_std"] = float(policy_action_noise_std)
        item["clip_policy_action_noise"] = (
            CLIP_POLICY_ACTION_NOISE if clip_policy_action_noise is None else bool(clip_policy_action_noise)
        )
    if trajectory_length is not None:
        item["trajectory_length"] = int(trajectory_length)
    if initial_states_source is not None:
        item["initial_states_source"] = initial_states_source
    if initial_states_pool is not None:
        item["initial_states_pool"] = initial_states_pool
    if resample_actions_every_frame:
        item["resample_actions_every_frame"] = True
    return item


def expand_g1_dataset_mix(experiment: Mapping) -> list[dict]:
    """Expand ``experiment['dataset_mix']`` into ordered dataset spec mappings."""
    dataset_mix = experiment.get("dataset_mix")
    if not isinstance(dataset_mix, Mapping):
        raise ValueError("G1 mix presets must define a dataset_mix mapping.")

    ratios = parse_mix_ratios(dataset_mix)
    train_total = int(experiment["train_transitions"])
    if train_total <= 0:
        raise ValueError("train_transitions must be positive.")

    source_counts = {
        key: count
        for key, count in zip(
            MIX_SOURCE_KEYS,
            allocate_counts(train_total, [ratios[key] for key in MIX_SOURCE_KEYS]),
        )
    }
    trajectory_length = int(experiment["trajectory_length"])
    if trajectory_length <= 0:
        raise ValueError("trajectory_length must be positive.")
    perturbed_length = int(dataset_mix.get("perturbed_trajectory_length", DEFAULT_PERTURBED_TRAJECTORY_LENGTH))
    if perturbed_length <= 0:
        raise ValueError("perturbed_trajectory_length must be positive.")
    reset_length = int(dataset_mix.get("reset_trajectory_length", DEFAULT_RESET_TRAJECTORY_LENGTH))
    if reset_length <= 0:
        raise ValueError("reset_trajectory_length must be positive.")
    random_fraction = float(dataset_mix.get("perturbed_random_fraction", DEFAULT_PERTURBED_RANDOM_FRACTION))
    if not 0.0 <= random_fraction <= 1.0:
        raise ValueError("perturbed_random_fraction must be in [0, 1].")

    checkpoint_dir = str(dataset_mix.get("policy_checkpoint_dir", DEFAULT_POLICY_CHECKPOINT_DIR))
    checkpoint_iters = tuple(
        int(value) for value in dataset_mix.get("policy_checkpoint_iters", DEFAULT_POLICY_CHECKPOINT_ITERS)
    )
    if source_counts["rl_ckpts"] > 0 and not checkpoint_iters:
        raise ValueError("rl_ckpts mix weight requires at least one policy_checkpoint_iters entry.")

    valid_transitions, always_valid = parse_valid_config(dataset_mix)

    specs: list[dict] = []
    policy_train_filename = _TRAIN_SOURCE_FILENAMES["policy"]
    pool_filename = "dataset_policy_pool.hdf5"
    # Ignore leftover ``policy_pool_transitions``: the pool is always as large
    # as the perturb train split, and is collected even when clean policy is
    # also in the mix.
    if source_counts["policy_pool_perturbed"] > 0:
        specs.append(
            _spec(
                filename=pool_filename,
                seed=4,
                split="pool",
                task_kind="deployment",
                zero_actions=False,
                use_policy=True,
                num_transitions=source_counts["policy_pool_perturbed"],
                trajectory_length=trajectory_length,
            )
        )

    if source_counts["policy"] > 0:
        specs.append(
            _spec(
                filename=policy_train_filename,
                seed=0,
                split="train",
                task_kind="deployment",
                zero_actions=False,
                use_policy=True,
                num_transitions=source_counts["policy"],
            )
        )
    if source_counts["random"] > 0:
        specs.append(
            _spec(
                filename=_TRAIN_SOURCE_FILENAMES["random"],
                seed=1,
                split="train",
                task_kind="default",
                zero_actions=False,
                use_policy=False,
                num_transitions=source_counts["random"],
                trajectory_length=reset_length,
            )
        )
    if source_counts["zero"] > 0:
        specs.append(
            _spec(
                filename=_TRAIN_SOURCE_FILENAMES["zero"],
                seed=2,
                split="train",
                task_kind="default",
                zero_actions=True,
                use_policy=False,
                num_transitions=source_counts["zero"],
                trajectory_length=reset_length,
            )
        )
    if source_counts["policy_noise"] > 0:
        specs.append(
            _spec(
                filename=_TRAIN_SOURCE_FILENAMES["policy_noise"],
                seed=3,
                split="train",
                task_kind="deployment",
                zero_actions=False,
                use_policy=True,
                num_transitions=source_counts["policy_noise"],
                policy_action_noise_std=POLICY_ACTION_NOISE_STD,
            )
        )
    if source_counts["rl_ckpts"] > 0:
        ckpt_counts = allocate_counts(source_counts["rl_ckpts"], [1.0] * len(checkpoint_iters))
        for index, (iteration, count) in enumerate(zip(checkpoint_iters, ckpt_counts)):
            if count <= 0:
                continue
            specs.append(
                _spec(
                    filename=_checkpoint_filename(iteration),
                    seed=100 + index,
                    split="train",
                    task_kind="deployment",
                    zero_actions=False,
                    use_policy=True,
                    num_transitions=count,
                    policy_checkpoint=_checkpoint_path(checkpoint_dir, iteration),
                )
            )

    perturbed_random, perturbed_zero = allocate_counts(
        source_counts["policy_pool_perturbed"],
        [random_fraction, 1.0 - random_fraction],
    )
    if perturbed_random > 0:
        specs.append(
            _spec(
                filename="dataset_pool_perturbed_random_train.hdf5",
                seed=5,
                split="train",
                task_kind="default",
                zero_actions=False,
                use_policy=False,
                num_transitions=perturbed_random,
                trajectory_length=perturbed_length,
                initial_states_source="input_states_pool",
                initial_states_pool=pool_filename,
                resample_actions_every_frame=True,
            )
        )
    if perturbed_zero > 0:
        specs.append(
            _spec(
                filename="dataset_pool_perturbed_zero_train.hdf5",
                seed=6,
                split="train",
                task_kind="default",
                zero_actions=True,
                use_policy=False,
                num_transitions=perturbed_zero,
                trajectory_length=perturbed_length,
                initial_states_source="input_states_pool",
                initial_states_pool=pool_filename,
                resample_actions_every_frame=True,
            )
        )

    if "policy" in always_valid:
        specs.append(
            _spec(
                filename="dataset_valid.hdf5",
                seed=10,
                split="valid",
                task_kind="deployment",
                zero_actions=False,
                use_policy=True,
                num_transitions=valid_transitions,
            )
        )
    if "random" in always_valid:
        specs.append(
            _spec(
                filename="dataset_random_action_valid.hdf5",
                seed=11,
                split="valid",
                task_kind="default",
                zero_actions=False,
                use_policy=False,
                num_transitions=valid_transitions,
                trajectory_length=reset_length,
            )
        )
    if "zero" in always_valid:
        specs.append(
            _spec(
                filename="dataset_zero_action_valid.hdf5",
                seed=12,
                split="valid",
                task_kind="default",
                zero_actions=True,
                use_policy=False,
                num_transitions=valid_transitions,
                trajectory_length=reset_length,
            )
        )
    if source_counts["policy_noise"] > 0:
        specs.append(
            _spec(
                filename="dataset_policy_noisy_valid.hdf5",
                seed=20,
                split="valid",
                task_kind="deployment",
                zero_actions=False,
                use_policy=True,
                num_transitions=valid_transitions,
                policy_action_noise_std=POLICY_ACTION_NOISE_STD,
            )
        )
    if perturbed_random > 0:
        specs.append(
            _spec(
                filename="dataset_pool_perturbed_random_valid.hdf5",
                seed=21,
                split="valid",
                task_kind="default",
                zero_actions=False,
                use_policy=False,
                num_transitions=valid_transitions,
                trajectory_length=perturbed_length,
                initial_states_source="input_states_pool",
                initial_states_pool="dataset_valid.hdf5",
                resample_actions_every_frame=True,
            )
        )
    if perturbed_zero > 0:
        specs.append(
            _spec(
                filename="dataset_pool_perturbed_zero_valid.hdf5",
                seed=22,
                split="valid",
                task_kind="default",
                zero_actions=True,
                use_policy=False,
                num_transitions=valid_transitions,
                trajectory_length=perturbed_length,
                initial_states_source="input_states_pool",
                initial_states_pool="dataset_valid.hdf5",
                resample_actions_every_frame=True,
            )
        )
    if "deployment" in always_valid:
        specs.append(
            _spec(
                filename="dataset_deployment_policy_valid.hdf5",
                seed=40,
                split="valid",
                task_kind="deployment",
                zero_actions=False,
                use_policy=True,
                num_transitions=valid_transitions,
            )
        )
    return specs


def mix_train_dataset_entries(
    experiment: Mapping,
    *,
    env_dataset_dir: str = "./data/datasets/G1-Native-ContactTokens",
) -> dict[str, dict]:
    """Build train_datasets entries for positive mix weights."""
    dataset_mix = experiment["dataset_mix"]
    ratios = parse_mix_ratios(dataset_mix)
    train_total = int(experiment["train_transitions"])
    source_counts = {
        key: count
        for key, count in zip(
            MIX_SOURCE_KEYS,
            allocate_counts(train_total, [ratios[key] for key in MIX_SOURCE_KEYS]),
        )
    }
    entries: dict[str, dict] = {}
    if source_counts["policy"] > 0:
        entries["policy_trajectory"] = {
            "path": f"{env_dataset_dir}/dataset_policy_train.hdf5",
            "weight": ratios["policy"],
            "max_capacity": source_counts["policy"],
        }
    if source_counts["random"] > 0:
        entries["random_action_trajectory"] = {
            "path": f"{env_dataset_dir}/dataset_random_action_train.hdf5",
            "weight": ratios["random"],
            "max_capacity": source_counts["random"],
        }
    if source_counts["zero"] > 0:
        entries["zero_action_trajectory"] = {
            "path": f"{env_dataset_dir}/dataset_zero_action_train.hdf5",
            "weight": ratios["zero"],
            "max_capacity": source_counts["zero"],
        }
    if source_counts["policy_noise"] > 0:
        entries["policy_noisy_trajectory"] = {
            "path": f"{env_dataset_dir}/dataset_policy_noisy_train.hdf5",
            "weight": ratios["policy_noise"],
            "max_capacity": source_counts["policy_noise"],
        }
    if source_counts["rl_ckpts"] > 0:
        checkpoint_iters = tuple(
            int(value) for value in dataset_mix.get("policy_checkpoint_iters", DEFAULT_POLICY_CHECKPOINT_ITERS)
        )
        ckpt_counts = allocate_counts(source_counts["rl_ckpts"], [1.0] * len(checkpoint_iters))
        ckpt_weight = ratios["rl_ckpts"] / max(sum(1 for count in ckpt_counts if count > 0), 1)
        for iteration, count in zip(checkpoint_iters, ckpt_counts):
            if count <= 0:
                continue
            entries[f"policy_checkpoint_{iteration:04d}"] = {
                "path": f"{env_dataset_dir}/{_checkpoint_filename(iteration)}",
                "weight": ckpt_weight,
                "max_capacity": count,
            }
    random_fraction = float(dataset_mix.get("perturbed_random_fraction", DEFAULT_PERTURBED_RANDOM_FRACTION))
    perturbed_random, perturbed_zero = allocate_counts(
        source_counts["policy_pool_perturbed"],
        [random_fraction, 1.0 - random_fraction],
    )
    if perturbed_random > 0:
        entries["pool_perturbed_random"] = {
            "path": f"{env_dataset_dir}/dataset_pool_perturbed_random_train.hdf5",
            "weight": ratios["policy_pool_perturbed"] * random_fraction,
            "max_capacity": perturbed_random,
        }
    if perturbed_zero > 0:
        entries["pool_perturbed_zero"] = {
            "path": f"{env_dataset_dir}/dataset_pool_perturbed_zero_train.hdf5",
            "weight": ratios["policy_pool_perturbed"] * (1.0 - random_fraction),
            "max_capacity": perturbed_zero,
        }
    return entries


def mix_valid_dataset_entries(
    experiment: Mapping,
    *,
    env_dataset_dir: str = "./data/datasets/G1-Native-ContactTokens",
) -> dict[str, str]:
    """Build valid_datasets paths: core policy/random/zero/deployment plus enabled extras."""
    dataset_mix = experiment["dataset_mix"]
    ratios = parse_mix_ratios(dataset_mix)
    train_total = int(experiment["train_transitions"])
    source_counts = {
        key: count
        for key, count in zip(
            MIX_SOURCE_KEYS,
            allocate_counts(train_total, [ratios[key] for key in MIX_SOURCE_KEYS]),
        )
    }
    _, always_valid = parse_valid_config(dataset_mix)
    random_fraction = float(dataset_mix.get("perturbed_random_fraction", DEFAULT_PERTURBED_RANDOM_FRACTION))
    perturbed_random, perturbed_zero = allocate_counts(
        source_counts["policy_pool_perturbed"],
        [random_fraction, 1.0 - random_fraction],
    )
    entries: dict[str, str] = {}
    for key in always_valid:
        if key == "deployment":
            continue
        name, filename = _CORE_VALID_FILES[key]
        entries[name] = f"{env_dataset_dir}/{filename}"
    if source_counts["policy_noise"] > 0:
        entries["policy_noisy_trajectory"] = f"{env_dataset_dir}/dataset_policy_noisy_valid.hdf5"
    if perturbed_random > 0:
        entries["pool_perturbed_random"] = f"{env_dataset_dir}/dataset_pool_perturbed_random_valid.hdf5"
    if perturbed_zero > 0:
        entries["pool_perturbed_zero"] = f"{env_dataset_dir}/dataset_pool_perturbed_zero_valid.hdf5"
    if "deployment" in always_valid:
        name, filename = _CORE_VALID_FILES["deployment"]
        entries[name] = f"{env_dataset_dir}/{filename}"
    return entries
