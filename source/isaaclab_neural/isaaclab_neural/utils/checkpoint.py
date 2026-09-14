# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import importlib
import importlib.metadata as metadata
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import yaml
from isaaclab_neural.models.models import ModelMixedInput
from isaaclab_neural.utils.running_mean_std import RunningMeanStd

from isaaclab.utils.dict import class_to_dict

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg

CHECKPOINT_VERSION = 2


def load_saved_agent_cfg_for_policy(
    checkpoint_path: str,
    fallback_agent_cfg: Mapping[str, Any] | RslRlBaseRunnerCfg | None,
    *,
    device: str,
    seed: int | None,
) -> Mapping[str, Any] | RslRlBaseRunnerCfg | None:
    """Load the policy checkpoint's adjacent ``params/agent.yaml`` when available.

    Args:
        checkpoint_path: Path to the policy checkpoint.
        fallback_agent_cfg: Task configuration used if no saved configuration exists.
        device: Device used for playback instead of the saved training device.
        seed: Explicit seed override, or ``None`` to preserve the saved seed.

    Returns:
        The saved runner configuration or the unchanged fallback configuration.

    Raises:
        ValueError: If the saved configuration is not a mapping.
    """
    agent_cfg_path = Path(checkpoint_path).expanduser().resolve().parent / "params" / "agent.yaml"
    if not agent_cfg_path.exists():
        return fallback_agent_cfg

    with agent_cfg_path.open("r", encoding="utf-8") as f:
        agent_cfg = yaml.safe_load(f)
    if not isinstance(agent_cfg, Mapping):
        raise ValueError(f"Expected a mapping in saved RSL-RL agent config: {agent_cfg_path}")

    agent_cfg = copy.deepcopy(dict(agent_cfg))
    agent_cfg["device"] = device
    if seed is not None:
        agent_cfg["seed"] = seed
    print(f"[policy] using saved RSL-RL agent config: {agent_cfg_path}")
    return agent_cfg


def normalize_runner_cfg(
    agent_cfg: Mapping[str, Any] | RslRlBaseRunnerCfg, *, device: str
) -> tuple[dict[str, Any], str, float | None, str]:
    """Normalize saved or registered RSL-RL configuration for policy playback.

    Args:
        agent_cfg: Saved runner mapping or registered runner configuration.
        device: Runtime device overriding the training device.

    Returns:
        Runner configuration, runner class name, action clipping limit, and device.
    """
    if isinstance(agent_cfg, Mapping):
        runner_cfg = copy.deepcopy(dict(agent_cfg))
        runner_cfg["device"] = device
        return (
            runner_cfg,
            runner_cfg.get("class_name", "OnPolicyRunner"),
            runner_cfg.get("clip_actions"),
            runner_cfg.get("device", device),
        )

    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    normalized_cfg: RslRlBaseRunnerCfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    normalized_cfg.device = device
    runner_cfg = class_to_dict(normalized_cfg)
    return (
        runner_cfg,
        getattr(normalized_cfg, "class_name", "OnPolicyRunner"),
        getattr(normalized_cfg, "clip_actions", None),
        getattr(normalized_cfg, "device", device),
    )


def install_legacy_module_aliases() -> None:
    """Alias old NeRD module paths so legacy pickled checkpoints can load."""
    aliases = {
        "models": "isaaclab_neural.models",
        "models.base_models": "isaaclab_neural.models.base_models",
        "models.distributions": "isaaclab_neural.models.distributions",
        "models.model_kan": "isaaclab_neural.models.model_kan",
        "models.model_transformer": "isaaclab_neural.models.model_transformer",
        "models.model_utils": "isaaclab_neural.models.model_utils",
        "models.models": "isaaclab_neural.models.models",
        "models.spatial_softmax": "isaaclab_neural.models.spatial_softmax",
        "utils": "isaaclab_neural.utils",
        "utils.running_mean_std": "isaaclab_neural.utils.running_mean_std",
    }
    for legacy_name, new_name in aliases.items():
        if legacy_name not in sys.modules:
            sys.modules[legacy_name] = importlib.import_module(new_name)
        if "." in legacy_name:
            parent_name, child_name = legacy_name.rsplit(".", 1)
            parent = sys.modules.get(parent_name)
            if parent is not None and not hasattr(parent, child_name):
                setattr(parent, child_name, sys.modules[legacy_name])


def save_checkpoint(path, model, robot_name, cfg, **training_state):
    """Save a NeRD checkpoint with state_dict and reconstruction metadata."""
    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "model_state_dict": model.state_dict(),
        "rms_info": model.get_rms_info(),
        "robot_name": robot_name,
        "cfg": cfg,
    }
    for key in [
        "optimizer_state_dict",
        "epoch",
        "best_valid_losses",
        "best_eval_error",
        "loss_weights",
        "contact_rms_counts",
    ]:
        if key in training_state:
            checkpoint[key] = training_state[key]
    torch.save(checkpoint, path)


def load_checkpoint(path, device="cpu"):
    """Load a NeRD checkpoint, detecting legacy and v2 formats."""
    install_legacy_module_aliases()
    raw = torch.load(path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "version" in raw:
        return raw
    if isinstance(raw, list):
        return {
            "version": 1,
            "legacy_model": raw[0],
            "robot_name": raw[1],
        }
    raise ValueError(f"Unrecognized checkpoint format at {path}")


def get_cfg_from_checkpoint(checkpoint, model_path):
    """Extract training config from checkpoint, falling back to adjacent cfg.yaml."""
    if checkpoint.get("cfg") is not None:
        return checkpoint["cfg"]
    train_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(model_path)), "../"))
    cfg_path = os.path.join(train_dir, "cfg.yaml")
    with open(cfg_path) as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def setup_rms_from_info(model, rms_info, device="cpu"):
    """Create RMS module structure on model from saved rms_info."""
    if "input_rms" in rms_info:
        rms_dict = {}
        for key, shape in rms_info["input_rms"].items():
            rms_dict[key] = RunningMeanStd(shape=tuple(shape), device=device)
        model.input_rms = nn.ModuleDict(rms_dict)
    if "output_rms" in rms_info:
        shape = rms_info["output_rms"]
        model.output_rms = RunningMeanStd(shape=tuple(shape), device=device)
    if "body_net_contact_force_rms" in rms_info:
        shape = rms_info["body_net_contact_force_rms"]
        model.body_net_contact_force_rms = RunningMeanStd(shape=tuple(shape), device=device)


def reconstruct_model_from_checkpoint(checkpoint, neural_solver, device="cpu"):
    """Reconstruct a neural model from a loaded NeRD checkpoint."""
    if checkpoint["version"] == 1:
        model = checkpoint["legacy_model"]
        model.to(device)
        model.eval()
        return model

    cfg = checkpoint["cfg"]
    input_sample = neural_solver.get_neural_model_inputs()
    output_dim = neural_solver.prediction_dim
    model = ModelMixedInput(
        input_sample=input_sample,
        output_dim=output_dim,
        input_cfg=cfg["inputs"],
        network_cfg=cfg["network"],
        contact_mode=neural_solver.contact_mode,
        num_bodies=getattr(
            neural_solver,
            "num_contact_bodies_per_env",
            getattr(neural_solver, "num_bodies_per_env", None),
        ),
        device=device,
    )

    setup_rms_from_info(model, checkpoint.get("rms_info", {}), device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def reconstruct_model_from_path(path, neural_solver, device="cpu"):
    """Load and reconstruct a NeRD model in one call."""
    checkpoint = load_checkpoint(path, device=device)
    return reconstruct_model_from_checkpoint(checkpoint, neural_solver, device=device)
