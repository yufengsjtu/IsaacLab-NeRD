# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib
import os
import sys

import torch
import torch.nn as nn
import yaml
from isaaclab_neural.models.models import ModelMixedInput
from isaaclab_neural.utils.running_mean_std import RunningMeanStd

CHECKPOINT_VERSION = 2


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
