# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for rough shared-per-body ablation configurations."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml
from isaaclab_neural.contacts.contact_set_schema import CONTACT_TOKEN_DIM
from isaaclab_neural.models.models import ModelMixedInput
from isaaclab_neural.train.trainers import VanillaTrainer
from isaaclab_neural.utils.torch_utils import num_params_torch_model

PACKAGE_ROOT = Path(__file__).parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
CFG_DIR = PACKAGE_ROOT / "isaaclab_neural" / "train" / "cfg" / "Anymal"
PRESET_DIR = REPOSITORY_ROOT / "osmo_scripts" / "presets"

ABLATION_CONDITIONS = {
    "large_20m": (
        "transformer_rough_native_shared_per_body.yaml",
        "anymal_rough_newton_native_shared_per_body.yaml",
        20_000_000,
        5_000,
        (6, 12, 384, 16, 64, [64]),
    ),
    "large_10m": (
        "transformer_rough_native_shared_per_body_10m.yaml",
        "anymal_rough_newton_native_shared_per_body_10m.yaml",
        10_000_000,
        2_500,
        (6, 12, 384, 16, 64, [64]),
    ),
    "medium_20m": (
        "transformer_rough_native_shared_per_body_medium.yaml",
        "anymal_rough_newton_native_shared_per_body_medium.yaml",
        20_000_000,
        5_000,
        (4, 8, 256, 12, 48, [48]),
    ),
    "medium_10m": (
        "transformer_rough_native_shared_per_body_medium_10m.yaml",
        "anymal_rough_newton_native_shared_per_body_medium_10m.yaml",
        10_000_000,
        2_500,
        (4, 8, 256, 12, 48, [48]),
    ),
    "small_20m": (
        "transformer_rough_native_shared_per_body_small.yaml",
        "anymal_rough_newton_native_shared_per_body_small.yaml",
        20_000_000,
        5_000,
        (3, 6, 192, 8, 32, [32]),
    ),
    "small_10m": (
        "transformer_rough_native_shared_per_body_small_10m.yaml",
        "anymal_rough_newton_native_shared_per_body_small_10m.yaml",
        10_000_000,
        2_500,
        (3, 6, 192, 8, 32, [32]),
    ),
}


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@pytest.mark.parametrize(
    ("condition", "condition_spec"),
    ABLATION_CONDITIONS.items(),
    ids=ABLATION_CONDITIONS,
)
def test_ablation_condition_matches_training_budget_and_preset(condition: str, condition_spec: tuple) -> None:
    del condition
    cfg_filename, preset_filename, max_capacity, num_iters, model_shape = condition_spec
    cfg = _load_yaml(CFG_DIR / cfg_filename)
    preset = _load_yaml(PRESET_DIR / preset_filename)
    transformer_cfg = cfg["network"]["transformer"]
    contact_cfg = cfg["inputs"]["contact_set"]

    assert cfg["algorithm"]["dataset"]["max_capacity"] == max_capacity
    assert cfg["algorithm"]["num_iters_per_epoch"] == num_iters
    assert (
        transformer_cfg["n_layer"],
        transformer_cfg["n_head"],
        transformer_cfg["n_embd"],
        contact_cfg["body_latent_dim"],
        contact_cfg["hidden_dim"],
        cfg["network"]["model"]["mlp"]["layer_sizes"],
    ) == model_shape
    assert transformer_cfg["n_embd"] % transformer_cfg["n_head"] == 0
    assert preset["experiment"]["train_transitions"] == 20_000_000
    assert preset["experiment"]["require_terrain_context"] is True
    assert Path(preset["experiment"]["train_cfg"]).name == cfg_filename


def test_ablation_model_parameter_counts_and_gradients_are_ordered() -> None:
    input_sample = {
        "states_embedding": torch.randn(1, 10, 37),
        "joint_f": torch.randn(1, 10, 12),
        "gravity_dir": torch.randn(1, 10, 3),
        "contact_tokens": torch.zeros(1, 10, 4, CONTACT_TOKEN_DIM),
    }
    input_sample["contact_tokens"][..., 0, 0] = 1.0
    input_sample["contact_tokens"][..., 0, 1] = 0.0
    parameter_counts = {}

    for tier, cfg_filename in {
        "large": "transformer_rough_native_shared_per_body.yaml",
        "medium": "transformer_rough_native_shared_per_body_medium.yaml",
        "small": "transformer_rough_native_shared_per_body_small.yaml",
    }.items():
        cfg = _load_yaml(CFG_DIR / cfg_filename)
        network_cfg = copy.deepcopy(cfg["network"])
        network_cfg["normalize_input"] = False
        network_cfg["normalize_output"] = False
        model = ModelMixedInput(
            input_sample=input_sample,
            output_dim=37,
            input_cfg=cfg["inputs"],
            network_cfg=network_cfg,
            contact_mode="newton_native",
            num_bodies=17,
            device="cpu",
        )
        output = model(input_sample)
        output.square().mean().backward()

        assert output.shape == (1, 10, 37)
        assert torch.isfinite(output).all()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()
        )
        parameter_counts[tier] = num_params_torch_model(model)

    assert 10_000_000 < parameter_counts["large"] < 12_000_000
    assert 3_000_000 < parameter_counts["medium"] < 4_000_000
    assert 1_000_000 < parameter_counts["small"] < 2_000_000
    assert parameter_counts["large"] > parameter_counts["medium"] > parameter_counts["small"]


def test_ablation_metadata_is_reported_to_wandb() -> None:
    cfg = _load_yaml(CFG_DIR / "transformer_rough_native_shared_per_body_medium_10m.yaml")
    cfg["algorithm"]["seed"] = 0
    trainer = VanillaTrainer.__new__(VanillaTrainer)
    trainer.cfg = cfg
    trainer.seed = 0
    trainer.num_iters_per_epoch = cfg["algorithm"]["num_iters_per_epoch"]
    trainer.dataset_max_capacity = cfg["algorithm"]["dataset"]["max_capacity"]
    trainer.neural_model = torch.nn.Linear(3, 2)

    metadata = trainer._wandb_config()

    assert metadata["seed"] == 0
    assert metadata["dataset_max_capacity"] == 10_000_000
    assert metadata["num_iters_per_epoch"] == 2_500
    assert metadata["model_num_parameters"] == 8
    assert metadata["contact_encoder_type"] == "shared_per_body"
    assert metadata["contact_body_latent_dim"] == 12
    assert metadata["contact_hidden_dim"] == 48
    assert metadata["transformer_n_layer"] == 4
    assert metadata["transformer_n_head"] == 8
    assert metadata["transformer_n_embd"] == 256
    assert metadata["model_mlp_layer_sizes"] == [48]
