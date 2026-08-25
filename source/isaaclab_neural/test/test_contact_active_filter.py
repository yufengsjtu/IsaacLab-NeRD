# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for solver-active views over reusable contact-token datasets."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml
from isaaclab_neural.contacts.contact_set_schema import (
    CONTACT_REPRESENTATION_RAW15,
    CONTACT_REPRESENTATION_TOKENS,
)
from isaaclab_neural.contacts.tensor_utils import filter_solver_active_contact_tokens
from isaaclab_neural.physics import NerdSolverCfg
from isaaclab_neural.solvers.factory import _valid_solver_args
from isaaclab_neural.solvers.transformer_neural_solver import TransformerNeuralSolver

PACKAGE_ROOT = Path(__file__).parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
CFG_DIR = PACKAGE_ROOT / "isaaclab_neural" / "train" / "cfg" / "Anymal"
PRESET_DIR = REPOSITORY_ROOT / "osmo_scripts" / "presets"


def _native15_tokens() -> torch.Tensor:
    tokens = torch.zeros(1, 1, 3, 17)
    tokens[..., :2, 0] = 1.0
    tokens[..., :2, 1] = 0.0
    tokens[0, 0, 0, 2:5] = torch.tensor([0.0, 0.0, 0.0])
    tokens[0, 0, 0, 5:8] = torch.tensor([0.0, 0.0, 0.02])
    tokens[0, 0, 0, 8:11] = torch.tensor([0.0, 0.0, -1.0])
    tokens[0, 0, 0, 15:17] = torch.tensor([0.03, 0.02])
    tokens[0, 0, 1, 2:5] = torch.tensor([0.0, 0.0, 0.0])
    tokens[0, 0, 1, 5:8] = torch.tensor([0.0, 0.0, 0.10])
    tokens[0, 0, 1, 8:11] = torch.tensor([0.0, 0.0, -1.0])
    tokens[0, 0, 1, 15:17] = torch.tensor([0.03, 0.02])
    return tokens


def test_raw15_solver_active_filter_is_recovered_from_existing_fields() -> None:
    tokens = _native15_tokens()

    filtered = filter_solver_active_contact_tokens(
        tokens,
        contact_representation=CONTACT_REPRESENTATION_RAW15,
    )

    torch.testing.assert_close(filtered[..., 0, :], tokens[..., 0, :])
    torch.testing.assert_close(filtered[..., 1:, :], torch.zeros_like(filtered[..., 1:, :]))
    torch.testing.assert_close(tokens, _native15_tokens())


def test_contact_token_solver_active_filter_uses_aligned_sidecar() -> None:
    tokens = torch.zeros(1, 1, 3, 17)
    tokens[..., :2, 0] = 1.0
    tokens[..., 0, 4] = 1.0
    tokens[..., 1, 4] = 2.0
    solver_active = torch.tensor([[[True, False, True]]])

    filtered = filter_solver_active_contact_tokens(
        tokens,
        contact_representation=CONTACT_REPRESENTATION_TOKENS,
        solver_active=solver_active,
    )

    torch.testing.assert_close(filtered[..., 0, :], tokens[..., 0, :])
    torch.testing.assert_close(filtered[..., 1:, :], torch.zeros_like(filtered[..., 1:, :]))


def test_contact_token_filter_can_exclude_only_active_robot_self_collisions() -> None:
    tokens = torch.zeros(1, 1, 5, 17)
    tokens[..., :, 0] = 1.0
    tokens[..., :, 4] = torch.arange(1.0, 6.0)
    # Robot self-collision: dynamic counterpart with a known robot body slot.
    tokens[..., 0, 2] = 3.0
    tokens[..., 0, 3] = 1.0
    # Foreign dynamic contact: dynamic counterpart without a robot body slot.
    tokens[..., 1, 2] = -1.0
    tokens[..., 1, 3] = 1.0
    # Static contact.
    tokens[..., 2, 2] = -1.0
    # Inactive robot self-collision.
    tokens[..., 3, 2] = 4.0
    tokens[..., 3, 3] = 1.0
    # Invalid robot self-collision.
    tokens[..., 4, 0] = 0.0
    tokens[..., 4, 2] = 5.0
    tokens[..., 4, 3] = 1.0
    solver_active = torch.tensor([[[True, True, True, False, True]]])
    original = tokens.clone()

    baseline = filter_solver_active_contact_tokens(
        tokens,
        contact_representation=CONTACT_REPRESENTATION_TOKENS,
        solver_active=solver_active,
    )
    filtered = filter_solver_active_contact_tokens(
        tokens,
        contact_representation=CONTACT_REPRESENTATION_TOKENS,
        solver_active=solver_active,
        exclude_robot_self_collisions=True,
    )

    torch.testing.assert_close(baseline[..., :3, :], tokens[..., :3, :])
    torch.testing.assert_close(filtered[..., 0, :], torch.zeros_like(filtered[..., 0, :]))
    torch.testing.assert_close(filtered[..., 1:3, :], tokens[..., 1:3, :])
    torch.testing.assert_close(filtered[..., 3:, :], torch.zeros_like(filtered[..., 3:, :]))
    torch.testing.assert_close(tokens, original)


def test_self_collision_exclusion_requires_contact_token_schema() -> None:
    with pytest.raises(ValueError, match="contact_tokens"):
        filter_solver_active_contact_tokens(
            _native15_tokens(),
            contact_representation=CONTACT_REPRESENTATION_RAW15,
            exclude_robot_self_collisions=True,
        )


def test_self_collision_exclusion_is_backward_compatible_and_factory_visible() -> None:
    legacy_cfg = NerdSolverCfg()
    experiment_cfg = NerdSolverCfg(
        contact_mode="newton_native",
        contact_representation=CONTACT_REPRESENTATION_TOKENS,
        contact_filter="solver_active",
        exclude_robot_self_collisions=True,
    )

    assert legacy_cfg.exclude_robot_self_collisions is False
    assert experiment_cfg.to_dict()["exclude_robot_self_collisions"] is True
    assert "exclude_robot_self_collisions" in _valid_solver_args(TransformerNeuralSolver)


def test_contact_token_solver_active_filter_requires_bool_aligned_sidecar() -> None:
    tokens = torch.zeros(1, 1, 3, 17)

    with pytest.raises(ValueError, match="contact_token_solver_active"):
        filter_solver_active_contact_tokens(
            tokens,
            contact_representation=CONTACT_REPRESENTATION_TOKENS,
        )
    with pytest.raises(ValueError, match="boolean"):
        filter_solver_active_contact_tokens(
            tokens,
            contact_representation=CONTACT_REPRESENTATION_TOKENS,
            solver_active=torch.zeros(1, 1, 3),
        )
    with pytest.raises(ValueError, match="shape"):
        filter_solver_active_contact_tokens(
            tokens,
            contact_representation=CONTACT_REPRESENTATION_TOKENS,
            solver_active=torch.zeros(1, 1, 2, dtype=torch.bool),
        )


@pytest.mark.parametrize(
    "config_name",
    (
        "transformer_rough_native_body_routed_active15.yaml",
        "transformer_rough_native_body_routed_active15_lr_1e-3.yaml",
        "transformer_rough_native_body_routed_active15_d16.yaml",
        "transformer_rough_native_body_routed_active15_d32.yaml",
    ),
)
def test_active15_configs_use_solver_active_view_of_raw15_dataset(config_name: str) -> None:
    cfg = yaml.safe_load((CFG_DIR / config_name).read_text(encoding="utf-8"))

    assert cfg["env"]["neural_solver_cfg"]["contact_representation"] == "active15_tokens"
    assert cfg["env"]["neural_solver_cfg"]["contact_filter"] == "solver_active"
    assert cfg["algorithm"]["dataset"]["contact_representation"] == "raw15_tokens"
    assert "/Anymal-C-Rough-Native-Raw15/" in cfg["algorithm"]["dataset"]["train_dataset_path"]
    assert "/Anymal-C-Rough-Native-Raw15/" in cfg["algorithm"]["eval"]["dataset_path"]


@pytest.mark.parametrize(
    ("config_name", "preset_name", "encoder_type"),
    (
        (
            "transformer_rough_native_shared_per_body_active_filter.yaml",
            "anymal_rough_newton_native_shared_per_body_active_filter.yaml",
            "shared_per_body",
        ),
        (
            "transformer_rough_native_contact_tokens_active_filter.yaml",
            "anymal_rough_newton_native_contact_tokens_active_filter.yaml",
            "global_attention",
        ),
    ),
)
def test_c_and_d_active_filter_variants_reuse_contact_token_dataset(
    config_name: str,
    preset_name: str,
    encoder_type: str,
) -> None:
    cfg = yaml.safe_load((CFG_DIR / config_name).read_text(encoding="utf-8"))
    preset = yaml.safe_load((PRESET_DIR / preset_name).read_text(encoding="utf-8"))
    base_cfg = yaml.safe_load((CFG_DIR / config_name.replace("_active_filter", "")).read_text(encoding="utf-8"))
    base_preset = yaml.safe_load((PRESET_DIR / preset_name.replace("_active_filter", "")).read_text(encoding="utf-8"))

    assert cfg["env"]["neural_solver_cfg"]["contact_filter"] == "solver_active"
    assert cfg["algorithm"]["dataset"]["contact_representation"] == "contact_tokens"
    assert cfg["inputs"]["contact_set"].get("encoder_type", "global_attention") == encoder_type
    assert preset["workflow"]["dataset_subdir"] == "anymal-c-rough-newton-native-contact-tokens"
    assert preset["experiment"]["dataset_contact_representation"] == "contact_tokens"
    assert preset["experiment"]["contact_filter"] == "solver_active"

    normalized_cfg = deepcopy(cfg)
    normalized_cfg["env"]["neural_solver_cfg"].pop("contact_filter")
    normalized_cfg["algorithm"]["dataset"].pop("contact_representation")
    assert normalized_cfg == base_cfg

    normalized_preset = deepcopy(preset)
    normalized_preset["workflow"]["base_name"] = base_preset["workflow"]["base_name"]
    normalized_preset["experiment"]["train_cfg"] = base_preset["experiment"]["train_cfg"]
    normalized_preset["experiment"].pop("dataset_contact_representation")
    normalized_preset["experiment"].pop("contact_filter")
    assert normalized_preset == base_preset


def test_contact_token_dataset_preset_generates_shared_cache_only() -> None:
    preset = yaml.safe_load(
        (PRESET_DIR / "anymal_rough_newton_native_contact_tokens_dataset.yaml").read_text(encoding="utf-8")
    )

    assert preset["workflow"]["dataset_subdir"] == "anymal-c-rough-newton-native-contact-tokens"
    assert preset["experiment"]["dataset_only"] is True
    assert preset["experiment"]["contact_representation"] == "contact_tokens"
    assert preset["experiment"]["max_contact_tokens"] == 64
    assert preset["resources"] == {
        "num_gpu": 1,
        "num_cpu": 15,
        "memory": "111Gi",
        "storage": "256Gi",
        "platform": "ovx-l40",
    }
