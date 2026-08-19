# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for owner-frame Raw15 extraction and body-routed encoding."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest
import torch
import yaml
from isaaclab_neural.contacts.active15_contact_encoder import Active15ContactEncoder
from isaaclab_neural.contacts.contact_set_schema import (
    CONTACT_REPRESENTATION_RAW15,
    RAW15_CATEGORICAL_CHANNELS,
    RAW15_FEATURE_DIM,
    RAW15_TOKEN_DIM,
)
from isaaclab_neural.contacts.raw15_contact_encoder import Raw15ContactEncoder
from isaaclab_neural.contacts.tensor_utils import ContactTokenMoments, normalize_raw15_contact_tokens
from isaaclab_neural.data.datasets import TrajectoryDataset
from isaaclab_neural.data.hdf5 import append_rollouts_to_hdf5, write_rollouts_to_hdf5
from isaaclab_neural.models.body_routed_raw15_model import BodyRoutedRaw15Encoder
from isaaclab_neural.models.models import ModelMixedInput
from isaaclab_neural.utils.checkpoint import load_checkpoint, reconstruct_model_from_checkpoint, save_checkpoint
from isaaclab_neural.utils.running_mean_std import RunningMeanStd

PACKAGE_ROOT = Path(__file__).parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
CFG_DIR = PACKAGE_ROOT / "isaaclab_neural" / "train" / "cfg" / "Anymal"
PRESET_DIR = REPOSITORY_ROOT / "osmo_scripts" / "presets"


def _extractor(encoder_type=Raw15ContactEncoder, max_tokens: int = 4):
    model = SimpleNamespace(
        body_count=2,
        body_world=torch.tensor([0, 0], dtype=torch.int32),
        body_com=torch.zeros(2, 3),
        shape_margin=torch.tensor([0.01, 0.01, 0.01]),
    )
    return encoder_type(
        model=model,
        primary_body_mask=torch.tensor([True, True]),
        shape_body_torch=torch.tensor([-1, 0, 1]),
        bodies_per_env=2,
        num_envs=1,
        max_contact_tokens=max_tokens,
        device="cpu",
    )


def _state() -> SimpleNamespace:
    half_sqrt = math.sqrt(0.5)
    return SimpleNamespace(
        body_q=torch.tensor(
            [
                [1.0, 2.0, 0.0, 0.0, 0.0, half_sqrt, half_sqrt],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ),
        body_qd=torch.tensor(
            [
                [2.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )


def _raw_contacts() -> dict[str, torch.Tensor]:
    return {
        "shape0": torch.tensor([1, 1, 1]),
        "shape1": torch.tensor([0, 2, 0]),
        "point0_world": torch.tensor(
            [
                [1.0, 3.0, 0.0],
                [1.0, 3.0, 0.0],
                [1.0, 3.0, 0.0],
            ]
        ),
        "point1_world": torch.tensor(
            [
                [1.0, 3.0, 0.02],
                [1.0, 3.0, 0.02],
                [1.0, 3.0, 0.10],
            ]
        ),
        "normal": torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        "thickness0": torch.tensor([0.03, 0.03, 0.03]),
        "thickness1": torch.tensor([0.02, 0.02, 0.02]),
    }


def _valid_token(body_slot: int, value: float = 0.0) -> torch.Tensor:
    token = torch.zeros(RAW15_TOKEN_DIM)
    token[0] = 1.0
    token[1] = float(body_slot)
    token[2:] = value
    return token


def _rollouts() -> dict:
    num_envs, steps, capacity = 2, 3, 4
    world_ids = torch.arange(num_envs, dtype=torch.long)
    tokens = torch.zeros(num_envs, steps, capacity, RAW15_TOKEN_DIM)
    tokens[0, 0, 0] = _valid_token(0, 0.1)
    body_ids = torch.full((num_envs, steps, capacity), -1, dtype=torch.long)
    body_ids[0, 0, 0] = 0
    token_world_ids = torch.full_like(body_ids, -1)
    token_world_ids[0, 0, 0] = 0
    return {
        "states": torch.zeros(num_envs, steps, 2),
        "next_states": torch.zeros(num_envs, steps, 2),
        "joint_f": torch.zeros(num_envs, steps, 1),
        "root_body_q": torch.zeros(num_envs, steps, 7),
        "root_body_qd": torch.zeros(num_envs, steps, 6),
        "gravity_dir": torch.zeros(num_envs, steps, 3),
        "contact_token_body_ids": body_ids,
        "contact_token_world_ids": token_world_ids,
        "trajectory_context": {
            "state_world_id": world_ids,
            "root_world_id": world_ids,
            "contact_world_id": world_ids,
        },
        "contacts": {
            "contact_tokens": tokens,
            "contact_token_overflow": torch.zeros(num_envs, steps, dtype=torch.long),
        },
    }


def test_raw15_keeps_inactive_raw_candidates_but_excludes_self_contacts() -> None:
    raw = _raw_contacts()
    raw_encoder = _extractor()
    raw_tokens = raw_encoder.encode(raw, _state())
    active_tokens = _extractor(Active15ContactEncoder).encode(raw, _state())
    raw_valid = raw_tokens[0, raw_tokens[0, :, 0] > 0.5]
    active_valid = active_tokens[0, active_tokens[0, :, 0] > 0.5]

    assert raw_valid.shape == (2, RAW15_TOKEN_DIM)
    assert active_valid.shape == (1, RAW15_TOKEN_DIM)
    # The positive-clearance raw candidate is retained without a distance cutoff.
    torch.testing.assert_close(raw_valid[:, 11], torch.tensor([-0.01, 0.07]), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(raw_valid[0], active_valid[0])
    # The middle candidate is a primary-primary self contact and stays excluded.
    assert raw_encoder.last_overflow.item() == 0

    truncated_encoder = _extractor(max_tokens=1)
    truncated = truncated_encoder.encode(raw, _state())
    assert truncated[..., 0].sum().item() == 1
    assert truncated_encoder.last_overflow.item() == 1


def test_raw15_model_contract_normalization_and_gradients() -> None:
    torch.manual_seed(0)
    encoder = BodyRoutedRaw15Encoder(num_bodies=3)
    tokens = torch.zeros(2, 4, RAW15_TOKEN_DIM)
    tokens[0, 0] = _valid_token(0, 0.1)
    tokens[0, 1] = _valid_token(2, 0.2)
    tokens[0, 2] = _valid_token(0, 0.3)
    tokens.requires_grad_()

    encoded = encoder.phi(tokens[..., 2:])
    expected = torch.zeros(2, 3, 64)
    expected[0, 0] = encoder.rho(encoded[0, 0] + encoded[0, 2])
    expected[0, 2] = encoder.rho(encoded[0, 1])
    output = encoder(tokens)

    torch.testing.assert_close(output, expected)
    assert output.shape == (2, 3, 64)
    assert sum(parameter.numel() for parameter in encoder.parameters()) == 6784
    assert encoder.phi[0].in_features == RAW15_FEATURE_DIM
    output.square().mean().backward()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()

    moments = ContactTokenMoments(
        RAW15_TOKEN_DIM,
        "cpu",
        categorical_channels=RAW15_CATEGORICAL_CHANNELS,
    )
    moments.update(tokens.detach())
    normalized = normalize_raw15_contact_tokens(tokens.detach(), moments.finalize("cpu"))
    torch.testing.assert_close(normalized[..., :2], tokens.detach()[..., :2])
    torch.testing.assert_close(normalized[1], torch.zeros_like(normalized[1]))


def test_raw15_hdf5_metadata_append_and_load(tmp_path: Path) -> None:
    path = tmp_path / "raw15.hdf5"
    write_rollouts_to_hdf5(
        path,
        _rollouts(),
        env_name="Anymal-C-Rough-Native-Raw15",
        contact_representation=CONTACT_REPRESENTATION_RAW15,
    )
    append_rollouts_to_hdf5(
        path,
        _rollouts(),
        env_name="Anymal-C-Rough-Native-Raw15",
        contact_representation=CONTACT_REPRESENTATION_RAW15,
    )

    with h5py.File(path, "r") as handle:
        attrs = handle["data"].attrs
        assert attrs["contact_representation"] == "raw15_tokens"
        assert attrs["contact_schema"] == "raw15_owner_body_v1"
        assert attrs["contact_frame"] == "owner_body_v1"
        assert attrs["contact_token_frame"] == "owner_body_v1"
        assert attrs["contact_velocity_point"] == "raw_point_midpoint_v1"
        assert attrs["contact_selection"] == "newton_raw_candidates_v1"
        assert handle["data"]["states"].shape[0] == 4

    dataset = TrajectoryDataset(
        path,
        sample_sequence_length=2,
        max_capacity=100,
        expected_contact_representation=CONTACT_REPRESENTATION_RAW15,
    )
    assert dataset[0]["contact_tokens"].shape == (2, 4, RAW15_TOKEN_DIM)
    with pytest.raises(ValueError, match="does not match the configured solver representation"):
        TrajectoryDataset(
            path,
            sample_sequence_length=2,
            expected_contact_representation="active15_tokens",
        )


def test_raw15_lr_variants_and_osmo_presets_share_one_dataset() -> None:
    standard = yaml.safe_load((CFG_DIR / "transformer_rough_native_body_routed_raw15.yaml").read_text())
    old_lr = yaml.safe_load((CFG_DIR / "transformer_rough_native_body_routed_raw15_lr_1e-3.yaml").read_text())
    standard_preset = yaml.safe_load((PRESET_DIR / "anymal_rough_newton_native_raw15.yaml").read_text())
    old_lr_preset = yaml.safe_load((PRESET_DIR / "anymal_rough_newton_native_raw15_lr_1e-3.yaml").read_text())
    dataset_preset = yaml.safe_load((PRESET_DIR / "anymal_rough_newton_native_raw15_dataset.yaml").read_text())

    assert standard["env"]["neural_solver_cfg"]["contact_representation"] == "raw15_tokens"
    assert standard["inputs"]["contact_set"]["encoder_type"] == "body_routed_raw15"
    assert standard["algorithm"]["optimizer"]["lr_start"] == "1e-4"
    assert standard["algorithm"]["optimizer"]["lr_end"] == "1e-5"
    assert old_lr["algorithm"]["optimizer"]["lr_start"] == "1e-3"
    assert old_lr["algorithm"]["optimizer"]["lr_end"] == "1e-4"

    normalized = copy.deepcopy(old_lr)
    normalized["algorithm"]["optimizer"] = copy.deepcopy(standard["algorithm"]["optimizer"])
    assert normalized == standard
    assert {
        standard_preset["workflow"]["dataset_subdir"],
        old_lr_preset["workflow"]["dataset_subdir"],
        dataset_preset["workflow"]["dataset_subdir"],
    } == {"anymal-c-rough-newton-native-raw15"}
    expected_dataset_root = "./data/datasets/Anymal-C-Rough-Native-Raw15/"
    assert standard["algorithm"]["dataset"]["train_dataset_path"].startswith(expected_dataset_root)
    assert standard["algorithm"]["eval"]["dataset_path"].startswith(expected_dataset_root)
    assert all(
        path.startswith(expected_dataset_root) for path in standard["algorithm"]["dataset"]["valid_datasets"].values()
    )
    assert dataset_preset["experiment"]["dataset_only"] is True
    assert dataset_preset["resources"]["num_gpu"] == 1
    assert dataset_preset["resources"]["num_cpu"] == 15
    assert dataset_preset["resources"]["memory"] == "111Gi"


def test_raw15_production_model_forward_backward_and_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(7)
    cfg = yaml.safe_load((CFG_DIR / "transformer_rough_native_body_routed_raw15.yaml").read_text())
    sample = {
        "states_embedding": torch.randn(1, 10, 37),
        "joint_f": torch.randn(1, 10, 18),
        "gravity_dir": torch.randn(1, 10, 3),
        "contact_tokens": torch.zeros(1, 10, 64, RAW15_TOKEN_DIM),
    }
    sample["contact_tokens"][0, :, 0] = _valid_token(0, 0.25)
    sample["contact_tokens"][0, :, 1] = _valid_token(16, -0.5)

    model = ModelMixedInput(
        input_sample=sample,
        output_dim=37,
        input_cfg=cfg["inputs"],
        network_cfg=cfg["network"],
        contact_mode="newton_native",
        contact_representation=CONTACT_REPRESENTATION_RAW15,
        num_bodies=17,
        device="cpu",
    )
    input_rms = {}
    for key in ("states_embedding", "joint_f", "gravity_dir"):
        rms = RunningMeanStd(shape=(sample[key].shape[-1],), device="cpu")
        rms.mean.fill_(0.25)
        rms.var.fill_(1.5)
        input_rms[key] = rms
    token_rms = RunningMeanStd(shape=(RAW15_TOKEN_DIM,), device="cpu")
    token_rms.mean.fill_(4.0)
    token_rms.var.fill_(2.0)
    input_rms["contact_tokens"] = token_rms
    output_rms = RunningMeanStd(shape=(37,), device="cpu")
    model.set_input_rms(input_rms)
    model.set_output_rms(output_rms)

    captured = {}
    handles = [
        model.contact_set_encoder.register_forward_pre_hook(
            lambda _module, args: captured.__setitem__("normalized_tokens", args[0].detach().clone())
        ),
        model.contact_set_encoder.register_forward_hook(
            lambda _module, _args, output: captured.__setitem__("body_latents", tuple(output.shape))
        ),
        model.transformer_model.register_forward_pre_hook(
            lambda _module, args: captured.__setitem__("transformer_input", tuple(args[0].shape))
        ),
        model.transformer_model.register_forward_hook(
            lambda _module, _args, output: captured.__setitem__("transformer_output", tuple(output.shape))
        ),
    ]
    output = model({key: value.clone() for key, value in sample.items()})
    for handle in handles:
        handle.remove()

    assert isinstance(model.contact_set_encoder, BodyRoutedRaw15Encoder)
    assert sum(parameter.numel() for parameter in model.parameters()) == 11_255_845
    assert captured["body_latents"] == (1, 10, 17, 64)
    assert captured["transformer_input"] == (1, 10, 1146)
    assert captured["transformer_output"] == (1, 10, 384)
    assert output.shape == (1, 10, 37)
    torch.testing.assert_close(captured["normalized_tokens"][..., :2], sample["contact_tokens"][..., :2])
    output.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.contact_set_encoder.parameters()
    )

    checkpoint_path = tmp_path / "raw15.pt"
    save_checkpoint(checkpoint_path, model, robot_name="Anymal-C", cfg=cfg)
    checkpoint = load_checkpoint(checkpoint_path)
    solver = SimpleNamespace(
        get_neural_model_inputs=lambda: {key: value.clone() for key, value in sample.items()},
        prediction_dim=37,
        contact_mode="newton_native",
        contact_representation=CONTACT_REPRESENTATION_RAW15,
        num_contact_bodies_per_env=17,
    )
    reconstructed = reconstruct_model_from_checkpoint(checkpoint, solver, device="cpu")
    with torch.no_grad():
        expected = model({key: value.clone() for key, value in sample.items()})
        actual = reconstructed({key: value.clone() for key, value in sample.items()})
    torch.testing.assert_close(actual, expected)

    incompatible_solver = copy.copy(solver)
    incompatible_solver.contact_representation = "active15_tokens"
    with pytest.raises(ValueError, match="must be configured together"):
        reconstruct_model_from_checkpoint(checkpoint, incompatible_solver, device="cpu")
