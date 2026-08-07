# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for geometry-based NeRD contact MDP terms."""

from types import SimpleNamespace

import torch
from isaaclab_neural.contacts.geometry_mdp import GeometryFeetAirTime, body_contacts
from isaaclab_neural.physics import NewtonNerdManager


def _install_adapter(monkeypatch) -> SimpleNamespace:
    adapter = SimpleNamespace(
        num_envs=2,
        bodies_per_env=4,
        device=torch.device("cpu"),
        contact_masks=torch.tensor([[True, True, False], [True, False, False]]),
        contact_depths=torch.tensor([[-0.01, 0.02, 0.0], [0.0, 0.0, 0.0]]),
        contact_body_ids_0=torch.tensor([[1, 2, -1], [3, -1, -1]]),
        contact_body_ids_1=torch.tensor([[-1, -1, -1], [1, -1, -1]]),
    )
    monkeypatch.setattr(NewtonNerdManager, "_nerd_contact_adapter", adapter)
    monkeypatch.setattr(NewtonNerdManager, "_nerd_contact_mode", "newton_native")
    return adapter


def test_body_contacts_matches_either_side_and_rejects_positive_separation(monkeypatch):
    _install_adapter(monkeypatch)

    actual = body_contacts([1, 2, 3])

    torch.testing.assert_close(
        actual,
        torch.tensor([[True, False, False], [True, False, True]]),
    )


def test_geometry_feet_air_time_rewards_touchdown_after_air(monkeypatch):
    adapter = _install_adapter(monkeypatch)
    adapter.contact_masks.zero_()
    command_manager = SimpleNamespace(get_command=lambda _: torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
    env = SimpleNamespace(num_envs=2, device="cpu", step_dt=0.1, command_manager=command_manager)
    body_cfg = SimpleNamespace(body_ids=[1])
    cfg = SimpleNamespace(params={"body_cfg": body_cfg})
    term = GeometryFeetAirTime(cfg, env)

    torch.testing.assert_close(
        term(env, body_cfg, "base_velocity", threshold=0.15),
        torch.zeros(2),
    )
    torch.testing.assert_close(
        term(env, body_cfg, "base_velocity", threshold=0.15),
        torch.zeros(2),
    )

    adapter.contact_masks[:, 0] = True
    adapter.contact_depths[:, 0] = -0.01
    adapter.contact_body_ids_0[:, 0] = 1
    torch.testing.assert_close(
        term(env, body_cfg, "base_velocity", threshold=0.15),
        torch.full((2,), 0.05),
    )
