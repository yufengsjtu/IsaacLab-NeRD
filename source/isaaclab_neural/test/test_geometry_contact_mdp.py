# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for geometry-based NeRD contact MDP terms."""

from types import SimpleNamespace

import torch
from isaaclab_neural.contacts.contact_set_schema import (
    CONTACT_TOKEN_BODY_SLOT_INDEX,
    CONTACT_TOKEN_DIM,
    CONTACT_TOKEN_GAP_INDEX,
    CONTACT_TOKEN_VALID_INDEX,
)
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
    return adapter


def test_body_contacts_matches_either_side_and_rejects_positive_separation(monkeypatch):
    _install_adapter(monkeypatch)

    actual = body_contacts([1, 2, 3])

    torch.testing.assert_close(
        actual,
        torch.tensor([[True, False, False], [True, False, True]]),
    )


def test_body_contacts_reads_contact_tokens(monkeypatch):
    adapter = _install_adapter(monkeypatch)
    adapter.contact_representation = "contact_tokens"
    adapter.contact_tokens = torch.zeros((2, 3, CONTACT_TOKEN_DIM))
    adapter.contact_masks.zero_()

    adapter.contact_tokens[0, 0, CONTACT_TOKEN_VALID_INDEX] = 1.0
    adapter.contact_tokens[0, 0, CONTACT_TOKEN_BODY_SLOT_INDEX] = 1
    adapter.contact_tokens[0, 0, CONTACT_TOKEN_GAP_INDEX] = -0.01
    adapter.contact_tokens[0, 1, CONTACT_TOKEN_VALID_INDEX] = 1.0
    adapter.contact_tokens[0, 1, CONTACT_TOKEN_BODY_SLOT_INDEX] = 2
    adapter.contact_tokens[0, 1, CONTACT_TOKEN_GAP_INDEX] = 0.02
    adapter.contact_tokens[0, 2, CONTACT_TOKEN_BODY_SLOT_INDEX] = 3
    adapter.contact_tokens[0, 2, CONTACT_TOKEN_GAP_INDEX] = -0.01

    adapter.contact_tokens[1, 0, CONTACT_TOKEN_VALID_INDEX] = 1.0
    adapter.contact_tokens[1, 0, CONTACT_TOKEN_BODY_SLOT_INDEX] = 3
    adapter.contact_tokens[1, 0, CONTACT_TOKEN_GAP_INDEX] = 0.0
    adapter.contact_tokens[1, 1, CONTACT_TOKEN_VALID_INDEX] = 1.0
    adapter.contact_tokens[1, 1, CONTACT_TOKEN_BODY_SLOT_INDEX] = 1
    adapter.contact_tokens[1, 1, CONTACT_TOKEN_GAP_INDEX] = -0.01

    torch.testing.assert_close(
        body_contacts([1, 2, 3]),
        torch.tensor([[True, False, False], [True, False, True]]),
    )


def test_body_contacts_prefers_newton_body_ids_over_encoder_slots(monkeypatch):
    adapter = _install_adapter(monkeypatch)
    adapter.contact_representation = "contact_tokens"
    adapter.contact_tokens = torch.zeros((1, 1, CONTACT_TOKEN_DIM))
    adapter.contact_tokens[0, 0, CONTACT_TOKEN_VALID_INDEX] = 1.0
    adapter.contact_tokens[0, 0, CONTACT_TOKEN_BODY_SLOT_INDEX] = 0
    adapter.contact_tokens[0, 0, CONTACT_TOKEN_GAP_INDEX] = -0.01
    adapter.contact_token_body_ids = torch.tensor([[7]])
    adapter.body_local_ids = torch.full((8,), -1, dtype=torch.long)
    adapter.body_local_ids[7] = 2

    torch.testing.assert_close(body_contacts([2]), torch.tensor([[True]]))
    torch.testing.assert_close(body_contacts([0]), torch.tensor([[False]]))


def test_body_contacts_ignores_padded_newton_body_ids(monkeypatch):
    adapter = _install_adapter(monkeypatch)
    adapter.contact_representation = "contact_tokens"
    adapter.contact_tokens = torch.zeros((1, 1, CONTACT_TOKEN_DIM))
    adapter.contact_tokens[0, 0, CONTACT_TOKEN_VALID_INDEX] = 1.0
    adapter.contact_tokens[0, 0, CONTACT_TOKEN_BODY_SLOT_INDEX] = 0
    adapter.contact_tokens[0, 0, CONTACT_TOKEN_GAP_INDEX] = -0.01
    adapter.contact_token_body_ids = torch.tensor([[-1]])
    adapter.body_local_ids = torch.arange(4, dtype=torch.long)

    torch.testing.assert_close(body_contacts([0]), torch.tensor([[False]]))


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


def test_geometry_feet_slide_tracks_history_and_resets(monkeypatch):
    from isaaclab_neural.contacts.geometry_mdp import GeometryFeetSlide

    callbacks = []
    contact = torch.tensor([[True, False], [False, True]])
    velocity = torch.tensor([[[3.0, 4.0, 99.0], [0.0, 2.0, 99.0]]]).repeat(2, 1, 1)
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        sim=SimpleNamespace(physics_manager=SimpleNamespace(register_post_step_callback=callbacks.append)),
        scene={"robot": SimpleNamespace(data=SimpleNamespace(body_lin_vel_w=SimpleNamespace(torch=velocity)))},
    )
    body_cfg = SimpleNamespace(name="robot", body_ids=[0, 1])
    cfg = SimpleNamespace(params={"body_cfg": body_cfg, "history_length": 3})
    monkeypatch.setattr("isaaclab_neural.contacts.geometry_mdp.body_contacts", lambda *args, **kwargs: contact)
    term = GeometryFeetSlide(cfg, env)
    callbacks[0]()
    torch.testing.assert_close(term(env, body_cfg), torch.tensor([5.0, 2.0]))
    contact.zero_()
    callbacks[0]()
    term.reset([0])
    torch.testing.assert_close(term(env, body_cfg), torch.tensor([0.0, 2.0]))
    callbacks[0]()
    torch.testing.assert_close(term(env, body_cfg), torch.tensor([0.0, 2.0]))
    callbacks[0]()
    torch.testing.assert_close(term(env, body_cfg), torch.zeros(2))
    contact.fill_(True)
    callbacks[0]()
    torch.testing.assert_close(term(env, body_cfg), torch.full((2,), 7.0))
    term.reset()
    torch.testing.assert_close(term(env, body_cfg), torch.zeros(2))


def test_biped_geometry_reward_tracks_physics_steps_and_partial_resets(monkeypatch):
    from isaaclab_neural.contacts.geometry_mdp import GeometryFeetAirTimePositiveBiped

    from isaaclab_tasks.core.velocity.mdp.rewards import feet_air_time_positive_biped

    callbacks = []
    command = torch.tensor([[0.5, 0.0], [0.5, 0.0]])
    contact = torch.zeros((2, 2), dtype=torch.bool)
    air = torch.zeros((2, 2))
    stance = torch.zeros((2, 2))
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            current_air_time=SimpleNamespace(torch=air), current_contact_time=SimpleNamespace(torch=stance)
        )
    )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        physics_dt=0.005,
        step_dt=0.02,
        sim=SimpleNamespace(physics_manager=SimpleNamespace(register_post_step_callback=callbacks.append)),
        command_manager=SimpleNamespace(get_command=lambda _: command),
        scene=SimpleNamespace(sensors={"contact_forces": sensor}),
    )
    body_cfg = SimpleNamespace(body_ids=[0, 1])
    sensor_cfg = SimpleNamespace(name="contact_forces", body_ids=[0, 1])
    monkeypatch.setattr("isaaclab_neural.contacts.geometry_mdp.body_contacts", lambda *args, **kwargs: contact)
    term = GeometryFeetAirTimePositiveBiped(SimpleNamespace(params={"body_cfg": body_cfg}), env)
    sequence = [(True, False), (True, True), (True, False), (True, False)] * 3
    for index, state in enumerate(sequence):
        contact[0] = torch.tensor(state)
        contact[1] = torch.tensor([False, True])
        air.copy_(torch.where(contact, 0.0, air + env.physics_dt))
        stance.copy_(torch.where(contact, stance + env.physics_dt, 0.0))
        callbacks[0]()
        if index % 4 == 3:
            expected = feet_air_time_positive_biped(env, "base_velocity", 0.025, sensor_cfg)
            torch.testing.assert_close(term(env, body_cfg, "base_velocity", 0.025), expected)
            torch.testing.assert_close(term(env, body_cfg, "base_velocity", 0.025), expected)
    term.reset([0])
    torch.testing.assert_close(term(env, body_cfg, "base_velocity", 0.025), torch.tensor([0.0, 0.025]))
    command.zero_()
    torch.testing.assert_close(term(env, body_cfg, "base_velocity", 0.025), torch.zeros(2))
    term.reset()
    command[:, 0] = 0.5
    torch.testing.assert_close(term(env, body_cfg, "base_velocity", 0.025), torch.zeros(2))
