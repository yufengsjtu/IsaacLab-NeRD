# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import pytest
from isaaclab_neural.contacts import AbstractContact, collision_detection_fixed_ground
from isaaclab_neural.physics import NerdSolverCfg
from isaaclab_neural.physics.nerd_solver_cfg import normalize_contact_mode


def test_fixed_ground_is_deprecated_and_rejected() -> None:
    with pytest.warns(DeprecationWarning, match="fixed_ground"):
        with pytest.raises(ValueError, match="incompatible contact representations"):
            normalize_contact_mode("fixed_ground")


def test_solver_cfg_rejects_legacy_contact_mode() -> None:
    with pytest.warns(DeprecationWarning, match="fixed_ground"):
        with pytest.raises(ValueError, match="incompatible contact representations"):
            NerdSolverCfg(contact_mode="fixed_ground", num_contacts_per_env=0)


def test_newton_native_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solver_cfg = NerdSolverCfg(contact_mode="newton_native", num_contacts_per_env=16)

    assert solver_cfg.contact_mode == "newton_native"
    assert solver_cfg.num_contacts_per_env == 16


def test_invalid_contact_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported contact mode"):
        normalize_contact_mode("unsupported")


def test_contact_free_capacity_is_supported() -> None:
    solver_cfg = NerdSolverCfg(contact_mode="newton_native", num_contacts_per_env=0)

    assert solver_cfg.num_contacts_per_env == 0


def test_negative_contact_capacity_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        NerdSolverCfg(contact_mode="newton_native", num_contacts_per_env=-1)


def test_solver_active_contact_filter_is_supported() -> None:
    solver_cfg = NerdSolverCfg(contact_filter="solver_active")

    assert solver_cfg.contact_filter == "solver_active"


def test_invalid_contact_filter_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported contact_filter"):
        NerdSolverCfg(contact_filter="unsupported")


@pytest.mark.parametrize(
    "legacy_call",
    [
        lambda: AbstractContact(object()),
        lambda: collision_detection_fixed_ground(object(), object(), object()),
    ],
)
def test_legacy_contact_tools_warn_and_reject(legacy_call) -> None:
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="fixed_ground"):
            legacy_call()
