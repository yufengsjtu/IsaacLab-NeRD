# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Initial OPD snapshots preserve explicit zero-update and RNG semantics."""

from types import SimpleNamespace

import pytest
import torch
from isaaclab_neural.utils.checkpoint import load_opd_policy_snapshot, save_opd_policy_snapshot


class Algorithm:
    def __init__(self):
        self.actor = torch.nn.Linear(2, 3)
        self.critic = torch.nn.Linear(2, 1)
        self.optimizer = torch.optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=1e-3)
        self.learning_rate = 1e-3

    def save(self):
        return {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load(self, payload, load_cfg, strict):
        self.actor.load_state_dict(payload["actor_state_dict"], strict=strict)
        self.critic.load_state_dict(payload["critic_state_dict"], strict=strict)
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])


def _runner():
    return SimpleNamespace(alg=Algorithm(), device="cpu", current_learning_iteration=0)


def test_initial_snapshot_preserves_weights_rng_and_zero_updates(tmp_path):
    torch.manual_seed(7)
    original = _runner()
    path = tmp_path / "initial.pt"
    save_opd_policy_snapshot(path, original, {"seed": 7}, updates_completed=0, provenance={"policy_seed": 7})
    expected_random = torch.rand(3)
    target = _runner()
    info = load_opd_policy_snapshot(path, target, restore_rng=True, require_initial=True)
    assert info["ppo_updates_completed"] == 0
    assert target.current_learning_iteration == 0
    assert not target.alg.optimizer.state
    torch.testing.assert_close(torch.rand(3), expected_random)
    for a, b in zip(original.alg.actor.parameters(), target.alg.actor.parameters()):
        torch.testing.assert_close(a, b)
    with pytest.raises(FileExistsError):
        save_opd_policy_snapshot(path, original, {}, updates_completed=0, provenance={})


def test_active_collector_resume_keeps_optimizer_progress_and_learning_rate(tmp_path):
    from isaaclab_neural.generate.active_opd import initialize_active_policy

    original = _runner()
    original.alg.actor(torch.ones(1, 2)).sum().backward()
    original.alg.optimizer.step()
    original.alg.learning_rate = 2e-5
    original.alg.optimizer.param_groups[0]["lr"] = 2e-5
    path = tmp_path / "policy.pt"
    protocol = {"physics_dt": 0.005}
    cfg = {"algorithm": {"learning_rate": 1e-4}}
    save_opd_policy_snapshot(path, original, cfg, updates_completed=7, provenance={"protocol": protocol})
    target = _runner()
    start = initialize_active_policy(target, cfg, str(path), resume=True, protocol=protocol)
    assert start == target.current_learning_iteration == 7
    assert target.alg.learning_rate == target.alg.optimizer.param_groups[0]["lr"] == 2e-5
    assert target.alg.optimizer.state
    for a, b in zip(original.alg.actor.parameters(), target.alg.actor.parameters()):
        assert torch.equal(a, b)
    with pytest.raises(ValueError, match="protocol"):
        initialize_active_policy(_runner(), cfg, str(path), resume=True, protocol={"physics_dt": 0.01})
    with pytest.raises(ValueError, match="untrained"):
        initialize_active_policy(_runner(), cfg, str(path), resume=False, protocol=protocol)


def test_continuation_restores_adaptive_lr_and_is_not_initial(tmp_path):
    original = _runner()
    original.alg.actor(torch.ones(1, 2)).sum().backward()
    original.alg.optimizer.step()
    original.alg.learning_rate = 2e-4
    original.alg.optimizer.param_groups[0]["lr"] = 2e-4
    path = tmp_path / "trained.pt"
    with pytest.raises(ValueError, match="empty optimizer"):
        save_opd_policy_snapshot(path, original, {}, updates_completed=0, provenance={})
    save_opd_policy_snapshot(path, original, {}, updates_completed=5, provenance={})
    target = _runner()
    load_opd_policy_snapshot(path, target)
    assert target.current_learning_iteration == 5
    assert target.alg.learning_rate == target.alg.optimizer.param_groups[0]["lr"] == 2e-4
    assert target.alg.optimizer.state
    with pytest.raises(ValueError, match="untrained"):
        load_opd_policy_snapshot(path, target, require_initial=True)
