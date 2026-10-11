# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO observers diagnose updates without changing gradients or random sampling."""

import copy

import torch
from isaaclab_neural.rl.rsl_rl.ppo_diagnostics import PpoDiagnostics
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict


def _algorithm():
    obs = TensorDict({"policy": torch.randn(4, 3)}, batch_size=[4])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = MLPModel(obs, groups, "actor", 2, hidden_dims=[8])
    actor.distribution = GaussianDistribution(2, init_std=1.0)
    critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[8])
    storage = RolloutStorage("rl", 4, 3, obs, [2], device="cpu")
    algorithm = PPO(actor, critic, storage, num_learning_epochs=2, num_mini_batches=2, device="cpu")
    with torch.no_grad():
        for _ in range(3):
            algorithm.act(obs)
            obs = TensorDict({"policy": torch.randn(4, 3)}, batch_size=[4])
            algorithm.process_env_step(obs, torch.randn(4), torch.zeros(4), {})
        algorithm.compute_returns(obs)
    return algorithm


def test_consistency_check_preserves_rng_and_detects_corruption():
    alg = _algorithm()
    diagnostics = PpoDiagnostics()
    state = torch.get_rng_state()
    report = diagnostics.check_consistency(alg)
    assert report["passed"]
    assert torch.equal(state, torch.get_rng_state())
    alg.storage.observations["policy"].add_(10)
    assert not diagnostics.check_consistency(alg)["passed"]


def test_minibatch_observer_does_not_change_update():
    original = _algorithm()
    observed = copy.deepcopy(original)
    state = torch.get_rng_state()
    original.update()
    final_rng = torch.get_rng_state()
    torch.set_rng_state(state)
    diagnostics = PpoDiagnostics()
    with diagnostics.observe_update(observed):
        observed.update()
    assert len(diagnostics.minibatches) == 4
    assert diagnostics.minibatches[0]["kl_mean"] < 1e-6
    assert "actor_gradient_before_clip" in diagnostics.minibatches[0]
    assert torch.equal(final_rng, torch.get_rng_state())
    for name, value in original.actor.state_dict().items():
        torch.testing.assert_close(value, observed.actor.state_dict()[name], atol=0, rtol=0)
    assert "mini_batch_generator" not in observed.storage.__dict__
