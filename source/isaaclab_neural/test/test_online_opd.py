# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded online promotion requires both actual progress and reference retention."""

import copy

from isaaclab_neural.rl.rsl_rl.online_opd import decide_candidate


def _report(value):
    metric = {"mean": value, "p95": value * 2, "count": 32, "non_finite": 0}
    return {
        "shards": [{"sha256": "same", "queries": 32, "horizon_physics_steps": 10}],
        "aggregate": {name: {"velocity": dict(metric)} for name in ("one_step", "held_target_rollout")},
    }


def test_online_gate_requires_gain_and_cumulative_reference_retention():
    config = {
        "numeric_tolerances": {"velocity": 0.0001},
        "current_regression": 0.05,
        "reference_regression": 0.1,
        "minimum_gain": 0.01,
    }
    current, reference = _report(1.0), _report(0.1)
    assert decide_candidate(current, _report(0.98), reference, _report(0.105), config)["accepted"]
    assert not decide_candidate(current, current, reference, reference, config)["accepted"]
    assert not decide_candidate(current, _report(0.9), reference, _report(0.12), config)["accepted"]
    bad = copy.deepcopy(current)
    bad["aggregate"]["one_step"]["velocity"]["non_finite"] = 1
    assert not decide_candidate(current, bad, reference, reference, config)["accepted"]


def test_reference_monitor_only_relaxes_finite_reference_regression():
    import pytest

    config = {
        "numeric_tolerances": {"velocity": 0.0001},
        "current_regression": 0.05,
        "reference_regression": 0.1,
        "minimum_gain": 0.01,
        "reference_mode": "monitor",
    }
    current, reference = _report(1.0), _report(0.1)
    decision = decide_candidate(current, _report(0.98), reference, _report(0.15), config)
    assert decision["accepted"]
    assert not decision["reference_vs_F_pre"]["passed"]
    assert not decision["strict_accepted"]
    assert decision["reference_mode"] == "monitor"
    assert not decide_candidate(current, current, reference, reference, config)["accepted"]
    assert not decide_candidate(current, _report(1.1), reference, reference, config)["accepted"]
    for invalid in ("non_finite", "count", "mean"):
        bad = copy.deepcopy(reference)
        bad["aggregate"]["held_target_rollout"]["velocity"][invalid] = {
            "non_finite": 1,
            "count": 0,
            "mean": float("inf"),
        }[invalid]
        assert not decide_candidate(current, _report(0.98), reference, bad, config)["accepted"]
    with pytest.raises(ValueError, match="reference_mode"):
        decide_candidate(current, _report(0.98), reference, reference, {**config, "reference_mode": "ignore"})


def test_ppo_block_uses_complete_rollout_and_one_update(monkeypatch):
    from types import SimpleNamespace

    import rsl_rl.utils
    import torch
    from isaaclab_neural.rl.rsl_rl.online_opd import _ppo_update

    monkeypatch.setattr(rsl_rl.utils, "check_nan", lambda *args: None)
    calls = []

    def act(obs):
        assert torch.is_inference_mode_enabled()
        calls.append("act")
        return torch.zeros(1, 1)

    def update():
        assert not torch.is_inference_mode_enabled()
        calls.append("update")
        return {"value": 1.0}

    algorithm = SimpleNamespace(
        train_mode=lambda: None,
        act=act,
        update=update,
        learning_rate=0.001,
        process_env_step=lambda *args: calls.append("store"),
        compute_returns=lambda obs: calls.append("returns"),
    )
    env = SimpleNamespace(device="cpu", step=lambda actions: (torch.zeros(1, 1), torch.ones(1), torch.zeros(1), {}))
    runner = SimpleNamespace(
        alg=algorithm, env=env, cfg={"num_steps_per_env": 3}, device="cpu", current_learning_iteration=4
    )
    recorder = SimpleNamespace(finish_control_step=lambda done: calls.append("record"))
    _, report = _ppo_update(runner, torch.zeros(1, 1), recorder, 4)
    assert calls == ["act", "record", "store"] * 3 + ["returns", "update"]
    assert recorder.policy_update_id == 4
    assert runner.current_learning_iteration == report["ppo_updates_completed"] == 5


def test_promotion_preserves_history_and_rng(monkeypatch):
    from types import SimpleNamespace

    import isaaclab_neural.rl.rsl_rl.online_opd as online
    import numpy as np
    import torch

    model = torch.nn.Linear(2, 1)
    before = torch.get_rng_state()
    numpy_before = np.random.get_state()
    history = [torch.ones(2)]
    solver = SimpleNamespace(neural_model=model, use_feature_cache=False, use_kv_cache=False, states_history=history)

    def load(*args, **kwargs):
        torch.rand(10)
        np.random.rand(10)
        return {"model_state_dict": {"weight": torch.ones(1, 2), "bias": torch.zeros(1)}}

    monkeypatch.setattr(online, "load_checkpoint", load)
    online._promote_without_reset(solver, "candidate.pt", "cpu")
    assert solver.states_history is history
    assert torch.equal(torch.get_rng_state(), before)
    np.testing.assert_equal(np.random.get_state(), numpy_before)
    torch.testing.assert_close(model.weight, torch.ones(1, 2))
    assert not model.training


def test_recover_consumed_runner_metadata_without_changing_saved_fields():
    import pytest
    from isaaclab_neural.rl.rsl_rl.online_opd import recover_runner_config

    original = {
        "algorithm": {"class_name": "PPO", "learning_rate": 0.001},
        "actor": {"class_name": "MLP", "distribution_cfg": {"class_name": "Gaussian"}},
        "obs_groups": {},
    }
    saved = {
        "algorithm": {"learning_rate": 0.001},
        "actor": {"distribution_cfg": {}},
        "obs_groups": {"actor": ["policy"]},
    }
    recovered, paths = recover_runner_config(saved, original)
    assert "algorithm.class_name" in paths
    assert recovered["actor"]["distribution_cfg"]["class_name"] == "Gaussian"
    assert recovered["obs_groups"] == saved["obs_groups"]
    assert "class_name" not in saved["algorithm"]
    saved["algorithm"]["learning_rate"] = 1.0
    with pytest.raises(ValueError, match="cannot recover blindly"):
        recover_runner_config(saved, original)
