# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded dynamics-curiosity scoring must not reward invalid or reset transitions."""

from types import SimpleNamespace

import pytest
import torch
from isaaclab_neural.generate.active_opd import bounded_model_error, dynamics_curiosity_reward


def test_stage_stream_consumes_fresh_rows_once_without_global_rng_changes():
    import numpy as np
    from isaaclab_neural.rl.rsl_rl.active_cycle import fresh_stage_batches

    state = np.random.get_state()
    batches = fresh_stage_batches(40, 32, updates=4, batch_size=8, seed=19)
    assert batches == fresh_stage_batches(40, 32, updates=4, batch_size=8, seed=19)
    assert fresh_stage_batches(40, 32, updates=1, batch_size=8, seed=19) == batches[:1]
    assert len({i for row in batches for i in row["current"]}) == 32
    assert sorted(i for row in batches for i in row["reference"]) == list(range(32))
    assert np.array_equal(state[1], np.random.get_state()[1])
    with pytest.raises(ValueError):
        fresh_stage_batches(40, 31, updates=4, batch_size=8, seed=19)


def test_reference_sequence_loss_averages_positions_and_never_broadcasts_anchor_labels():
    from isaaclab_neural.rl.rsl_rl.active_cycle import weighted_prediction_mse

    prediction = torch.ones(2, 10, 3, requires_grad=True)
    target = torch.zeros_like(prediction)
    weights = torch.tensor([1.0, 2.0, 3.0])
    loss = weighted_prediction_mse(prediction, target, weights)
    expected = weights.square().mean()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(weighted_prediction_mse(prediction[:, -1], target[:, -1], weights), expected)
    loss.backward()
    torch.testing.assert_close(prediction.grad, (2 * weights.square() / prediction.numel()).expand_as(prediction))
    with pytest.raises(ValueError):
        weighted_prediction_mse(prediction, target[:, -1], weights)
    with pytest.raises(ValueError):
        weighted_prediction_mse(prediction, target[:, -1:], weights)


def test_active_cycle_defaults_align_both_sources_and_validation_with_pretraining():
    from isaaclab_neural.rl.rsl_rl.active_cycle import active_supervision_settings, weighted_prediction_mse

    assert active_supervision_settings({}) == {
        "current_supervision": "all_positions",
        "reference_supervision": "all_positions",
        "current_validation_supervision": "all_positions",
        "pretraining_loss": True,
    }
    for key, value in (
        ("current_supervision", "anchor"),
        ("reference_supervision", "anchor"),
        ("current_validation_supervision", "anchor"),
        ("pretraining_loss", False),
        ("retain_current_targets", False),
        ("retain_reference_targets", False),
    ):
        with pytest.raises(ValueError, match="full-history"):
            active_supervision_settings({key: value})
    prediction = torch.linspace(-0.4, 0.7, 60).reshape(2, 10, 3).requires_grad_()
    target = prediction.detach().square() + 0.031
    weights = torch.tensor([0.17, 2.13, 0.003])
    actual = weighted_prediction_mse(prediction, target, weights)
    expected = torch.nn.MSELoss()(prediction * weights, target * weights)
    assert torch.equal(actual, expected)
    assert torch.equal(weighted_prediction_mse(prediction, target, weights, pretraining=True), expected)
    with pytest.raises(ValueError, match="pretraining"):
        weighted_prediction_mse(prediction, target, weights, pretraining=False)
    (g1,) = torch.autograd.grad(actual, prediction, retain_graph=True)
    (g2,) = torch.autograd.grad(expected, prediction)
    assert torch.equal(g1, g2)


def test_full_history_replay_cannot_drop_labels_or_accept_anchor_only_cache():
    from isaaclab_neural.eval.learnability import _context_cache
    from isaaclab_neural.rl.rsl_rl.active_cycle import _join_caches

    target = torch.arange(30, dtype=torch.float32).reshape(1, 10, 3)
    cache = {
        "inputs": {"states": target.clone()},
        "target": target[:, -1],
        "truth": target[:, -1],
        "sequence_target": target,
        "sequence_label_origin": "observed_mj_post_step",
        "source_hashes": {"raw": "hash"},
        "families": ["episode"],
    }
    joined = _join_caches([cache, cache])
    torch.testing.assert_close(joined["sequence_target"], target.repeat(2, 1, 1))
    assert joined["sequence_label_origin"] == "observed_mj_post_step"
    legacy = {key: value for key, value in cache.items() if key != "sequence_target"}
    for caches in ([cache, legacy], [legacy, legacy]):
        with pytest.raises(ValueError, match="anchor-only"):
            _join_caches(caches)
    with pytest.raises(ValueError, match="anchor"):
        _join_caches([{**cache, "target": cache["target"] + 1}])
    with pytest.raises(ValueError, match="verified"):
        _context_cache(object(), [], None, "cpu", keep_sequence_targets=True)


def test_full_history_microbatches_match_pretrained_two_source_gradients():
    from isaaclab_neural.rl.rsl_rl.active_cycle import _backward_history_loss

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor([0.2, 0.3, 0.4]))

        def forward(self, inputs):
            return inputs["states"] * self.scale

    model = Model()
    reference_model = Model()
    weights = torch.tensor([0.3, 2.0, 1.5])
    indices = [4, 1, 3]
    expected_loss = 0.0
    for offset in (0.0, 0.1):
        states = torch.linspace(-1, 1, 150).reshape(5, 10, 3) + offset
        targets = states.square()
        cache = {
            "inputs": {"states": states},
            "sequence_target": targets,
            "target": targets[:, -1],
            "truth": targets[:, -1],
        }
        observed = _backward_history_loss(model, cache, indices, weights, "cpu", 2)
        expected = torch.nn.MSELoss()(
            reference_model({"states": states[indices]}) * weights, targets[indices] * weights
        )
        assert observed == pytest.approx(float(expected.detach()))
        expected_loss = expected_loss + 0.5 * expected
    expected_loss.backward()
    torch.testing.assert_close(model.scale.grad, reference_model.scale.grad)


@pytest.mark.parametrize("beta", [0.0, 3.0])
@pytest.mark.parametrize("drift", [0.0, 0.1])
@pytest.mark.parametrize("old_weight", [0.5, 2.5])
def test_reference_distillation_keeps_teacher_frozen_and_preserves_true_loss(beta, drift, old_weight):
    from isaaclab_neural.rl.rsl_rl.active_cycle import _backward_history_loss

    class Model(torch.nn.Module):
        def __init__(self, shift):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor([0.2, 0.3, 0.4]) + shift)

        def forward(self, inputs):
            return inputs["states"] * self.scale

    student, expected_model = Model(drift), Model(drift)
    teacher = Model(0.0).eval().requires_grad_(False)
    frozen = {key: value.clone() for key, value in teacher.state_dict().items()}
    weights = torch.tensor([0.3, 2.0, 1.5])
    indices = [4, 1, 3]
    expected_loss = 0.0
    log = {}
    for source, shift in (("current", 0.3), ("reference", -0.1)):
        states = torch.linspace(-1, 1, 150).reshape(5, 10, 3) + shift
        truth = states.square()
        cache = {
            "inputs": {"states": states.clone()},
            "sequence_target": truth,
            "target": truth[:, -1],
            "truth": truth[:, -1],
        }
        is_reference = source == "reference"
        _backward_history_loss(
            student,
            cache,
            indices,
            weights,
            "cpu",
            2,
            source=source,
            reference_teacher=teacher if is_reference else None,
            distillation_beta=beta if is_reference else 0.0,
            retention_metrics=log if is_reference else None,
            truth_weight=old_weight if is_reference else 0.5,
        )
        prediction = expected_model({"states": states[indices]})
        expected_loss = expected_loss + (old_weight if is_reference else 0.5) * torch.nn.MSELoss()(
            prediction * weights, truth[indices] * weights
        )
        if is_reference:
            keep = torch.nn.MSELoss()(prediction * weights, teacher({"states": states[indices]}).detach() * weights)
            (keep_gradient,) = torch.autograd.grad(keep, expected_model.scale, retain_graph=True)
            assert bool(keep_gradient.abs().sum() > 0) == bool(drift)
            expected_loss = expected_loss + beta * keep
            if beta:
                assert log["loss"] == pytest.approx(float(keep.detach()))
        assert torch.equal(states, cache["inputs"]["states"])
    expected_loss.backward()
    torch.testing.assert_close(student.scale.grad, expected_model.scale.grad)
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert all(torch.equal(value, teacher.state_dict()[key]) for key, value in frozen.items())
    with pytest.raises(ValueError, match="reference"):
        _backward_history_loss(
            student,
            cache,
            indices,
            weights,
            "cpu",
            2,
            source="current",
            reference_teacher=teacher,
            distillation_beta=1.0,
        )


def test_zero_distillation_does_not_call_teacher():
    from isaaclab_neural.rl.rsl_rl.active_cycle import _backward_history_loss

    parameter = torch.nn.Parameter(torch.ones(3))

    def model(inputs):
        return inputs["states"] * parameter

    states = torch.ones(1, 10, 3)
    cache = {
        "inputs": {"states": states},
        "sequence_target": states * 2,
        "target": states[:, -1] * 2,
        "truth": states[:, -1] * 2,
    }
    _backward_history_loss(
        model,
        cache,
        [0],
        torch.ones(3),
        "cpu",
        1,
        source="reference",
        reference_teacher=object(),
        distillation_beta=0.0,
    )
    torch.testing.assert_close(parameter.grad, torch.full((3,), -1 / 3))


def test_feature_noise_does_not_change_physics_or_clean_keep_targets():
    from isaaclab_neural.rl.rsl_rl.active_cycle import _backward_history_loss, _state_feature_inputs

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(3))
            self.input_rms = {"states_embedding": SimpleNamespace(var=torch.ones(3))}
            self.seen = []

        def forward(self, inputs):
            self.seen.append(inputs["states_embedding"].detach().clone())
            return inputs["states_embedding"] * self.scale

    student = Model()
    teacher = Model().eval().requires_grad_(False)
    states = torch.ones(2, 10, 3)
    cache = {
        "inputs": {"states": states, "states_embedding": states.clone(), "joint_f": states.clone()},
        "sequence_target": states * 2,
        "target": states[:, -1] * 2,
        "truth": states[:, -1] * 2,
    }
    generator = torch.Generator().manual_seed(9)
    rng = torch.get_rng_state()
    augmented = _state_feature_inputs(student, cache["inputs"], 0.01, generator)
    assert not torch.equal(augmented["states_embedding"], states)
    assert torch.equal(augmented["states"], states) and torch.equal(augmented["joint_f"], states)
    assert torch.equal(torch.get_rng_state(), rng)
    log = {}
    _backward_history_loss(
        student,
        cache,
        [0, 1],
        torch.ones(3),
        "cpu",
        2,
        source="reference",
        reference_teacher=teacher,
        distillation_beta=10,
        retention_metrics=log,
        feature_noise_std=0.01,
        noise_generator=generator,
    )
    assert len(student.seen) == 2 and torch.equal(student.seen[-1], states)
    assert torch.equal(teacher.seen[0], states) and log["loss"] == 0
    assert torch.equal(cache["inputs"]["states_embedding"], states)


def test_rollout_gradient_accumulation_does_not_readd_local_loss():
    from isaaclab_neural.rl.rsl_rl.active_cycle import _backward_rollout_loss

    parameter = torch.nn.Parameter(torch.tensor(2.0))

    class Student:
        num_envs = 2

        def sequence_loss_terms(self, records, max_physics_steps):
            assert max_physics_steps == 4
            sequence = torch.stack([(parameter - value).square() for value in records])
            return {"sequence": sequence, "local": sequence * 1000}

    records = [torch.tensor(float(value)) for value in (1, 3, 5)]
    raw = _backward_rollout_loss(Student(), records, [2, 0, 1], 3.0, 4)
    expected = torch.stack([(parameter - value).square() for value in records]).mean()
    assert raw == pytest.approx(float(expected.detach()))
    expected_gradient = torch.autograd.grad(expected * 3, parameter)[0]
    torch.testing.assert_close(parameter.grad, expected_gradient)
    parameter.grad = None
    assert _backward_rollout_loss(None, None, None, 0.0, 4) == 0.0 and parameter.grad is None


def test_stage_worker_rejects_obsolete_supervision_before_loading_models(tmp_path):
    import json

    from isaaclab_neural.rl.rsl_rl.active_cycle import fit_active_stage

    config = tmp_path / "legacy.json"
    config.write_text(json.dumps({"current_supervision": "anchor"}))
    with pytest.raises(ValueError, match="full-history"):
        fit_active_stage(None, SimpleNamespace(active_fit_config=config))


def test_stage_gate_requires_frozen_old_retention_and_current_gain():
    from isaaclab_neural.rl.rsl_rl.active_cycle import decide_active_stage

    baseline = {"old_recipe": 1.0, "current": 4.0, "history": [2.0]}
    active = {"old_recipe": 0.99, "current": 3.0, "history": [1.0]}
    candidate = {"old_recipe": 0.98, "current": 2.5, "history": [0.95]}
    assert decide_active_stage(baseline, active, candidate, minimum_gain=0.005)["accepted"]
    assert not decide_active_stage(baseline, active, {**candidate, "old_recipe": 1.01}, minimum_gain=0.005)["accepted"]
    assert not decide_active_stage(baseline, active, {**candidate, "history": [1.01]}, minimum_gain=0.005)["accepted"]
    with pytest.raises(ValueError):
        decide_active_stage(baseline, active, {**candidate, "current": float("nan")}, minimum_gain=0.005)


def test_stage_boundary_switches_only_accepted_scorers_and_rejects_stale_models():
    from isaaclab_neural.rl.rsl_rl.active_cycle import next_active_snapshot

    before = {"path": "previous.pt", "sha256": "previous"}
    fitted = {"parent_sha256": "previous", "accepted": False, "candidate": {"path": "trial.pt", "sha256": "trial"}}
    assert next_active_snapshot(before, fitted) == before
    assert next_active_snapshot(before, {**fitted, "accepted": True}) == fitted["candidate"]
    with pytest.raises(ValueError):
        next_active_snapshot(before, {**fitted, "parent_sha256": "stale"})


def test_model_error_uses_frozen_scales_and_bounded_log_reward():
    target = torch.zeros(3, 2)
    predicted = torch.tensor([[0.0, 0.0], [1.0, 2.0], [100.0, 100.0]])
    raw, bonus = bounded_model_error(predicted, target, torch.tensor([2.0, 1.0]), cap=10.0)
    torch.testing.assert_close(raw, torch.tensor([0.0, 4.0, 25000.0]))
    assert bonus[0] == 0 and bonus[-1] == 1
    torch.testing.assert_close(bonus[1], torch.log1p(torch.tensor(4.0)) / torch.log1p(torch.tensor(10.0)))
    assert torch.isfinite(bonus).all() and ((bonus >= 0) & (bonus <= 1)).all()


def test_nonfinite_prediction_is_not_a_curiosity_windfall():
    with pytest.raises(FloatingPointError):
        bounded_model_error(torch.tensor([[float("nan")]]), torch.zeros(1, 1), torch.ones(1), cap=10.0)
    with pytest.raises(ValueError):
        bounded_model_error(torch.ones(1, 2), torch.zeros(1, 2), torch.ones(2), cap=0.0)


def test_root_leg_reward_excludes_other_channels_without_rescaling_or_mutating_checkpoint_weights():
    from isaaclab_neural.generate.active_opd import active_error_weights

    names = ["left_hip_pitch_joint", "right_knee_joint", "torso_joint", "left_shoulder_pitch_joint", "left_zero_joint"]
    width = 13 + 2 * len(names)
    weights = torch.arange(1, width + 1, dtype=torch.float32)
    original = weights.clone()
    scoped = active_error_weights(weights, names, "root_legs")
    full = active_error_weights(weights, names, "full")
    prediction = torch.zeros(2, width)
    prediction[0, [9, 10, 11, 20, 21, 22]] = 10
    prediction[1, [0, 7, 12, 18]] = 2
    target = torch.zeros_like(prediction)
    raw, bonus = bounded_model_error(prediction, target, scoped, cap=10.0)
    expected, _ = bounded_model_error(prediction, target, full, cap=10.0)
    assert raw[0] == bonus[0] == 0 and expected[0] > 0
    torch.testing.assert_close(raw[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(weights, original, rtol=0, atol=0)
    assert torch.equal(
        scoped.nonzero().flatten(), torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18, 19])
    )
    full.zero_()
    assert torch.equal(weights, original)
    with pytest.raises(ValueError):
        active_error_weights(weights, names, "typo")
    prediction[0, 11] = float("nan")
    with pytest.raises(FloatingPointError):
        bounded_model_error(prediction, target, scoped, cap=10.0)


def test_report_keeps_full_error_visible_when_reward_uses_only_selected_channels():
    from isaaclab_neural.generate.active_opd import DynamicsErrorRecorder

    observer = SimpleNamespace(
        errors=[torch.tensor([100.0, 4.0])],
        reward_errors=[torch.tensor([1.0, 2.0])],
        coordinate_error_sum=torch.tensor([207.0, 1.0], dtype=torch.float64),
        bonuses=[torch.tensor([0.1, 0.2])],
        scored_frames=[torch.tensor([4, 4])],
        cap=10.0,
        error_scope="root_legs",
        error_groups={"root": [0], "hands": [1]},
        verified_callbacks=64,
        env=SimpleNamespace(num_envs=2),
        ids=[0, 1],
        capacity=128,
    )
    report = DynamicsErrorRecorder.report(observer, clear=True)
    assert report["normalized_error_mean"] == 52 and report["normalized_error_scope"] == "full_state"
    assert report["reward_error_mean"] == 1.5 and report["cap_fraction"] == 0
    assert sum(report["group_full_loss_contribution"].values()) == 52
    assert report["group_full_loss_fraction"]["hands"] == pytest.approx(1 / 208)
    assert not observer.errors and not observer.reward_errors and not observer.coordinate_error_sum.any()


def test_terminal_and_unwarmed_worlds_cannot_receive_bonus():
    observer = SimpleNamespace(bonus_sum=torch.tensor([4.0, 4.0, 0.0, 1.0]), decimation=4)
    env = SimpleNamespace(dynamics_error_observer=observer, reset_buf=torch.tensor([False, True, False, False]))
    torch.testing.assert_close(dynamics_curiosity_reward(env), torch.tensor([1.0, 0.0, 0.0, 0.25]))


@pytest.mark.parametrize(
    "profile,base_enabled,bonus_weight", [("base", True, 0.0), ("error", False, 10.0), ("base_error", True, 10.0)]
)
def test_reward_profiles_change_rewards_without_changing_physical_contract(profile, base_enabled, bonus_weight):
    from isaaclab_neural.generate.active_opd import configure_active_collection

    from isaaclab.managers import RewardTermCfg

    rewards = SimpleNamespace(
        tracking=RewardTermCfg(func=dynamics_curiosity_reward, weight=2.0),
        penalty=RewardTermCfg(func=dynamics_curiosity_reward, weight=-1.0),
        disabled=None,
    )
    cfg = SimpleNamespace(
        rewards=rewards,
        decimation=4,
        sim=SimpleNamespace(dt=0.005, physics=SimpleNamespace(solver_cfg=SimpleNamespace(), num_substeps=2)),
        observations=SimpleNamespace(policy=SimpleNamespace()),
        terminations=object(),
        actions=object(),
    )
    terminations, actions = cfg.terminations, cfg.actions
    args = SimpleNamespace(active_reward_profile=profile, active_reward_weight=0.2)
    configure_active_collection(cfg, args)
    assert cfg.rewards.tracking.weight == (2.0 if base_enabled else 0.0)
    assert cfg.rewards.penalty.weight == (-1.0 if base_enabled else 0.0)
    assert cfg.rewards.dynamics_curiosity.weight == pytest.approx(bonus_weight)
    assert cfg.rewards.disabled is None
    assert cfg.terminations is terminations and cfg.actions is actions
    assert cfg.sim.dt == 0.005 and cfg.decimation == 4 and cfg.sim.physics.num_substeps == 2
    assert args.active_reward_audit["original_base_weights"] == {"tracking": 2.0, "penalty": -1.0}


def test_rl_coverage_collection_preserves_stock_observations_and_rewards():
    from isaaclab_neural.generate.active_opd import configure_active_collection

    rewards = SimpleNamespace(tracking=object())
    physics = SimpleNamespace(
        solver_cfg=SimpleNamespace(disable_sensors=False), num_substeps=2, deterministic_mode="not_guaranteed"
    )
    cfg = SimpleNamespace(
        rewards=rewards,
        sim=SimpleNamespace(physics=physics),
        observations=SimpleNamespace(policy=SimpleNamespace(enable_corruption=True)),
    )
    args = SimpleNamespace(active_mode="collect_benchmark")
    configure_active_collection(cfg, args)
    assert cfg.rewards is rewards and not hasattr(rewards, "dynamics_curiosity")
    assert cfg.observations.policy.enable_corruption
    assert not physics.solver_cfg.disable_sensors and physics.num_substeps == 2
    assert physics.deterministic_mode == "not_guaranteed"
    assert args.active_reward_audit["error_weight_per_control"] == 0


def test_score_audit_layout_preserves_original_batch_size_and_row_slots():
    from isaaclab_neural.generate.active_opd import _score_audit_batches

    identities = [(0, 0, 10, 0), (2, 0, 10, 0), (0, 0, 11, 0), (3, 0, 12, 0)]
    layouts = {identities[0]: (8, 0), identities[1]: (8, 2), identities[2]: (8, 0), identities[3]: (7, 3)}
    batches = list(_score_audit_batches(range(4), identities, layouts))
    observed = []
    for size, rows in batches:
        positions = [position for _, position in rows]
        assert len(positions) == len(set(positions))
        for index, position in rows:
            assert (size, position) == layouts[identities[index]]
            observed.append(index)
    assert sorted(observed) == list(range(4)) and len(batches) == 3


def test_selected_recording_worlds_rotate_without_policy_rng_or_role_leakage():
    import numpy as np
    from isaaclab_neural.generate.active_opd import active_recording_worlds

    torch_state = torch.get_rng_state()
    numpy_state = np.random.get_state()
    first = active_recording_worlds(128, 16, seed=100, iteration=0)
    second = active_recording_worlds(128, 16, seed=100, iteration=1)
    assert first == active_recording_worlds(128, 16, seed=100, iteration=0)
    assert first != second and len(set(first)) == len(set(second)) == 16
    assert sum(i < 96 for i in first) == sum(i < 96 for i in second) == 12
    assert min(first + second) >= 0 and max(first + second) < 128
    assert active_recording_worlds(8, 32, seed=100, iteration=7) == list(range(8))
    assert torch.equal(torch_state, torch.get_rng_state())
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])
    with pytest.raises(ValueError):
        active_recording_worlds(128, 0, seed=100, iteration=0)


def test_epoch_replay_keeps_first_fresh_epoch_and_reports_real_reuse():
    import numpy as np
    from isaaclab_neural.rl.rsl_rl.active_cycle import epoch_stage_batches, fresh_stage_batches

    before = np.random.get_state()
    rows = epoch_stage_batches(16, 16, updates=12, batch_size=4, seed=19)
    assert rows[:4] == fresh_stage_batches(16, 16, updates=4, batch_size=4, seed=19)
    for offset in (0, 4, 8):
        for source in ("current", "reference"):
            assert sorted(i for row in rows[offset : offset + 4] for i in row[source]) == list(range(16))
    assert rows == epoch_stage_batches(16, 16, updates=12, batch_size=4, seed=19)
    assert np.array_equal(before[1], np.random.get_state()[1])
    with pytest.raises(ValueError):
        fresh_stage_batches(16, 16, updates=12, batch_size=4, seed=19)
    with pytest.raises(ValueError):
        epoch_stage_batches(16, 12, updates=12, batch_size=4, seed=19)
    with pytest.raises(ValueError):
        epoch_stage_batches(17, 17, updates=12, batch_size=4, seed=19)


def test_preservation_stream_uses_exact_mixture_quotas_without_fake_unique_rows():
    from isaaclab_neural.rl.rsl_rl.active_cycle import preservation_batches

    counts = {"recipe": 16, "real": 8, "teacher": 32}
    weights = {"recipe": 0.5, "real": 0.25, "teacher": 0.25}
    rows = preservation_batches(counts, weights, updates=16, batch_size=8, seed=123)
    assert rows == preservation_batches(counts, weights, updates=16, batch_size=8, seed=123)
    assert all(
        {key: len(value) for key, value in row.items()} == {"recipe": 4, "real": 2, "teacher": 2} for row in rows
    )
    for key, size in counts.items():
        flattened = [i for row in rows for i in row[key]]
        assert sorted(set(flattened)) == list(range(size))
        assert len(set(flattened.count(i) for i in range(size))) == 1
    with pytest.raises(ValueError):
        preservation_batches(counts, {**weights, "recipe": 0.6}, updates=1, batch_size=8, seed=123)
    with pytest.raises(ValueError):
        preservation_batches({"tiny": 1}, {"tiny": 1.0}, updates=1, batch_size=8, seed=123)


@pytest.mark.parametrize("micro", [1, 2, 4])
def test_input_only_preservation_matches_pooled_gradient_and_keeps_teacher_frozen(micro):
    from isaaclab_neural.rl.rsl_rl.active_cycle import _backward_preservation_inputs

    class Model(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(value))

        def forward(self, inputs):
            return inputs["states"] * self.scale

    model = Model(2.0)
    teacher = Model(1.0).eval().requires_grad_(False)
    caches = {
        "recipe": {"inputs": {"states": torch.ones(4, 10, 3)}},
        "real": {"inputs": {"states": torch.full((4, 10, 3), 3.0)}},
    }
    indices = {"recipe": [0, 1, 2], "real": [3]}
    result = _backward_preservation_inputs(model, teacher, caches, indices, torch.ones(3), "cpu", micro, beta=5.0)
    assert result["loss"] == pytest.approx(3.0)
    assert result["weighted_loss"] == pytest.approx(15.0)
    assert result["sources"]["recipe"]["inputs"] == 3 and result["sources"]["real"]["inputs"] == 1
    assert float(model.scale.grad) == pytest.approx(30.0)
    assert teacher.scale.grad is None and float(teacher.scale) == 1.0
    assert torch.equal(caches["real"]["inputs"]["states"], torch.full((4, 10, 3), 3.0))


def test_progress_can_advance_without_weakening_immutable_artifacts(tmp_path):
    import json

    from isaaclab_neural.rl.rsl_rl.online_opd import _write_json, _write_status

    progress = tmp_path / "training_status.json"
    _write_status(progress, {"optimizer_updates": 0})
    _write_status(progress, {"optimizer_updates": 1})
    assert json.loads(progress.read_text())["optimizer_updates"] == 1
    assert not progress.with_suffix(".json.partial").exists()
    artifact = tmp_path / "step_0001.json"
    _write_json(artifact, {"step": 1})
    with pytest.raises(FileExistsError):
        _write_json(artifact, {"step": 2})
    assert json.loads(artifact.read_text())["step"] == 1


def test_recipe_only_control_does_not_relabel_unverified_current_data():
    import copy

    from isaaclab_neural.rl.rsl_rl.active_cycle import _validate_recipe_truth_control

    config = {"old_only_truth_control": True}
    source = {"sha256": "verified", "source_hashes": {"real.hdf5": "hash"}}
    prepared = {
        "current": source,
        "reference": copy.deepcopy(source),
        "current_inventory": {"role": "old_only_control"},
    }
    cache = {"origin": "resampled_recipe_replay"}
    _validate_recipe_truth_control(config, prepared, cache)
    with pytest.raises(ValueError):
        _validate_recipe_truth_control(config, prepared, {"origin": "teacher_rollout"})
    changed = copy.deepcopy(prepared)
    changed["current"]["sha256"] = "other"
    with pytest.raises(ValueError):
        _validate_recipe_truth_control(config, changed, cache)
    with pytest.raises(ValueError):
        _validate_recipe_truth_control(config, {**prepared, "current_inventory": {}}, cache)


def test_preservation_cache_cannot_be_validation_or_masquerade_as_truth():
    from isaaclab_neural.rl.rsl_rl.active_cycle import _validate_preservation_cache

    cache = {
        "kind": "teacher_inputs_v1",
        "teacher_sha256": "frozen",
        "split": "train",
        "input_origin": "teacher_rollout",
        "inputs": {"states": torch.zeros(2, 10, 3)},
        "source_hashes": {"training.pt": "hash"},
        "identities": ["a", "b"],
    }
    assert _validate_preservation_cache(cache, "frozen") == 2
    for changed in (
        {"split": "validation"},
        {"teacher_sha256": "another"},
        {"sequence_target": torch.zeros(2, 10, 3)},
        {"identities": ["a", "a"]},
        {"input_origin": "oracle_prediction"},
    ):
        with pytest.raises(ValueError):
            _validate_preservation_cache({**cache, **changed}, "frozen")


def test_research_fit_checkpoints_are_explicit_without_expanding_pilot_defaults():
    import copy

    from isaaclab_neural.rl.rsl_rl.active_cycle import active_fit_checkpoints

    pilot = {"updates": 128, "learning_rate": 1e-5, "check_every": 16, "history_fraction": 0.25}
    assert active_fit_checkpoints(pilot) == (1, *range(16, 129, 16))
    for changed in ({"updates": 129}, {"learning_rate": 3e-5}, {"check_every": 0}):
        with pytest.raises(ValueError):
            active_fit_checkpoints({**pilot, **changed})
    research = {
        **pilot,
        "updates": 4096,
        "learning_rate": 1e-4,
        "truth_sampling": "epoch",
        "training_manifest": "immutable_train.json",
        "defer_audit": True,
        "record_step_zero": True,
        "research_study": {
            "kind": "fixed_data_strength_v1",
            "checkpoint_updates": [128, 512, 1024, 2048, 4096],
            "minimum_free_bytes": 100 * 2**30,
            "maximum_output_bytes": 2 * 2**30,
        },
    }
    assert active_fit_checkpoints(research) == (128, 512, 1024, 2048, 4096)
    for changed in (
        {"updates": 4097},
        {"updates": True},
        {"learning_rate": 2e-4},
        {"learning_rate": float("nan")},
        {"truth_sampling": "fresh"},
        {"training_manifest": ""},
        {"defer_audit": False},
        {"record_step_zero": False},
        {"history_fraction": 0.75},
    ):
        with pytest.raises(ValueError):
            active_fit_checkpoints({**research, **changed})
    for changed in (
        {"kind": "unbounded"},
        {"checkpoint_updates": [128, 512]},
        {"checkpoint_updates": [128, 128, 4096]},
        {"checkpoint_updates": [4096, 128]},
        {"checkpoint_updates": [0, 4096]},
        {"checkpoint_updates": [4097]},
        {"checkpoint_updates": [*range(1, 10), 4096]},
        {"minimum_free_bytes": 0},
        {"maximum_output_bytes": 0},
    ):
        invalid = copy.deepcopy(research)
        invalid["research_study"].update(changed)
        with pytest.raises(ValueError):
            active_fit_checkpoints(invalid)


def test_fit_storage_guard_checks_reservation_and_never_deletes_existing_files(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from isaaclab_neural.rl.rsl_rl import active_cycle

    checkpoint = tmp_path / "preserved.pt"
    checkpoint.write_bytes(b"preserved model")
    allocated = checkpoint.stat().st_blocks * 512
    monkeypatch.setattr(active_cycle.shutil, "disk_usage", lambda path: SimpleNamespace(free=150))
    result = active_cycle.check_active_fit_storage(
        tmp_path, minimum_free_bytes=100, maximum_output_bytes=allocated + 10, reserve_bytes=10
    )
    assert result == {"free_bytes": 150, "output_bytes": allocated, "reserve_bytes": 10}
    with pytest.raises(RuntimeError, match="Storage"):
        active_cycle.check_active_fit_storage(
            tmp_path, minimum_free_bytes=100, maximum_output_bytes=allocated + 9, reserve_bytes=10
        )
    monkeypatch.setattr(active_cycle.shutil, "disk_usage", lambda path: SimpleNamespace(free=109))
    with pytest.raises(RuntimeError, match="Storage"):
        active_cycle.check_active_fit_storage(
            tmp_path, minimum_free_bytes=100, maximum_output_bytes=allocated + 10, reserve_bytes=10
        )
    assert checkpoint.read_bytes() == b"preserved model"


def test_research_arms_start_fresh_while_legacy_resume_preserves_adam_contract(monkeypatch):
    from isaaclab_neural.rl.rsl_rl import active_cycle

    model = torch.nn.Linear(2, 1)
    monkeypatch.setattr(active_cycle, "file_sha256", lambda path: "f_pre")
    config = {"learning_rate": 1e-5, "initial_dynamics": "initial.pt", "parent_sha256": "f_pre"}
    optimizer, prior = active_cycle._active_fit_optimizer(model, {"cfg": {}}, config)
    assert prior is None and not optimizer.state
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    saved = {
        "cfg": {"active_cycle": {"initial_sha256": "f_pre", "optimizer_updates_total": 1}},
        "optimizer_state_dict": optimizer.state_dict(),
    }
    resumed, prior = active_cycle._active_fit_optimizer(model, saved, config)
    assert prior["optimizer_updates_total"] == 1
    assert all(float(state["step"]) == 1 for state in resumed.state.values())
    with pytest.raises(ValueError, match="learning rate"):
        active_cycle._active_fit_optimizer(model, saved, {**config, "learning_rate": 3e-5})
    research = {**config, "research_study": {"kind": "fixed_data_strength_v1"}}
    fresh, prior = active_cycle._active_fit_optimizer(model, {"cfg": {}}, research)
    assert prior is None and not fresh.state
    with pytest.raises(ValueError, match="fresh Adam"):
        active_cycle._active_fit_optimizer(model, saved, research)
    with pytest.raises(ValueError, match="fresh Adam"):
        active_cycle._active_fit_optimizer(model, {"cfg": {}}, {**research, "parent_sha256": "fine_tuned"})


def test_coverage_features_remove_world_translation_and_yaw_without_hiding_leg_state():
    import math

    from isaaclab_neural.generate.active_opd import active_coverage_features

    names = ["left_knee_joint", "left_zero_joint"]
    state = torch.tensor([[2.0, 3.0, 0.75, 0.0, 0.0, 0.0, 1.0, 0.2, 0.8, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4, 1.2]])
    actions = torch.tensor([[0.3, 0.7]])
    contact = torch.tensor([[True, False]])
    features = active_coverage_features(state, actions, contact, names)
    expected = torch.tensor(
        [[0.75, 0.0, 0.0, -1.0, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3, math.sin(0.2), math.cos(0.2), 0.4, 0.3, 1.0, 0.0]]
    )
    torch.testing.assert_close(features, expected)
    rotated = state.clone()
    rotated[:, :2] = torch.tensor([[100.0, -200.0]])
    rotated[:, 3:7] = torch.tensor([[0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]])
    rotated[:, 9:12] = torch.tensor([[-2.0, 1.0, 3.0]])
    rotated[:, 12:15] = torch.tensor([[-0.2, 0.1, 0.3]])
    torch.testing.assert_close(active_coverage_features(rotated, actions, contact, names), features)
    modified = state.clone()
    modified[:, 8] += 1.0
    modified[:, 16] += 2.0
    changed_action = actions.clone()
    changed_action[:, 1] += 3.0
    torch.testing.assert_close(active_coverage_features(modified, changed_action, contact, names), features)
    assert not torch.equal(
        active_coverage_features(modified, changed_action, contact, names, scope="full"),
        active_coverage_features(state, actions, contact, names, scope="full"),
    )
    with pytest.raises(ValueError):
        active_coverage_features(state[:, :-1], actions, contact, names)
    with pytest.raises(ValueError):
        active_coverage_features(state, actions, contact, names, scope="unrecognized")


def test_collector_sequence_objective_retains_prefixes_with_anchor_weighted_gradients():
    from isaaclab_neural.rl.rsl_rl.active_cycle import _CollectorSequenceObjective

    parameter = torch.tensor(0.3, requires_grad=True)
    lengths = [4, 8, 12, 16, 4]
    objective = _CollectorSequenceObjective.__new__(_CollectorSequenceObjective)
    objective.records = [{"oracle_states": torch.zeros(length, 87)} for length in lengths]
    objective.batches = [list(range(len(lengths)))]
    objective.local_weight, objective.rollout_weight, objective.root_scales = 0.25, 0.4, {}
    seen = []

    def terms(records, *, max_physics_steps, root_correction_scales):
        sizes = [len(row["oracle_states"]) for row in records]
        assert len(set(sizes)) == 1 and max_physics_steps == 16
        seen.extend(sizes)
        values = torch.tensor(sizes, dtype=torch.float32)
        return {"local": parameter * values, "root_sequence": 2 * parameter * values}

    objective.student = SimpleNamespace(num_envs=2, sequence_loss_terms=terms)
    result = objective.backward(0)
    mean_length = sum(lengths) / len(lengths)
    assert sorted(seen) == sorted(lengths)
    torch.testing.assert_close(parameter.grad, torch.tensor((0.25 + 2 * 0.4) * mean_length))
    assert result["weighted_loss"] == pytest.approx(float(parameter.detach()) * (0.25 + 2 * 0.4) * mean_length)


def test_neural_collector_diagnostics_observe_current_returns_before_update():
    from contextlib import nullcontext

    from isaaclab_neural.generate.neural_collector import _collector_update

    order = []
    algorithm = SimpleNamespace(
        learning_rate=0.001,
        compute_returns=lambda obs: order.append("returns"),
        update=lambda: order.append("update") or {"value": 0.1},
    )

    def check(algorithm):
        assert order == ["returns"]
        order.append("consistency")
        return {"passed": True}

    diagnostics = SimpleNamespace(
        check_consistency=check, observe_update=lambda algorithm: nullcontext(), minibatches=[]
    )
    result = _collector_update(SimpleNamespace(alg=algorithm), {}, diagnostics)
    assert order == ["returns", "consistency", "update"]
    assert result["consistency"]["passed"]


def test_neural_collector_rewards_only_bounded_positive_optimism():
    from isaaclab_neural.generate.neural_collector import neural_collector_optimism

    truth = torch.zeros(3, 87)
    truth[:, 2], truth[:, 6] = 0.75, 1.0
    predicted = truth.clone()
    predicted[0, 2] += 0.04
    predicted[1, 2] -= 0.04
    result = neural_collector_optimism(predicted, truth, height_scale_m=0.01, upright_scale=0.1, cap=3.0)
    assert 0 < result["score"][0] <= 1
    assert result["score"][1:].tolist() == [0.0, 0.0]
    torch.testing.assert_close(result["height_gap_m"], torch.tensor([0.04, -0.04, 0.0]))
    assert not result["score"].requires_grad
    predicted[0, 2] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        neural_collector_optimism(predicted, truth, height_scale_m=0.01, upright_scale=0.1, cap=3.0)


def test_neural_collector_window_bonus_credits_the_last_control_and_masks_termination():
    from isaaclab_neural.generate.neural_collector import neural_collector_reward_matrix

    records = [
        {"env_id": 0, "frame_ids": torch.arange(96, 112), "terminal": False},
        {"env_id": 1, "frame_ids": torch.arange(112, 128), "terminal": False},
    ]
    scores = torch.tensor([0.6, 0.2])
    matrix = neural_collector_reward_matrix(records, scores, worlds=2, rollout_controls=8, first_physics_frame=96)
    expected = torch.zeros(8, 2)
    expected[3, 0], expected[7, 1] = 0.6, 0.2
    torch.testing.assert_close(matrix, expected)
    records[1]["terminal"] = True
    expected[7, 1] = 0.0
    torch.testing.assert_close(
        neural_collector_reward_matrix(records, scores, worlds=2, rollout_controls=8, first_physics_frame=96), expected
    )
    with pytest.raises(ValueError):
        neural_collector_reward_matrix(records, -scores, worlds=2, rollout_controls=8, first_physics_frame=96)
    with pytest.raises(ValueError):
        neural_collector_reward_matrix(
            [records[0], records[0]], scores, worlds=2, rollout_controls=8, first_physics_frame=96
        )
    with pytest.raises(ValueError):
        neural_collector_reward_matrix(records, scores, worlds=2, rollout_controls=4, first_physics_frame=96)


def test_oracle_query_validity_reports_rejections_without_clipping_states():
    from isaaclab_neural.generate.active_opd import active_query_validity

    states = torch.zeros(6, 15)
    states[:, 2] = 0.3
    states[:, 6] = 1.0
    states[1, 7] = 1.5
    states[2, 2] = -0.1
    states[3, 6] = 0.8
    states[5, -1] = float("nan")
    gaps = torch.tensor([0.0, 0.0, 0.0, 0.0, -0.08, 0.0])
    before = states.clone()
    checks = active_query_validity(
        states,
        torch.tensor([[-1.0, 1.0]]),
        gaps,
        minimum_root_height_m=0.02,
        maximum_penetration_m=0.05,
        joint_limit_tolerance_rad=0.001,
        quaternion_norm_tolerance=0.001,
    )
    valid = torch.stack(list(checks.values())).all(dim=0)
    assert valid.tolist() == [True, False, False, False, False, False]
    assert not checks["within_hard_joint_limits"][1]
    assert not checks["root_above_floor"][2]
    assert not checks["unit_quaternion"][3]
    assert not checks["no_deep_contact_penetration"][4]
    assert not checks["finite"][5]
    torch.testing.assert_close(states, before, rtol=0, atol=0, equal_nan=True)
    with pytest.raises(ValueError):
        active_query_validity(states[:, :-1], torch.tensor([[-1.0, 1.0]]), gaps, joint_limit_tolerance_rad=0.001)
    with pytest.raises(TypeError, match="joint_limit_tolerance_rad"):
        active_query_validity(states, torch.tensor([[-1.0, 1.0]]), gaps)
